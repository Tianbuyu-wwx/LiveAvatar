# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""DuplexSession — full-duplex star-topology session over WebSocket.

Wires a :class:`~liveavatar.runtime.worker.RealtimeWorker` (VAD / EOU /
ASR / LLM / TTS / Avatar) to the publish service's audio WebSocket,
complementing the push-mode pipeline with a true full-duplex mode::

    browser mic ─▶ WS binary (PCM 16k mono) ─▶ push_pcm ─▶ worker.input_queue
                                                             │
                                                   worker._run loop
                                             (VAD/EOU/ASR → LLM → TTS)
                                                             │
    browser ◀─ WS binary PCM ◀─ out_queue ◀─ drain loop ◀─ worker.output_queue
                + JSON events (vad/asr/eou/control/error)

Spokes are all optional and resolved from env (see
:class:`DuplexSettings`):

- ASR:   ``LIVEAVATAR_ASR_URL`` → remote RealtimeAsr microservice;
         otherwise the in-process reference ``ScriptedAsrAdapter`` (echo).
- LLM:   ``LIVEAVATAR_LLM_BASE_URL`` + ``LIVEAVATAR_LLM_MODEL`` →
         :class:`~liveavatar.text_source.OpenAIChatTextSource`; otherwise
         the ASR text goes straight to TTS (echo mode).
- TTS:   ``LIVEAVATAR_VOICE_CHAR`` → :class:`NvcStreamingTtsAdapter` on a
         :class:`VoicePool` (GPT-SoVITS in-process; set
         ``LIVEAVATAR_VOICE_DEVICE=cpu`` to stay off the GPU); otherwise
         the deterministic FakeTts.
- AEC:   ``LIVEAVATAR_AEC=1`` → NLMS echo cancellation (numpy-only).
- Avatar: ``LIVEAVATAR_DUPLEX_AVATAR=1`` → AvatarStreamingAdapter feeding
         a WebSocketSink served at ``/v1/sessions/{sid}/video``.

Epoch authority stays in the worker: ``cancel_epoch`` triggers
``worker.advance_epoch``, which cancels in-flight LLM/TTS tasks, purges
queues and (via ``avatar_adapter.cancel_epoch``) stops stale video within
one frame.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from liveavatar.runtime.metrics import SessionMetrics
from liveavatar.runtime.queues import BoundedAsyncQueue
from liveavatar.runtime.worker import RealtimeWorker
from liveavatar.spokes import (
    build_tts_adapter,
    default_voice_pool_config,
    resolve_aec,
    resolve_avatar_adapter,
    resolve_remote_asr,
    resolve_text_source,
    resolve_voice_pool,
    static_fallback_worker,
)

logger = logging.getLogger("liveavatar.duplex")

# PCM uplink framing (audio_in may be absent in light installs).
try:
    from liveavatar.audio_in.frame import PCMFrame

    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False


# 20 ms @ 16 kHz mono s16le — the canonical PCMFrame granularity.
_FRAME_MS = 20
_SAMPLES_PER_FRAME = 16000 * _FRAME_MS // 1000
_BYTES_PER_FRAME = _SAMPLES_PER_FRAME * 2


@dataclass
class DuplexSettings:
    """Full-duplex spoke configuration (env: ``LIVEAVATAR_*``)."""

    asr_url: str = ""
    enable_aec: bool = False
    char_id: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_system_prompt: str = ""
    with_avatar: bool = False

    @classmethod
    def from_env(cls) -> DuplexSettings:
        return cls(
            asr_url=os.getenv("LIVEAVATAR_ASR_URL", "").strip(),
            enable_aec=os.getenv("LIVEAVATAR_AEC", "").lower() in ("1", "true", "yes"),
            char_id=os.getenv("LIVEAVATAR_VOICE_CHAR", "").strip(),
            llm_base_url=os.getenv("LIVEAVATAR_LLM_BASE_URL", "").strip(),
            llm_api_key=os.getenv("LIVEAVATAR_LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LIVEAVATAR_LLM_MODEL", "").strip(),
            llm_system_prompt=os.getenv("LIVEAVATAR_LLM_SYSTEM_PROMPT", ""),
            with_avatar=os.getenv("LIVEAVATAR_DUPLEX_AVATAR", "").lower()
            in ("1", "true", "yes"),
        )

    def describe(self) -> dict[str, str]:
        """What the session actually wired (for the create-session response)."""
        return {
            "tts": "voice_pool" if self.char_id else "fake",
            "asr": "remote" if self.asr_url else "reference",
            "llm": "openai-chat" if (self.llm_base_url and self.llm_model) else "echo",
            "aec": "nlms" if self.enable_aec else "off",
            "avatar": "on" if self.with_avatar else "audio_only",
        }


class DuplexSession:
    """One full-duplex session: WS audio ⇄ worker spokes ⇄ WS audio+events."""

    def __init__(
        self,
        session_id: str,
        avatar_id: str,
        *,
        settings: DuplexSettings | None = None,
        worker: RealtimeWorker | None = None,
        voice_pool: Any = None,
        avatar_pool: Any = None,
        sink: Any = None,
        metrics: SessionMetrics | None = None,
        lease_renew_interval: float = 30.0,
    ) -> None:
        self.session_id = session_id
        self.avatar_id = avatar_id
        self.settings = settings or DuplexSettings.from_env()
        self.metrics = metrics or SessionMetrics(session_id)
        self.lease_renew_interval = lease_renew_interval

        # ── Spoke resolution (shared with runtime assemblies) ────────
        remote = resolve_remote_asr(self.settings.asr_url, session_id, logger=logger)
        self._asr_client: Any = remote.client if remote else None
        remote_vad = remote.vad if remote else None
        remote_eou = remote.eou if remote else None
        remote_asr = remote.asr if remote else None

        aec: Any = resolve_aec(self.settings.enable_aec, logger=logger)

        # Voice pool: externally-owned takes precedence, else construct one
        # (only meaningful with a char_id configured).
        self._voice_pool: Any = None
        self._owns_voice_pool = False
        self._tts_adapter: Any = None
        self._lease_renewer: asyncio.Task | None = None
        if voice_pool is not None:
            self._voice_pool = voice_pool
        elif self.settings.char_id:
            self._voice_pool, self._owns_voice_pool = resolve_voice_pool(
                None, default_voice_pool_config(), None
            )
        tts_for_worker = build_tts_adapter(
            self._voice_pool, session_id, self.settings.char_id
        )
        if tts_for_worker is not None:
            self._tts_adapter = tts_for_worker

        # LLM spoke.
        text_source: Any = resolve_text_source(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            system_prompt=self.settings.llm_system_prompt,
            logger=logger,
        )

        # Avatar spoke: caller passes the shared pool + a publisher sink.
        self._avatar_pool = avatar_pool
        self._avatar_adapter: Any = None
        self._avatar_lease_renewer: asyncio.Task | None = None
        self.sink = sink
        self._avatar_adapter = resolve_avatar_adapter(
            self._avatar_pool,
            session_id,
            avatar_id,
            sink,
            fallback_worker=static_fallback_worker(avatar_id),
        )

        self.worker = worker or RealtimeWorker(
            session_id,
            metrics=self.metrics,
            vad=remote_vad,
            eou=remote_eou,
            asr=remote_asr,
            aec=aec,
            tts=tts_for_worker,
            avatar_adapter=self._avatar_adapter,
            text_source=text_source,
        )
        if worker is not None and self._avatar_adapter is not None:
            self.worker.avatar_adapter = self._avatar_adapter

        # Downlink queue consumed by the audio WS sender task.
        self.out_queue: BoundedAsyncQueue[dict[str, Any]] = BoundedAsyncQueue(512)
        self._running = False
        self._drain_task: asyncio.Task | None = None
        # WS uplink frame state (see push_pcm).
        self._pcm_buf = b""
        self._pts_us = 0
        self._seq = 0
        self._frame_samples = 0

    # ------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._running:
            return
        if self._voice_pool is not None and self._owns_voice_pool:
            await self._voice_pool.start()
        if self._tts_adapter is not None:
            await self._tts_adapter.acquire()
            self._lease_renewer = asyncio.create_task(self._renew_lease_loop())
        if self._avatar_adapter is not None:
            await self._avatar_adapter.start()
            self._avatar_lease_renewer = asyncio.create_task(
                self._renew_avatar_lease_loop()
            )
        if self._asr_client is not None:
            await self._asr_client.connect()
        await self.worker.start()
        if self.sink is not None and hasattr(self.sink, "start"):
            await self.sink.start()
        self._running = True
        self._drain_task = asyncio.create_task(self._drain_output())
        logger.info(
            "duplex_session_started",
            extra={"session_id": self.session_id, **self.settings.describe()},
        )

    async def stop(self) -> None:
        if not self._running and self._drain_task is None:
            return
        self._running = False
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None
        await self.worker.stop()
        if self._lease_renewer is not None:
            self._lease_renewer.cancel()
            try:
                await self._lease_renewer
            except asyncio.CancelledError:
                pass
            self._lease_renewer = None
        if self._tts_adapter is not None:
            try:
                await self._tts_adapter.release()
            except Exception:
                logger.exception("tts_release_failed", extra={"session_id": self.session_id})
        if self._voice_pool is not None and self._owns_voice_pool:
            await self._voice_pool.stop()
        if self._avatar_lease_renewer is not None:
            self._avatar_lease_renewer.cancel()
            try:
                await self._avatar_lease_renewer
            except asyncio.CancelledError:
                pass
            self._avatar_lease_renewer = None
        if self._avatar_adapter is not None:
            try:
                await self._avatar_adapter.stop()
            except Exception:
                logger.exception(
                    "avatar_adapter_stop_failed", extra={"session_id": self.session_id}
                )
        if self.sink is not None and hasattr(self.sink, "stop"):
            try:
                await self.sink.stop()
            except Exception:
                logger.exception("sink_stop_failed", extra={"session_id": self.session_id})
        if self._asr_client is not None:
            await self._asr_client.close()
        logger.info("duplex_session_stopped", extra={"session_id": self.session_id})

    # ----------------------------------------------------------- uplink

    async def push_pcm(self, pcm: bytes) -> int:
        """Feed WS binary PCM (16 kHz mono s16le, arbitrary chunk size).

        Buffers and re-chunks into 20 ms canonical frames. Returns the
        number of complete frames pushed to the worker.
        """
        if not _HAS_AUDIO:
            return 0
        self._pcm_buf += pcm
        pushed = 0
        while len(self._pcm_buf) >= _BYTES_PER_FRAME:
            chunk = self._pcm_buf[:_BYTES_PER_FRAME]
            self._pcm_buf = self._pcm_buf[_BYTES_PER_FRAME:]
            frame = PCMFrame(
                session_id=self.session_id,
                epoch=self.worker.epoch,
                seq=self._seq,
                pts_us=self._pts_us,
                deadline_us=self._pts_us + _FRAME_MS * 1000,
                pcm_s16le=chunk,
            )
            self._seq += 1
            self._pts_us += _FRAME_MS * 1000
            self._frame_samples += _SAMPLES_PER_FRAME
            if await self.worker.push_frame(frame):
                pushed += 1
        return pushed

    def cancel_epoch(self) -> int:
        """Barge-in: advance the epoch (cancels LLM/TTS/video, purges queues)."""
        return self.worker.advance_epoch()

    # --------------------------------------------------------- downlink

    async def _drain_output(self) -> None:
        """Worker.output_queue → out_queue (binary PCM + JSON events)."""
        while self._running:
            try:
                event = await self.worker.output_queue.dequeue()
            except asyncio.CancelledError:
                break
            payload = _extract_payload(event)
            event_type = event.get("event_type", "")
            if event_type == "tts_audio":
                try:
                    pcm = bytes.fromhex(payload.get("pcm_s16le", ""))
                except (ValueError, TypeError):
                    continue
                # enqueue() is sync (returns bool).
                self.out_queue.enqueue(
                    {
                        "kind": "pcm",
                        "epoch": payload.get("epoch", 0),
                        "pts_us": payload.get("pts_us", 0),
                        "duration_us": payload.get("duration_us", 0),
                        "data": pcm,
                    }
                )
                if self._avatar_adapter is not None and not hasattr(
                    self.worker.tts, "synthesize_stream"
                ):
                    # Video fan-out is normally done inside the worker's
                    # streaming-TTS task; sync TTS (FakeTts) has no such
                    # task, so replicate it here for parity.
                    try:
                        await self._avatar_adapter.push_pcm(
                            pcm, payload.get("pts_us", 0), payload.get("epoch", 0)
                        )
                    except Exception:
                        logger.exception(
                            "avatar_push_pcm_error", extra={"session_id": self.session_id}
                        )
            else:
                self.out_queue.enqueue(
                    {"kind": "event", "event_type": event_type, "payload": payload}
                )

    # ----------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "avatar_id": self.avatar_id,
            "mode": "duplex",
            "epoch": self.worker.epoch,
            "samples_pushed": self._frame_samples,
            "worker": vars(self.worker.stats),
            "spokes": self.settings.describe(),
        }

    # ---------------------------------------------------------- internals

    async def _renew_lease_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.lease_renew_interval)
                if not self._running:
                    break
                await self._tts_adapter.acquire()
        except asyncio.CancelledError:
            pass

    async def _renew_avatar_lease_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.lease_renew_interval)
                if not self._running:
                    break
                if self._avatar_pool is not None and self._avatar_adapter is not None:
                    try:
                        await self._avatar_pool.acquire(
                            self.session_id, self._avatar_adapter._avatar_id
                        )
                    except Exception:
                        logger.exception(
                            "avatar_lease_renew_failed",
                            extra={"session_id": self.session_id},
                        )
        except asyncio.CancelledError:
            pass


def _extract_payload(event: dict[str, Any]) -> dict[str, Any]:
    """Unwrap a contracts Envelope into a flat payload dict."""
    payload = event.get("payload")
    if isinstance(payload, dict):
        key = f"{event.get('event_type', '')}_event"
        if key in payload:
            return payload[key]
    return event

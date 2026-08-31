"""LiveKit worker runtime: full duplex session orchestrator.

Wires the RealtimeWorker together with the LiveKit adapters into a single
runnable session that forms the complete star topology::

    student_mic ─▶ LiveKitParticipantAdapter ─▶ worker.input_queue
                                                    │
                                          worker._run loop
                                          (VAD/EOU/ASR/TTS)
                                                    │
                                                    ▼
    tutor track ◀─ TutorAudioPublisher ◀─ drain loop ◀─ worker.output_queue
                                                    ▲
    data_received ─▶ ControlChannelAdapter ─▶ worker.control_queue

Spokes (all injected, all optional):
- ASR: reference EnergyVad/ScriptedAsr/SilenceEou in-process, or a remote
  RealtimeAsr-compatible microservice when ``asr_url`` is provided.
- TTS: :class:`NvcStreamingTtsAdapter` backed by a VoicePool (GPT-SoVITS
  in-process) when ``voice_pool``/``voice_pool_config`` + ``char_id`` are
  provided; otherwise FakeTts.
- Avatar: video avatar pool + streaming adapter when ``avatar_pool`` etc.
  are provided; audio-only otherwise.

Epoch authority stays in the worker: ``advance_epoch`` (triggered by a
confirmed interrupt from the control channel) cancels the Tutor publisher
through the ``on_epoch_advance`` callback and the avatar adapter directly,
so stale audio AND video stop within one frame.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from liveavatar.runtime.control_channel import ControlChannelAdapter
from liveavatar.runtime.livekit_adapter import LiveKitParticipantAdapter
from liveavatar.runtime.metrics import SessionMetrics
from liveavatar.runtime.tutor_publisher import TutorAudioPublisher
from liveavatar.runtime.worker import RealtimeWorker

logger = logging.getLogger("liveavatar.runtime.livekit")

from liveavatar.spokes import (  # noqa: E402
    HAS_AVATAR,
    build_tts_adapter,
    resolve_aec,
    resolve_avatar_adapter,
    resolve_avatar_pool,
    resolve_remote_asr,
    resolve_voice_pool,
)


@dataclass
class RuntimeStats:
    """Counters for the runtime drain loop."""

    tts_chunks_published: int = 0
    other_events: int = 0
    errors: int = 0


class LiveKitWorkerRuntime:
    """Full session runtime wiring worker + LiveKit adapters + drain loop.

    Lifecycle::

        runtime = LiveKitWorkerRuntime(session_id, room_url, token)
        await runtime.start()    # connect adapters, publish tutor track, drain
        ...
        await runtime.stop()     # cancel drain, disconnect adapters, stop worker
    """

    def __init__(
        self,
        session_id: str,
        room_url: str,
        token: str,
        *,
        worker: RealtimeWorker | None = None,
        capacity: int = 100,
        metrics: SessionMetrics | None = None,
        asr_url: str | None = None,
        enable_aec: bool = False,
        voice_pool: Any = None,
        voice_pool_config: Any = None,
        voice_pool_worker_factory: Any = None,
        char_id: str | None = None,
        lease_renew_interval: float = 30.0,
        # ── Avatar (Phase 3): video avatar pool + streaming adapter ──
        avatar_pool: Any = None,
        avatar_pool_config: Any = None,
        avatar_pool_worker_factory: Any = None,
        avatar_id: str | None = None,
        avatar_adapter: Any = None,
        avatar_video_publisher: Any = None,
        avatar_fallback_worker: Any = None,
        avatar_degrade_after_errors: int = 3,
        text_source: Any = None,
    ) -> None:
        self.session_id = session_id
        self.metrics = metrics or SessionMetrics(session_id)

        # If asr_url is provided, create remote adapters backed by a shared
        # RealtimeAsrClient. The client connects in start(); the adapters are
        # injected into the worker so it uses the remote VAD/EOU/ASR pipeline.
        remote = resolve_remote_asr(asr_url, session_id, logger=logger)
        self._asr_client: Any = remote.client if remote else None
        remote_vad: Any = remote.vad if remote else None
        remote_eou: Any = remote.eou if remote else None
        remote_asr: Any = remote.asr if remote else None

        # NLMS AEC for acoustic echo cancellation (numpy-only, in-process).
        aec: Any = resolve_aec(enable_aec, logger=logger)

        # ── Voice pool + streaming TTS adapter (Step 5) ──────────────
        # Resolve the pool: an externally-owned pool (shared across sessions)
        # takes precedence; otherwise construct one from config (owned by this
        # runtime, started/stopped here). The pool is started in start() and,
        # if owned, stopped in stop().
        self._voice_pool: Any = None
        self._owns_pool: bool = False
        self._tts_adapter: Any = None
        self._lease_renew_interval = lease_renew_interval
        self._lease_renewer: asyncio.Task | None = None
        self._voice_pool, self._owns_pool = resolve_voice_pool(
            voice_pool, voice_pool_config, voice_pool_worker_factory
        )
        if self._voice_pool is None and char_id is not None:
            logger.warning(
                "char_id provided but no voice_pool/voice_pool_config; "
                "falling back to default FakeTts"
            )

        # Construct the streaming TTS adapter (pool mode) when both a pool
        # and char_id are available AND no pre-built worker was passed. If a
        # pre-built worker was supplied alongside pool+char_id, we honor the
        # worker as-is (caller manages TTS) and log a warning.
        tts_for_worker: Any = None
        if worker is None:
            tts_for_worker = build_tts_adapter(
                self._voice_pool, session_id, char_id or ""
            )
            if tts_for_worker is not None:
                self._tts_adapter = tts_for_worker
        elif char_id is not None and self._voice_pool is not None:
            logger.warning(
                "pre-built worker passed alongside voice_pool+char_id; "
                "ignoring voice pool injection (caller manages TTS)"
            )

        # ── Avatar pool + streaming adapter (Phase 3 Step 5) ──────────
        # Mirrors the voice pool ownership model: external pools are shared
        # across sessions and not stopped by the runtime; self-owned pools
        # are fully managed here. The AvatarStreamingAdapter is wired into
        # the RealtimeWorker so each TTS PCM chunk is fanned out to the
        # avatar worker in parallel with audio playback.
        self._avatar_pool: Any = None
        self._owns_avatar_pool: bool = False
        self._avatar_adapter: Any = None
        self._avatar_video_publisher: Any = None
        self._avatar_lease_renewer: asyncio.Task | None = None
        self._avatar_degrade_after_errors = avatar_degrade_after_errors

        self._avatar_pool, self._owns_avatar_pool = resolve_avatar_pool(
            avatar_pool, avatar_pool_config, avatar_pool_worker_factory
        )
        if self._avatar_pool is None and avatar_id is not None:
            logger.warning(
                "avatar_id provided but no avatar_pool/avatar_pool_config; "
                "running in audio-only mode"
            )

        # Pre-built adapter (caller-managed) takes precedence — useful when
        # the runtime shouldn't own the adapter lifecycle (e.g. tests).
        if avatar_adapter is not None:
            self._avatar_adapter = avatar_adapter
        elif avatar_id is not None:
            adapter = resolve_avatar_adapter(
                self._avatar_pool,
                session_id,
                avatar_id,
                avatar_video_publisher,
                fallback_worker=avatar_fallback_worker,
                degrade_after_errors=avatar_degrade_after_errors,
            )
            if adapter is not None:
                self._avatar_adapter = adapter
                self._avatar_video_publisher = avatar_video_publisher

        self.worker = worker or RealtimeWorker(
            session_id,
            capacity=capacity,
            metrics=self.metrics,
            vad=remote_vad,
            eou=remote_eou,
            asr=remote_asr,
            aec=aec,
            tts=tts_for_worker,
            avatar_adapter=self._avatar_adapter,
            text_source=text_source,
        )
        # If a pre-built worker was passed, attach metrics so interrupt/flush
        # tracing is active even on custom workers. Also wire the avatar
        # adapter onto a pre-built worker so PCM fan-out still happens.
        self.worker.metrics = self.metrics
        if (
            worker is not None
            and self._avatar_adapter is not None
            and getattr(self.worker, "avatar_adapter", None) is None
        ):
            self.worker.avatar_adapter = self._avatar_adapter
        self.adapter = LiveKitParticipantAdapter(
            self.worker, session_id, room_url, token, metrics=self.metrics
        )
        self.publisher: TutorAudioPublisher | None = None
        self.control: ControlChannelAdapter | None = None
        self._drain_task: asyncio.Task | None = None
        self._running = False
        self.stats = RuntimeStats()

    async def start(self) -> None:
        """Connect all adapters, wire the epoch callback, and start draining."""
        if self._running:
            return
        # Start the voice pool (owned case) and acquire the character lease
        # before the worker begins processing, so the streaming TTS adapter
        # has a bound worker for the first ASR final.
        if self._voice_pool is not None and self._owns_pool:
            await self._voice_pool.start()
        if self._tts_adapter is not None:
            await self._tts_adapter.acquire()
            # Renew the lease periodically so long conversations don't lose
            # their worker to the reaper.
            self._lease_renewer = asyncio.create_task(self._renew_lease_loop())

        # Start the avatar pool (owned case) and acquire the lease before the
        # worker begins, so the first TTS PCM chunk has a bound avatar worker.
        if self._avatar_pool is not None and self._owns_avatar_pool:
            await self._avatar_pool.start()
        if self._avatar_adapter is not None:
            await self._avatar_adapter.start()
            self._avatar_lease_renewer = asyncio.create_task(
                self._renew_avatar_lease_loop()
            )

        # Connect the remote ASR client first (if configured) so the worker's
        # adapters are ready before the first frame arrives.
        if self._asr_client is not None:
            await self._asr_client.connect()
        await self.worker.start()
        await self.adapter.start()

        # Tutor publisher shares the adapter's room local participant.
        self.publisher = TutorAudioPublisher(
            self.adapter.local_participant, self.session_id, metrics=self.metrics
        )
        await self.publisher.start()

        # Avatar video publisher shares the same room participant. Caller
        # may also pass a pre-built publisher (e.g. tests with a fake).
        if self._avatar_video_publisher is not None and HAS_AVATAR:
            # If the publisher was passed in but not yet started, wire it to
            # the adapter's local participant and start it.
            if self._avatar_video_publisher.local_participant is None:
                self._avatar_video_publisher.local_participant = (
                    self.adapter.local_participant
                )
            try:
                await self._avatar_video_publisher.start()
            except Exception:
                logger.exception(
                    "avatar_video_publisher_start_failed",
                    extra={"session_id": self.session_id},
                )

        # Worker is the epoch authority; advance_epoch cancels both the
        # Tutor audio publisher and (via the worker's avatar_adapter field)
        # the Avatar video publisher. The Tutor publisher's cancel_epoch is
        # hooked here; the avatar adapter is cancelled directly by the
        # worker's advance_epoch (which calls avatar_adapter.cancel_epoch).
        self.worker.on_epoch_advance = self.publisher.cancel_epoch

        # Control channel shares the adapter's room.
        self.control = ControlChannelAdapter(
            self.worker, self.adapter.room, self.session_id
        )
        self.control.start()

        self._running = True
        self._drain_task = asyncio.create_task(self._drain_output())
        logger.info(
            "runtime_started",
            extra={
                "session_id": self.session_id,
                "avatar_enabled": self._avatar_adapter is not None,
            },
        )

    async def _drain_output(self) -> None:
        """Drain worker.output_queue: tts_audio → publisher, else count."""
        while self._running:
            try:
                event = await self.worker.output_queue.dequeue()
            except asyncio.CancelledError:
                break
            event_type = event.get("event_type")
            if event_type == "tts_audio":
                await self._handle_tts_audio(event)
            elif event_type == "error":
                self.stats.errors += 1
            else:
                self.stats.other_events += 1

    def _extract_payload(self, event: dict[str, Any]) -> dict[str, Any]:
        """Extract the nested payload from a RealtimeContracts Envelope.

        Supports both the new ``payload.{event_type}_event`` wrapper and the
        legacy flat dict for backward compatibility.
        """
        payload = event.get("payload")
        if isinstance(payload, dict):
            key = f"{event.get('event_type', '')}_event"
            if key in payload:
                return payload[key]
        return event

    async def _handle_tts_audio(self, event: dict[str, Any]) -> None:
        """Decode a tts_audio event and publish it as a Tutor chunk.

        Also feeds the PCM to the AEC far-end reference and marks the tutor
        as speaking (for PTT mode).
        """
        payload = self._extract_payload(event)
        pcm_hex = payload.get("pcm_s16le", "")
        try:
            pcm = bytes.fromhex(pcm_hex)
        except (ValueError, TypeError):
            self.stats.errors += 1
            return
        epoch = payload.get("epoch", 0)

        # Feed far-end reference to AEC (for echo cancellation).
        self.worker.push_far_end(pcm)

        # Mark tutor as speaking for PTT mode. Schedule a reset after the
        # chunk duration so PTT suppresses mic during playback.
        duration_us = payload.get("duration_us", 20000)
        self.worker.set_tutor_speaking(True)
        asyncio.create_task(self._reset_tutor_speaking(duration_us))

        if self.publisher is not None:
            await self.publisher.publish_chunk(pcm, epoch)
        self.stats.tts_chunks_published += 1

    async def _reset_tutor_speaking(self, delay_us: int) -> None:
        """Reset tutor_speaking flag after a delay (for PTT mode)."""
        try:
            await asyncio.sleep(delay_us / 1_000_000)
            self.worker.set_tutor_speaking(False)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Tear down in reverse order: drain, control, publisher, adapter, worker."""
        self._running = False
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None
        if self.control is not None:
            await self.control.stop()
        if self.publisher is not None:
            await self.publisher.stop()
        # Stop the Avatar video publisher (unpublish video track) before the
        # adapter — frames in flight will be dropped by the publisher's
        # epoch check (epoch advanced by worker.stop → advance_epoch).
        if self._avatar_video_publisher is not None:
            try:
                await self._avatar_video_publisher.stop()
            except Exception:
                logger.exception(
                    "avatar_video_publisher_stop_failed",
                    extra={"session_id": self.session_id},
                )
        await self.adapter.stop()
        await self.worker.stop()
        # Release the voice lease after the worker stops so no in-flight
        # synthesis is orphaned, then stop the pool if we own it.
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
                logger.exception(
                    "tts_adapter_release_failed",
                    extra={"session_id": self.session_id},
                )
        if self._voice_pool is not None and self._owns_pool:
            await self._voice_pool.stop()
        # Release the avatar lease and stop the avatar pool (owned case).
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
                    "avatar_adapter_stop_failed",
                    extra={"session_id": self.session_id},
                )
        if self._avatar_pool is not None and self._owns_avatar_pool:
            await self._avatar_pool.stop()
        # Close the remote ASR client last (after the worker stops sending).
        if self._asr_client is not None:
            await self._asr_client.close()
        # Emit final metrics summary + queue high-water marks.
        self._log_metrics_summary()
        logger.info(
            "runtime_stopped",
            extra={"session_id": self.session_id},
        )

    async def _renew_lease_loop(self) -> None:
        """Periodically renew the voice lease so the reaper doesn't reclaim it.

        ``NvcStreamingTtsAdapter.acquire()`` renews when a lease is already
        held. Runs until cancelled by :meth:`stop`.
        """
        try:
            while self._running:
                await asyncio.sleep(self._lease_renew_interval)
                if not self._running:
                    break
                await self._tts_adapter.acquire()
        except asyncio.CancelledError:
            pass

    async def _renew_avatar_lease_loop(self) -> None:
        """Periodically renew the avatar lease so the reaper doesn't reclaim it.

        Calls ``avatar_pool.acquire(session_id, avatar_id)`` which renews an
        existing lease in-place (mirrors VoicePool semantics). Runs until
        cancelled by :meth:`stop`.
        """
        try:
            while self._running:
                await asyncio.sleep(self._lease_renew_interval)
                if not self._running:
                    break
                if self._avatar_pool is not None and self._avatar_adapter is not None:
                    # acquire() renews when a lease already exists.
                    try:
                        await self._avatar_pool.acquire(
                            self.session_id,
                            self._avatar_adapter._avatar_id,
                        )
                    except Exception:
                        logger.exception(
                            "avatar_lease_renew_failed",
                            extra={"session_id": self.session_id},
                        )
        except asyncio.CancelledError:
            pass

    def _log_metrics_summary(self) -> None:
        """Log the session metrics summary and queue high-water marks."""
        summary = self.metrics.summary()
        summary["input_queue_high_water"] = self.worker.input_queue.stats.high_water
        summary["output_queue_high_water"] = self.worker.output_queue.stats.high_water
        summary["control_queue_high_water"] = self.worker.control_queue.stats.high_water
        logger.info("session_metrics_summary", extra=summary)

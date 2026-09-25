# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Reference orchestration worker.

REF-2 split — the :class:`RealtimeWorker` hub lives here (state, lifecycle,
queues, epoch authority, main loop); the inbound frame/control handling is
mixed in from :mod:`liveavatar.runtime.ingest` and the outbound LLM→TTS
turn path from :mod:`liveavatar.runtime.llm_turn`. Both are pure code
moves: the public interface and the epoch semantics are unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from liveavatar.runtime.contracts import Envelope, ErrorEvent, EventType
from liveavatar.runtime.fake_tts import FakeTts
from liveavatar.runtime.ingest import IngestMixin
from liveavatar.runtime.llm_turn import LlmTurnMixin
from liveavatar.runtime.queues import BoundedAsyncQueue
from liveavatar.runtime.valley import find_valley_cut

logger = logging.getLogger("liveavatar.runtime.worker")

# Epoch-advance callback type: called with the new epoch after a barge-in.
EpochAdvanceCallback = Callable[[int], None]

# Optional sibling import; RealtimeAudio path is added by paths.py
try:
    from liveavatar.audio_in.frame import PCMFrame
    from liveavatar.audio_in.reference.asr import ScriptedAsrAdapter
    from liveavatar.audio_in.reference.eou import SilenceEouDetector
    from liveavatar.audio_in.reference.vad import EnergyVad

    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False
    PCMFrame = Any  # type: ignore


@dataclass
class WorkerStats:
    input_frames: int = 0
    asr_events: int = 0
    vad_events: int = 0
    eou_events: int = 0
    tts_segments: int = 0
    cancelled_segments: int = 0
    control_events_out: int = 0
    playback_acks: int = 0
    # AEC / PTT counters.
    aec_frames: int = 0
    aec_bypassed: int = 0
    ptt_suppressed: int = 0


class RealtimeWorker(IngestMixin, LlmTurnMixin):
    """Single-session reference worker.

    Runs an asyncio loop consuming PCM frames and emitting ASR/VAD/EOU/control events.

    Adapters (VAD/EOU/ASR) can be injected for testing or for remote backends.
    If not provided, the reference implementations (EnergyVad / ScriptedAsr /
    SilenceEou) are used when ``realtime_audio`` is importable.
    """

    def __init__(
        self,
        session_id: str,
        capacity: int = 100,
        *,
        metrics: Any = None,
        vad: Any = None,
        eou: Any = None,
        asr: Any = None,
        aec: Any = None,
        tts: Any = None,
        avatar_adapter: Any = None,
        text_source: Any = None,
        history_limit: int = 12,
    ) -> None:
        self.session_id = session_id
        self.epoch = 0
        self.input_queue: BoundedAsyncQueue[Any] = BoundedAsyncQueue(capacity)
        self.output_queue: BoundedAsyncQueue[dict[str, Any]] = BoundedAsyncQueue(capacity)
        self.control_queue: BoundedAsyncQueue[dict[str, Any]] = BoundedAsyncQueue(capacity)
        self.stats = WorkerStats()
        self.metrics = metrics
        self._running = False
        self._task: asyncio.Task | None = None
        self._vad_active = False
        # Monotonic sequence number for outgoing Envelope events.
        self._seq_counter = 0
        # Active streaming-TTS background tasks (Step 4). Each ASR final
        # spawns one task that consumes ``tts.synthesize_stream()`` and
        # enqueues ``tts_audio`` events incrementally. Cancelled on epoch
        # advance / stop for prompt interrupt.
        self._tts_tasks: set[asyncio.Task] = set()
        # AEC (Acoustic Echo Cancellation) — optional. When provided, each
        # mic frame is processed through AEC before VAD/EOU/ASR.
        self._aec = aec
        # Headphone mode — when True, AEC is bypassed (no acoustic echo).
        # Set via control channel from the browser.
        self._headphones: bool = False
        # PTT (Push-to-Talk) mode — when True, mic frames are suppressed
        # during Tutor playback (walkie-talkie mode).
        self._ptt_mode: bool = False
        # Tutor speaking flag — set by the runtime when TTS is published.
        self._tutor_speaking: bool = False
        # Epoch-advance callback (called with the new epoch). Set by the
        # orchestrator to e.g. cancel the Tutor publisher. Keeps the worker
        # as the single source of truth for epoch authority.
        self.on_epoch_advance: EpochAdvanceCallback | None = None
        # P-C: latest client-reported playback position (from playback_ack
        # events) and the last energy-valley cut decision, carried on the
        # flush control event so a downstream player can play the pending
        # tail up to the valley before stopping.
        self._consumed_pts_us: int = 0
        self._last_valley_cut: dict[str, Any] | None = None

        # Use injected adapters, or fall back to reference implementations.
        if vad is not None:
            self.vad = vad
        elif _HAS_AUDIO:
            self.vad = EnergyVad(threshold_db=-50.0, release_db=-55.0)
        else:
            self.vad = None

        if eou is not None:
            self.eou = eou
        elif _HAS_AUDIO:
            self.eou = SilenceEouDetector(silence_needed_us=400000)
        else:
            self.eou = None

        if asr is not None:
            self.asr = asr
        elif _HAS_AUDIO:
            self.asr = ScriptedAsrAdapter()
        else:
            self.asr = None

        # TTS: inject an async-capable adapter (NvcStreamingTtsAdapter) for
        # streaming synthesis, or fall back to the deterministic FakeTts.
        # The worker auto-detects ``synthesize_stream`` and routes to the
        # background-task path when available (Step 4).
        self.tts = tts if tts is not None else FakeTts()
        # Avatar adapter (Phase 3 Step 4): when provided, each TTS PCM chunk
        # is forwarded to ``avatar_adapter.push_pcm`` so the Avatar worker
        # produces synchronized video frames in parallel with audio playback.
        # ``None`` means audio-only mode (no video track).
        self.avatar_adapter = avatar_adapter
        # LLM spoke (TextSource protocol): when provided, an ASR final is
        # first streamed through ``text_source.stream_text`` and split into
        # sentence-bounded pieces, each dispatched to TTS independently —
        # first-audio latency overlaps LLM generation. ``None`` means the
        # ASR text goes straight to TTS (echo mode, reference behavior).
        self.text_source = text_source
        # Rolling dialogue history (user/assistant turns) passed to the
        # TextSource. Capped at ``history_limit`` messages.
        self._history: list[dict[str, str]] = []
        self.history_limit = max(2, history_limit)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        # Cancel any in-flight streaming-TTS tasks before stopping the main
        # loop so they don't outlive the worker.
        self._cancel_tts_tasks()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def push_frame(self, frame: Any) -> bool:
        return self.input_queue.enqueue(frame, epoch=getattr(frame, "epoch", 0))

    async def push_control(self, event: dict[str, Any]) -> bool:
        return self.control_queue.enqueue(event)

    def _next_seq(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def _make_envelope(
        self, event_type: EventType, payload: Any, *, pts_us: int = 0
    ) -> dict[str, Any]:
        """Build an Envelope dict for the output queue."""
        return Envelope(
            session_id=self.session_id,
            event_type=event_type,
            turn=1,
            epoch=self.epoch,
            seq=self._next_seq(),
            pts_us=pts_us,
            payload=payload,
        ).to_dict()

    def advance_epoch(self) -> int:
        m = self.metrics
        # P-C: pick the energy-valley cut BEFORE the cancel drops the
        # pending tail (needs the old-epoch segments still alive). Pure
        # numpy over ≤ a few hundred ms of PCM — well inside the
        # interrupt latency budget.
        valley = self._pick_valley_cut()
        self._last_valley_cut = valley
        if m is not None and valley is not None:
            m.record_valley_cut(bool(valley["found"]), float(valley["rollback_ms"]))
        self.epoch += 1
        self.input_queue.advance_epoch(self.epoch)
        self.output_queue.advance_epoch(self.epoch)
        # P-A timeline: queues purged for the new epoch.
        if m is not None:
            m.timeline_mark("audio_flush")
        self.tts.cancel_epoch(self.epoch)
        # Promptly cancel in-flight streaming-TTS tasks so torch inference
        # stops ASAP (the adapter's cancel_token also breaks the generator,
        # but cancelling the asyncio task wakes it immediately). Kept before
        # the avatar cancel so the P-A timeline layers (tts_stop →
        # video_invalidate) match true execution order.
        self._cancel_tts_tasks()
        # P-A timeline: TTS cancel requested + in-flight tasks cancelled.
        if m is not None:
            m.timeline_mark("tts_stop")
        # Cancel Avatar video inference in lockstep with TTS so an interrupt
        # stops both audio and video within one frame. The adapter forwards
        # the new epoch to the AvatarVideoPublisher (drops stale-epoch frames
        # already in capture) and cancels the in-flight CancelToken (breaks
        # the worker's synthesize_video_stream generator promptly).
        if self.avatar_adapter is not None:
            self.avatar_adapter.cancel_epoch(self.epoch)
            # P-A timeline: stale-epoch frames invalidated at the adapter/sink.
            if m is not None:
                m.timeline_mark("video_invalidate")
        if self.asr:
            self.asr.advance_epoch(self.epoch)
        # Reset VAD and EOU state for the new epoch.
        if self.vad:
            self.vad.reset()
        if self.eou:
            self.eou.reset()
        # Reset AEC filter state for the new epoch.
        if self._aec:
            self._aec.reset()
        if self.on_epoch_advance:
            try:
                self.on_epoch_advance(self.epoch)
            except Exception:
                pass
        return self.epoch

    def _pick_valley_cut(self) -> dict[str, Any] | None:
        """P-C: energy-valley cut decision for the pending TTS tail.

        Returns ``None`` when there is nothing to refine (no TTS adapter
        support for ``pending_audio_tail``, or no pending audio) — the
        flush then keeps the plain hard-cut semantics.
        """
        tts = self.tts
        getter = getattr(tts, "pending_audio_tail", None)
        if not callable(getter):
            return None
        try:
            pending = getter(self._consumed_pts_us)
        except Exception:
            logger.exception(
                "valley_pending_audio_error",
                extra={"session_id": self.session_id},
            )
            return None
        if not pending:
            return None
        sr = int(getattr(tts, "sample_rate", 16000) or 16000)
        cut = find_valley_cut(pending, sample_rate=sr)
        return {
            "found": cut.found,
            "reason": cut.reason,
            "rollback_ms": round(cut.rollback_ms, 2),
            # Absolute point on the playback timeline where the pending
            # tail should stop — actionable for a downstream player.
            "cut_pts_us": self._consumed_pts_us + int(cut.cut_sample / sr * 1_000_000),
        }

    # ----------------------------------------------------- AEC / PTT / phones

    def push_far_end(self, pcm: bytes) -> None:
        """Feed Tutor audio as far-end reference to the AEC filter.

        Called by the runtime whenever TTS audio is published to the Tutor
        track. The AEC uses this to estimate and cancel the acoustic echo.
        """
        if self._aec:
            self._aec.push_far_end(pcm)

    def set_headphones(self, value: bool) -> None:
        """Enable/disable headphone mode (bypasses AEC when True)."""
        self._headphones = value
        if self._aec:
            self._aec.enabled = not value

    def set_ptt_mode(self, value: bool) -> None:
        """Enable/disable Push-to-Talk mode."""
        self._ptt_mode = value

    def set_tutor_speaking(self, value: bool) -> None:
        """Mark whether the Tutor is currently playing audio (for PTT)."""
        self._tutor_speaking = value

    async def _run(self) -> None:
        while self._running:
            try:
                # Prefer control events over audio frames.
                control = self.control_queue.try_dequeue()
                if control:
                    await self._handle_control(control)

                frame = self.input_queue.try_dequeue()
                if frame is None:
                    await asyncio.wait_for(self.input_queue._event.wait(), timeout=0.05)
                    continue

                if getattr(frame, "epoch", 0) < self.epoch:
                    continue

                self.stats.input_frames += 1
                await self._process_frame(frame)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.output_queue.enqueue(
                    self._make_envelope(
                        "error",
                        ErrorEvent(source="worker._run", message=str(exc)),
                    ),
                    epoch=self.epoch,
                )

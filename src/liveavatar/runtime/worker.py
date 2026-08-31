"""Reference orchestration worker."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from liveavatar.runtime.contracts import (
    AsrEvent,
    ControlEvent,
    Envelope,
    EouEvent,
    ErrorEvent,
    EventType,
    InterruptEvent,
    PlaybackAck,
    TtsAudioEvent,
    VadEvent,
    WordTiming,
)
from liveavatar.runtime.fake_tts import FakeTts
from liveavatar.runtime.queues import BoundedAsyncQueue
from liveavatar.text_source import sentence_stream

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


class RealtimeWorker:
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
        self.epoch += 1
        self.input_queue.advance_epoch(self.epoch)
        self.output_queue.advance_epoch(self.epoch)
        self.tts.cancel_epoch(self.epoch)
        # Cancel Avatar video inference in lockstep with TTS so an interrupt
        # stops both audio and video within one frame. The adapter forwards
        # the new epoch to the AvatarVideoPublisher (drops stale-epoch frames
        # already in capture) and cancels the in-flight CancelToken (breaks
        # the worker's synthesize_video_stream generator promptly).
        if self.avatar_adapter is not None:
            self.avatar_adapter.cancel_epoch(self.epoch)
        # Promptly cancel in-flight streaming-TTS tasks so torch inference
        # stops ASAP (the adapter's cancel_token also breaks the generator,
        # but cancelling the asyncio task wakes it immediately).
        self._cancel_tts_tasks()
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

    async def _handle_control(self, event: dict[str, Any]) -> None:
        intent = event.get("intent")
        if intent == "cancel_and_flush" or event.get("kind") == "confirmed":
            # Record interrupt start (before advance_epoch, which cancels
            # publisher + TTS + purges queues).
            if self.metrics is not None:
                self.metrics.record_interrupt()
                logger.info(
                    "trace_interrupt",
                    extra={
                        "session_id": self.session_id,
                        "old_epoch": self.epoch,
                        "interrupt_count": self.metrics.interrupt_count,
                    },
                )
            old = self.epoch
            self.advance_epoch()
            self.stats.cancelled_segments += self.tts.cancel_epoch(self.epoch)
            self.output_queue.enqueue(
                self._make_envelope(
                    "control",
                    ControlEvent(
                        intent="flush",
                        metadata={"old_epoch": old, "new_epoch": self.epoch},
                    ),
                ),
                epoch=self.epoch,
            )
            self.stats.control_events_out += 1
            # Record flush completion (interrupt-to-flush window closed).
            if self.metrics is not None:
                itf_ms = self.metrics.record_flush()
                logger.info(
                    "trace_flush",
                    extra={
                        "session_id": self.session_id,
                        "old_epoch": old,
                        "new_epoch": self.epoch,
                        "interrupt_to_flush_ms": itf_ms,
                    },
                )
        elif intent == "duck" or event.get("kind") == "provisional":
            self.output_queue.enqueue(
                self._make_envelope(
                    "interrupt",
                    InterruptEvent(kind="provisional", duck_gain=0.3),
                ),
                epoch=self.epoch,
            )
            self.stats.control_events_out += 1
        elif intent == "playback_ack":
            self.stats.playback_acks += 1
            self.output_queue.enqueue(
                self._make_envelope(
                    "playback_ack",
                    PlaybackAck(
                        segment_seq=event.get("segment_seq", 0),
                        consumed_pts_us=event.get("consumed_pts_us", 0),
                    ),
                ),
                epoch=self.epoch,
            )
        elif intent == "set_headphones":
            self.set_headphones(bool(event.get("value", False)))
        elif intent == "set_ptt":
            self.set_ptt_mode(bool(event.get("value", False)))
        elif intent == "close":
            self._running = False

    async def _process_frame(self, frame: Any) -> None:
        if not _HAS_AUDIO:
            return

        # PTT (Push-to-Talk): suppress mic during Tutor playback.
        if self._ptt_mode and self._tutor_speaking:
            self.stats.ptt_suppressed += 1
            return

        # AEC (Acoustic Echo Cancellation): cancel echo before VAD/EOU/ASR.
        # Bypassed when headphones are detected (no acoustic echo).
        if self._aec is not None and not self._headphones:
            cleaned = self._aec.process(frame.pcm_s16le)
            frame = dataclasses.replace(frame, pcm_s16le=cleaned)
            self.stats.aec_frames += 1
        elif self._aec is not None:
            self.stats.aec_bypassed += 1

        vad_events = self.vad.push_frame(frame)
        for ve in vad_events:
            self.stats.vad_events += 1
            self._vad_active = ve["kind"] == "speech_start"
            self.output_queue.enqueue(
                self._make_envelope(
                    "vad",
                    VadEvent(
                        kind=ve.get("kind", "speech_start"), energy_db=ve.get("energy_db", -96.0)
                    ),
                    pts_us=frame.pts_us,
                ),
                epoch=self.epoch,
            )

        eou_events = self.eou.push_frame(frame, vad_active=self._vad_active)
        for ee in eou_events:
            self.stats.eou_events += 1
            self.output_queue.enqueue(
                self._make_envelope(
                    "eou",
                    EouEvent(
                        confidence=ee.get("confidence", 0.0), silence_us=ee.get("silence_us", 0)
                    ),
                    pts_us=frame.pts_us,
                ),
                epoch=self.epoch,
            )

        asr_events = self.asr.push_frame(frame)
        for ae in asr_events:
            self.stats.asr_events += 1
            self.output_queue.enqueue(
                self._make_envelope(
                    "asr",
                    AsrEvent(
                        phase=ae.get("phase", "partial"),
                        text=ae.get("text", ""),
                        stability=ae.get("stability", 0.0),
                        revision=ae.get("revision", 0),
                        words=[WordTiming(**w) for w in ae.get("words", [])],
                    ),
                    pts_us=frame.pts_us,
                ),
                epoch=self.epoch,
            )
            if ae.get("phase") == "final":
                if self.text_source is not None:
                    task = asyncio.create_task(
                        self._run_llm_turn(ae["text"], self.epoch, frame.pts_us)
                    )
                    self._tts_tasks.add(task)
                    task.add_done_callback(self._tts_tasks.discard)
                else:
                    self._dispatch_tts(ae["text"], self.epoch, frame.pts_us)

    # ----------------------------------------------------- TTS dispatch

    async def _run_llm_turn(self, text: str, epoch: int, pts_us: int) -> None:
        """Background task: stream the utterance through the LLM spoke.

        Consumes ``text_source.stream_text`` incrementally, splits the
        stream into sentence-bounded pieces (``sentence_stream``) and
        dispatches each piece to TTS — first-audio starts before the LLM
        finishes. Cancelled on epoch advance / stop (registered in
        ``_tts_tasks``). The completed exchange is appended to the rolling
        history passed to subsequent turns.
        """
        history = list(self._history)
        reply_parts: list[str] = []
        try:
            async for piece in sentence_stream(
                self.text_source.stream_text(text, history=history)
            ):
                if epoch < self.epoch or not self._running:
                    return
                reply_parts.append(piece)
                self._dispatch_tts(piece, epoch, pts_us)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "llm_turn_error",
                extra={"session_id": self.session_id, "epoch": epoch},
            )
            self.output_queue.enqueue(
                self._make_envelope(
                    "error",
                    ErrorEvent(
                        source="worker._run_llm_turn",
                        message=f"llm_turn: {exc}",
                    ),
                ),
                epoch=self.epoch,
            )
            return
        reply = "".join(reply_parts).strip()
        if reply:
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": reply})
            excess = len(self._history) - self.history_limit
            if excess > 0:
                del self._history[:excess]

    def _dispatch_tts(self, text: str, epoch: int, pts_us: int) -> None:
        """Route an ASR-final utterance to the TTS backend.

        If the TTS backend exposes ``synthesize_stream`` (async generator),
        spawn a background task that consumes it incrementally — the event
        loop stays free for ASR/VAD/EOU/control processing between chunks
        (Step 4 non-blocking path).

        Otherwise fall back to the synchronous ``synthesize()`` call that
        returns a full segment list (FakeTts compat / transitional path).
        """
        if hasattr(self.tts, "synthesize_stream") and callable(
            getattr(self.tts, "synthesize_stream", None)
        ):
            task = asyncio.create_task(
                self._run_tts_stream(text, epoch, pts_us)
            )
            self._tts_tasks.add(task)
            task.add_done_callback(self._tts_tasks.discard)
            return

        # Sync path (FakeTts): blocks the loop, returns full list.
        segments = self.tts.synthesize(text, epoch, pts_us)
        for seg in segments:
            self._emit_tts_segment(seg)

    async def _run_tts_stream(
        self, text: str, epoch: int, pts_us: int
    ) -> None:
        """Background task: consume ``synthesize_stream`` and emit segments.

        The async generator runs torch inference in a worker thread (via
        ``asyncio.to_thread`` inside ``NvcWorker``), so this task merely
        awaits each chunk. Between chunks we check ``self.epoch`` so a
        confirmed interrupt that bumped the epoch stops emission promptly
        even if the cancel_token hasn't fired yet.

        Avatar fan-out (Phase 3 Step 4): when ``self.avatar_adapter`` is
        set, each PCM chunk is forwarded to ``avatar_adapter.push_pcm``
        so the Avatar worker produces synchronized video frames in
        parallel with audio playback. Audio remains the master clock:
        ``push_pcm`` is non-blocking (drops on backpressure) so a slow
        Avatar worker never slows TTS emission.
        """
        try:
            async for seg in self.tts.synthesize_stream(text, epoch, pts_us):
                if epoch < self.epoch:
                    # Epoch advanced mid-stream — stop emitting; the
                    # already-produced segments are reaped by cancel_epoch.
                    break
                self._emit_tts_segment(seg)
                # Fan out PCM to the Avatar inference pipeline. Best-effort:
                # returns False on stale-epoch or queue full — neither
                # should block or break TTS emission.
                if self.avatar_adapter is not None:
                    try:
                        await self.avatar_adapter.push_pcm(
                            seg.pcm_s16le, seg.pts_us, seg.epoch
                        )
                    except Exception:
                        logger.exception(
                            "avatar_push_pcm_error",
                            extra={
                                "session_id": self.session_id,
                                "epoch": seg.epoch,
                                "segment_seq": seg.segment_seq,
                            },
                        )
        except asyncio.CancelledError:
            # Raised by advance_epoch/stop for prompt interrupt.
            # Segment accounting is handled by tts.cancel_epoch(); nothing
            # to increment here (the in-flight partial chunk is abandoned).
            raise
        except Exception as exc:
            logger.exception(
                "tts_stream_error",
                extra={"session_id": self.session_id, "epoch": epoch},
            )
            self.output_queue.enqueue(
                self._make_envelope(
                    "error",
                    ErrorEvent(
                        source="worker._run_tts_stream",
                        message=f"tts_stream: {exc}",
                    ),
                ),
                epoch=self.epoch,
            )

    def _emit_tts_segment(self, seg: Any) -> None:
        """Enqueue one TTS segment as a ``tts_audio`` output event."""
        self.stats.tts_segments += 1
        self.output_queue.enqueue(
            self._make_envelope(
                "tts_audio",
                TtsAudioEvent(
                    segment_seq=seg.segment_seq,
                    epoch=seg.epoch,
                    pts_us=seg.pts_us,
                    duration_us=seg.duration_us,
                    pcm_s16le=seg.pcm_s16le,
                ),
                pts_us=seg.pts_us,
            ),
            epoch=self.epoch,
        )

    def _cancel_tts_tasks(self) -> None:
        """Cancel all in-flight streaming-TTS background tasks."""
        if not self._tts_tasks:
            return
        for task in list(self._tts_tasks):
            if not task.done():
                task.cancel()
        self._tts_tasks.clear()

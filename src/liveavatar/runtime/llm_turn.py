# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""LLM turn + streaming-TTS emission for :class:`RealtimeWorker` (REF-2).

Pure code move out of ``runtime/worker.py`` — the outbound generation path:
an ASR final streams through the LLM spoke (``_run_llm_turn``), is split
into sentence-bounded pieces and dispatched to TTS (``_dispatch_tts``);
streaming backends emit ``tts_audio`` events incrementally
(``_run_tts_stream``) while fanning PCM out to the avatar adapter.

Epoch semantics are untouched: these methods run as members of
``RealtimeWorker._tts_tasks``, check ``self.epoch`` between chunks and are
cancelled by ``advance_epoch`` / ``stop`` (``_cancel_tts_tasks``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from liveavatar.runtime.contracts import ErrorEvent, TtsAudioEvent
from liveavatar.runtime.worker_api import RealtimeWorkerCore
from liveavatar.text_source import sentence_stream

# Same logger name as worker.py so log attribution is unchanged by the move.
logger = logging.getLogger("liveavatar.runtime.worker")


class LlmTurnMixin:
    """Outbound turn path (LLM → TTS), mixed into :class:`RealtimeWorker`."""

    async def _run_llm_turn(
        self: RealtimeWorkerCore, text: str, epoch: int, pts_us: int
    ) -> None:
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

    def _dispatch_tts(self: RealtimeWorkerCore, text: str, epoch: int, pts_us: int) -> None:
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
        self: RealtimeWorkerCore, text: str, epoch: int, pts_us: int
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

    def _emit_tts_segment(self: RealtimeWorkerCore, seg: Any) -> None:
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

    def _cancel_tts_tasks(self: RealtimeWorkerCore) -> None:
        """Cancel all in-flight streaming-TTS background tasks."""
        if not self._tts_tasks:
            return
        for task in list(self._tts_tasks):
            if not task.done():
                task.cancel()
        self._tts_tasks.clear()

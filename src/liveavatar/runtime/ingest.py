# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Inbound event handling for :class:`RealtimeWorker` (REF-2).

Pure code move out of ``runtime/worker.py`` — how the main loop reacts to
what it dequeues:

- ``_process_frame``: one mic frame through AEC → VAD/EOU/ASR fan-out;
  an ASR final spawns the LLM/TTS turn path (see ``llm_turn.py``).
- ``_handle_control``: control intents — the confirmed interrupt calls
  ``advance_epoch`` (the single epoch authority, kept in ``worker.py``),
  provisional interrupts duck, playback_ack feeds the P-C valley cut.

Epoch semantics are untouched; this module only forwards to the authority.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any

from liveavatar.runtime.contracts import (
    AsrEvent,
    ControlEvent,
    EouEvent,
    InterruptEvent,
    PlaybackAck,
    VadEvent,
    WordTiming,
)
from liveavatar.runtime.worker_api import RealtimeWorkerCore

# Same logger name as worker.py so log attribution is unchanged by the move.
logger = logging.getLogger("liveavatar.runtime.worker")

# Availability probe — mirrors worker.py's guarded import (audio_in optional).
try:
    from liveavatar.audio_in.frame import PCMFrame  # noqa: F401

    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False


class IngestMixin:
    """Inbound frame/control handling, mixed into :class:`RealtimeWorker`."""

    async def _handle_control(self: RealtimeWorkerCore, event: dict[str, Any]) -> None:
        intent = event.get("intent")
        if intent == "cancel_and_flush" or event.get("kind") == "confirmed":
            # Record interrupt start (before advance_epoch, which cancels
            # publisher + TTS + purges queues).
            if self.metrics is not None:
                self.metrics.record_interrupt()
                # P-A: begin a fresh five-layer timeline; "detect" is stamped
                # at the top of this branch (first confirmed-interrupt site).
                self.metrics.timeline_reset()
                self.metrics.timeline_mark("detect")
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
            flush_meta: dict[str, Any] = {"old_epoch": old, "new_epoch": self.epoch}
            if self._last_valley_cut is not None:
                # P-C: downstream players may play the pending tail up to
                # ``cut_pts_us`` (the energy valley) before stopping.
                flush_meta["valley"] = self._last_valley_cut
            self.output_queue.enqueue(
                self._make_envelope(
                    "control",
                    ControlEvent(intent="flush", metadata=flush_meta),
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
            consumed = int(event.get("consumed_pts_us", 0) or 0)
            if consumed > self._consumed_pts_us:
                # P-C: keep the freshest playback position for the
                # energy-valley cut selection.
                self._consumed_pts_us = consumed
            self.output_queue.enqueue(
                self._make_envelope(
                    "playback_ack",
                    PlaybackAck(
                        segment_seq=event.get("segment_seq", 0),
                        consumed_pts_us=consumed,
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

    async def _process_frame(self: RealtimeWorkerCore, frame: Any) -> None:
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

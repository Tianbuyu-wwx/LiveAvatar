# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Session-level metrics for the realtime loopback (Sprint 1 step 5).

Records four key observability signals:

- **first packet**: monotonic time of the first student_mic PCMFrame pushed
  to the worker (capture-to-ingest latency baseline).
- **first playback**: monotonic time of the first Tutor audio frame captured
  to the playback sink (ingest-to-playback latency baseline).
- **interrupt-to-flush**: duration from a confirmed interrupt
  (``worker.advance_epoch``) to the flush control event emitted back to the
  browser (interrupt responsiveness SLO).
- **queue high-water**: max depth reached by input/output/control queues
  (backpressure/capacity signal); tracked in ``BoundedAsyncQueue.stats``.

All timestamps use ``time.monotonic_ns()`` for drift-free duration math.
Each recorder is idempotent (first-packet/first-playback fire once) so
callers can invoke unconditionally on every frame without branching.

This module is pure data — trace logging lives at the call sites where
domain context (track_sid, seq, segment_seq) is available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SessionMetrics:
    """Per-session observability counters and timestamps.

    Pass the same instance to the worker, adapter, and publisher so they
    share one coherent timeline. All recorders are simple attribute writes
    (GIL-protected), safe to call from any asyncio task.
    """

    session_id: str

    # First student mic packet (monotonic ns).
    first_packet_ns: int | None = None
    # First Tutor audio frame published (monotonic ns).
    first_playback_ns: int | None = None
    # Last confirmed-interrupt timestamp (monotonic ns).
    last_interrupt_ns: int | None = None
    # Most recent interrupt-to-flush duration (ns), None until measured.
    interrupt_to_flush_ns: int | None = None
    # Cumulative counts.
    interrupt_count: int = 0
    flush_count: int = 0

    # P-A: five-layer interruption timeline (None = instrumentation off).
    # Layers in true execution order inside the realtime worker:
    #   detect           — interrupt confirmed (control event observed)
    #   audio_flush      — input/output queues purged for the new epoch
    #   tts_stop         — TTS cancel requested + asyncio tasks cancelled
    #   video_invalidate — avatar adapter/sink dropped stale-epoch frames
    #   new_frame        — first frame of the new epoch published
    # All stamps are time.monotonic_ns(); durations derive from consecutive
    # layer deltas (paper §5.1 latency decomposition table).
    timeline_stamps: dict[str, int] = field(default_factory=dict)
    timeline_interrupt_seq: int = 0

    # P-C: energy-valley cut stats (one record per confirmed interrupt
    # where the TTS adapter exposed a pending audio tail). ``found`` =
    # a valley (or EOU prior) was selected; otherwise the hard cut kept.
    valley_cut_count: int = 0
    valley_hit_count: int = 0
    valley_rollback_ms: float | None = None

    def record_valley_cut(self, found: bool, rollback_ms: float) -> None:
        """Record one energy-valley cut decision (P-C, paper §4.3)."""
        self.valley_cut_count += 1
        if found:
            self.valley_hit_count += 1
        self.valley_rollback_ms = rollback_ms

    @property
    def valley_hit_rate(self) -> float | None:
        """Fraction of cuts where a valley was found, or None."""
        if self.valley_cut_count == 0:
            return None
        return self.valley_hit_count / self.valley_cut_count

    def timeline_mark(self, layer: str) -> int:
        """Stamp one timeline layer (first write wins, idempotent)."""
        stamps = self.timeline_stamps
        if layer not in stamps:
            stamps[layer] = time.monotonic_ns()
        return stamps[layer]

    def timeline_reset(self) -> None:
        """Start a fresh timeline window (next confirmed interrupt)."""
        self.timeline_interrupt_seq += 1
        self.timeline_stamps = {}

    def timeline_decompose_ms(self) -> dict[str, float | list[str]]:
        """Return layer deltas in ms for the current/last interrupt.

        Deltas are consecutive-layer differences (detect→audio_flush→
        tts_stop→video_invalidate→new_frame). Layers never stamped are
        omitted; a layer stamped *before* its predecessor (should not
        happen in the worker's linear interrupt path) yields a negative
        delta and is clamped to 0.0 with the raw value preserved.
        """
        stamps = self.timeline_stamps
        order = [
            "detect",
            "audio_flush",
            "tts_stop",
            "video_invalidate",
            "new_frame",
        ]
        known = [name for name in order if name in stamps]
        out: dict[str, float | list[str]] = {}
        for prev, cur in zip(known, known[1:], strict=False):
            delta = (stamps[cur] - stamps[prev]) / 1e6
            out[f"{prev}_to_{cur}_ms"] = max(delta, 0.0)
        if len(known) >= 2:
            total = (stamps[known[-1]] - stamps[known[0]]) / 1e6
            out["total_ms"] = max(total, 0.0)
        out["layers_recorded"] = known
        return out


    def record_first_packet(self) -> bool:
        """Record first student mic packet. Returns True if this was the first."""
        if self.first_packet_ns is not None:
            return False
        self.first_packet_ns = time.monotonic_ns()
        return True

    def record_first_playback(self) -> bool:
        """Record first Tutor audio frame. Returns True if this was the first."""
        if self.first_playback_ns is not None:
            return False
        self.first_playback_ns = time.monotonic_ns()
        return True

    def record_interrupt(self) -> None:
        """Record a confirmed interrupt (start of interrupt-to-flush window)."""
        self.last_interrupt_ns = time.monotonic_ns()
        self.interrupt_count += 1

    def record_flush(self) -> float | None:
        """Record a flush event.

        Returns interrupt-to-flush duration in ms, or None when no interrupt
        was recorded (e.g. a flush without a preceding confirmed interrupt).
        """
        if self.last_interrupt_ns is None:
            return None
        now = time.monotonic_ns()
        self.interrupt_to_flush_ns = now - self.last_interrupt_ns
        self.flush_count += 1
        return self.interrupt_to_flush_ns / 1e6

    @property
    def first_to_playback_ms(self) -> float | None:
        """Latency from first packet to first playback in ms, or None."""
        if self.first_packet_ns is None or self.first_playback_ns is None:
            return None
        return (self.first_playback_ns - self.first_packet_ns) / 1e6

    @property
    def interrupt_to_flush_ms(self) -> float | None:
        """Last interrupt-to-flush duration in ms, or None."""
        if self.interrupt_to_flush_ns is None:
            return None
        return self.interrupt_to_flush_ns / 1e6

    def summary(self) -> dict:
        """Return a flat dict summary suitable for logging or /metrics export."""
        out: dict[str, object] = {
            "session_id": self.session_id,
            "first_packet_ns": self.first_packet_ns,
            "first_playback_ns": self.first_playback_ns,
            "first_packet_to_first_playback_ms": self.first_to_playback_ms,
            "interrupt_to_flush_ms": self.interrupt_to_flush_ms,
            "interrupt_count": self.interrupt_count,
            "flush_count": self.flush_count,
        }
        if self.valley_cut_count:
            out["valley_hit_rate"] = self.valley_hit_rate
            out["valley_rollback_ms"] = self.valley_rollback_ms
        if self.timeline_stamps:
            out["interruption_timeline"] = self.timeline_decompose_ms()
        return out

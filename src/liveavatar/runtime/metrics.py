"""Session-level metrics for the realtime loopback (Sprint 1 step 5).

Records four key observability signals:

- **first packet**: monotonic time of the first student_mic PCMFrame pushed
  to the worker (capture-to-ingest latency baseline).
- **first playback**: monotonic time of the first Tutor audio frame captured
  to the LiveKit AudioSource (ingest-to-playback latency baseline).
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
from dataclasses import dataclass


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
        return {
            "session_id": self.session_id,
            "first_packet_ns": self.first_packet_ns,
            "first_playback_ns": self.first_playback_ns,
            "first_packet_to_first_playback_ms": self.first_to_playback_ms,
            "interrupt_to_flush_ms": self.interrupt_to_flush_ms,
            "interrupt_count": self.interrupt_count,
            "flush_count": self.flush_count,
        }

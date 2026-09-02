# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Bounded audio frame buffer with epoch-aware drop policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .frame import PCMFrame


class DropPolicy(str, Enum):
    BLOCK = "block"
    DROP_NEWEST = "drop_newest"
    DROP_OLDEST = "drop_oldest"


@dataclass
class BufferStats:
    received: int = 0
    dropped: int = 0
    dropped_deadline: int = 0
    dropped_epoch: int = 0
    dropped_full: int = 0
    duplicates: int = 0
    gaps: int = 0
    last_seq: int = -1
    current_epoch: int = 0


class BoundedFrameBuffer:
    """Thread-unsafe bounded frame buffer.

    Designed to be used inside a single asyncio task or with explicit locking.
    """

    def __init__(
        self,
        capacity: int,
        drop_policy: DropPolicy = DropPolicy.DROP_OLDEST,
        clock_now_us: Callable[[], int] | None = None,
    ) -> None:
        self.capacity = max(1, capacity)
        self.drop_policy = drop_policy
        self._clock_now_us = clock_now_us or (lambda: 0)
        self._frames: list[PCMFrame] = []
        self.stats = BufferStats()

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def advance_epoch(self, new_epoch: int) -> None:
        """Drop all frames from older epochs and reset state."""
        if new_epoch <= self.stats.current_epoch:
            return
        removed = [f for f in self._frames if f.epoch < new_epoch]
        self._frames = [f for f in self._frames if f.epoch >= new_epoch]
        self.stats.dropped_epoch += len(removed)
        self.stats.current_epoch = new_epoch
        self.stats.last_seq = -1

    def push(self, frame: PCMFrame) -> bool:
        """Push a frame. Returns True if accepted, False if dropped."""
        self.stats.received += 1

        now_us = self._clock_now_us()
        if frame.deadline_us > 0 and now_us > frame.deadline_us:
            self.stats.dropped += 1
            self.stats.dropped_deadline += 1
            return False

        if frame.epoch < self.stats.current_epoch:
            self.stats.dropped += 1
            self.stats.dropped_epoch += 1
            return False

        if frame.seq <= self.stats.last_seq:
            self.stats.duplicates += 1
            # Accept out-of-order only if not duplicate; for simplicity drop duplicates.
            return False
        if self.stats.last_seq >= 0 and frame.seq != self.stats.last_seq + 1:
            self.stats.gaps += 1

        if len(self._frames) >= self.capacity:
            if self.drop_policy == DropPolicy.BLOCK:
                return False
            if self.drop_policy == DropPolicy.DROP_NEWEST:
                self.stats.dropped += 1
                self.stats.dropped_full += 1
                return False
            if self.drop_policy == DropPolicy.DROP_OLDEST:
                self._frames.pop(0)
                self.stats.dropped += 1
                self.stats.dropped_full += 1

        self._frames.append(frame)
        self.stats.last_seq = frame.seq
        return True

    def pop(self) -> PCMFrame | None:
        if not self._frames:
            return None
        return self._frames.pop(0)

    def peek(self) -> PCMFrame | None:
        if not self._frames:
            return None
        return self._frames[0]

    def drain(self) -> list[PCMFrame]:
        frames = self._frames.copy()
        self._frames.clear()
        return frames

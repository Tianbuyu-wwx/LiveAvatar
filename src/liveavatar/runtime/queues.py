# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Bounded input/output queues with epoch-aware discard."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class QueueStats:
    enqueued: int = 0
    dequeued: int = 0
    dropped_full: int = 0
    dropped_epoch: int = 0
    high_water: int = 0  # max depth ever reached (queue backpressure signal)


class BoundedAsyncQueue(Generic[T]):
    """Async queue with capacity and epoch discard."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._queue: deque[T] = deque()
        self._event = asyncio.Event()
        self.stats = QueueStats()
        self.current_epoch: int = 0

    def _maybe_set_event(self) -> None:
        if self._queue:
            self._event.set()
        else:
            self._event.clear()

    def enqueue(self, item: T, epoch: int | None = None) -> bool:
        if epoch is not None and epoch < self.current_epoch:
            self.stats.dropped_epoch += 1
            return False
        if len(self._queue) >= self.capacity:
            self.stats.dropped_full += 1
            return False
        self._queue.append(item)
        self.stats.enqueued += 1
        if len(self._queue) > self.stats.high_water:
            self.stats.high_water = len(self._queue)
        self._event.set()
        return True

    async def dequeue(self) -> T:
        await self._event.wait()
        item = self._queue.popleft()
        self.stats.dequeued += 1
        self._maybe_set_event()
        return item

    def try_dequeue(self) -> T | None:
        if not self._queue:
            return None
        item = self._queue.popleft()
        self.stats.dequeued += 1
        self._maybe_set_event()
        return item

    def advance_epoch(self, new_epoch: int) -> int:
        if new_epoch <= self.current_epoch:
            return 0
        removed = sum(1 for item in self._queue if getattr(item, "epoch", 0) < new_epoch)
        self._queue = deque(item for item in self._queue if getattr(item, "epoch", 0) >= new_epoch)
        self.stats.dropped_epoch += removed
        self.current_epoch = new_epoch
        self._maybe_set_event()
        return removed

    def __len__(self) -> int:
        return len(self._queue)

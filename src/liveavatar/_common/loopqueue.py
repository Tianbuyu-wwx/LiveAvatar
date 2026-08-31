"""Loop-free bounded FIFO queue.

``asyncio.Queue`` caches the first event loop that awaits ``get()`` /
``put()`` and then raises ``RuntimeError: ... is bound to a different
event loop`` when used from another loop. This queue never caches a loop:
every waiter creates its future on its *own* running loop and is woken via
``call_soon_threadsafe`` on the waiter's loop, so buffered items, pending
waiters and blocked producers all survive loop/thread changes.

Used by :class:`~liveavatar.adapter.AvatarStreamingAdapter` (PCM chunks)
and :class:`~liveavatar.ws_sink.WebSocketSink` (per-client fan-out).
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class LoopFreeQueue(Generic[T]):
    """Bounded FIFO queue safe across event loops and threads.

    Implements the ``asyncio.Queue``-compatible subset used in this
    project: ``put`` (backpressure) / ``get`` / ``get_nowait`` / ``empty``
    / ``qsize``, plus two extensions:

    - ``offer(item)``: never blocks; drops the OLDEST item when full
      (real-time fan-out semantics — a slow consumer must never stall the
      producer). Returns the number of dropped items.
    - ``close()``: delivers ``None`` to every pending getter and marks the
      queue finished; subsequent ``put`` raises, ``get`` keeps draining.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._maxsize = max(1, maxsize) if maxsize else 0
        self._items: deque[T | None] = deque()
        self._getters: list[asyncio.Future[T | None]] = []
        self._putters: list[tuple[T | None, asyncio.Future[None]]] = []
        self._closed = False
        self._dropped_total = 0

    # ------------------------------------------------------------ introspection

    def qsize(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        return not self._items

    @property
    def dropped_total(self) -> int:
        """Total items dropped by ``offer`` (drop-oldest) since creation."""
        return self._dropped_total

    @property
    def closed(self) -> bool:
        return self._closed

    # ---------------------------------------------------------------- put side

    async def put(self, item: T | None) -> None:
        """Enqueue; block while the buffer is full (backpressure)."""
        if self._closed:
            raise RuntimeError("queue is closed")
        if self._handoff_to_getter(item):
            return
        if not self._maxsize or len(self._items) < self._maxsize:
            self._items.append(item)
            return
        p_fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._putters.append((item, p_fut))
        try:
            await p_fut
        finally:
            try:
                self._putters.remove((item, p_fut))
            except ValueError:
                pass

    def offer(self, item: T | None) -> int:
        """Enqueue without ever blocking; drop-oldest when full.

        Returns the number of dropped items (0 or 1). A closed queue
        drops everything (returns 1).
        """
        if self._closed:
            self._dropped_total += 1
            return 1
        if self._handoff_to_getter(item):
            return 0
        dropped = 0
        if self._maxsize and len(self._items) >= self._maxsize:
            self._items.popleft()
            self._dropped_total += 1
            dropped = 1
        self._items.append(item)
        return dropped

    def close(self) -> None:
        """Finish the queue: wake pending getters with ``None`` sentinels."""
        if self._closed:
            return
        self._closed = True
        for fut in list(self._getters):
            self._set(fut, None)
        self._getters.clear()

    # ---------------------------------------------------------------- get side

    async def get(self) -> T | None:
        """Dequeue; block when empty. Returns ``None`` after ``close()``."""
        if self._items:
            item = self._items.popleft()
            self._wake_putter()
            return item
        if self._closed:
            return None  # drained past the EOF sentinel
        fut: asyncio.Future[T | None] = asyncio.get_running_loop().create_future()
        self._getters.append(fut)
        try:
            return await fut
        finally:
            try:
                self._getters.remove(fut)
            except ValueError:
                pass

    def get_nowait(self) -> T | None:
        """Dequeue without blocking; raises ``asyncio.QueueEmpty`` when empty."""
        if not self._items:
            raise asyncio.QueueEmpty
        item = self._items.popleft()
        self._wake_putter()
        return item

    # ----------------------------------------------------------------- internals

    def _handoff_to_getter(self, item: T | None) -> bool:
        """Directly resolve one waiting getter; True when handed off."""
        while self._getters:
            fut = self._getters.pop(0)
            if not fut.cancelled():
                self._set(fut, item)
                return True
        return False

    def _wake_putter(self) -> None:
        """A slot freed up: move one waiting item into the buffer."""
        while self._putters:
            item, fut = self._putters.pop(0)
            if fut.cancelled():
                continue
            self._items.append(item)
            self._set(fut, None)
            return

    def _set(self, fut: asyncio.Future[Any], value: Any) -> None:
        """Resolve ``fut`` on its own loop (works across loops/threads)."""
        try:
            fut.get_loop().call_soon_threadsafe(
                LoopFreeQueue._set_result, fut, value
            )
        except RuntimeError:
            # Waiter's loop already closed — drop the dead waiter.
            pass

    @staticmethod
    def _set_result(fut: asyncio.Future[Any], value: Any) -> None:
        if not fut.cancelled():
            fut.set_result(value)

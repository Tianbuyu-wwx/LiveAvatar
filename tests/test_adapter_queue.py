"""Tests for the adapter's loop-agnostic bounded queue (R2 CI regression).

``asyncio.Queue`` caches its first event loop and crashes with
``RuntimeError: ... is bound to a different event loop`` when the adapter
is driven from multiple loops (TestClient portals). These tests reproduce
that scenario directly — producer and consumer run on different loops in
different threads — and would fail with the old queue implementation.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from liveavatar.adapter import _BoundedQueue, _PendingChunk


def _chunk(pcm: bytes = b"x", pts: int = 0, epoch: int = 0) -> _PendingChunk:
    return _PendingChunk(pcm_s16le=pcm, pts_us=pts, epoch=epoch)


def _run(coro):
    """Run a coroutine on its own fresh event loop (its own thread-safe)."""
    return asyncio.run(coro)


class TestSameLoopBasics:
    async def _fifo_order(self, q: _BoundedQueue) -> list[bytes]:
        for i in range(5):
            await q.put(_chunk(pcm=bytes([i])))
        return [(await q.get()).pcm_s16le for _ in range(5)]

    def test_fifo_order(self):
        q = _BoundedQueue(maxsize=8)
        assert _run(self._fifo_order(q)) == [bytes([i]) for i in range(5)]

    def test_get_nowait_raises_when_empty(self):
        q = _BoundedQueue(maxsize=4)
        with pytest.raises(asyncio.QueueEmpty):
            q.get_nowait()

    def test_qsize_empty(self):
        q = _BoundedQueue(maxsize=4)
        assert q.empty() and q.qsize() == 0
        _run(q.put(_chunk()))
        assert not q.empty() and q.qsize() == 1

    def test_put_blocks_when_full_then_get_wakes_it(self):
        q = _BoundedQueue(maxsize=2)
        done: list[int] = []

        async def main() -> None:
            await q.put(_chunk(pcm=b"0"))
            await q.put(_chunk(pcm=b"1"))
            third = asyncio.ensure_future(q.put(_chunk(pcm=b"2")))
            await asyncio.sleep(0.05)
            assert not third.done()  # blocked: buffer full
            assert (await q.get()).pcm_s16le == b"0"
            await asyncio.wait_for(third, 1)  # woken by the get above
            done.append(1)
            assert [(await q.get()).pcm_s16le for _ in range(2)] == [b"1", b"2"]

        _run(main())
        assert done == [1]

    def test_get_nowait_wakes_blocked_putter(self):
        q = _BoundedQueue(maxsize=1)

        async def main() -> None:
            await q.put(_chunk(pcm=b"a"))
            putter = asyncio.ensure_future(q.put(_chunk(pcm=b"b")))
            await asyncio.sleep(0.05)
            assert q.get_nowait().pcm_s16le == b"a"
            await asyncio.wait_for(putter, 1)
            assert q.get_nowait().pcm_s16le == b"b"
            with pytest.raises(asyncio.QueueEmpty):
                q.get_nowait()

        _run(main())


class TestCrossLoop:
    """The CI failure pattern: waiter in one loop, producer in another."""

    def test_consumer_waits_in_thread_b_producer_in_loop_c(self):
        q = _BoundedQueue(maxsize=4)
        got: list[bytes] = []

        async def consume() -> None:
            got.append((await asyncio.wait_for(q.get(), 3)).pcm_s16le)

        async def produce() -> None:
            await q.put(_chunk(pcm=b"cross"))

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run, consume())
            time.sleep(0.1)  # let the getter register on loop B
            _run(produce())  # different loop (C) — asyncio.Queue would raise
            fut.result(timeout=3)

        assert got == [b"cross"]

    def test_buffered_items_survive_loop_change(self):
        q = _BoundedQueue(maxsize=8)

        async def produce_two() -> None:
            await q.put(_chunk(pcm=b"1"))
            await q.put(_chunk(pcm=b"2"))

        _run(produce_two())  # loop A — closed afterwards

        async def consume_two() -> list[bytes]:
            return [(await asyncio.wait_for(q.get(), 2)).pcm_s16le for _ in range(2)]

        assert _run(consume_two()) == [b"1", b"2"]  # loop B

    def test_full_queue_survives_loop_change(self):
        """Blocked putter on a dying loop is cancelled — queue stays consistent.

        Mirrors asyncio.Queue cancellation semantics: a ``put`` cancelled
        before delivery does not enqueue its item. The surviving buffered
        chunk must still be readable from a different loop without any
        cross-loop crash (the asyncio.Queue failure mode).
        """
        q = _BoundedQueue(maxsize=1)

        async def fill() -> None:
            await q.put(_chunk(pcm=b"a"))
            putter = asyncio.ensure_future(q.put(_chunk(pcm=b"b")))
            await asyncio.sleep(0.05)
            assert not putter.done()
            # Loop A ends here, cancelling the blocked putter.

        _run(fill())

        async def drain() -> list[bytes]:
            out = [(await asyncio.wait_for(q.get(), 2)).pcm_s16le]
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.get(), 0.1)
            return out

        assert _run(drain()) == [b"a"]
        assert q.empty()

    def test_dead_waiter_loop_does_not_lose_chunks(self):
        """Getter registered on a loop that dies before put — item buffers."""
        q = _BoundedQueue(maxsize=4)
        started = threading.Event()

        async def register_then_die() -> None:
            started.set()
            # Create the getter future, then abandon the loop immediately.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.get(), 0.05)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_run, register_then_die()).result(timeout=3)

        # Loop gone; the waiter either timed out (removed) or is dead.
        async def produce_and_consume() -> bytes:
            await q.put(_chunk(pcm=b"alive"))
            return (await asyncio.wait_for(q.get(), 1)).pcm_s16le

        assert _run(produce_and_consume()) == b"alive"

    def test_many_items_cross_loop_stress(self):
        q = _BoundedQueue(maxsize=16)
        n = 200
        got: list[int] = []
        errors: list[BaseException] = []

        async def consume() -> None:
            try:
                for _ in range(n):
                    got.append(int.from_bytes((await q.get()).pcm_s16le, "big"))
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        async def produce() -> None:
            for i in range(n):
                await q.put(_chunk(pcm=i.to_bytes(2, "big")))

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run, consume())
            _run(produce())
            fut.result(timeout=10)

        assert not errors
        assert got == list(range(n))

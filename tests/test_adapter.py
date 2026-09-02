# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for AvatarStreamingAdapter.

Validates the PCM → AvatarWorker → AvatarVideoPublisher pipeline using
fake workers and a fake publisher (no torch, no cv2).

Coverage:
- Direct mode (worker injected, no pool).
- Pool mode (lease acquired via a fake pool).
- PCM chunk → frames produced → frames published (in-order, PTS advancing).
- Epoch cancellation: stale-epoch chunks dropped, in-flight generator cancelled.
- Backpressure: queue-full drops chunks without blocking.
- Degradation chain: N consecutive errors → switch to fallback worker.
- Degradation reset on epoch advance.
- stop() releases lease and cancels consumer.
"""

from __future__ import annotations

import asyncio
import unittest

from liveavatar.adapter import AvatarStreamingAdapter, _PendingChunk
from liveavatar.worker import AvatarAssets, AvatarFrame, AvatarWorker
from tests.conftest import make_assets as _make_assets
from tests.conftest import pcm

# ────────────────────────────────────────────────────── helpers

class _CountingAvatarWorker(AvatarWorker):
    """Fake worker that yields ``batch_size`` frames per chunk, tagged with pts.

    Each frame's ``frame_data`` encodes the chunk's pts_us for verification:
    ``pts_us.to_bytes(8, "little") + b"frame" + frame_index_byte``.
    """

    def __init__(
        self,
        assets: AvatarAssets,
        *,
        batch_size: int = 4,
        width: int = 4,
        height: int = 4,
        delay: float = 0.0,
    ) -> None:
        super().__init__(
            assets, target_fps=25, width=width, height=height, batch_size=batch_size
        )
        self.delay = delay
        self.chunks_seen: list[bytes] = []

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        self.chunks_seen.append(pcm_s16le)
        if self.delay > 0:
            import time

            time.sleep(self.delay)
        return [
            (b"\x00" * (self.width * self.height * 3), True)
            for _ in range(self.batch_size)
        ]


class _ErrorAvatarWorker(AvatarWorker):
    """Fake worker that always raises during inference (for degradation tests)."""

    def __init__(self, assets: AvatarAssets) -> None:
        super().__init__(assets, width=4, height=4, batch_size=1)

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        raise RuntimeError("simulated inference failure")


class _FakeVideoPublisher:
    """Stand-in publisher that records captured frames and supports cancel_epoch."""

    def __init__(self) -> None:
        self.captured: list[AvatarFrame] = []
        self._current_epoch = 0

    async def publish_frame(self, frame: AvatarFrame, epoch: int) -> bool:
        if epoch < self._current_epoch:
            return False
        self.captured.append(frame)
        return True

    def cancel_epoch(self, new_epoch: int) -> None:
        if new_epoch > self._current_epoch:
            self._current_epoch = new_epoch

    @property
    def current_epoch(self) -> int:
        return self._current_epoch


class _FakePool:
    """Fake AvatarPool that returns a pre-built worker."""

    def __init__(self, worker: AvatarWorker) -> None:
        self._worker = worker
        self.acquired = False
        self.released = False

    async def acquire(self, session_id: str, avatar_id: str, **kwargs):
        self.acquired = True
        return _FakeLease(self._worker)

    async def release_async(self, session_id: str) -> bool:
        self.released = True
        return True


class _FakeLease:
    def __init__(self, worker: AvatarWorker) -> None:
        self.worker = worker


async def _wait_for(predicate, timeout: float = 1.0, interval: float = 0.01) -> bool:
    """Poll a predicate until True or timeout."""
    elapsed = 0.0
    while not predicate():
        if elapsed >= timeout:
            return False
        await asyncio.sleep(interval)
        elapsed += interval
    return True


# ──────────────────────────────────────────────────── direct mode


class TestDirectMode(unittest.IsolatedAsyncioTestCase):
    """Tests for worker-injected (no pool) adapter."""

    async def test_push_pcm_produces_frames(self):
        worker = _CountingAvatarWorker(_make_assets(), batch_size=4)
        adapter = AvatarStreamingAdapter(
            worker=worker, session_id="s1", avatar_id="nahida"
        )
        await adapter.start()
        try:
            ok = await adapter.push_pcm(pcm(value=100), pts_us=0, epoch=0)
            self.assertTrue(ok)
            # Wait for consumer to process.
            await _wait_for(lambda: adapter.stats.frames_produced >= 4)
            self.assertEqual(adapter.stats.frames_produced, 4)
            self.assertEqual(len(adapter.published_frames), 4)
            # Each frame's pts advances by 1/fps microseconds.
            self.assertEqual(adapter.published_frames[0].pts_us, 0)
            self.assertEqual(adapter.published_frames[1].pts_us, 40_000)
            self.assertEqual(adapter.published_frames[3].pts_us, 120_000)
        finally:
            await adapter.stop()

    async def test_publisher_receives_frames(self):
        worker = _CountingAvatarWorker(_make_assets(), batch_size=2)
        pub = _FakeVideoPublisher()
        adapter = AvatarStreamingAdapter(
            worker=worker, publisher=pub, session_id="s1", avatar_id="nahida"
        )
        await adapter.start()
        try:
            await adapter.push_pcm(pcm(value=50), pts_us=0, epoch=0)
            await _wait_for(lambda: len(pub.captured) >= 2)
            self.assertEqual(len(pub.captured), 2)
            self.assertEqual(adapter.stats.frames_published, 2)
        finally:
            await adapter.stop()

    async def test_multiple_chunks_pts_advances(self):
        worker = _CountingAvatarWorker(_make_assets(), batch_size=2)
        adapter = AvatarStreamingAdapter(
            worker=worker, session_id="s1", avatar_id="nahida"
        )
        await adapter.start()
        try:
            await adapter.push_pcm(pcm(value=10), pts_us=0, epoch=0)
            await adapter.push_pcm(pcm(value=20), pts_us=80_000, epoch=0)
            await _wait_for(lambda: adapter.stats.frames_produced >= 4)
            self.assertEqual(adapter.stats.frames_produced, 4)
            # 4 frames total, 2 per chunk with pts 0, 40ms, 80ms, 120ms.
            pts_seq = [f.pts_us for f in adapter.published_frames]
            self.assertEqual(pts_seq, [0, 40_000, 80_000, 120_000])
        finally:
            await adapter.stop()

    async def test_start_idempotent(self):
        worker = _CountingAvatarWorker(_make_assets())
        adapter = AvatarStreamingAdapter(worker=worker, session_id="s1")
        await adapter.start()
        await adapter.start()  # no-op
        try:
            self.assertIsNotNone(adapter._consumer_task)
        finally:
            await adapter.stop()


# ──────────────────────────────────────────────────── pool mode


class TestPoolMode(unittest.IsolatedAsyncioTestCase):
    """Tests for lease-managed (pool) adapter."""

    async def test_pool_acquire_and_release(self):
        worker = _CountingAvatarWorker(_make_assets())
        pool = _FakePool(worker)
        adapter = AvatarStreamingAdapter(
            pool=pool, session_id="s1", avatar_id="nahida"
        )
        self.assertFalse(pool.acquired)
        await adapter.start()
        self.assertTrue(pool.acquired)
        self.assertTrue(adapter.stats.lease_acquired)
        await adapter.stop()
        self.assertTrue(pool.released)
        self.assertFalse(adapter.stats.lease_acquired)

    async def test_pool_mode_pushes_frames(self):
        worker = _CountingAvatarWorker(_make_assets(), batch_size=2)
        pool = _FakePool(worker)
        adapter = AvatarStreamingAdapter(
            pool=pool, session_id="s1", avatar_id="nahida"
        )
        await adapter.start()
        try:
            await adapter.push_pcm(pcm(value=80), pts_us=0, epoch=0)
            await _wait_for(lambda: adapter.stats.frames_produced >= 2)
            self.assertEqual(adapter.stats.frames_produced, 2)
        finally:
            await adapter.stop()


# ──────────────────────────────────────────────────── cancellation


class TestEpochCancellation(unittest.IsolatedAsyncioTestCase):
    """Tests for cancel_epoch behavior."""

    async def test_stale_epoch_chunk_dropped_at_push(self):
        worker = _CountingAvatarWorker(_make_assets())
        adapter = AvatarStreamingAdapter(worker=worker, session_id="s1")
        await adapter.start()
        try:
            adapter.cancel_epoch(5)
            ok = await adapter.push_pcm(pcm(value=10), pts_us=0, epoch=3)
            self.assertFalse(ok)
            self.assertEqual(adapter.stats.frames_dropped_epoch, 1)
            # Worker should never see the stale chunk.
            self.assertEqual(len(worker.chunks_seen), 0)
        finally:
            await adapter.stop()

    async def test_cancel_epoch_forwards_to_publisher(self):
        worker = _CountingAvatarWorker(_make_assets())
        pub = _FakeVideoPublisher()
        adapter = AvatarStreamingAdapter(
            worker=worker, publisher=pub, session_id="s1"
        )
        await adapter.start()
        try:
            adapter.cancel_epoch(7)
            self.assertEqual(pub.current_epoch, 7)
        finally:
            await adapter.stop()

    async def test_cancel_epoch_drains_queue(self):
        worker = _CountingAvatarWorker(_make_assets(), batch_size=2)
        adapter = AvatarStreamingAdapter(
            worker=worker, session_id="s1", queue_capacity=20
        )
        await adapter.start()
        try:
            # Push several chunks without waiting for processing.
            for i in range(5):
                await adapter.push_pcm(pcm(value=i), pts_us=i * 80_000, epoch=0)
            # Cancel — should drain the queue.
            adapter.cancel_epoch(1)
            # Give the consumer a moment to confirm no further processing.
            await asyncio.sleep(0.05)
            produced = adapter.stats.frames_produced
            await asyncio.sleep(0.05)
            # No new frames after the cancel.
            self.assertEqual(adapter.stats.frames_produced, produced)
        finally:
            await adapter.stop()

    async def test_cancel_epoch_monotonic(self):
        worker = _CountingAvatarWorker(_make_assets())
        adapter = AvatarStreamingAdapter(worker=worker, session_id="s1")
        await adapter.start()
        try:
            adapter.cancel_epoch(5)
            adapter.cancel_epoch(3)  # lower — ignored
            self.assertEqual(adapter.current_epoch, 5)
            adapter.cancel_epoch(5)  # equal — ignored
            self.assertEqual(adapter.current_epoch, 5)
        finally:
            await adapter.stop()

    async def test_cancel_token_cancelled_on_epoch_advance(self):
        """When cancel_epoch fires, the in-flight CancelToken is cancelled."""
        worker = _CountingAvatarWorker(_make_assets(), batch_size=2)
        adapter = AvatarStreamingAdapter(worker=worker, session_id="s1")
        await adapter.start()
        try:
            token_before = adapter._cancel_token
            adapter.cancel_epoch(1)
            self.assertTrue(token_before.cancelled)
            # New token created for the next epoch.
            self.assertIsNot(adapter._cancel_token, token_before)
            self.assertFalse(adapter._cancel_token.cancelled)
        finally:
            await adapter.stop()


# ──────────────────────────────────────────────────── backpressure


class TestBackpressure(unittest.IsolatedAsyncioTestCase):
    """Tests for queue-full backpressure."""

    async def test_queue_full_drops_chunk(self):
        # Worker with a delay so the queue fills up.
        worker = _CountingAvatarWorker(_make_assets(), batch_size=2, delay=0.05)
        adapter = AvatarStreamingAdapter(
            worker=worker,
            session_id="s1",
            queue_capacity=2,
            push_timeout_s=0.01,
        )
        await adapter.start()
        try:
            # Push faster than the worker can process.
            results = []
            for i in range(10):
                ok = await adapter.push_pcm(pcm(value=i), pts_us=i * 80_000, epoch=0)
                results.append(ok)
            # At least one drop should have happened (queue cap = 2).
            self.assertIn(False, results)
            self.assertGreater(adapter.stats.frames_dropped_error, 0)
        finally:
            await adapter.stop()


# ──────────────────────────────────────────────────── degradation


class TestDegradation(unittest.IsolatedAsyncioTestCase):
    """Tests for the MuseTalk → static fallback chain."""

    async def test_consecutive_errors_trigger_degradation(self):
        primary = _ErrorAvatarWorker(_make_assets("primary"))
        fallback = _CountingAvatarWorker(_make_assets("fallback"), batch_size=1)
        adapter = AvatarStreamingAdapter(
            worker=primary,
            fallback_worker=fallback,
            session_id="s1",
            degrade_after_errors=3,
        )
        await adapter.start()
        try:
            # Push 5 chunks — first 3 fail on primary (triggering degradation),
            # then the fallback takes over for chunks 4 and 5 (which succeed).
            for i in range(5):
                await adapter.push_pcm(pcm(value=i), pts_us=i * 40_000, epoch=0)
                await asyncio.sleep(0.02)  # let the consumer process
            # 3 errors recorded (all on the primary before degradation).
            self.assertEqual(adapter.stats.inference_errors, 3)
            # After 3 consecutive errors, degraded flag is set.
            self.assertTrue(adapter.stats.degraded)
            self.assertEqual(adapter.stats.degradation_count, 1)
            # Fallback worker should have processed chunks 4 and 5.
            self.assertGreaterEqual(len(fallback.chunks_seen), 1)
        finally:
            await adapter.stop()

    async def test_epoch_advance_resets_degradation(self):
        primary = _ErrorAvatarWorker(_make_assets("primary"))
        fallback = _CountingAvatarWorker(_make_assets("fallback"), batch_size=1)
        adapter = AvatarStreamingAdapter(
            worker=primary,
            fallback_worker=fallback,
            session_id="s1",
            degrade_after_errors=2,
        )
        await adapter.start()
        try:
            # Trigger degradation.
            for i in range(3):
                await adapter.push_pcm(pcm(value=i), pts_us=i * 40_000, epoch=0)
                await asyncio.sleep(0.02)
            self.assertTrue(adapter.stats.degraded)
            # Advance epoch — degradation resets, primary gets another chance.
            adapter.cancel_epoch(1)
            self.assertFalse(adapter.stats.degraded)
            self.assertEqual(adapter.stats.consecutive_errors, 0)
        finally:
            await adapter.stop()

    async def test_successful_inference_resets_error_streak(self):
        """A single success between errors resets the consecutive counter."""
        # Use a worker that fails twice then succeeds.
        assets = _make_assets("flaky")
        worker = _CountingAvatarWorker(assets, batch_size=1)
        call_count = [0]
        original_infer = worker._infer_batch

        def flaky_infer(pcm):
            call_count[0] += 1
            if call_count[0] in (1, 2):
                raise RuntimeError("flaky failure")
            return original_infer(pcm)

        worker._infer_batch = flaky_infer  # type: ignore[assignment]

        fallback = _CountingAvatarWorker(_make_assets("fallback"), batch_size=1)
        adapter = AvatarStreamingAdapter(
            worker=worker,
            fallback_worker=fallback,
            session_id="s1",
            degrade_after_errors=3,
        )
        await adapter.start()
        try:
            # Push 5 chunks: fail, fail, success, success, success.
            for i in range(5):
                await adapter.push_pcm(pcm(value=i), pts_us=i * 40_000, epoch=0)
                await asyncio.sleep(0.02)
            # Two errors recorded but no degradation (streak reset by success).
            self.assertEqual(adapter.stats.inference_errors, 2)
            self.assertFalse(adapter.stats.degraded)
        finally:
            await adapter.stop()


# ──────────────────────────────────────────────────── construction


class TestConstruction(unittest.TestCase):
    def test_requires_pool_or_worker(self):
        with self.assertRaises(ValueError):
            AvatarStreamingAdapter(session_id="s1")


class TestConstructionAsync(unittest.IsolatedAsyncioTestCase):
    async def test_start_without_pool_or_worker_raises(self):
        # Construct with worker, then null it out to test the runtime guard.
        worker = _CountingAvatarWorker(_make_assets())
        adapter = AvatarStreamingAdapter(worker=worker, session_id="s1")
        adapter._direct_worker = None  # simulate misconfiguration
        with self.assertRaises(RuntimeError):
            await adapter.start()


# ──────────────────────────────────────────────────── _PendingChunk


class TestPendingChunk(unittest.TestCase):
    def test_pending_chunk_fields(self):
        chunk = _PendingChunk(pcm_s16le=b"\x00\x00", pts_us=1000, epoch=3)
        self.assertEqual(chunk.pcm_s16le, b"\x00\x00")
        self.assertEqual(chunk.pts_us, 1000)
        self.assertEqual(chunk.epoch, 3)


if __name__ == "__main__":
    unittest.main()

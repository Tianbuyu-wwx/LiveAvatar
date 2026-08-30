"""Tests for worker eviction: manual evict_worker + max_loaded_workers LRU."""

from __future__ import annotations

import asyncio
import unittest

from liveavatar.config import AvatarPoolConfig
from liveavatar.pool import AvatarPool
from liveavatar.worker import AvatarAssets, AvatarWorker
from tests.test_pool import _make_assets, _make_worker_factory


class TestWorkerEviction(unittest.IsolatedAsyncioTestCase):
    """evict_worker + max_loaded_workers overflow eviction."""

    def _make_pool(self, **kwargs) -> AvatarPool:
        config = AvatarPoolConfig(
            avatar_data_root="/nonexistent",
            max_workers=4,
            acquire_timeout=1.0,
            **kwargs,
        )
        pool = AvatarPool(config, worker_factory=_make_worker_factory())
        pool._avatars = {aid: _make_assets(aid) for aid in ("a1", "a2", "a3")}
        return pool

    async def test_evict_worker_unloads(self):
        pool = self._make_pool()
        await pool.start()
        try:
            await pool.acquire("s1", "a1")
            await pool.release_async("s1")
            evicted: list[str] = []
            original = pool._on_worker_evicted
            pool._on_worker_evicted = lambda w: (evicted.append(w.avatar_id), original(w))
            self.assertTrue(await pool.evict_worker("a1"))
            self.assertNotIn("a1", pool._workers)
            self.assertEqual(evicted, ["a1"])
        finally:
            await pool.stop()

    async def test_evict_with_active_lease_fails(self):
        pool = self._make_pool()
        await pool.start()
        try:
            await pool.acquire("s1", "a1")
            self.assertFalse(await pool.evict_worker("a1"))
            self.assertIn("a1", pool._workers)
        finally:
            await pool.stop()

    async def test_evict_unknown_or_unloaded_fails(self):
        pool = self._make_pool()
        await pool.start()
        try:
            self.assertFalse(await pool.evict_worker("ghost"))
            self.assertFalse(await pool.evict_worker("a1"))  # never loaded
        finally:
            await pool.stop()

    async def test_evicted_worker_is_reloaded_on_demand(self):
        pool = self._make_pool()
        await pool.start()
        try:
            await pool.acquire("s1", "a1")
            await pool.release_async("s1")
            self.assertTrue(await pool.evict_worker("a1"))
            await pool.acquire("s2", "a1")
            self.assertIn("a1", pool._workers)
            await pool.release_async("s2")
        finally:
            await pool.stop()

    async def test_overflow_eviction_respects_cap_and_lru(self):
        pool = self._make_pool(max_loaded_workers=1)
        await pool.start()
        try:
            await pool.acquire("s1", "a1")
            await pool.release_async("s1")
            await asyncio.sleep(0.01)
            await pool.acquire("s2", "a2")  # a2 keeps its active lease
            evicted: list[str] = []
            original = pool._on_worker_evicted
            pool._on_worker_evicted = lambda w: (evicted.append(w.avatar_id), original(w))
            n = await pool._evict_overflow()
            self.assertEqual(n, 1)
            self.assertEqual(evicted, ["a1"])  # older loaded_at evicted first
            self.assertNotIn("a1", pool._workers)
            self.assertIn("a2", pool._workers)
        finally:
            await pool.stop()

    async def test_no_overflow_when_disabled(self):
        pool = self._make_pool()  # max_loaded_workers=0 (default)
        await pool.start()
        try:
            for i, aid in enumerate(("a1", "a2", "a3")):
                await pool.acquire(f"s{i}", aid)
            self.assertEqual(await pool._evict_overflow(), 0)
            self.assertEqual(len(pool._workers), 3)
        finally:
            for i in range(3):
                await pool.release_async(f"s{i}")
            await pool.stop()

    async def test_overflow_skips_waited_resources(self):
        pool = self._make_pool(max_loaded_workers=1)
        await pool.start()
        try:
            await pool.acquire("s1", "a1")
            await pool.release_async("s1")
            await pool.acquire("s2", "a2")
            # Simulate a waiter on a1 so it must not be evicted.
            from collections import deque

            from liveavatar._common.pool import Waiter

            waiter = Waiter("s3", "a1", asyncio.get_running_loop().time() + 5.0)
            pool._waiters["a1"] = deque([waiter])
            n = await pool._evict_overflow()
            self.assertEqual(n, 0)
            self.assertIn("a1", pool._workers)
        finally:
            await pool.stop()
            pool._waiters.clear()


class TestEvictHookUnloadsAvatarData(unittest.IsolatedAsyncioTestCase):
    async def test_avatar_pool_hook_calls_unload(self):
        """AvatarPool._on_worker_evicted calls worker.unload_avatar_data."""
        unloaded: list[str] = []

        class _UnloadWorker(_FakeUnloadWorker):
            def unload_avatar_data(self) -> None:
                unloaded.append(self.avatar_id)

        config = AvatarPoolConfig(avatar_data_root="/nonexistent", max_workers=4)
        pool = AvatarPool(
            config, worker_factory=lambda a: _UnloadWorker(a)
        )
        pool._avatars = {"a1": _make_assets("a1")}
        await pool.start()
        try:
            await pool.acquire("s1", "a1")
            await pool.release_async("s1")
            self.assertTrue(await pool.evict_worker("a1"))
            self.assertEqual(unloaded, ["a1"])
        finally:
            await pool.stop()


class _FakeUnloadWorker(AvatarWorker):
    """Worker with a countable unload_avatar_data hook."""

    def __init__(self, assets: AvatarAssets) -> None:
        super().__init__(assets, width=4, height=4, batch_size=1)

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        return [(b"\x00" * 48, True)]

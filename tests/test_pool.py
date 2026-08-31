"""Tests for AvatarPool: acquire, release, wait queue, reaper, limits.

Aligned to the current source API in ``liveavatar/pool.py``:
- ``AvatarAssets`` fields: ``data_dir`` / ``full_imgs_dir`` / ``mask_dir``.
- ``discover_avatars`` requires ``coords.pkl`` + ``latents.pt`` +
  ``mask_coords.pkl``.
- ``AvatarWorker`` subclass implements sync ``_infer_batch``.
- ``AvatarPool.stats()`` returns a snapshot dict.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from liveavatar.config import AvatarPoolConfig
from liveavatar.pool import (
    AvatarNotFound,
    AvatarPool,
    AvatarPoolExhausted,
    GpuMemoryExhausted,
    discover_avatars,
)
from liveavatar.worker import AvatarAssets, AvatarWorker
from tests.conftest import make_assets as _make_assets

# ── Fake worker for pool tests (no torch) ──


class _FakeAvatarWorker(AvatarWorker):
    """Minimal worker that yields a single BGR24 frame instantly."""

    def __init__(self, assets: AvatarAssets) -> None:
        super().__init__(assets, width=4, height=4, batch_size=1)

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        return [(b"\x00" * 48, True)]


# ── Worker factory for tests ──


def _make_worker_factory():
    """Return a factory that creates a fake AvatarWorker."""

    def factory(assets: AvatarAssets) -> AvatarWorker:
        return _FakeAvatarWorker(assets)

    return factory


# ── Helpers ──


def _create_temp_avatars(
    avatar_ids: list[str],
) -> tuple[str, tempfile.TemporaryDirectory]:
    """Create a temp avatars/ dir with dummy avatar folders.

    Each folder gets ``coords.pkl`` + ``latents.pt`` + ``mask_coords.pkl``
    (the minimum required by ``discover_avatars``). Returns
    (path, temp_dir_object); the caller must keep the temp_dir_object alive
    and call cleanup() when done.
    """
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for aid in avatar_ids:
        d = root / aid
        d.mkdir(parents=True)
        (d / "coords.pkl").write_bytes(b"fake")
        (d / "latents.pt").write_bytes(b"fake")
        (d / "mask_coords.pkl").write_bytes(b"fake")
    return tmp.name, tmp


# ────────────────────────────────────────────────────── discover_avatars


class TestDiscoverAvatars(unittest.TestCase):
    def test_finds_valid_avatars(self):
        tmp_dir, tmp_obj = _create_temp_avatars(["nahida", "jingyuan"])
        try:
            avatars = discover_avatars(tmp_dir)
            self.assertEqual(sorted(avatars.keys()), ["jingyuan", "nahida"])
        finally:
            tmp_obj.cleanup()

    def test_resolves_all_asset_paths(self):
        tmp_dir, tmp_obj = _create_temp_avatars(["nahida"])
        try:
            avatars = discover_avatars(tmp_dir)
            a = avatars["nahida"]
            self.assertEqual(a.avatar_id, "nahida")
            self.assertTrue(a.coords_path.endswith("coords.pkl"))
            self.assertTrue(a.latents_path.endswith("latents.pt"))
            self.assertTrue(a.mask_coords_path.endswith("mask_coords.pkl"))
            self.assertTrue(a.full_imgs_dir.endswith("full_imgs"))
            self.assertTrue(a.mask_dir.endswith("mask"))
        finally:
            tmp_obj.cleanup()

    def test_skips_incomplete_folders(self):
        """Folders missing mask_coords.pkl are skipped."""
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        # Complete avatar.
        (root / "good").mkdir()
        (root / "good" / "coords.pkl").write_bytes(b"")
        (root / "good" / "latents.pt").write_bytes(b"")
        (root / "good" / "mask_coords.pkl").write_bytes(b"")
        # Missing mask_coords.pkl.
        (root / "bad").mkdir()
        (root / "bad" / "coords.pkl").write_bytes(b"")
        (root / "bad" / "latents.pt").write_bytes(b"")
        try:
            avatars = discover_avatars(tmp.name)
            self.assertEqual(list(avatars.keys()), ["good"])
        finally:
            tmp.cleanup()

    def test_nonexistent_root_returns_empty(self):
        self.assertEqual(discover_avatars("/nonexistent/path"), {})


# ──────────────────────────────────────────────────── acquire / release


class TestAvatarPoolAcquireRelease(unittest.IsolatedAsyncioTestCase):
    def _make_pool(self, max_workers: int = 4, **kwargs) -> AvatarPool:
        config = AvatarPoolConfig(
            avatar_data_root="/nonexistent",  # we inject avatars manually
            max_workers=max_workers,
            lease_ttl=60.0,
            reap_interval=1.0,
            acquire_timeout=1.0,
            **kwargs,
        )
        pool = AvatarPool(config, worker_factory=_make_worker_factory())
        # Inject avatars directly for testing.
        pool._avatars = {
            "nahida": _make_assets("nahida"),
            "jingyuan": _make_assets("jingyuan"),
        }
        return pool

    async def test_acquire_returns_lease(self):
        pool = self._make_pool()
        await pool.start()
        try:
            lease = await pool.acquire("s1", "nahida")
            self.assertEqual(lease.session_id, "s1")
            self.assertEqual(lease.avatar_id, "nahida")
            self.assertIsNotNone(lease.worker)
        finally:
            await pool.stop()

    async def test_acquire_same_avatar_renews(self):
        pool = self._make_pool()
        await pool.start()
        try:
            lease1 = await pool.acquire("s1", "nahida")
            old_count = lease1.renew_count
            lease2 = await pool.acquire("s1", "nahida")
            self.assertIs(lease1, lease2)
            self.assertEqual(lease2.renew_count, old_count + 1)
        finally:
            await pool.stop()

    async def test_acquire_different_avatar_releases_old(self):
        pool = self._make_pool()
        await pool.start()
        try:
            lease1 = await pool.acquire("s1", "nahida")
            lease2 = await pool.acquire("s1", "jingyuan")
            self.assertEqual(lease2.avatar_id, "jingyuan")
            self.assertNotEqual(lease1.avatar_id, lease2.avatar_id)
            # The old lease is replaced by the new one under "s1".
            self.assertIn("s1", pool.active_leases)
            self.assertEqual(pool.active_leases["s1"].avatar_id, "jingyuan")
        finally:
            await pool.stop()

    async def test_release_frees_worker(self):
        pool = self._make_pool()
        await pool.start()
        try:
            await pool.acquire("s1", "nahida")
            self.assertIn("s1", pool.active_leases)
            pool.release("s1")
            await asyncio.sleep(0.05)  # let async release complete
            self.assertNotIn("s1", pool.active_leases)
        finally:
            await pool.stop()

    async def test_release_unknown_session_returns_false(self):
        pool = self._make_pool()
        await pool.start()
        try:
            self.assertFalse(pool.release("nonexistent"))
        finally:
            await pool.stop()

    async def test_avatar_not_found_raises(self):
        pool = self._make_pool()
        await pool.start()
        try:
            with self.assertRaises(AvatarNotFound):
                await pool.acquire("s1", "nonexistent")
        finally:
            await pool.stop()

    async def test_concurrent_different_avatars_no_block(self):
        """Two sessions using different avatars acquire instantly."""
        pool = self._make_pool()
        await pool.start()
        try:
            l1, l2 = await asyncio.gather(
                pool.acquire("s1", "nahida"),
                pool.acquire("s2", "jingyuan"),
            )
            self.assertEqual(l1.avatar_id, "nahida")
            self.assertEqual(l2.avatar_id, "jingyuan")
        finally:
            await pool.stop()


# ──────────────────────────────────────────────────────── wait queue


class TestAvatarPoolWaitQueue(unittest.IsolatedAsyncioTestCase):
    def _make_pool(self, acquire_timeout: float = 0.5) -> AvatarPool:
        config = AvatarPoolConfig(
            avatar_data_root="/nonexistent",
            max_workers=4,
            lease_ttl=60.0,
            reap_interval=0.1,
            acquire_timeout=acquire_timeout,
        )
        pool = AvatarPool(config, worker_factory=_make_worker_factory())
        pool._avatars = {
            "nahida": _make_assets("nahida"),
        }
        return pool

    async def test_second_session_waits(self):
        """Session B waits while session A holds the only worker."""
        pool = self._make_pool(acquire_timeout=2.0)
        await pool.start()
        try:
            lease_a = await pool.acquire("sA", "nahida")
            self.assertFalse(lease_a.worker.busy)  # not synthesizing yet

            # sB should block.
            started = asyncio.Event()

            async def _try_acquire():
                started.set()
                return await pool.acquire("sB", "nahida")

            task = asyncio.create_task(_try_acquire())
            await started.wait()
            # Give it a moment to enter the wait queue.
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())

            # Release sA → sB should get the worker.
            pool.release("sA")
            await asyncio.sleep(0.05)

            lease_b = await asyncio.wait_for(task, timeout=1.0)
            self.assertEqual(lease_b.session_id, "sB")
            self.assertEqual(lease_b.avatar_id, "nahida")
        finally:
            await pool.stop()

    async def test_acquire_timeout_raises_exhausted(self):
        """When the wait times out, AvatarPoolExhausted is raised."""
        pool = self._make_pool(acquire_timeout=0.1)
        await pool.start()
        try:
            await pool.acquire("sA", "nahida")
            with self.assertRaises(AvatarPoolExhausted):
                await pool.acquire("sB", "nahida")
        finally:
            await pool.stop()

    async def test_release_dispatches_to_waiter_fifo(self):
        """Releasing a lease fulfills waiters in FIFO order."""
        pool = self._make_pool(acquire_timeout=5.0)
        await pool.start()
        try:
            await pool.acquire("sA", "nahida")

            # Enqueue sB.
            task_b = asyncio.create_task(pool.acquire("sB", "nahida"))
            await asyncio.sleep(0.05)
            self.assertFalse(task_b.done())

            # Enqueue sC.
            task_c = asyncio.create_task(pool.acquire("sC", "nahida"))
            await asyncio.sleep(0.05)

            # Release → sB gets it first (FIFO).
            pool.release("sA")
            lease_b = await asyncio.wait_for(task_b, timeout=1.0)
            self.assertEqual(lease_b.session_id, "sB")

            # sC still waiting.
            self.assertFalse(task_c.done())

            # Release sB → sC gets it.
            pool.release("sB")
            lease_c = await asyncio.wait_for(task_c, timeout=1.0)
            self.assertEqual(lease_c.session_id, "sC")
        finally:
            await pool.stop()


# ──────────────────────────────────────────────────────────── reaper


class TestAvatarPoolReaper(unittest.IsolatedAsyncioTestCase):
    def _make_pool(
        self, lease_ttl: float = 0.2, reap_interval: float = 0.05
    ) -> AvatarPool:
        config = AvatarPoolConfig(
            avatar_data_root="/nonexistent",
            max_workers=4,
            lease_ttl=lease_ttl,
            reap_interval=reap_interval,
            acquire_timeout=1.0,
        )
        pool = AvatarPool(config, worker_factory=_make_worker_factory())
        pool._avatars = {
            "nahida": _make_assets("nahida"),
        }
        return pool

    async def test_expired_lease_is_reaped(self):
        pool = self._make_pool(lease_ttl=0.1, reap_interval=0.05)
        await pool.start()
        try:
            await pool.acquire("s1", "nahida")
            self.assertIn("s1", pool.active_leases)

            # Wait for lease to expire + reaper to run.
            await asyncio.sleep(0.3)
            self.assertNotIn("s1", pool.active_leases)
        finally:
            await pool.stop()

    async def test_reaped_lease_dispatches_waiter(self):
        pool = self._make_pool(lease_ttl=0.1, reap_interval=0.05)
        await pool.start()
        try:
            await pool.acquire("sA", "nahida")
            # sB waits.
            task_b = asyncio.create_task(
                pool.acquire("sB", "nahida", timeout=2.0)
            )
            await asyncio.sleep(0.05)

            # Wait for sA's lease to expire and reaper to reclaim.
            lease_b = await asyncio.wait_for(task_b, timeout=1.0)
            self.assertEqual(lease_b.session_id, "sB")
        finally:
            await pool.stop()


# ──────────────────────────────────────────────────────── max_workers


class TestAvatarPoolMaxWorkers(unittest.IsolatedAsyncioTestCase):
    async def test_max_workers_limits_loaded_avatars(self):
        config = AvatarPoolConfig(
            avatar_data_root="/nonexistent",
            max_workers=1,
            lease_ttl=60.0,
            reap_interval=1.0,
            acquire_timeout=1.0,
        )
        pool = AvatarPool(config, worker_factory=_make_worker_factory())
        pool._avatars = {
            "nahida": _make_assets("nahida"),
            "jingyuan": _make_assets("jingyuan"),
        }
        await pool.start()
        try:
            # Load first avatar — OK.
            await pool.acquire("s1", "nahida")
            pool.release("s1")
            await asyncio.sleep(0.05)

            # Try to load second — should fail (max_workers=1).
            with self.assertRaises(GpuMemoryExhausted):
                await pool.acquire("s2", "jingyuan")
        finally:
            await pool.stop()


# ──────────────────────────────────────────────────────────── renew


class TestAvatarPoolRenew(unittest.IsolatedAsyncioTestCase):
    async def test_renew_extends_lease(self):
        config = AvatarPoolConfig(
            avatar_data_root="/nonexistent",
            max_workers=4,
            lease_ttl=1.0,
            reap_interval=0.5,
            acquire_timeout=1.0,
        )
        pool = AvatarPool(config, worker_factory=_make_worker_factory())
        pool._avatars = {
            "nahida": _make_assets("nahida"),
        }
        await pool.start()
        try:
            lease = await pool.acquire("s1", "nahida")
            old_deadline = lease.deadline
            await asyncio.sleep(0.02)  # ensure monotonic clock advances
            renewed = pool.renew("s1")
            self.assertIsNotNone(renewed)
            self.assertGreater(renewed.deadline, old_deadline)
            self.assertEqual(renewed.renew_count, 1)
        finally:
            await pool.stop()

    async def test_renew_unknown_session_returns_none(self):
        config = AvatarPoolConfig(avatar_data_root="/nonexistent", max_workers=4)
        pool = AvatarPool(config, worker_factory=_make_worker_factory())
        self.assertIsNone(pool.renew("nonexistent"))


# ──────────────────────────────────────────────────────────── stats


class TestAvatarPoolStats(unittest.IsolatedAsyncioTestCase):
    async def test_stats_snapshot(self):
        config = AvatarPoolConfig(
            avatar_data_root="/nonexistent",
            max_workers=4,
            lease_ttl=60.0,
            reap_interval=1.0,
            acquire_timeout=1.0,
        )
        pool = AvatarPool(config, worker_factory=_make_worker_factory())
        pool._avatars = {
            "nahida": _make_assets("nahida"),
        }
        await pool.start()
        try:
            await pool.acquire("s1", "nahida")
            stats = pool.stats()
            self.assertTrue(stats["started"])
            self.assertEqual(stats["loaded_workers"], 1)
            self.assertIn("nahida", stats["loaded_worker_ids"])
            self.assertEqual(stats["active_leases"], 1)
            self.assertIn("s1", stats["active_lease_sessions"])
        finally:
            await pool.stop()


# ──────────────────────────────────────────────────── cross-talk free


class TestNoCrossTalk(unittest.IsolatedAsyncioTestCase):
    """The core guarantee: two sessions using different avatars never
    interfere with each other's worker state."""

    async def test_concurrent_different_avatars_isolated(self):
        config = AvatarPoolConfig(
            avatar_data_root="/nonexistent",
            max_workers=4,
            lease_ttl=60.0,
            reap_interval=1.0,
            acquire_timeout=1.0,
        )
        pool = AvatarPool(config, worker_factory=_make_worker_factory())
        pool._avatars = {
            "nahida": _make_assets("nahida"),
            "jingyuan": _make_assets("jingyuan"),
        }
        await pool.start()
        try:
            # Acquire both simultaneously.
            l1, l2 = await asyncio.gather(
                pool.acquire("s1", "nahida"),
                pool.acquire("s2", "jingyuan"),
            )
            # Workers are distinct instances.
            self.assertIsNot(l1.worker, l2.worker)
            self.assertEqual(l1.worker.avatar_id, "nahida")
            self.assertEqual(l2.worker.avatar_id, "jingyuan")

            # Synthesize concurrently — no interference.
            async def _synth(lease):
                frames = []
                async for frame in lease.worker.synthesize_video_stream(
                    b"pcm", pts_us=0, epoch=0
                ):
                    frames.append(frame)
                return frames

            c1, c2 = await asyncio.gather(_synth(l1), _synth(l2))
            self.assertEqual(len(c1), 1)
            self.assertEqual(len(c2), 1)

            # Each worker's stats are independent.
            self.assertEqual(l1.worker.stats.frames_produced, 1)
            self.assertEqual(l2.worker.stats.frames_produced, 1)
        finally:
            await pool.stop()


if __name__ == "__main__":
    unittest.main()

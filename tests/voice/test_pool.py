"""Tests for VoicePool: acquire, release, wait queue, reaper, limits."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections.abc import Generator
from pathlib import Path

import numpy as np

from liveavatar.voice.config import VoicePoolConfig
from liveavatar.voice.pool import (
    CharacterNotFound,
    GpuMemoryExhausted,
    VoicePool,
    VoicePoolExhausted,
    discover_characters,
)
from liveavatar.voice.worker import CharacterAssets, NvcWorker

# ── Fake TTS for pool tests ──

class _FakeTts:
    """Yields a single chunk instantly."""

    def run(self, inputs: dict) -> Generator[tuple[int, np.ndarray], None, None]:
        yield (16000, np.zeros(320, dtype=np.float32))


# ── Worker factory for tests (no torch) ──

def _make_worker_factory(target_sr: int = 16000):
    """Return a factory that creates NvcWorker with a fake TTS."""
    def factory(assets: CharacterAssets) -> NvcWorker:
        return NvcWorker(assets, _FakeTts(), target_sample_rate=target_sr)
    return factory


# ── Temp weights dir for character discovery tests ──

def _create_temp_weights(characters: list[str]) -> tuple[str, tempfile.TemporaryDirectory]:
    """Create a temp weights/ dir with dummy character folders.

    Returns (path, temp_dir_object). The caller must keep the temp_dir_object
    alive and call cleanup() when done.
    """
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for char_id in characters:
        char_dir = root / char_id
        char_dir.mkdir(parents=True)
        (char_dir / "model.ckpt").write_bytes(b"fake")
        (char_dir / "model.pth").write_bytes(b"fake")
        (char_dir / f"{char_id}_ref.wav").write_bytes(b"fake")
    return tmp.name, tmp


class TestDiscoverCharacters(unittest.TestCase):
    def test_finds_valid_characters(self):
        tmp_dir, tmp_obj = _create_temp_weights(["nahida", "jingyuan"])
        try:
            chars = discover_characters(tmp_dir)
            self.assertEqual(sorted(chars.keys()), ["jingyuan", "nahida"])
        finally:
            tmp_obj.cleanup()

    def test_ref_text_is_filename_without_extension(self):
        tmp_dir, tmp_obj = _create_temp_weights(["nahida"])
        try:
            chars = discover_characters(tmp_dir)
            self.assertEqual(chars["nahida"].ref_text, "nahida_ref")
        finally:
            tmp_obj.cleanup()

    def test_skips_incomplete_folders(self):
        """Folders missing .wav are skipped."""
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        # Complete character.
        (root / "good").mkdir()
        (root / "good" / "a.ckpt").write_bytes(b"")
        (root / "good" / "a.pth").write_bytes(b"")
        (root / "good" / "a.wav").write_bytes(b"")
        # Missing wav.
        (root / "bad").mkdir()
        (root / "bad" / "a.ckpt").write_bytes(b"")
        (root / "bad" / "a.pth").write_bytes(b"")
        try:
            chars = discover_characters(tmp.name)
            self.assertEqual(list(chars.keys()), ["good"])
        finally:
            tmp.cleanup()

    def test_nonexistent_root_returns_empty(self):
        self.assertEqual(discover_characters("/nonexistent/path"), {})


class TestVoicePoolAcquireRelease(unittest.IsolatedAsyncioTestCase):
    def _make_pool(self, max_workers: int = 4, **kwargs) -> VoicePool:
        config = VoicePoolConfig(
            weights_root="/nonexistent",  # we inject characters manually
            max_workers=max_workers,
            lease_ttl=60.0,
            reap_interval=1.0,
            acquire_timeout=1.0,
            **kwargs,
        )
        pool = VoicePool(config, worker_factory=_make_worker_factory())
        # Inject characters directly for testing.
        pool._characters = {
            "nahida": CharacterAssets(
                char_id="nahida",
                gpt_path="fake.ckpt",
                sovits_path="fake.pth",
                ref_audio_path="fake.wav",
                ref_text="你好",
            ),
            "jingyuan": CharacterAssets(
                char_id="jingyuan",
                gpt_path="fake.ckpt",
                sovits_path="fake.pth",
                ref_audio_path="fake.wav",
                ref_text="我是景元",
            ),
        }
        return pool

    async def test_acquire_returns_lease(self):
        pool = self._make_pool()
        await pool.start()
        try:
            lease = await pool.acquire("s1", "nahida")
            self.assertEqual(lease.session_id, "s1")
            self.assertEqual(lease.char_id, "nahida")
            self.assertIsNotNone(lease.worker)
        finally:
            await pool.stop()

    async def test_acquire_same_char_renews(self):
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

    async def test_acquire_different_char_releases_old(self):
        pool = self._make_pool()
        await pool.start()
        try:
            lease1 = await pool.acquire("s1", "nahida")
            lease2 = await pool.acquire("s1", "jingyuan")
            self.assertEqual(lease2.char_id, "jingyuan")
            self.assertNotEqual(lease1.char_id, lease2.char_id)
            # The old lease is replaced by the new one under "s1".
            self.assertIn("s1", pool.active_leases)
            self.assertEqual(pool.active_leases["s1"].char_id, "jingyuan")
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

    async def test_character_not_found_raises(self):
        pool = self._make_pool()
        await pool.start()
        try:
            with self.assertRaises(CharacterNotFound):
                await pool.acquire("s1", "nonexistent")
        finally:
            await pool.stop()

    async def test_concurrent_different_chars_no_block(self):
        """Two sessions using different characters acquire instantly."""
        pool = self._make_pool()
        await pool.start()
        try:
            l1, l2 = await asyncio.gather(
                pool.acquire("s1", "nahida"),
                pool.acquire("s2", "jingyuan"),
            )
            self.assertEqual(l1.char_id, "nahida")
            self.assertEqual(l2.char_id, "jingyuan")
        finally:
            await pool.stop()


class TestVoicePoolWaitQueue(unittest.IsolatedAsyncioTestCase):
    def _make_pool(self, acquire_timeout: float = 0.5) -> VoicePool:
        config = VoicePoolConfig(
            weights_root="/nonexistent",
            max_workers=4,
            lease_ttl=60.0,
            reap_interval=0.1,
            acquire_timeout=acquire_timeout,
        )
        pool = VoicePool(config, worker_factory=_make_worker_factory())
        pool._characters = {
            "nahida": CharacterAssets(
                char_id="nahida",
                gpt_path="f.ckpt",
                sovits_path="f.pth",
                ref_audio_path="f.wav",
                ref_text="你好",
            ),
        }
        return pool

    async def test_second_session_waits(self):
        """Session B waits while session A holds the only worker."""
        pool = self._make_pool(acquire_timeout=2.0)
        await pool.start()
        try:
            lease_a = await pool.acquire("sA", "nahida")
            self.assertTrue(lease_a.worker.busy is False)  # not synthesizing yet

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
            self.assertEqual(lease_b.char_id, "nahida")
        finally:
            await pool.stop()

    async def test_acquire_timeout_raises_exhausted(self):
        """When the wait times out, VoicePoolExhausted is raised."""
        pool = self._make_pool(acquire_timeout=0.1)
        await pool.start()
        try:
            await pool.acquire("sA", "nahida")
            with self.assertRaises(VoicePoolExhausted):
                await pool.acquire("sB", "nahida")
        finally:
            await pool.stop()

    async def test_release_dispatches_to_waiter(self):
        """Releasing a lease fulfills the next waiter."""
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


class TestVoicePoolReaper(unittest.IsolatedAsyncioTestCase):
    def _make_pool(self, lease_ttl: float = 0.2, reap_interval: float = 0.05) -> VoicePool:
        config = VoicePoolConfig(
            weights_root="/nonexistent",
            max_workers=4,
            lease_ttl=lease_ttl,
            reap_interval=reap_interval,
            acquire_timeout=1.0,
        )
        pool = VoicePool(config, worker_factory=_make_worker_factory())
        pool._characters = {
            "nahida": CharacterAssets(
                char_id="nahida",
                gpt_path="f.ckpt",
                sovits_path="f.pth",
                ref_audio_path="f.wav",
                ref_text="你好",
            ),
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
            task_b = asyncio.create_task(pool.acquire("sB", "nahida", timeout=2.0))
            await asyncio.sleep(0.05)

            # Wait for sA's lease to expire and reaper to reclaim.
            lease_b = await asyncio.wait_for(task_b, timeout=1.0)
            self.assertEqual(lease_b.session_id, "sB")
        finally:
            await pool.stop()


class TestVoicePoolMaxWorkers(unittest.IsolatedAsyncioTestCase):
    async def test_max_workers_limits_loaded_characters(self):
        config = VoicePoolConfig(
            weights_root="/nonexistent",
            max_workers=1,
            lease_ttl=60.0,
            reap_interval=1.0,
            acquire_timeout=1.0,
        )
        pool = VoicePool(config, worker_factory=_make_worker_factory())
        pool._characters = {
            "nahida": CharacterAssets("nahida", "f.ckpt", "f.pth", "f.wav", "你好"),
            "jingyuan": CharacterAssets("jingyuan", "f.ckpt", "f.pth", "f.wav", "景元"),
        }
        await pool.start()
        try:
            # Load first character — OK.
            await pool.acquire("s1", "nahida")
            pool.release("s1")
            await asyncio.sleep(0.05)

            # Try to load second — should fail (max_workers=1).
            with self.assertRaises(GpuMemoryExhausted):
                await pool.acquire("s2", "jingyuan")
        finally:
            await pool.stop()


class TestVoicePoolRenew(unittest.IsolatedAsyncioTestCase):
    async def test_renew_extends_lease(self):
        config = VoicePoolConfig(
            weights_root="/nonexistent",
            max_workers=4,
            lease_ttl=1.0,
            reap_interval=0.5,
            acquire_timeout=1.0,
        )
        pool = VoicePool(config, worker_factory=_make_worker_factory())
        pool._characters = {
            "nahida": CharacterAssets("nahida", "f.ckpt", "f.pth", "f.wav", "你好"),
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
        config = VoicePoolConfig(weights_root="/nonexistent", max_workers=4)
        pool = VoicePool(config, worker_factory=_make_worker_factory())
        self.assertIsNone(pool.renew("nonexistent"))


class TestVoicePoolStats(unittest.IsolatedAsyncioTestCase):
    async def test_stats_snapshot(self):
        config = VoicePoolConfig(
            weights_root="/nonexistent",
            max_workers=4,
            lease_ttl=60.0,
            reap_interval=1.0,
            acquire_timeout=1.0,
        )
        pool = VoicePool(config, worker_factory=_make_worker_factory())
        pool._characters = {
            "nahida": CharacterAssets("nahida", "f.ckpt", "f.pth", "f.wav", "你好"),
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


class TestNoCrossTalk(unittest.IsolatedAsyncioTestCase):
    """The core guarantee: two sessions using different characters never
    interfere with each other's worker state."""

    async def test_concurrent_different_chars_isolated(self):
        config = VoicePoolConfig(
            weights_root="/nonexistent",
            max_workers=4,
            lease_ttl=60.0,
            reap_interval=1.0,
            acquire_timeout=1.0,
        )
        pool = VoicePool(config, worker_factory=_make_worker_factory())
        pool._characters = {
            "nahida": CharacterAssets("nahida", "f.ckpt", "f.pth", "f.wav", "纳西妲"),
            "jingyuan": CharacterAssets("jingyuan", "f.ckpt", "f.pth", "f.wav", "景元"),
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
            self.assertEqual(l1.worker.char_id, "nahida")
            self.assertEqual(l2.worker.char_id, "jingyuan")

            # Synthesize concurrently — no interference.
            async def _synth(lease):
                chunks = []
                async for pcm in lease.worker.synthesize_stream("hello"):
                    chunks.append(pcm)
                return chunks

            c1, c2 = await asyncio.gather(_synth(l1), _synth(l2))
            self.assertEqual(len(c1), 1)
            self.assertEqual(len(c2), 1)

            # Each worker's stats are independent.
            self.assertEqual(l1.worker.stats.syntheses_completed, 1)
            self.assertEqual(l2.worker.stats.syntheses_completed, 1)
        finally:
            await pool.stop()


if __name__ == "__main__":
    unittest.main()

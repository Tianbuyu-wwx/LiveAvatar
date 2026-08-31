"""Step 5 integration tests: LiveKitWorkerRuntime voice-pool wiring.

Verifies that the runtime correctly constructs an ``NvcStreamingTtsAdapter``
in pool mode, injects it into the worker, and manages the lease lifecycle
(acquire on start, renew, release on stop).

Uses a real ``VoicePool`` with a temp weights dir and a fake worker factory
(no torch/GPU dependency). The LiveKit connection layer is bypassed by
exercising the pool/lease lifecycle directly (mirroring the exact sequence
``start()`` / ``stop()`` execute for the voice-pool portion).
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections.abc import Generator
from pathlib import Path

import numpy as np

from liveavatar.runtime.fake_tts import FakeTts
from liveavatar.runtime.livekit_runtime import LiveKitWorkerRuntime
from liveavatar.runtime.worker import RealtimeWorker
from liveavatar.tts import NvcStreamingTtsAdapter
from liveavatar.voice.config import VoicePoolConfig
from liveavatar.voice.pool import VoicePool
from liveavatar.voice.worker import CharacterAssets, NvcWorker

# ──────────────────────────────────────── fakes

class _FakeTts:
    """Yields a single PCM chunk instantly (no torch)."""

    def run(self, inputs: dict) -> Generator[tuple[int, np.ndarray], None, None]:
        yield (16000, np.zeros(320, dtype=np.float32))


def _make_worker_factory(target_sr: int = 16000):
    """Factory that builds NvcWorker with a fake TTS (no torch)."""

    def factory(assets: CharacterAssets) -> NvcWorker:
        return NvcWorker(assets, _FakeTts(), target_sample_rate=target_sr)

    return factory


def _create_temp_weights(characters: list[str]) -> tuple[str, tempfile.TemporaryDirectory]:
    """Create a temp weights/ dir with dummy character folders."""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for char_id in characters:
        char_dir = root / char_id
        char_dir.mkdir(parents=True)
        (char_dir / "model.ckpt").write_bytes(b"fake")
        (char_dir / "model.pth").write_bytes(b"fake")
        (char_dir / f"{char_id}_ref.wav").write_bytes(b"fake")
    return tmp.name, tmp


def _make_pool(characters: list[str]) -> tuple[VoicePool, tempfile.TemporaryDirectory]:
    """Build a real VoicePool over a temp weights dir with fake workers."""
    weights_root, tmp_obj = _create_temp_weights(characters)
    config = VoicePoolConfig(
        weights_root=weights_root,
        device="cpu",
        is_half=False,
        max_workers=4,
        lease_ttl=2.0,  # short for reaper tests
        reap_interval=0.5,
        acquire_timeout=1.0,
    )
    pool = VoicePool(config, worker_factory=_make_worker_factory())
    return pool, tmp_obj


# ──────────────────────────────────────── construction tests

class TestRuntimeConstruction(unittest.TestCase):
    """Verify __init__ wiring of voice_pool / char_id / adapter."""

    def test_pool_mode_constructs_adapter(self):
        pool, tmp = _make_pool(["nahida"])
        try:
            rt = LiveKitWorkerRuntime(
                "s1", "ws://x", "tok", voice_pool=pool, char_id="nahida"
            )
            self.assertIsInstance(rt.worker.tts, NvcStreamingTtsAdapter)
            self.assertIs(rt._tts_adapter, rt.worker.tts)
            self.assertIs(rt._voice_pool, pool)
            self.assertFalse(rt._owns_pool)  # external pool
            self.assertEqual(rt._tts_adapter._char_id, "nahida")
        finally:
            tmp.cleanup()

    def test_owned_pool_from_config(self):
        weights_root, tmp = _create_temp_weights(["nahida"])
        try:
            config = VoicePoolConfig(
                weights_root=weights_root, device="cpu", is_half=False
            )
            rt = LiveKitWorkerRuntime(
                "s2", "ws://x", "tok",
                voice_pool_config=config, char_id="nahida",
            )
            self.assertTrue(rt._owns_pool)
            self.assertIsInstance(rt._voice_pool, VoicePool)
            self.assertIsInstance(rt.worker.tts, NvcStreamingTtsAdapter)
        finally:
            tmp.cleanup()

    def test_no_pool_defaults_to_faketts(self):
        rt = LiveKitWorkerRuntime("s3", "ws://x", "tok")
        self.assertIsInstance(rt.worker.tts, FakeTts)
        self.assertIsNone(rt._tts_adapter)
        self.assertIsNone(rt._voice_pool)

    def test_char_id_without_pool_falls_back(self):
        rt = LiveKitWorkerRuntime("s4", "ws://x", "tok", char_id="nahida")
        # No pool → default FakeTts, no adapter.
        self.assertIsInstance(rt.worker.tts, FakeTts)
        self.assertIsNone(rt._tts_adapter)

    def test_prebuilt_worker_with_pool_is_honored(self):
        pool, tmp = _make_pool(["nahida"])
        try:
            worker = RealtimeWorker("s5", capacity=10)  # default FakeTts
            rt = LiveKitWorkerRuntime(
                "s5", "ws://x", "tok",
                worker=worker, voice_pool=pool, char_id="nahida",
            )
            # Pre-built worker used as-is; tts NOT injected.
            self.assertIs(rt.worker, worker)
            self.assertIsInstance(rt.worker.tts, FakeTts)
            self.assertIsNone(rt._tts_adapter)
        finally:
            tmp.cleanup()


# ──────────────────────────────────────── lease lifecycle tests

class TestLeaseLifecycle(unittest.IsolatedAsyncioTestCase):
    """Verify the pool/lease sequence that start()/stop() execute."""

    async def test_acquire_and_release_lifecycle(self):
        pool, tmp = _make_pool(["nahida"])
        try:
            await pool.start()
            rt = LiveKitWorkerRuntime(
                "sess_lease", "ws://x", "tok",
                voice_pool=pool, char_id="nahida",
            )
            adapter = rt._tts_adapter
            self.assertIsNotNone(adapter)

            # Mirror start(): acquire lease → adapter has a bound worker.
            await adapter.acquire()
            self.assertTrue(adapter.has_worker)
            self.assertEqual(adapter.char_id, "nahida")

            # The pool records the lease.
            self.assertIn("sess_lease", pool._leases)

            # Mirror stop(): release lease → pool reclaims worker.
            await adapter.release()
            self.assertNotIn("sess_lease", pool._leases)

            await pool.stop()
        finally:
            tmp.cleanup()

    async def test_owned_pool_start_stop(self):
        """Runtime-owned pool is started/stopped across the lifecycle."""
        weights_root, tmp = _create_temp_weights(["nahida"])
        try:
            config = VoicePoolConfig(
                weights_root=weights_root,
                device="cpu",
                is_half=False,
                lease_ttl=2.0,
                reap_interval=0.5,
            )
            rt = LiveKitWorkerRuntime(
                "sess_owned", "ws://x", "tok",
                voice_pool_config=config,
                voice_pool_worker_factory=_make_worker_factory(),
                char_id="nahida",
            )
            self.assertTrue(rt._owns_pool)

            # Mirror start()'s pool/lease portion.
            await rt._voice_pool.start()
            await rt._tts_adapter.acquire()
            self.assertTrue(rt._tts_adapter.has_worker)
            self.assertTrue(rt._voice_pool._started)

            # Mirror stop()'s pool/lease portion.
            await rt._tts_adapter.release()
            await rt._voice_pool.stop()
            self.assertFalse(rt._voice_pool._started)
        finally:
            tmp.cleanup()

    async def test_streaming_synthesis_works_after_acquire(self):
        """End-to-end: after acquire, the adapter streams PCM via the pool."""
        pool, tmp = _make_pool(["nahida"])
        try:
            await pool.start()
            rt = LiveKitWorkerRuntime(
                "sess_stream", "ws://x", "tok",
                voice_pool=pool, char_id="nahida",
            )
            await rt._tts_adapter.acquire()

            segments = []
            async for seg in rt._tts_adapter.synthesize_stream(
                "你好", epoch=0, pts_us=0
            ):
                segments.append(seg)
            self.assertEqual(len(segments), 1)  # _FakeTts yields one chunk
            self.assertGreater(len(segments[0].pcm_s16le), 0)

            await rt._tts_adapter.release()
            await pool.stop()
        finally:
            tmp.cleanup()


# ──────────────────────────────────────── lease renewer tests

class TestLeaseRenewer(unittest.IsolatedAsyncioTestCase):
    """Verify the renewer task is created/cancelled and renews the lease."""

    async def test_renewer_renews_lease(self):
        pool, tmp = _make_pool(["nahida"])
        try:
            await pool.start()
            rt = LiveKitWorkerRuntime(
                "sess_renew", "ws://x", "tok",
                voice_pool=pool, char_id="nahida",
                lease_renew_interval=0.05,  # fast for testing
            )
            await rt._tts_adapter.acquire()
            original_deadline = rt._tts_adapter._lease.deadline

            # Start the renewer (mirrors start()'s task creation).
            rt._running = True
            rt._lease_renewer = asyncio.create_task(rt._renew_lease_loop())

            # Wait > 1 renew interval.
            await asyncio.sleep(0.12)
            self.assertTrue(rt._lease_renewer.done() is False)  # still running

            # The lease deadline was pushed forward (renewed).
            self.assertGreater(rt._tts_adapter._lease.deadline, original_deadline)
            self.assertGreaterEqual(
                rt._tts_adapter._lease.renew_count, 1
            )

            # Mirror stop(): cancel renewer + release.
            rt._running = False
            rt._lease_renewer.cancel()
            try:
                await rt._lease_renewer
            except asyncio.CancelledError:
                pass
            await rt._tts_adapter.release()
            await pool.stop()
        finally:
            tmp.cleanup()

    async def test_renewer_cancelled_on_stop(self):
        pool, tmp = _make_pool(["nahida"])
        try:
            await pool.start()
            rt = LiveKitWorkerRuntime(
                "sess_cancel", "ws://x", "tok",
                voice_pool=pool, char_id="nahida",
                lease_renew_interval=0.05,
            )
            await rt._tts_adapter.acquire()
            rt._running = True
            rt._lease_renewer = asyncio.create_task(rt._renew_lease_loop())
            await asyncio.sleep(0.02)

            # Simulate the stop() renewer-cancellation block.
            rt._running = False
            rt._lease_renewer.cancel()
            try:
                await rt._lease_renewer
            except asyncio.CancelledError:
                pass
            # The task is finished (cancelled); stop() would set it to None.
            self.assertTrue(rt._lease_renewer.done())
            rt._lease_renewer = None
            await rt._tts_adapter.release()
            await pool.stop()
        finally:
            tmp.cleanup()


# ──────────────────────────────────────── backward compat

class TestBackwardCompat(unittest.IsolatedAsyncioTestCase):
    """No voice_pool → no lease operations, default FakeTts path."""

    async def test_no_pool_start_stop_is_noop_for_lease(self):
        rt = LiveKitWorkerRuntime("sess_compat", "ws://x", "tok")
        # No _tts_adapter, no _voice_pool → renewer/lease code is skipped.
        self.assertIsNone(rt._tts_adapter)
        self.assertIsNone(rt._voice_pool)
        self.assertIsNone(rt._lease_renewer)
        # _renew_lease_loop would NPE if called without adapter, but start()
        # only spawns it when _tts_adapter is set, so it's never called.

    async def test_default_worker_uses_faketts_sync_path(self):
        """Default worker (FakeTts) doesn't spawn streaming tasks."""
        worker = RealtimeWorker("sess_default", capacity=10)
        await worker.start()
        try:
            self.assertIsInstance(worker.tts, FakeTts)
            self.assertEqual(len(worker._tts_tasks), 0)
        finally:
            await worker.stop()


if __name__ == "__main__":
    unittest.main()

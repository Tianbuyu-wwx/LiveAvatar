"""Tests for AvatarPipeline (orchestrator) with a fake pool + fake workers.

Covers: open/close session lifecycle, auto PTS clock, epoch cancellation,
publisher_factory injection, stats, and error paths.
"""

from __future__ import annotations

import asyncio
import unittest

from liveavatar.config import AvatarPoolConfig
from liveavatar.pipeline import AvatarPipeline
from liveavatar.pool import AvatarNotFound
from liveavatar.worker import AvatarAssets, AvatarWorker

# ── fakes ──


def _make_assets(avatar_id: str = "yongen") -> AvatarAssets:
    base = f"avatars/{avatar_id}/"
    return AvatarAssets(
        avatar_id=avatar_id,
        data_dir=base,
        full_imgs_dir=base + "full_imgs",
        coords_path=base + "coords.pkl",
        latents_path=base + "latents.pt",
        mask_dir=base + "mask",
        mask_coords_path=base + "mask_coords.pkl",
    )


class _CountingWorker(AvatarWorker):
    """Yields batch_size frames per chunk, BGR24 4x4."""

    def __init__(self, assets: AvatarAssets, *, batch_size: int = 4) -> None:
        super().__init__(assets, target_fps=25, width=4, height=4, batch_size=batch_size)

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        return [(b"\x00" * 48, True) for _ in range(self.batch_size)]


class _RecordingSink:
    """Publisher-compatible sink capturing frames (like adapter capture mode)."""

    def __init__(self) -> None:
        self.frames = []
        self._current_epoch = 0

    async def publish_frame(self, frame, epoch: int) -> bool:
        if epoch < self._current_epoch:
            return False
        self.frames.append(frame)
        return True

    def cancel_epoch(self, new_epoch: int) -> None:
        if new_epoch > self._current_epoch:
            self._current_epoch = new_epoch

    @property
    def current_epoch(self) -> int:
        return self._current_epoch


class _FakeLease:
    def __init__(self, worker: AvatarWorker) -> None:
        self.worker = worker


class _FakePool:
    """AvatarPool-compatible fake."""

    def __init__(self, worker: AvatarWorker | None = None) -> None:
        self._worker = worker or _CountingWorker(_make_assets())
        self.released: list[str] = []
        self.stopped = False

    @property
    def available_avatars(self) -> list[str]:
        return ["yongen"]

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self.stopped = True

    async def acquire(self, session_id: str, avatar_id: str, **kwargs) -> _FakeLease:
        return _FakeLease(self._worker)

    async def release_async(self, session_id: str) -> bool:
        self.released.append(session_id)
        return True

    def stats(self) -> dict:
        return {"fake": True}


class _NotFoundPool(_FakePool):
    async def acquire(self, session_id: str, avatar_id: str, **kwargs):
        raise AvatarNotFound(f"avatar '{avatar_id}' not found")


async def _wait_for(predicate, timeout: float = 1.0) -> bool:
    elapsed = 0.0
    while not predicate():
        if elapsed >= timeout:
            return False
        await asyncio.sleep(0.01)
        elapsed += 0.01
    return True


def _pcm(samples: int = 320) -> bytes:
    """320 samples @16k = 20ms of audio."""
    return b"\x01\x00" * samples


# ── tests ──


class TestSessionLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_open_and_close_session(self):
        pool = _FakePool()
        pipeline = AvatarPipeline(AvatarPoolConfig(avatar_data_root="/x"), pool=pool)
        await pipeline.start()
        try:
            state = await pipeline.open_session("s1", "yongen")
            self.assertIn("s1", pipeline.sessions)
            self.assertEqual(state.avatar_id, "yongen")
            self.assertIsNone(state.publisher)  # capture mode

            ok = await pipeline.close_session("s1")
            self.assertTrue(ok)
            self.assertNotIn("s1", pipeline.sessions)
            self.assertIn("s1", pool.released)
        finally:
            await pipeline.stop()
        self.assertTrue(pool.stopped)

    async def test_open_duplicate_session_raises(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_FakePool()
        )
        await pipeline.start()
        try:
            await pipeline.open_session("s1", "yongen")
            with self.assertRaises(ValueError):
                await pipeline.open_session("s1", "yongen")
        finally:
            await pipeline.stop()

    async def test_close_unknown_session_returns_false(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_FakePool()
        )
        await pipeline.start()
        try:
            self.assertFalse(await pipeline.close_session("nope"))
        finally:
            await pipeline.stop()

    async def test_avatar_not_found_propagates(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_NotFoundPool()
        )
        await pipeline.start()
        try:
            with self.assertRaises(AvatarNotFound):
                await pipeline.open_session("s1", "ghost")
        finally:
            await pipeline.stop()


class TestAudioFlow(unittest.IsolatedAsyncioTestCase):
    async def test_push_pcm_auto_pts_clock(self):
        """Default pts advances by chunk duration (320 samples = 20ms)."""
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_FakePool()
        )
        await pipeline.start()
        try:
            await pipeline.open_session("s1", "yongen")
            ok = await pipeline.push_pcm("s1", _pcm(320))
            self.assertTrue(ok)
            await _wait_for(
                lambda: pipeline.get_session("s1").adapter.stats.frames_produced >= 4
            )
            adapter = pipeline.get_session("s1").adapter
            frames = adapter.published_frames
            self.assertEqual(len(frames), 4)
            # Chunk pts=0, 4 frames at 25fps → 0, 40k, 80k, 120k us.
            self.assertEqual([f.pts_us for f in frames], [0, 40_000, 80_000, 120_000])
            # Session PTS clock advanced by 20ms.
            self.assertEqual(pipeline.get_session("s1").next_pts_us, 20_000)
            self.assertEqual(pipeline.get_session("s1").samples_pushed, 320)
        finally:
            await pipeline.stop()

    async def test_push_pcm_explicit_pts(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_FakePool()
        )
        await pipeline.start()
        try:
            await pipeline.open_session("s1", "yongen")
            await pipeline.push_pcm("s1", _pcm(320), pts_us=1_000_000)
            await _wait_for(
                lambda: pipeline.get_session("s1").adapter.stats.frames_produced >= 1
            )
            frames = pipeline.get_session("s1").adapter.published_frames
            self.assertEqual(frames[0].pts_us, 1_000_000)
        finally:
            await pipeline.stop()

    async def test_push_unknown_session_raises(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_FakePool()
        )
        await pipeline.start()
        try:
            with self.assertRaises(KeyError):
                await pipeline.push_pcm("ghost", _pcm())
        finally:
            await pipeline.stop()

    async def test_publisher_factory_receives_frames(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"),
            pool=_FakePool(),
            publisher_factory=lambda state: _RecordingSink(),
        )
        await pipeline.start()
        try:
            await pipeline.open_session("s1", "yongen")
            state = pipeline.get_session("s1")
            self.assertIsInstance(state.publisher, _RecordingSink)
            await pipeline.push_pcm("s1", _pcm(320))
            await _wait_for(lambda: len(state.publisher.frames) >= 4)
            self.assertEqual(len(state.publisher.frames), 4)
        finally:
            await pipeline.stop()


class TestEpochFlow(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_epoch_advances_session_and_adapter(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_FakePool()
        )
        await pipeline.start()
        try:
            await pipeline.open_session("s1", "yongen")
            pipeline.cancel_epoch("s1", 3)
            state = pipeline.get_session("s1")
            self.assertEqual(state.epoch, 3)
            self.assertEqual(state.adapter.current_epoch, 3)

            # Push with an explicit stale epoch → dropped.
            ok = await pipeline.push_pcm("s1", _pcm(), epoch=1)
            self.assertFalse(ok)

            # Default epoch push uses session epoch (3) → accepted.
            ok = await pipeline.push_pcm("s1", _pcm())
            self.assertTrue(ok)
        finally:
            await pipeline.stop()

    async def test_cancel_epoch_monotonic(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_FakePool()
        )
        await pipeline.start()
        try:
            await pipeline.open_session("s1", "yongen")
            pipeline.cancel_epoch("s1", 5)
            pipeline.cancel_epoch("s1", 2)  # lower → ignored
            self.assertEqual(pipeline.get_session("s1").epoch, 5)
        finally:
            await pipeline.stop()

    async def test_cancel_unknown_session_is_noop(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_FakePool()
        )
        await pipeline.start()
        try:
            pipeline.cancel_epoch("ghost", 9)  # must not raise
        finally:
            await pipeline.stop()


class TestStats(unittest.IsolatedAsyncioTestCase):
    async def test_session_stats_and_pipeline_stats(self):
        pipeline = AvatarPipeline(
            AvatarPoolConfig(avatar_data_root="/x"), pool=_FakePool()
        )
        await pipeline.start()
        try:
            await pipeline.open_session("s1", "yongen")
            stats = pipeline.session_stats("s1")
            self.assertEqual(stats["session_id"], "s1")
            self.assertEqual(stats["avatar_id"], "yongen")
            self.assertIn("adapter", stats)

            pstats = pipeline.stats()
            self.assertIn("pool", pstats)
            self.assertIn("s1", pstats["sessions"])

            with self.assertRaises(KeyError):
                pipeline.session_stats("ghost")
        finally:
            await pipeline.stop()


if __name__ == "__main__":
    unittest.main()

"""Tests for AvatarWorker base class with a fake subclass (no torch).

Aligned to the current source API in ``liveavatar/worker.py``:
- ``AvatarWorker.__init__`` takes an ``AvatarAssets`` dataclass.
- Subclasses implement the synchronous ``_infer_batch`` which returns
  ``list[tuple[bytes, bool]]`` (frame_data_bgr24, is_speaking).
- The base ``synthesize_video_stream`` runs ``_infer_batch`` via
  ``asyncio.to_thread`` and yields ``AvatarFrame`` objects.
- Stats use ``frames_produced`` / ``cancel_count`` / ``pcm_chunks_consumed``.
"""

from __future__ import annotations

import asyncio
import time
import unittest

from liveavatar.lease import CancelToken
from liveavatar.worker import (
    AvatarAssets,
    AvatarFrame,
    AvatarWorker,
    AvatarWorkerStats,
)

# ── helpers ──


def _make_assets(avatar_id: str = "nahida") -> AvatarAssets:
    """Build a minimal AvatarAssets with fake paths (no files accessed)."""
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


# ── Fake subclass that yields synthetic BGR24 frames ──


class FakeAvatarWorker(AvatarWorker):
    """Deterministic worker that yields canned BGR24 frames from PCM.

    Mimics what a real ``MuseTalkAvatarWorker._infer_batch`` would do:
    consume PCM bytes and return ``batch_size`` ``(frame_data, is_speaking)``
    tuples. Uses a tiny resolution so frame buffers are cheap.
    """

    def __init__(
        self,
        assets: AvatarAssets | None = None,
        *,
        n_frames: int = 3,
        delay: float = 0.0,
        width: int = 4,
        height: int = 4,
        fps: int = 25,
        batch_size: int = 4,
        is_speaking: bool = True,
    ) -> None:
        super().__init__(
            assets or _make_assets(),
            target_fps=fps,
            width=width,
            height=height,
            batch_size=batch_size,
        )
        self.n_frames = n_frames
        self.delay = delay
        self.is_speaking_flag = is_speaking
        self.last_pcm: bytes | None = None
        self.run_count: int = 0

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        self.run_count += 1
        self.last_pcm = pcm_s16le
        if self.delay > 0:
            # Runs in a worker thread via asyncio.to_thread — safe to block.
            time.sleep(self.delay)
        frames: list[tuple[bytes, bool]] = []
        for i in range(self.n_frames):
            # BGR24 raw bytes: height * width * 3, filled with frame index.
            frame_bytes = bytes([i & 0xFF]) * (self.height * self.width * 3)
            frames.append((frame_bytes, self.is_speaking_flag))
        return frames


def _make_worker(
    *,
    n_frames: int = 3,
    delay: float = 0.0,
    avatar_id: str = "nahida",
    is_speaking: bool = True,
    fps: int = 25,
) -> FakeAvatarWorker:
    return FakeAvatarWorker(
        _make_assets(avatar_id),
        n_frames=n_frames,
        delay=delay,
        is_speaking=is_speaking,
        fps=fps,
    )


# ── tests ──


class TestAvatarWorkerBasics(unittest.TestCase):
    def test_avatar_id_pinned_from_assets(self):
        worker = _make_worker()
        self.assertEqual(worker.avatar_id, "nahida")

    def test_busy_false_when_idle(self):
        worker = _make_worker()
        self.assertFalse(worker.busy)

    def test_stats_initial_zero(self):
        worker = _make_worker()
        self.assertEqual(worker.stats.frames_produced, 0)
        self.assertEqual(worker.stats.cancel_count, 0)
        self.assertEqual(worker.stats.pcm_chunks_consumed, 0)
        self.assertEqual(worker.stats.inference_batches, 0)
        self.assertEqual(worker.stats.errors, 0)


class TestSynthesizeVideoStream(unittest.IsolatedAsyncioTestCase):
    async def test_yields_all_frames(self):
        worker = _make_worker(n_frames=3)
        frames = []
        async for frame in worker.synthesize_video_stream(b"pcm", pts_us=100, epoch=7):
            frames.append(frame)
        self.assertEqual(len(frames), 3)
        self.assertEqual(worker.stats.frames_produced, 3)
        self.assertEqual(worker.stats.cancel_count, 0)
        self.assertEqual(worker.stats.inference_batches, 1)
        self.assertEqual(worker.stats.pcm_chunks_consumed, 1)

    async def test_frame_fields_are_set(self):
        worker = _make_worker(n_frames=1, is_speaking=True)
        frames = []
        async for frame in worker.synthesize_video_stream(b"pcm", pts_us=42, epoch=5):
            frames.append(frame)
        self.assertEqual(len(frames), 1)
        f = frames[0]
        self.assertIsInstance(f.frame_data, bytes)
        # BGR24 = height * width * 3 = 4 * 4 * 3 = 48
        self.assertEqual(len(f.frame_data), 4 * 4 * 3)
        self.assertEqual(f.pts_us, 42)
        self.assertEqual(f.epoch, 5)
        self.assertEqual(f.width, 4)
        self.assertEqual(f.height, 4)
        self.assertTrue(f.is_speaking)

    async def test_pts_advances_per_frame(self):
        """Each frame gets pts_us incremented by 1/fps seconds (in microseconds)."""
        worker = _make_worker(n_frames=3, fps=25)
        frames = []
        async for frame in worker.synthesize_video_stream(b"pcm", pts_us=1000, epoch=0):
            frames.append(frame)
        # 1_000_000 // 25 = 40000 us per frame
        self.assertEqual(
            [f.pts_us for f in frames],
            [1000, 1000 + 40_000, 1000 + 80_000],
        )

    async def test_pcm_forwarded_to_infer_batch(self):
        worker = _make_worker(n_frames=1)
        async for _ in worker.synthesize_video_stream(b"hello-pcm", pts_us=0, epoch=99):
            pass
        self.assertEqual(worker.last_pcm, b"hello-pcm")
        self.assertEqual(worker.run_count, 1)

    async def test_busy_flag_during_synthesis(self):
        worker = _make_worker(n_frames=5, delay=0.02)

        busy_seen = asyncio.Event()

        async def _check_busy():
            while not worker.busy:
                await asyncio.sleep(0.001)
            busy_seen.set()

        check_task = asyncio.create_task(_check_busy())
        async for _ in worker.synthesize_video_stream(b"pcm", pts_us=0, epoch=0):
            pass
        await asyncio.wait_for(busy_seen.wait(), timeout=1.0)
        await check_task
        self.assertFalse(worker.busy)

    async def test_pre_cancelled_token_yields_nothing(self):
        """A token cancelled before synthesis starts skips inference entirely."""
        worker = _make_worker(n_frames=5)
        token = CancelToken()
        token.cancel()
        frames = []
        async for frame in worker.synthesize_video_stream(
            b"pcm", pts_us=0, epoch=0, cancel_token=token
        ):
            frames.append(frame)
        self.assertEqual(len(frames), 0)
        self.assertEqual(worker.stats.cancel_count, 1)
        self.assertEqual(worker.stats.frames_produced, 0)
        # Inference was skipped (pre-inference cancellation check).
        self.assertEqual(worker.run_count, 0)
        self.assertEqual(worker.stats.inference_batches, 0)

    async def test_cancellation_during_inference_yields_nothing(self):
        """Token cancelled while _infer_batch runs → 0 frames yielded."""
        worker = _make_worker(n_frames=10, delay=0.05)
        token = CancelToken()
        frames = []

        async def _cancel_during_inference():
            await asyncio.sleep(0.02)
            token.cancel()

        cancel_task = asyncio.create_task(_cancel_during_inference())
        async for frame in worker.synthesize_video_stream(
            b"pcm", pts_us=0, epoch=0, cancel_token=token
        ):
            frames.append(frame)
        await cancel_task
        # Inference completed (delay finished) but first frame check sees
        # cancelled → 0 frames yielded.
        self.assertEqual(len(frames), 0)
        self.assertEqual(worker.stats.cancel_count, 1)
        self.assertEqual(worker.run_count, 1)  # inference did run

    async def test_inference_lock_serializes_calls(self):
        """Two concurrent synthesize_video_stream calls run sequentially."""
        worker = _make_worker(n_frames=3, delay=0.02)

        async def _synth():
            frames = []
            async for frame in worker.synthesize_video_stream(b"pcm", pts_us=0, epoch=0):
                frames.append(frame)
            return len(frames)

        results = await asyncio.gather(_synth(), _synth())
        self.assertEqual(results, [3, 3])
        self.assertEqual(worker.stats.frames_produced, 6)
        self.assertEqual(worker.stats.inference_batches, 2)

    async def test_default_token_created_when_none_passed(self):
        """Passing no cancel_token still works (a fresh token is created)."""
        worker = _make_worker(n_frames=2)
        frames = []
        async for frame in worker.synthesize_video_stream(b"pcm", pts_us=0, epoch=0):
            frames.append(frame)
        self.assertEqual(len(frames), 2)

    async def test_silence_frames_tracked_separately(self):
        """Frames with is_speaking=False are counted in frames_silence."""
        worker = _make_worker(n_frames=4, is_speaking=False)
        async for _ in worker.synthesize_video_stream(b"pcm", pts_us=0, epoch=0):
            pass
        self.assertEqual(worker.stats.frames_produced, 4)
        self.assertEqual(worker.stats.frames_silence, 4)


class TestAvatarFrame(unittest.TestCase):
    def test_frame_is_dataclass_with_all_fields(self):
        f = AvatarFrame(
            frame_data=b"\x00" * 12,
            pts_us=123,
            epoch=4,
            width=2,
            height=2,
        )
        self.assertEqual(f.frame_data, b"\x00" * 12)
        self.assertEqual(f.pts_us, 123)
        self.assertEqual(f.epoch, 4)
        self.assertEqual(f.width, 2)
        self.assertEqual(f.height, 2)
        self.assertTrue(f.is_speaking)  # default

    def test_frame_is_speaking_false(self):
        f = AvatarFrame(
            frame_data=b"\x00" * 12,
            pts_us=0,
            epoch=0,
            width=2,
            height=2,
            is_speaking=False,
        )
        self.assertFalse(f.is_speaking)


class TestAvatarWorkerStats(unittest.TestCase):
    def test_default_values(self):
        stats = AvatarWorkerStats()
        self.assertEqual(stats.pcm_chunks_consumed, 0)
        self.assertEqual(stats.frames_produced, 0)
        self.assertEqual(stats.frames_silence, 0)
        self.assertEqual(stats.inference_batches, 0)
        self.assertEqual(stats.cancel_count, 0)
        self.assertEqual(stats.errors, 0)
        self.assertEqual(stats.total_inference_ms, 0.0)


if __name__ == "__main__":
    unittest.main()

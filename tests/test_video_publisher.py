"""Tests for AvatarVideoPublisher: I420 conversion + publish/epoch paths.

No livekit required — publish paths use a stubbed _capture_frame / fake source.
"""

from __future__ import annotations

import asyncio
import unittest

from liveavatar.video_publisher import (
    AvatarVideoPublisher,
    _bgr24_to_i420,
    _numpy_bgr24_to_i420,
)
from liveavatar.worker import AvatarFrame

W, H = 8, 8


def _frame(
    pixel: tuple[int, int, int] = (0, 128, 255),
    pts: int = 0,
    epoch: int = 0,
    is_speaking: bool = True,
) -> AvatarFrame:
    return AvatarFrame(
        frame_data=bytes(pixel) * (W * H),
        pts_us=pts,
        epoch=epoch,
        width=W,
        height=H,
        is_speaking=is_speaking,
    )


def _numpy_i420(bgr_data: bytes, width: int, height: int) -> bytes:
    """Delegate to the library's numpy fallback under test."""
    return _numpy_bgr24_to_i420(bgr_data, width, height)


class TestBgrToI420(unittest.TestCase):
    def test_output_length(self):
        data = bytes((0, 128, 255)) * (W * H)
        out = _bgr24_to_i420(data, W, H)
        self.assertEqual(len(out), W * H + (W // 2) * (H // 2) * 2)

    def test_gray_is_flat(self):
        # Pure gray 128 → Y=126, U=V=128 (BT.601 limited range) on both paths.
        data = bytes((128, 128, 128)) * (W * H)
        out = _bgr24_to_i420(data, W, H)
        y_plane = out[: W * H]
        self.assertEqual(set(y_plane), {126})
        chroma = out[W * H :]
        self.assertEqual(set(chroma), {128})

    def test_cv2_matches_numpy_fallback(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not installed")
        data = bytes((30, 200, 90)) * (W * H)
        fast = _bgr24_to_i420(data, W, H)
        slow = _numpy_i420(data, W, H)
        diffs = [abs(a - b) for a, b in zip(fast, slow, strict=True)]
        self.assertLessEqual(max(diffs), 2)  # rounding tolerance


class _RecordingSource:
    def __init__(self) -> None:
        self.captured: list[AvatarFrame] = []
        self.delay: float = 0.0

    def capture_frame(self, video_frame, timestamp_us: int = 0) -> None:
        self.captured.append((video_frame, timestamp_us))


class TestPublisher(unittest.IsolatedAsyncioTestCase):
    def _make_publisher(self) -> tuple[AvatarVideoPublisher, _RecordingSource]:
        pub = AvatarVideoPublisher(None, "s1", width=W, height=H)
        source = _RecordingSource()
        pub._source = source

        # Stub the LiveKit-dependent capture with a recording async fn
        # (the documented test seam — no livekit required).
        async def recording_capture(frame: AvatarFrame) -> None:
            if source.delay:
                await asyncio.sleep(source.delay)
            source.captured.append((frame, frame.pts_us))

        pub._capture_frame = recording_capture  # type: ignore[method-assign]
        return pub, source

    async def test_publish_frame_ok(self):
        pub, source = self._make_publisher()
        ok = await pub.publish_frame(_frame(pts=1000), epoch=0)
        self.assertTrue(ok)
        self.assertEqual(len(source.captured), 1)
        video_frame, ts = source.captured[0]
        self.assertEqual(ts, 1000)
        self.assertEqual(pub.stats.frames_published, 1)
        self.assertEqual(pub.stats.bytes_published, W * H * 3)
        self.assertEqual(pub.stats.frames_silence, 0)

    async def test_publish_silence_counted(self):
        pub, _ = self._make_publisher()
        await pub.publish_frame(_frame(is_speaking=False), epoch=0)
        self.assertEqual(pub.stats.frames_silence, 1)

    async def test_stale_epoch_dropped(self):
        pub, source = self._make_publisher()
        pub.cancel_epoch(5)
        ok = await pub.publish_frame(_frame(epoch=4), epoch=4)
        self.assertFalse(ok)
        self.assertEqual(pub.stats.frames_dropped_epoch, 1)
        self.assertEqual(source.captured, [])

    async def test_cancel_epoch_monotonic(self):
        pub, _ = self._make_publisher()
        pub.cancel_epoch(3)
        pub.cancel_epoch(1)
        self.assertEqual(pub.current_epoch, 3)
        pub.cancel_epoch(3)
        self.assertEqual(pub.current_epoch, 3)

    async def test_capture_timeout_dropped(self):
        pub, source = self._make_publisher()
        pub.capture_timeout_s = 0.05

        async def timeout_capture(frame: AvatarFrame) -> None:
            raise asyncio.TimeoutError()

        pub._capture_frame = timeout_capture  # type: ignore[method-assign]
        ok = await pub.publish_frame(_frame(), epoch=0)
        self.assertFalse(ok)
        self.assertEqual(pub.stats.frames_dropped_timeout, 1)

    async def test_publish_before_start_raises(self):
        pub = AvatarVideoPublisher(None, "s1", width=W, height=H)
        with self.assertRaises(RuntimeError):
            await pub.publish_frame(_frame(), epoch=0)

    async def test_stop_without_publication_is_noop(self):
        pub = AvatarVideoPublisher(None, "s1", width=W, height=H)
        await pub.stop()  # must not raise
        self.assertFalse(pub.stats.track_published)

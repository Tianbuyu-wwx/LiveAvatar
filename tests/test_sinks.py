# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for sinks (RTMP) — subprocess faked, no ffmpeg needed."""

from __future__ import annotations

import unittest

from liveavatar.sinks import PublishSink, RtmpSink
from liveavatar.worker import AvatarFrame


def _frame(pixel: tuple[int, int, int] = (1, 2, 3), epoch: int = 0) -> AvatarFrame:
    return AvatarFrame(
        frame_data=bytes(pixel) * (8 * 8),
        pts_us=0,
        epoch=epoch,
        width=8,
        height=8,
    )


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = self
        self.written: list[bytes] = []
        self.closed = False
        self.waited = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def close(self) -> None:
        self.closed = True

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0


class TestRtmpSink(unittest.IsolatedAsyncioTestCase):
    def _make_sink(self) -> tuple[RtmpSink, _FakeProc]:
        sink = RtmpSink("rtmp://host/live/key", width=8, height=8, target_fps=25)
        proc = _FakeProc()
        sink._proc = proc  # type: ignore[assignment]
        return sink, proc

    def test_implements_protocol(self):
        sink = RtmpSink("rtmp://x")
        self.assertIsInstance(sink, PublishSink)

    def test_odd_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            RtmpSink("rtmp://x", width=7, height=8)

    def test_ffmpeg_cmd_shape(self):
        sink = RtmpSink("rtmp://host/k", width=512, height=512, target_fps=25)
        cmd = sink._ffmpeg_cmd()
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("bgr24", cmd)
        self.assertIn("512x512", cmd)
        self.assertIn("rtmp://host/k", cmd)

    async def test_publish_and_stop(self):
        sink, proc = self._make_sink()
        self.assertTrue(await sink.publish_frame(_frame(), epoch=0))
        self.assertEqual(len(proc.written), 1)
        self.assertEqual(len(proc.written[0]), 8 * 8 * 3)
        await sink.stop()
        self.assertTrue(proc.closed)
        self.assertTrue(proc.waited)
        body = sink.stats()
        self.assertEqual(body["frames_published"], 1)
        self.assertEqual(body["frames_seen"], 1)

    async def test_stale_epoch_dropped(self):
        sink, proc = self._make_sink()
        sink.cancel_epoch(5)
        self.assertEqual(sink.current_epoch, 5)
        self.assertFalse(await sink.publish_frame(_frame(epoch=4), epoch=4))
        self.assertEqual(proc.written, [])
        self.assertEqual(sink.stats()["frames_dropped_epoch"], 1)
        # Monotonic.
        sink.cancel_epoch(2)
        self.assertEqual(sink.current_epoch, 5)

    async def test_wrong_frame_size_raises(self):
        sink, _ = self._make_sink()
        bad = AvatarFrame(frame_data=b"\x00", pts_us=0, epoch=0, width=1, height=1)
        with self.assertRaises(ValueError):
            await sink.publish_frame(bad, epoch=0)

    async def test_publish_before_start_raises(self):
        sink = RtmpSink("rtmp://x")
        with self.assertRaises(RuntimeError):
            await sink.publish_frame(_frame(), epoch=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

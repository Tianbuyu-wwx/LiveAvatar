# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for WebSocketSink (self-developed transport, R2 M1).

Pure CPU: a fake encoder avoids cv2; one test exercises the real
MjpegFrameEncoder on a tiny frame.
"""

from __future__ import annotations

import asyncio
import unittest

from liveavatar.video_protocol import (
    CODEC_MJPEG_FULL,
    FLAG_EPOCH_BOUNDARY,
    FLAG_KEYFRAME,
    has_flag,
    unpack_video_frame,
)
from liveavatar.worker import AvatarFrame
from liveavatar.ws_sink import (
    MjpegFrameEncoder,
    VideoClient,
    WebSocketSink,
)


class _FakeEncoder:
    """Deterministic encoder: payload = frame_data (or marker)."""

    codec = CODEC_MJPEG_FULL

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def encode(
        self, frame: AvatarFrame, *, keyframe: bool, quality: int
    ) -> bytes:
        self.calls += 1
        if self.fail:
            raise RuntimeError("encode boom")
        return bytes([keyframe]) + frame.frame_data[:4]


def _frame(
    pts_us: int = 0, width: int = 4, height: int = 4, epoch: int = 0
) -> AvatarFrame:
    return AvatarFrame(
        frame_data=b"\x00" * (width * height * 3),
        pts_us=pts_us,
        epoch=epoch,
        width=width,
        height=height,
    )


class WebSocketSinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_frame_is_keyframe(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        assert isinstance(client, VideoClient)
        ok = await sink.publish_frame(_frame(pts_us=0), epoch=0)
        self.assertTrue(ok)
        wire = client.queue.get_nowait()
        header, _payload = unpack_video_frame(wire)
        self.assertTrue(has_flag(header.flags, FLAG_KEYFRAME))

    async def test_subsequent_frames_not_keyframes(self) -> None:
        sink = WebSocketSink(
            encoder=_FakeEncoder(), keyframe_interval_us=1_000_000
        )
        client = sink.add_client()
        await sink.publish_frame(_frame(pts_us=0), epoch=0)
        await sink.publish_frame(_frame(pts_us=40_000), epoch=0)
        await sink.publish_frame(_frame(pts_us=80_000), epoch=0)
        flags = [
            unpack_video_frame(client.queue.get_nowait())[0].flags
            for _ in range(3)
        ]
        self.assertTrue(has_flag(flags[0], FLAG_KEYFRAME))
        self.assertFalse(has_flag(flags[1], FLAG_KEYFRAME))
        self.assertFalse(has_flag(flags[2], FLAG_KEYFRAME))

    async def test_fan_out_multiple_clients(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        c1 = sink.add_client()
        c2 = sink.add_client()
        await sink.publish_frame(_frame(), epoch=0)
        w1 = c1.queue.get_nowait()
        w2 = c2.queue.get_nowait()
        self.assertEqual(w1, w2)
        self.assertEqual(c1.frames_sent, 1)
        self.assertEqual(c2.frames_sent, 1)
        self.assertEqual(sink.client_count, 2)

    async def test_seq_increments_and_wraps(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        for i in range(3):
            await sink.publish_frame(_frame(pts_us=i * 40_000), epoch=0)
        seqs = [unpack_video_frame(client.queue.get_nowait())[0].seq for _ in range(3)]
        self.assertEqual(seqs, [0, 1, 2])
        # Force wrap-around: header carries seq before incrementing.
        sink._seq = 65535
        await sink.publish_frame(_frame(pts_us=1_000_000), epoch=0)
        header, _ = unpack_video_frame(client.queue.get_nowait())
        self.assertEqual(header.seq, 65535)
        await sink.publish_frame(_frame(pts_us=1_040_000), epoch=0)
        header, _ = unpack_video_frame(client.queue.get_nowait())
        self.assertEqual(header.seq, 0)

    async def test_stale_epoch_dropped(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        await sink.publish_frame(_frame(), epoch=2)
        ok = await sink.publish_frame(_frame(), epoch=1)
        self.assertFalse(ok)
        stats = sink.stats()
        self.assertEqual(stats["frames_dropped_epoch"], 1)
        self.assertEqual(client.queue.qsize(), 1)

    async def test_cancel_epoch_forces_keyframe_boundary(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder(), keyframe_interval_us=10**9)
        client = sink.add_client()
        await sink.publish_frame(_frame(pts_us=0), epoch=0)
        await sink.publish_frame(_frame(pts_us=40_000), epoch=0)
        sink.cancel_epoch(1)
        self.assertEqual(sink.current_epoch, 1)
        await sink.publish_frame(_frame(pts_us=80_000), epoch=1)
        headers = [
            unpack_video_frame(client.queue.get_nowait())[0] for _ in range(3)
        ]
        # Frame 1: connect keyframe. Frame 2: intra. Frame 3: boundary keyframe.
        self.assertTrue(has_flag(headers[0].flags, FLAG_KEYFRAME))
        self.assertFalse(has_flag(headers[1].flags, FLAG_KEYFRAME))
        self.assertTrue(has_flag(headers[2].flags, FLAG_KEYFRAME))
        self.assertTrue(has_flag(headers[2].flags, FLAG_EPOCH_BOUNDARY))
        self.assertEqual(headers[2].epoch, 1)

    async def test_cancel_epoch_ignores_backwards(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        sink.cancel_epoch(5)
        sink.cancel_epoch(2)
        self.assertEqual(sink.current_epoch, 5)
        # Boundary pending: next frame is a keyframe.
        client = sink.add_client()
        await sink.publish_frame(_frame(), epoch=5)
        header, _ = unpack_video_frame(client.queue.get_nowait())
        self.assertTrue(has_flag(header.flags, FLAG_EPOCH_BOUNDARY))

    async def test_slow_client_drops_oldest(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder(), client_queue_size=2)
        client = sink.add_client()
        for i in range(5):
            await sink.publish_frame(_frame(pts_us=i * 40_000), epoch=0)
        self.assertEqual(client.queue.qsize(), 2)
        self.assertEqual(client.frames_sent, 2)
        self.assertEqual(client.frames_dropped, 3)
        self.assertEqual(sink.stats()["client_frames_dropped"], 3)
        # Oldest dropped: first queued frame is seq=3.
        header, _ = unpack_video_frame(client.queue.get_nowait())
        self.assertEqual(header.seq, 3)

    async def test_request_keyframe(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder(), keyframe_interval_us=10**9)
        client = sink.add_client()
        await sink.publish_frame(_frame(pts_us=0), epoch=0)  # connect keyframe
        await sink.publish_frame(_frame(pts_us=40_000), epoch=0)
        sink.request_keyframe(client)
        await sink.publish_frame(_frame(pts_us=80_000), epoch=0)
        flags = [
            unpack_video_frame(client.queue.get_nowait())[0].flags
            for _ in range(3)
        ]
        self.assertTrue(has_flag(flags[0], FLAG_KEYFRAME))
        self.assertFalse(has_flag(flags[1], FLAG_KEYFRAME))
        self.assertTrue(has_flag(flags[2], FLAG_KEYFRAME))

    async def test_keyframe_interval(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder(), keyframe_interval_us=100_000)
        client = sink.add_client()
        await sink.publish_frame(_frame(pts_us=0), epoch=0)
        await sink.publish_frame(_frame(pts_us=50_000), epoch=0)
        await sink.publish_frame(_frame(pts_us=120_000), epoch=0)
        flags = [
            unpack_video_frame(client.queue.get_nowait())[0].flags
            for _ in range(3)
        ]
        self.assertTrue(has_flag(flags[0], FLAG_KEYFRAME))
        self.assertFalse(has_flag(flags[1], FLAG_KEYFRAME))
        self.assertTrue(has_flag(flags[2], FLAG_KEYFRAME))

    async def test_no_clients_no_keyframe_but_published(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        ok = await sink.publish_frame(_frame(), epoch=0)
        self.assertTrue(ok)
        stats = sink.stats()
        self.assertEqual(stats["frames_published"], 1)
        self.assertEqual(stats["clients"], 0)

    async def test_encode_error_counted(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder(fail=True))
        client = sink.add_client()
        ok = await sink.publish_frame(_frame(), epoch=0)
        self.assertFalse(ok)
        self.assertEqual(sink.stats()["encode_errors"], 1)
        self.assertTrue(client.queue.empty())

    async def test_stop_closes_queues_and_drops_frames(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        await sink.start()
        await sink.publish_frame(_frame(), epoch=0)
        await sink.stop()
        # Buffered frames still drain after close, then the EOF sentinel.
        wire = await asyncio.wait_for(client.queue.get(), timeout=1.0)
        self.assertIsNotNone(wire)
        got = await asyncio.wait_for(client.queue.get(), timeout=1.0)
        self.assertIsNone(got)
        ok = await sink.publish_frame(_frame(), epoch=0)
        self.assertFalse(ok)
        self.assertEqual(sink.stats()["frames_dropped_closed"], 1)

    async def test_remove_client_closes_queue(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        self.assertEqual(sink.client_count, 1)
        sink.remove_client(client)
        self.assertEqual(sink.client_count, 0)
        self.assertTrue(client.queue.closed)

    async def test_publish_on_one_loop_get_on_another(self) -> None:
        """Cross-loop fan-out: publish on loop A, consume on loop B."""
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        await sink.publish_frame(_frame(), epoch=0)

        async def consume() -> bytes | None:
            return await asyncio.wait_for(client.queue.get(), timeout=2.0)

        wire = await consume()
        self.assertIsNotNone(wire)
        header, _ = unpack_video_frame(wire)
        self.assertTrue(has_flag(header.flags, FLAG_KEYFRAME))


class MjpegEncoderTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_encoder_roundtrip(self) -> None:
        """Real cv2 encoder produces a decodable JPEG wire frame."""
        import cv2
        import numpy as np

        sink = WebSocketSink(encoder=MjpegFrameEncoder())
        client = sink.add_client()
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[:, :, 0] = 200  # blue-ish frame
        frame = AvatarFrame(
            frame_data=img.tobytes(), pts_us=0, epoch=0, width=8, height=8
        )
        ok = await sink.publish_frame(frame, epoch=0)
        self.assertTrue(ok)
        wire = client.queue.get_nowait()
        header, payload = unpack_video_frame(wire)
        self.assertEqual(header.codec, CODEC_MJPEG_FULL)
        self.assertEqual(header.width, 8)
        self.assertEqual(header.height, 8)
        # JPEG decodes back to the same geometry.
        arr = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(arr.shape, (8, 8, 3))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

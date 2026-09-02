# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for the region-delta codec (R2 M4) — pure CPU.

DoD checks:
- roundtrip PSNR ≥ 30 dB across a synthetic speech stream;
- bandwidth ≤ 2 Mbps @25fps (avg ≤ 10 kB/frame) on 200 synthetic frames;
- any dropped patch frame keeps subsequent frames independently decodable.
"""

from __future__ import annotations

import os
import struct
import tempfile
import unittest

import cv2
import numpy as np

from liveavatar.region_codec import (
    RegionFrameEncoder,
    RegionSpec,
    load_region_json,
    region_spec_from_masks,
    write_region_json,
)
from liveavatar.video_protocol import (
    CODEC_REGION_DELTA,
    FLAG_KEYFRAME,
    unpack_region_frame,
    unpack_region_payload,
)
from liveavatar.worker import AvatarFrame

W, H = 256, 256
REGION = RegionSpec(x=W // 2 - 32, y=H // 2, w=64, h=48)


def _frame_at(t: int) -> AvatarFrame:
    """Static background + an animated mouth ellipse inside REGION."""
    img = np.full((H, W, 3), 80, dtype=np.uint8)
    img[:, :, 0] = np.arange(W)[None, :] % 256  # textured background
    open_h = 12 + int(8 * np.sin(t * 0.7))
    cv2.ellipse(
        img,
        (W // 2, H // 2 + 24),
        (26, max(2, open_h)),
        0,
        0,
        360,
        (40, 40, 200),
        -1,
    )
    return AvatarFrame(
        frame_data=img.tobytes(), pts_us=t * 40_000, epoch=0, width=W, height=H
    )


def _decode_canvas(base: np.ndarray | None, patches: list) -> np.ndarray:
    """Reference decoder: full-canvas patches update the base; region
    patches composite on top of a copy of it."""
    if patches[0].x == 0 and patches[0].y == 0:
        canvas = cv2.imdecode(
            np.frombuffer(patches[0].jpeg, np.uint8), cv2.IMREAD_COLOR
        )
        return canvas.copy()  # new base
    assert base is not None, "region patch before any full frame"
    canvas = base.copy()
    for p in patches:
        crop = cv2.imdecode(
            np.frombuffer(p.jpeg, np.uint8), cv2.IMREAD_COLOR
        )
        canvas[p.y : p.y + p.h, p.x : p.x + p.w] = crop
    return canvas


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return 99.0
    return 10.0 * np.log10(255.0**2 / mse)


class RegionSpecTests(unittest.TestCase):
    def test_grown_clamps_to_canvas(self) -> None:
        edge = RegionSpec(x=0, y=0, w=10, h=10)
        g = edge.grown(8, W, H)
        self.assertEqual((g.x, g.y, g.w, g.h), (0, 0, 18, 18))
        corner = RegionSpec(x=W - 5, y=H - 5, w=5, h=5)
        g2 = corner.grown(8, W, H)
        self.assertEqual((g2.x, g2.y, g2.w, g2.h), (W - 13, H - 13, 13, 13))

    def test_region_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "region.json")
            write_region_json(path, REGION)
            self.assertEqual(load_region_json(path), REGION)

    def test_region_spec_from_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask = np.zeros((H, W), np.uint8)
            mask[130:170, 100:160] = 255
            cv2.imwrite(os.path.join(tmp, "0.png"), mask)
            spec = region_spec_from_masks(tmp, W, H)
            assert spec is not None
            self.assertEqual((spec.x, spec.y, spec.w, spec.h), (100, 130, 60, 40))

    def test_region_spec_rejects_huge_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask = np.full((H, W), 255, np.uint8)
            cv2.imwrite(os.path.join(tmp, "0.png"), mask)
            self.assertIsNone(region_spec_from_masks(tmp, W, H))
        # Empty dir → None.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(region_spec_from_masks(tmp, W, H))


class RegionEncoderStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = RegionFrameEncoder(REGION)
        self.base: np.ndarray | None = None
        self.psnrs: list[float] = []
        self.total_bytes = 0
        self.full_frames = 0
        self.n = 200

    def _push(self, t: int, keyframe: bool) -> np.ndarray:
        frame = _frame_at(t)
        payload = self.encoder.encode(frame, keyframe=keyframe, quality=80)
        self.total_bytes += 26 + len(payload)
        header, patches = unpack_region_frame(
            struct.pack(
                "<BBBBHIqHHI",
                1,
                0,
                CODEC_REGION_DELTA,
                80,
                t,
                0,
                frame.pts_us,
                W,
                H,
                len(payload),
            )
            + payload
        )
        self.assertEqual(header.codec, CODEC_REGION_DELTA)
        if patches[0].x == 0 and patches[0].y == 0:
            self.full_frames += 1
        canvas = _decode_canvas(self.base, patches)
        if patches[0].x == 0 and patches[0].y == 0:
            self.base = canvas.copy()
        ref = np.frombuffer(frame.frame_data, np.uint8).reshape(H, W, 3)
        self.psnrs.append(_psnr(canvas, ref))
        return canvas

    def test_psnr_and_bandwidth(self) -> None:
        # First frame: keyframe (full canvas), then region patches.
        self._push(0, keyframe=True)
        for t in range(1, self.n):
            self._push(t, keyframe=False)

        # DoD: PSNR ≥ 30 dB on every frame.
        self.assertGreaterEqual(min(self.psnrs), 30.0, f"min PSNR {min(self.psnrs):.2f}")

        # DoD: ≤ 2 Mbps @ 25 fps → avg wire size ≤ 10 000 B/frame.
        avg = self.total_bytes / self.n
        self.assertLessEqual(avg, 10_000, f"avg frame {avg:.0f} B")

        # Sanity: the encoder actually exploits the static background.
        self.assertEqual(self.full_frames, 1)
        self.assertLess(avg, 6_000, "region frames should be well below full-frame size")

    def test_background_change_triggers_full_frame(self) -> None:
        self._push(0, keyframe=True)
        self._push(1, keyframe=False)
        # Disturb the background outside the mouth rect.
        frame = _frame_at(2)
        img = np.frombuffer(frame.frame_data, np.uint8).reshape(H, W, 3).copy()
        img[10:40, 10:40] = 255
        changed = AvatarFrame(
            frame_data=img.tobytes(), pts_us=80_000, epoch=0, width=W, height=H
        )
        payload = self.encoder.encode(changed, keyframe=False, quality=80)
        patches = unpack_region_payload(payload)
        self.assertEqual(patches[0].x, 0)  # full-canvas patch
        self.assertEqual(patches[0].w, W)

    def test_dropped_patch_frames_stay_decodable(self) -> None:
        """Skip decoding 10 consecutive patch frames; the stream still
        reconstructs every later frame correctly (independent frames)."""
        self._push(0, keyframe=True)
        skip_until = 50
        for t in range(1, self.n):
            frame = _frame_at(t)
            payload = self.encoder.encode(frame, keyframe=False, quality=80)
            if t < skip_until:
                continue  # "dropped" — never decoded
            _header, patches = unpack_region_frame(
                struct.pack(
                    "<BBBBHIqHHI", 1, 0, CODEC_REGION_DELTA, 80, t, 0,
                    frame.pts_us, W, H, len(payload),
                )
                + payload
            )
            canvas = _decode_canvas(self.base, patches)
            if patches[0].x == 0 and patches[0].y == 0:
                self.base = canvas.copy()
            ref = np.frombuffer(frame.frame_data, np.uint8).reshape(H, W, 3)
            self.assertGreaterEqual(_psnr(canvas, ref), 30.0)


class RegionEncoderSinkIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sink_with_region_encoder(self) -> None:
        """RegionFrameEncoder plugs into WebSocketSink; wire frames decode."""
        from liveavatar.ws_sink import WebSocketSink

        sink = WebSocketSink(encoder=RegionFrameEncoder(REGION))
        client = sink.add_client()
        ok = await sink.publish_frame(_frame_at(0), epoch=0)
        self.assertTrue(ok)
        wire = client.queue.get_nowait()
        header, patches = unpack_region_frame(wire)
        self.assertEqual(header.codec, CODEC_REGION_DELTA)
        self.assertTrue(header.flags & FLAG_KEYFRAME)
        self.assertEqual(patches[0].w, W)  # first frame is full-canvas
        ok = await sink.publish_frame(_frame_at(1), epoch=0)
        self.assertTrue(ok)
        _header, patches = unpack_region_frame(client.queue.get_nowait())
        grown = REGION.grown(8, W, H)
        self.assertEqual(
            (patches[0].w, patches[0].h), (grown.w, grown.h)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

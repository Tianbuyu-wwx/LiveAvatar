"""Tests for StaticAvatarWorker (degradation fallback) — no GPU, no torch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from liveavatar.static_worker import StaticAvatarWorker
from tests.conftest import make_assets


class TestStaticWorker(unittest.TestCase):
    @staticmethod
    def _assets(full_imgs_dir: str = "/nonexistent"):
        return make_assets("static", full_imgs_dir=full_imgs_dir)

    def test_black_frame_when_no_assets(self):
        w = StaticAvatarWorker(
            self._assets(), width=8, height=8, batch_size=2
        )
        self.assertEqual(len(w._static_frame), 8 * 8 * 3)
        self.assertEqual(w._static_frame, b"\x00" * (8 * 8 * 3))

    def test_default_frame_passthrough_exact_size(self):
        frame = bytes((10, 20, 30)) * (8 * 8)
        w = StaticAvatarWorker(
            self._assets(), width=8, height=8, default_frame_bgr=frame
        )
        self.assertEqual(w._static_frame, frame)

    def test_default_frame_resized_when_wrong_size(self):
        # 4x4 input resized to 8x8 via cv2.
        frame = bytes((5, 10, 15)) * (4 * 4)
        w = StaticAvatarWorker(
            self._assets(), width=8, height=8, default_frame_bgr=frame
        )
        self.assertEqual(len(w._static_frame), 8 * 8 * 3)

    def test_resize_garbage_shape_returns_black(self):
        # 5-byte buffer cannot form a square or width-aligned shape.
        w = StaticAvatarWorker(
            self._assets(),
            width=8,
            height=8,
            default_frame_bgr=b"\x01\x02\x03\x04\x05",
        )
        self.assertEqual(w._static_frame, b"\x00" * (8 * 8 * 3))

    def test_reads_first_full_img_from_disk(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("cv2 not installed")
        with tempfile.TemporaryDirectory() as tmp:
            imgs = Path(tmp) / "full_imgs"
            imgs.mkdir()
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            img[:, :] = (1, 2, 3)
            cv2.imwrite(str(imgs / "000.jpg"), img)
            w = StaticAvatarWorker(
                self._assets(str(imgs)), width=8, height=8
            )
            self.assertEqual(len(w._static_frame), 8 * 8 * 3)

    def test_empty_full_imgs_falls_back_to_black(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "full_imgs").mkdir()
            w = StaticAvatarWorker(
                self._assets(str(Path(tmp) / "full_imgs")), width=8, height=8
            )
            self.assertEqual(w._static_frame, b"\x00" * (8 * 8 * 3))

    def test_infer_batch_yields_static_frames(self):
        frame = bytes((7, 8, 9)) * (8 * 8)
        w = StaticAvatarWorker(
            make_assets("static", full_imgs_dir="/nonexistent"), width=8, height=8, batch_size=4,
            default_frame_bgr=frame,
        )
        out = w._infer_batch(b"\x01\x00" * 320)
        self.assertEqual(len(out), 4)
        for data, speaking in out:
            self.assertEqual(data, frame)
            self.assertFalse(speaking)
        # Frame index advances per batch.
        w._infer_batch(b"\x01\x00" * 320)
        self.assertEqual(w._frame_index, 8)

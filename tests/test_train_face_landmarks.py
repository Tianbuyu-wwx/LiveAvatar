# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Integration smoke for the M2 landmark training script — tiny CPU run."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

try:
    import torch

    _HAVE_TORCH = True
except ImportError:  # light CI env
    _HAVE_TORCH = False

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tests"))

if _HAVE_TORCH:
    import train_face_landmarks as tfl  # noqa: E402

    from test_face_landmarks import GT_CROP, POINT_COLORS  # noqa: E402


def _face_image(size: int = 128) -> np.ndarray:
    img = np.full((size, size, 3), 25, np.uint8)
    for (x, y), color in zip(GT_CROP, POINT_COLORS, strict=True):
        cx, cy = int(x * size), int(y * size)
        img[cy - 2 : cy + 3, cx - 2 : cx + 3] = color
    return img


@unittest.skipUnless(_HAVE_TORCH, "torch not installed (light CI env)")
class TrainLandmarksTests(unittest.TestCase):
    def test_two_epochs_cpu_end_to_end(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="_landmarks_"))
        try:
            entries = []
            rng = np.random.default_rng(0)
            for i in range(6):
                img = _face_image(128).copy()
                img[:8] = rng.integers(0, 30, (8, 128, 3), dtype=np.uint8)  # variety
                p = tmp / f"img_{i}.png"
                cv2.imencode(".png", img)[1].tofile(str(p))
                entries.append(
                    {
                        "image": str(p),
                        "width": 128,
                        "height": 128,
                        "source": "synthetic",
                        "pseudo": False,
                        "boxes": [[0.0, 0.0, 1.0, 1.0]],
                        "points5": GT_CROP.tolist(),
                    }
                )
            manifest = tmp / "manifest.jsonl"
            manifest.write_text(
                "\n".join(json.dumps(e) for e in entries), encoding="utf-8"
            )
            out = tmp / "landmarks5.pt"
            rc = tfl.main(
                [
                    "--manifest", str(manifest),
                    "--epochs", "2",
                    "--batch", "2",
                    "--size", "128",
                    "--width", "16",
                    "--device", "cpu",
                    "--out", str(out),
                ]
            )
            self.assertEqual(0, rc)
            self.assertTrue(out.exists())
            ckpt = torch.load(out, map_location="cpu", weights_only=True)
            self.assertEqual(16, ckpt["width"])
            self.assertEqual(128, ckpt["input_size"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

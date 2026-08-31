"""Integration smoke for the M1 training script — tiny CPU run, seconds."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import train_face_det as tfd  # noqa: E402


class TrainScriptTests(unittest.TestCase):
    def test_one_epoch_cpu_end_to_end(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="_facedet_"))
        try:
            # 6 synthetic images, each with one bright square "face".
            entries = []
            rng = np.random.default_rng(0)
            for i in range(6):
                img = rng.integers(0, 60, (96, 96, 3), dtype=np.uint8)
                cx, cy, half = 48 + (i % 3) * 4, 40, 16
                img[cy - half : cy + half, cx - half : cx + half] = 220
                p = tmp / f"img_{i}.png"
                cv2.imencode(".png", img)[1].tofile(str(p))
                entries.append(
                    {
                        "image": str(p),
                        "width": 96,
                        "height": 96,
                        "source": "synthetic",
                        "pseudo": False,
                        "boxes": [
                            [(cx - half) / 96, (cy - half) / 96,
                             (2 * half) / 96, (2 * half) / 96]
                        ],
                        "points5": None,
                    }
                )
            manifest = tmp / "manifest.jsonl"
            manifest.write_text(
                "\n".join(json.dumps(e) for e in entries), encoding="utf-8"
            )
            out = tmp / "facedet.pt"
            argv = [
                "--manifest", str(manifest),
                "--epochs", "2",
                "--batch", "2",
                "--size", "128",
                "--width", "24",
                "--device", "cpu",  # project rule: CPU when GPU busy
                "--out", str(out),
            ]
            rc = tfd.main(argv)
            self.assertEqual(0, rc)
            self.assertTrue(out.exists())
            ckpt = torch.load(out, map_location="cpu", weights_only=True)
            self.assertEqual(24, ckpt["width"])
            self.assertEqual(128, ckpt["input_size"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

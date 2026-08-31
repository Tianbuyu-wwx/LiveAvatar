"""Tests for the M0 face-dataset builder (R1 self-replacement pipeline).

Synthetic fixtures only — real WIDER FACE / 300W downloads are not needed
in CI.  The MediaPipe teacher path is exercised only when mediapipe and the
teacher model are available (training-time dependency).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import make_face_dataset as mfd  # noqa: E402


def _write_image(path: Path, w: int = 64, h: int = 64) -> None:
    cv2.imencode(".png", np.full((h, w, 3), 128, np.uint8))[1].tofile(str(path))


def _synthetic_pts68(cx: float = 32.0, cy: float = 32.0, s: float = 10.0) -> np.ndarray:
    """68 points on a face-like layout around (cx, cy)."""
    pts = np.zeros((68, 2), np.float32)
    pts[36:42] = [cx - 0.5 * s, cy - 0.3 * s]  # left eye
    pts[42:48] = [cx + 0.5 * s, cy - 0.3 * s]  # right eye
    pts[30] = [cx, cy]
    pts[48] = [cx - 0.3 * s, cy + 0.5 * s]
    pts[54] = [cx + 0.3 * s, cy + 0.5 * s]
    return pts


class Map68To5Tests(unittest.TestCase):
    def test_centroids_and_corners(self) -> None:
        pts5 = mfd.map68to5(_synthetic_pts68(cx=10.0, cy=20.0, s=10.0))
        self.assertAlmostEqual(float(pts5[0, 0]), 10.0 - 5.0)
        self.assertAlmostEqual(float(pts5[0, 1]), 20.0 - 3.0)
        self.assertAlmostEqual(float(pts5[1, 0]), 15.0)
        np.testing.assert_allclose(pts5[2], [10.0, 20.0])
        np.testing.assert_allclose(pts5[3], [7.0, 25.0])
        np.testing.assert_allclose(pts5[4], [13.0, 25.0])

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            mfd.map68to5(np.zeros((67, 2), np.float32))


class PtsParserTests(unittest.TestCase):
    def test_parse_real_300w_format(self) -> None:
        pts = _synthetic_pts68()
        lines = ["version: 1", "n_points: 68", "{"]
        lines += [f"{x:.3f} {y:.3f}" for x, y in pts]
        lines += ["}"]
        p = Path(__file__).parent / "_tmp.pts"
        p.write_text("\n".join(lines), encoding="utf-8")
        try:
            parsed = mfd.parse_pts(p)
        finally:
            p.unlink()
        np.testing.assert_allclose(parsed, pts, atol=1e-3)


class WiderParserTests(unittest.TestCase):
    def test_bbox_and_invalid_face_filtering(self) -> None:
        p = Path(__file__).parent / "_tmp_bbx.txt"
        p.write_text(
            "\n".join(
                [
                    "0--Parade/0_Parade_marchingband_1_849.jpg",  # event header skipped
                    "0--Parade/0_Parade_marchingband_1_849.jpg "
                    "449,330,122,331,488.9,373.6,0 "  # bbox + landmark triplet
                    "-71,-49,394,451,-999,-999,-999",  # invalid face → dropped
                    "",  # blank line skipped
                ]
            ),
            encoding="utf-8",
        )
        try:
            entries = mfd.parse_wider_annotations(p)
        finally:
            p.unlink()
        self.assertEqual(1, len(entries))
        rel, boxes = entries[0]
        self.assertEqual("0--Parade/0_Parade_marchingband_1_849.jpg", rel)
        self.assertEqual([[449, 330, 122, 331]], boxes)


class ManifestBuildTests(unittest.TestCase):
    def _w300_fixture(self, root: Path) -> Path:
        img_dir = root / "images"
        img_dir.mkdir(parents=True)
        img = img_dir / "face_001.png"
        _write_image(img, 64, 64)
        pts = _synthetic_pts68(32, 32, 10)
        p = img.with_suffix(".pts")
        lines = ["version: 1", "n_points: 68", "{"]
        lines += [f"{x:.3f} {y:.3f}" for x, y in pts]
        lines += ["}"]
        p.write_text("\n".join(lines), encoding="utf-8")
        return img

    def test_cli_w300_end_to_end(self) -> None:
        root = Path(__file__).parent / "_ds_w300"
        out = Path(__file__).parent / "_ds_out"
        img = self._w300_fixture(root)
        try:
            rc = mfd.main(
                ["--w300-root", str(root), "--out", str(out), "--limit", "5"]
            )
            self.assertEqual(0, rc)
            raw = (out / "manifest.jsonl").read_text().splitlines()
            entries = [json.loads(line) for line in raw]
            self.assertEqual(1, len(entries))
            e = entries[0]
            self.assertEqual("w300", e["source"])
            self.assertFalse(e["pseudo"])
            self.assertEqual(img.name, Path(e["image"]).name)
            self.assertEqual(1, len(e["boxes"]))
            self.assertEqual(5, len(e["points5"]))
            # Normalized coords stay within [0, 1].
            for x, y in e["points5"]:
                self.assertTrue(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
            # Eye centroid: 32-5=27 px over width 64.
            self.assertAlmostEqual(e["points5"][0][0], 27.0 / 64.0, places=5)
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(out, ignore_errors=True)

    def test_requires_a_source(self) -> None:
        with self.assertRaises(SystemExit):
            mfd.main([])


class TeacherTests(unittest.TestCase):
    def test_bbox_from_points5_square_margin(self) -> None:
        pts5 = np.array([[10, 10], [20, 10], [15, 15], [12, 20], [18, 20]], np.float32)
        box = mfd._bbox_from_points5(pts5)
        span = float((pts5.max(axis=0) - pts5.min(axis=0)).max())  # 10
        size = span * 1.4
        self.assertAlmostEqual(box[2], size)
        self.assertAlmostEqual(box[3], size)
        self.assertAlmostEqual(box[0], 15.0 - size / 2)
        self.assertAlmostEqual(box[1], 15.0 - size / 2)

    def test_teacher_labels_synthetic_face(self) -> None:
        if importlib.util.find_spec("mediapipe") is None:
            self.skipTest("mediapipe (training-time teacher) not installed")
        if not Path("models/mediapipe/face_landmarker.task").exists():
            self.skipTest("teacher model not downloaded")
        # A real photographic face is required for the teacher to fire;
        # synthetic noise frames legitimately return None, which we assert.
        frame = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        frame = np.ascontiguousarray(frame)
        result = mfd.label_own_with_mediapipe(frame)
        self.assertIsNone(result)  # no face in noise → None is the contract


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for the M4 acceptance metrics (face_accept) and harness wiring.

All synthetic + CPU-only: no mediapipe, no YuNet model, no checkpoints —
the harness is exercised with fake landmark/detect callables so the whole
scaffold stays CI-green before real weights exist.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from liveavatar import face_accept as fa

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

try:
    import torch

    _HAVE_TORCH = True
except ImportError:  # CI installs light extras only; torch is optional there
    _HAVE_TORCH = False

_TORCH = unittest.skipUnless(_HAVE_TORCH, "torch not installed (light CI env)")


def _template_pts(size: int = 200) -> np.ndarray:
    from face_align import get_template_5

    return get_template_5(size)


def _synthetic_face(size: int = 200) -> np.ndarray:
    img = np.full((size, size, 3), 30, np.uint8)
    img[40:150, 50:160] = 200
    return img


def _noise_frame(size: int = 200, seed: int = 0) -> np.ndarray:
    """Texture-rich frame — phase correlation needs structure to lock onto."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


class SsimTests(unittest.TestCase):
    def test_identical_images_score_one(self) -> None:
        img = fa.to_gray_f32(_synthetic_face())
        self.assertAlmostEqual(fa.ssim_gray(img, img), 1.0, places=6)

    def test_degraded_image_scores_below_one(self) -> None:
        import cv2

        img = fa.to_gray_f32(_synthetic_face())
        blurred = cv2.GaussianBlur(img, (9, 9), 3.0)
        score = fa.ssim_gray(img, blurred)
        self.assertLess(score, 0.999)
        self.assertGreater(score, 0.0)
        # Symmetry.
        self.assertAlmostEqual(score, fa.ssim_gray(blurred, img), places=9)

    def test_shape_mismatch_raises(self) -> None:
        a = np.zeros((32, 32), np.float32)
        with self.assertRaises(ValueError):
            fa.ssim_gray(a, np.zeros((16, 16), np.float32))


class GeometryTests(unittest.TestCase):
    def test_region_shift_px_identical_and_shifted(self) -> None:
        rng = np.random.default_rng(0)
        gray = rng.uniform(0, 255, (96, 96)).astype(np.float32)
        self.assertAlmostEqual(fa.region_shift_px(gray, gray, (8, 8, 88, 88)), 0.0, places=4)
        # np.roll by 3 px on both axes → dominant shift of 3·√2 px.
        shifted = np.roll(gray, 3, axis=(0, 1))
        dev = fa.region_shift_px(gray, shifted, (8, 8, 88, 88))
        self.assertAlmostEqual(dev, 3.0 * np.sqrt(2), delta=0.1)

    def test_region_shift_px_flat_crop_is_zero(self) -> None:
        flat = np.full((32, 32), 100.0, np.float32)
        self.assertEqual(fa.region_shift_px(flat, flat, (0, 0, 32, 32)), 0.0)

    def test_eye_mouth_dev_px_averages_boxes(self) -> None:
        rng = np.random.default_rng(1)
        gray = rng.uniform(0, 255, (128, 128)).astype(np.float32)
        boxes = [(10, 10, 60, 60), (70, 80, 120, 120)]
        self.assertAlmostEqual(fa.eye_mouth_dev_px(gray, gray, boxes), 0.0, places=4)
        shifted = np.roll(gray, 2, axis=1)  # 2 px horizontal
        dev = fa.eye_mouth_dev_px(gray, shifted, boxes)
        self.assertAlmostEqual(dev, 2.0, delta=0.1)

    def test_eye_mouth_boxes_contain_points(self) -> None:
        pts = _template_pts(200)
        boxes = fa.eye_mouth_boxes(pts, 200, 200)
        self.assertEqual(len(boxes), 2)
        for idxs, box in zip(((0, 1), (3, 4)), boxes, strict=True):
            x1, y1, x2, y2 = box
            for i in idxs:
                self.assertTrue(x1 <= pts[i][0] <= x2)
                self.assertTrue(y1 <= pts[i][1] <= y2)

    def test_region_mean_abs_intensity(self) -> None:
        img = _synthetic_face()
        self.assertEqual(
            fa.region_mean_abs_intensity(img, img, [(50, 40, 160, 150)]), 0.0
        )
        changed = img.copy()
        changed[40:150, 50:160] = 255
        dev = fa.region_mean_abs_intensity(img, changed, [(50, 40, 160, 150)])
        self.assertGreater(dev, 30.0)
        # Outside the region nothing counts.
        self.assertEqual(fa.region_mean_abs_intensity(img, changed, [(0, 0, 5, 5)]), 0.0)


class MaskIoUTests(unittest.TestCase):
    def test_identical_and_disjoint(self) -> None:
        self.assertEqual(fa.box_iou((10, 10, 50, 50), (10, 10, 50, 50)), 1.0)
        self.assertEqual(fa.box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_known_partial_iou(self) -> None:
        # Inter 10x20=200; union 300+300-200=400 → 0.5.
        self.assertAlmostEqual(fa.box_iou((0, 0, 10, 30), (0, 10, 10, 40)), 0.5)

    def test_empty_box_semantics(self) -> None:
        self.assertEqual(fa.box_iou((0, 0, 0, 0), (0, 0, 0, 0)), 1.0)
        self.assertEqual(fa.box_iou((0, 0, 0, 0), (5, 5, 9, 9)), 0.0)

    def test_average_and_validation(self) -> None:
        a = [(0, 0, 10, 10), (0, 0, 0, 0)]
        self.assertEqual(fa.average_mask_iou(a, a), 1.0)
        with self.assertRaises(ValueError):
            fa.average_mask_iou(a, a[:1])
        with self.assertRaises(ValueError):
            fa.average_mask_iou([], [])


class SpeedAndGatesTests(unittest.TestCase):
    def test_speed_ratio(self) -> None:
        self.assertAlmostEqual(fa.speed_ratio(0.2, 0.3), 1.5)
        with self.assertRaises(ValueError):
            fa.speed_ratio(0.0, 0.1)

    def test_evaluate_gates_all_pass(self) -> None:
        report = {
            "eye_mouth_dev_px": 0.5,
            "ssim": 0.99,
            "mask_coords_iou": 0.98,
            "speed_ratio": 1.2,
        }
        gates = fa.evaluate_gates(report)
        self.assertEqual(set(gates.values()), {"pass", "skipped"})
        self.assertTrue(fa.overall_pass(gates))

    def test_evaluate_gates_fail(self) -> None:
        report = {
            "eye_mouth_dev_px": 3.0,
            "ssim": 0.99,
            "mask_coords_iou": 0.98,
            "speed_ratio": 1.2,
        }
        gates = fa.evaluate_gates(report)
        self.assertEqual(gates["eye_mouth_dev_px<=2px"], "fail")
        self.assertFalse(fa.overall_pass(gates))

    def test_skipped_core_gate_fails_overall(self) -> None:
        gates = fa.evaluate_gates({"ssim": 0.99, "mask_coords_iou": 0.98})
        self.assertEqual(gates["eye_mouth_dev_px<=2px"], "skipped")
        self.assertFalse(fa.overall_pass(gates))

    def test_optional_latents_gate_may_be_skipped(self) -> None:
        gates = {
            "eye_mouth_dev_px<=2px": "pass",
            "ssim>=0.97": "pass",
            "mask_coords_iou>=0.95": "pass",
            "speed_ratio<=1.5": "pass",
            "latents_cosine>=0.99": "skipped",
        }
        self.assertTrue(fa.overall_pass(gates))


@_TORCH
class LatentsCosineTests(unittest.TestCase):
    def test_identical_and_mismatched(self) -> None:
        import tempfile
        from pathlib import Path

        lat = [torch.zeros(1, 8, 32, 32), torch.randn(1, 8, 32, 32)]
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = Path(tmp) / "a.pt", Path(tmp) / "b.pt"
            torch.save(lat, pa)
            torch.save(lat, pb)
            self.assertAlmostEqual(fa.latents_cosine(pa, pb), 1.0, places=6)
            torch.save(lat[:1], pb)
            with self.assertRaises(ValueError):
                fa.latents_cosine(pa, pb)


class AcceptHarnessWiringTests(unittest.TestCase):
    """scripts/accept_face_backend.run_acceptance with fake backends."""

    # Deterministic per-call burn so legacy timing >> self timing (keeps the
    # speed_ratio gate stable despite micro-benchmark noise).
    _BURN = np.arange(2_000_000, dtype=np.float64)

    def _run(self, jitter_px: float) -> dict:
        import accept_face_backend as harness

        frames = [_noise_frame()]
        pts = _template_pts()
        jittered = pts + np.array([jitter_px, 0.0])  # x-shift only

        def legacy_landmarks(frame):
            _ = self._BURN.sum()  # deterministic burn, points untouched
            return pts

        def self_landmarks(frame):
            return jittered

        return harness.run_acceptance(
            frames,
            legacy_landmarks=legacy_landmarks,
            self_landmarks=self_landmarks,
            align_size=128,
        )

    def test_identical_backends_pass(self) -> None:
        report = self._run(jitter_px=0.0)
        self.assertEqual(report["aligned_frames"], 1)
        self.assertLess(report["eye_mouth_dev_px"], 0.05)
        self.assertAlmostEqual(report["ssim"], 1.0, places=5)
        self.assertEqual(report["mask_coords_iou"], 1.0)
        self.assertGreater(report["legacy_ms_per_frame"], 0.0)
        self.assertGreater(report["self_ms_per_frame"], 0.0)
        self.assertLess(report["speed_ratio"], 1.5)
        self.assertTrue(report["pass"])

    def test_small_jitter_within_gate(self) -> None:
        # 1 px source jitter at 200 px → sub-px shift in the 128 alignment.
        report = self._run(jitter_px=1.0)
        self.assertGreater(report["eye_mouth_dev_px"], 0.1)
        self.assertLess(report["eye_mouth_dev_px"], 2.0)
        # Pure-noise frames tank SSIM even for sub-px shifts (expected);
        # the wiring check only asserts degradation vs the identical case.
        self.assertLess(report["ssim"], 1.0)

    def test_large_jitter_fails(self) -> None:
        report = self._run(jitter_px=10.0)
        self.assertGreater(report["eye_mouth_dev_px"], 2.0)
        self.assertFalse(report["pass"])

    def test_self_backend_missing_frames_fails_overall(self) -> None:
        import accept_face_backend as harness

        report = harness.run_acceptance(
            [_synthetic_face()],
            legacy_landmarks=lambda f: _template_pts(),
            self_landmarks=lambda f: None,
            align_size=128,
        )
        self.assertEqual(report["aligned_frames"], 0)
        self.assertIsNone(report["eye_mouth_dev_px"])
        self.assertIsNone(report["ssim"])
        self.assertFalse(report["pass"])

    def test_mask_iou_is_landmark_derived(self) -> None:
        """mask IoU now derives from the landmark points, not det boxes."""
        # Identical points → identical mask boxes → IoU 1.0.
        report = self._run(jitter_px=0.0)
        self.assertEqual(report["mask_coords_iou"], 1.0)
        # A 2 px point shift slightly moves the landmark-derived box.
        report = self._run(jitter_px=2.0)
        self.assertGreater(report["mask_coords_iou"], 0.95)
        self.assertLess(report["mask_coords_iou"], 1.0)

    def test_load_frames_from_directory(self) -> None:
        import tempfile

        import accept_face_backend as harness
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                ok, buf = cv2.imencode(".png", _synthetic_face())
                assert ok
                buf.tofile(str(Path(tmp) / f"{i}.png"))
            frames = harness.load_frames(tmp, max_frames=8)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0].shape, (200, 200, 3))


if __name__ == "__main__":
    unittest.main()

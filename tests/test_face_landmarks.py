"""Tests for the self-developed 5-point landmark student (R1 M2). CPU only."""

from __future__ import annotations

import random
import unittest
from pathlib import Path

import numpy as np

try:
    import torch

    from liveavatar.face_landmarks import (
        LM_STRIDE,
        NUM_POINTS,
        FaceCropTransform,
        LandmarkNet5Self,
        gaussian_targets,
        landmark_loss,
        landmarks5,
        nme,
    )
    _HAVE_TORCH = True
except ImportError:  # light CI env — torch is optional there
    _HAVE_TORCH = False

_TORCH = unittest.skipUnless(_HAVE_TORCH, "torch not installed (light CI env)")

# 5 template points as *full-image* normalized coordinates, laid out at
# realistic face proportions but well separated (smoke-test-friendly).
GT_CROP = np.array(
    [[0.25, 0.30], [0.75, 0.30], [0.50, 0.50], [0.35, 0.72], [0.65, 0.72]],
    np.float32,
)
# The overfit uses box [0,0,1,1] with the default 25% margin, so the network
# works in crop space — targets must go through the same mapping.
TEST_BOX = [0.0, 0.0, 1.0, 1.0]
GT_IN_CROP = (GT_CROP + 0.25) / 1.5  # FaceCropTransform(TEST_BOX).points(GT_CROP)


# One distinct color per landmark channel — gives each heatmap channel an
# unambiguous visual cue (mirrors how the student locks onto teacher points).
POINT_COLORS = [
    (255, 40, 40),    # red — left eye
    (40, 255, 40),    # green — right eye
    (40, 40, 255),    # blue — nose
    (255, 255, 40),   # yellow — mouth left
    (40, 255, 255),   # cyan — mouth right
]


def _synthetic_face(size: int = 128) -> np.ndarray:
    """Dark background with one colored dot per landmark position."""
    img = np.full((size, size, 3), 25, np.uint8)
    for (x, y), color in zip(GT_CROP, POINT_COLORS, strict=True):
        cx, cy = int(x * size), int(y * size)
        img[cy - 2 : cy + 3, cx - 2 : cx + 3] = color
    return img


@_TORCH
class ArchitectureTests(unittest.TestCase):
    def test_forward_shapes_and_budget(self) -> None:
        model = LandmarkNet5Self()
        n_params = sum(p.numel() for p in model.parameters())
        self.assertLess(n_params, 500_000)  # M2 gate: <0.5M params
        x = torch.randn(2, 3, 128, 128)
        hm, off = model(x)
        grid = 128 // LM_STRIDE
        self.assertEqual((2, NUM_POINTS, grid, grid), tuple(hm.shape))
        self.assertEqual((2, NUM_POINTS * 2, grid, grid), tuple(off.shape))


@_TORCH
class DecodeTests(unittest.TestCase):
    def test_argmax_plus_offset_decode(self) -> None:
        grid = 8
        hm = torch.zeros(1, NUM_POINTS, grid, grid)
        off = torch.zeros(1, NUM_POINTS * 2, grid, grid)
        # Point 0 at cell (y=2, x=3) with offset (0.25, 0.5).
        hm[0, 0, 2, 3] = 5.0
        off = off.reshape(1, NUM_POINTS, 2, grid, grid)
        off[0, 0, 0, 2, 3] = 0.25
        off[0, 0, 1, 2, 3] = 0.5
        pts = LandmarkNet5Self.decode_landmarks(
            hm, off.reshape(1, NUM_POINTS * 2, grid, grid)
        )[0]
        self.assertAlmostEqual(float(pts[0, 0]), (3 + 0.25) / grid, places=6)
        self.assertAlmostEqual(float(pts[0, 1]), (2 + 0.5) / grid, places=6)
        # Other points default to cell 0 (zero heatmap argmax) — valid.

    def test_gaussian_peak_at_gt_cell(self) -> None:
        gt = torch.tensor([[[0.75, 0.25]]])  # → cell (x=6, y=2) on 8×8
        target = gaussian_targets(gt, 8, 8, sigma=1.0)
        peak = target[0, 0].argmax()
        self.assertEqual(2 * 8 + 6, int(peak))


@_TORCH
class PooledDecodeTests(unittest.TestCase):
    """3×3 avg-pool before argmax (round-3 anti-jitter measure)."""

    def test_pool_flips_argmax_to_stable_neighbour(self) -> None:
        grid = 8
        hm = torch.zeros(1, NUM_POINTS, grid, grid)
        off = torch.zeros(1, NUM_POINTS * 2, grid, grid)
        # Isolated peak at (y=2, x=6): plain argmax picks it, but its 3×3
        # neighbourhood average (peak + 8 zero cells) loses to a broad
        # plateau centred at (y=2, x=2) — pooling picks the stable plateau.
        hm[0, 0, 2, 6] = 5.0
        hm[0, 0, 1:4, 1:4] = 1.2
        pts_plain = LandmarkNet5Self.decode_landmarks(hm, off)[0]
        pts_pooled = LandmarkNet5Self.decode_landmarks(hm, off, pool=True)[0]
        self.assertAlmostEqual(float(pts_plain[0, 0]), 6 / grid, places=6)
        self.assertAlmostEqual(float(pts_pooled[0, 0]), 2 / grid, places=6)

    def test_pool_keeps_lone_gaussian_peak_within_one_cell(self) -> None:
        """A well-separated Gaussian peak survives pooling (no drift)."""
        grid = 32
        gt = torch.tensor([[[0.75, 0.25], [0.3, 0.6], [0.5, 0.45], [0.4, 0.8], [0.6, 0.8]]])
        hm = gaussian_targets(gt, grid, grid, sigma=1.5) * 4.0 - 2.0
        off = torch.zeros(1, NUM_POINTS * 2, grid, grid)
        plain = LandmarkNet5Self.decode_landmarks(hm, off)[0]
        pooled = LandmarkNet5Self.decode_landmarks(hm, off, pool=True)[0]
        # Same cell picked → offsets identical; coords must agree exactly.
        np.testing.assert_allclose(pooled.numpy(), plain.numpy(), atol=1e-6)

    def test_pool_ignored_in_soft_mode(self) -> None:
        grid = 8
        hm = torch.full((1, NUM_POINTS, grid, grid), -12.0)
        hm[0, 0, 2, 3] = 12.0
        off = torch.zeros(1, NUM_POINTS * 2, grid, grid)
        a = LandmarkNet5Self.decode_landmarks(hm, off, mode="soft", pool=True)[0]
        b = LandmarkNet5Self.decode_landmarks(hm, off, mode="soft", pool=False)[0]
        np.testing.assert_allclose(a.numpy(), b.numpy())


@_TORCH
class SoftDecodeTests(unittest.TestCase):
    def test_soft_argmax_peak_location(self) -> None:
        grid = 8
        hm = torch.full((1, NUM_POINTS, grid, grid), -12.0)
        hm[0, 0, 2, 3] = 12.0  # confident single-cell peak
        off = torch.zeros(1, NUM_POINTS * 2, grid, grid)
        pts = LandmarkNet5Self.decode_landmarks(hm, off, mode="soft")[0]
        # Expected coordinate of a one-hot at cell index i is i/grid — the
        # same continuous convention as gaussian_targets (gx = x_norm * grid).
        self.assertAlmostEqual(float(pts[0, 0]), 3 / grid, delta=1e-3)
        self.assertAlmostEqual(float(pts[0, 1]), 2 / grid, delta=1e-3)

    def test_soft_argmax_between_cells(self) -> None:
        """Two equal peaks → expected coord at the midpoint (sub-pixel)."""
        grid = 8
        hm = torch.full((1, NUM_POINTS, grid, grid), -12.0)
        hm[0, 0, 3, 3] = 12.0
        hm[0, 0, 3, 4] = 12.0
        off = torch.zeros(1, NUM_POINTS * 2, grid, grid)
        pts = LandmarkNet5Self.decode_landmarks(hm, off, mode="soft")[0]
        self.assertAlmostEqual(float(pts[0, 0]), 3.5 / grid, delta=1e-3)
        self.assertAlmostEqual(float(pts[0, 1]), 3 / grid, delta=1e-3)

    def test_soft_argmax_matches_gaussian_gt(self) -> None:
        """Realistic calibrated logits: sigmoid(logits) ≈ Gaussian target,
        background pushed negative (as the focal loss trains it)."""
        grid = 16
        gt = torch.tensor(GT_CROP).unsqueeze(0)
        target = gaussian_targets(gt, grid, grid, sigma=1.0)
        hm = torch.logit(target.clamp(1e-4, 1.0 - 1e-4))
        off = torch.zeros(1, NUM_POINTS * 2, grid, grid)
        pts = LandmarkNet5Self.decode_landmarks(hm, off, mode="soft")[0]
        np.testing.assert_allclose(pts.numpy(), GT_CROP, atol=0.02)


class _FixedRng(random.Random):
    """Deterministic RNG: always flip, fixed brightness gain/offset."""

    def random(self) -> float:  # noqa: D102 — always below the 0.5 flip gate
        return 0.0

    def uniform(self, a: float, b: float) -> float:  # noqa: D102
        return a


class AugmentTests(unittest.TestCase):
    """Flip semantics of scripts.train_face_landmarks.augment_batch (no torch)."""

    @classmethod
    def setUpClass(cls) -> None:
        import sys

        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import train_face_landmarks as tfl  # noqa: E402

        cls.tfl = tfl

    def test_flip_swaps_eye_and_mouth_identities(self) -> None:
        imgs = np.zeros((1, 3, 4, 4), np.float32)
        pts = np.array([GT_CROP], np.float32)  # (1, 5, 2) crop-normalized
        out_imgs, out_pts = self.tfl.augment_batch(imgs, pts, _FixedRng())
        # x mirrored; identities swapped: L eye ↔ R eye, mouth L ↔ mouth R.
        np.testing.assert_allclose(out_pts[0, 0], [1 - GT_CROP[1, 0], GT_CROP[1, 1]], atol=1e-6)
        np.testing.assert_allclose(out_pts[0, 1], [1 - GT_CROP[0, 0], GT_CROP[0, 1]], atol=1e-6)
        np.testing.assert_allclose(out_pts[0, 2], [1 - GT_CROP[2, 0], GT_CROP[2, 1]], atol=1e-6)
        np.testing.assert_allclose(out_pts[0, 3], [1 - GT_CROP[4, 0], GT_CROP[4, 1]], atol=1e-6)
        np.testing.assert_allclose(out_pts[0, 4], [1 - GT_CROP[3, 0], GT_CROP[3, 1]], atol=1e-6)
        # Image rows preserved (flip is horizontal only).
        self.assertEqual(out_imgs.shape, imgs.shape)


@_TORCH
class LossTests(unittest.TestCase):
    def test_zero_loss_when_offsets_match_targets(self) -> None:
        grid = 16
        gt = torch.tensor(GT_CROP).unsqueeze(0)  # (1, 5, 2)
        hm = gaussian_targets(gt, grid, grid, sigma=1.0).requires_grad_(True)
        cell_x = (gt[..., 0] * grid).floor()
        cell_y = (gt[..., 1] * grid).floor()
        dx = gt[..., 0] * grid - cell_x
        dy = gt[..., 1] * grid - cell_y
        off = torch.zeros(1, NUM_POINTS, 2, grid, grid)
        for c in range(NUM_POINTS):
            off[0, c, 0, int(cell_y[0, c]), int(cell_x[0, c])] = dx[0, c]
            off[0, c, 1, int(cell_y[0, c]), int(cell_x[0, c])] = dy[0, c]
        off = off.reshape(1, NUM_POINTS * 2, grid, grid).requires_grad_(True)
        loss, parts = landmark_loss(hm, off, gt, coord_weight=0.25)
        self.assertLess(parts["off"], 1e-5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()  # graph must be differentiable end to end


@_TORCH
class CropTransformTests(unittest.TestCase):
    BOX = [0.2, 0.25, 0.5, 0.5]  # normalized xywh, points inside

    def test_points_roundtrip(self) -> None:
        t = FaceCropTransform(self.BOX, out_size=128, margin=0.25)
        pts = GT_CROP.copy()  # all inside the expanded box
        back = t.points_back(t.points(pts))
        np.testing.assert_allclose(back, pts, atol=1e-6)

    def test_crop_size_and_content(self) -> None:
        t = FaceCropTransform(self.BOX, out_size=64)
        crop = t.crop(_synthetic_face(128))
        self.assertEqual((64, 64, 3), crop.shape)


@_TORCH
class OverfitTests(unittest.TestCase):
    def test_cpu_overfit_landmarks(self) -> None:
        torch.manual_seed(0)
        model = LandmarkNet5Self(width=16)
        img = _synthetic_face(128)
        # Same pipeline as inference: crop by the face box, targets in crop space.
        transform = FaceCropTransform(TEST_BOX, out_size=128)
        crop = transform.crop(img)
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        gt = torch.tensor(GT_IN_CROP).unsqueeze(0)
        opt = torch.optim.Adam(model.parameters(), lr=8e-3)
        first = None
        for step in range(700):
            hm, off = model(tensor)
            # Peaks first (focal), then coord loss refines the offsets.
            loss, _ = landmark_loss(
                hm, off, gt, coord_weight=0.0 if step < 200 else 0.5
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss.detach())
        self.assertLess(float(loss.detach()), first * 0.2)

        pred = landmarks5(model, img, [0.0, 0.0, 1.0, 1.0], input_size=128)
        error = nme(pred, GT_CROP)
        self.assertLess(error, 0.03)  # <3% normalized mean error after overfit


if __name__ == "__main__":
    unittest.main()

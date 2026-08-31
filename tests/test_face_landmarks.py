"""Tests for the self-developed 5-point landmark student (R1 M2). CPU only."""

from __future__ import annotations

import unittest

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

"""Smoke tests for the self-developed tiny face detector (R1 M1).

All tests run on CPU in seconds: forward shapes, anchor/decode roundtrip,
self-written NMS, and a short CPU overfit on one synthetic face.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch

    from liveavatar.face_self import (
        ANCHOR_SIZES,
        NUM_ANCHORS,
        STRIDE,
        TinyFaceDetector,
        cxcywh_to_xyxy,
        detection_loss,
        nms,
    )
    _HAVE_TORCH = True
except ImportError:  # CI installs light extras only; torch is optional there
    _HAVE_TORCH = False

_TORCH = unittest.skipUnless(_HAVE_TORCH, "torch not installed (light CI env)")


def _synthetic_face_image(size: int = 256) -> np.ndarray:
    """Dark background, bright face-like square in the upper center."""
    img = np.full((size, size, 3), 30, np.uint8)
    cx, cy, half = size // 2, size // 3, size // 8
    img[cy - half : cy + half, cx - half : cx + half] = 220
    return img


@_TORCH
class ArchitectureTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = TinyFaceDetector()
        n_params = sum(p.numel() for p in model.parameters())
        self.assertLess(n_params, 1_000_000)  # <1MB fp32 budget (M1 gate)
        x = torch.randn(2, 3, 256, 256)
        cls, box = model(x)
        k = (256 // STRIDE) ** 2
        self.assertEqual((2, NUM_ANCHORS, k), tuple(cls.shape))
        self.assertEqual((2, NUM_ANCHORS * 4, k), tuple(box.shape))

    def test_anchor_layout_and_decode_roundtrip(self) -> None:
        gh = gw = 4
        anchors = TinyFaceDetector.make_anchors(gh, gw)
        self.assertEqual((NUM_ANCHORS * gh * gw, 4), tuple(anchors.shape))
        # Zero offsets → boxes exactly on anchors.
        decoded = TinyFaceDetector.decode_offsets(
            torch.zeros(1, NUM_ANCHORS * 4, gh * gw), anchors
        )[0]
        np.testing.assert_allclose(decoded.numpy(), anchors.numpy(), atol=1e-6)
        # Manual offset: +0.5 * aw in x on the first anchor.
        off = torch.zeros(1, NUM_ANCHORS * 4, gh * gw)
        off[0, 0, 0] = 0.5
        decoded = TinyFaceDetector.decode_offsets(off, anchors)[0]
        self.assertAlmostEqual(
            float(decoded[0, 0]), float(anchors[0, 0] + 0.5 * anchors[0, 2]), places=6
        )


@_TORCH
class NmsTests(unittest.TestCase):
    def test_suppresses_overlap_keeps_distinct(self) -> None:
        boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = nms(boxes, scores, iou_thr=0.4)
        self.assertEqual([0, 2], keep)

    def test_empty_input(self) -> None:
        self.assertEqual([], nms(np.zeros((0, 4), np.float32), np.zeros((0,))))

    def test_cxcywh_conversion(self) -> None:
        xyxy = cxcywh_to_xyxy(np.array([[5, 5, 10, 10]], np.float32))
        np.testing.assert_allclose(xyxy, [[0, 0, 10, 10]])


@_TORCH
class OverfitTests(unittest.TestCase):
    def test_cpu_overfit_single_face_two_phases(self) -> None:
        """The loss must drop substantially within a few CPU steps on one
        synthetic face — proves anchor matching, decode and loss wiring."""
        torch.manual_seed(0)
        model = TinyFaceDetector(width=24)  # smaller for test speed
        size = 128
        anchors = TinyFaceDetector.make_anchors(size // STRIDE, size // STRIDE)
        img = torch.from_numpy(_synthetic_face_image(size)).permute(2, 0, 1)
        img = img.unsqueeze(0).float() / 255.0
        gt = torch.tensor([[0.5, 1 / 3, 0.25, 0.25]])  # matches the drawn square
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        first_loss = None
        for step in range(60):
            cls, box = model(img)
            loss, _ = detection_loss(cls, box, anchors, [gt])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step == 0:
                first_loss = float(loss.detach())
        self.assertLess(float(loss.detach()), first_loss * 0.2)
        self.assertTrue(np.isfinite(float(loss.detach())))

        # Detection after overfit: the known face must be found with a tight box.
        from liveavatar.face_self import detect

        rgb = _synthetic_face_image(size)
        results = detect(model, rgb, conf_threshold=0.5, input_size=size)
        self.assertTrue(results)
        xyxy, score = results[0]
        # Ground-truth square in pixels: x [48, 80], y [21.3, 42.7] (size 128).
        cx_gt, cy_gt = 64.0, size / 3
        w_gt = 0.25 * size
        cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
        self.assertLess(abs(cx - cx_gt), 12.0)
        self.assertLess(abs(cy - cy_gt), 12.0)
        self.assertLess(abs((xyxy[2] - xyxy[0]) - w_gt), 18.0)
        self.assertGreater(score, 0.5)

    def test_anchor_sizes_cover_domain(self) -> None:
        # Five scales from tiny (WIDER median 0.028) to large (w300 up to
        # 0.74) — under 0.5-IoU matching every box class needs a nearby
        # anchor, and detection_loss additionally force-matches each GT's
        # best anchor (SSD-style guarantee).
        self.assertEqual(5, len(ANCHOR_SIZES))
        self.assertEqual(5, NUM_ANCHORS)
        self.assertTrue(all(0 < s < 1 for s in ANCHOR_SIZES))

    def test_forced_best_match_supervises_tiny_face(self) -> None:
        """A GT far below any anchor still yields a positive via forced match."""
        anchors = TinyFaceDetector.make_anchors(16, 16)  # 256 input
        # Tiny face: 3% of the side — below every anchor's 0.5-IoU reach.
        gt = torch.tensor([[0.5, 0.5, 0.03, 0.03]])
        cls = torch.zeros(1, NUM_ANCHORS, 16 * 16)
        box = torch.zeros(1, NUM_ANCHORS * 4, 16 * 16)
        _, pos_mask = detection_loss(cls, box, anchors, [gt])
        self.assertTrue(pos_mask.any())


if __name__ == "__main__":
    unittest.main()

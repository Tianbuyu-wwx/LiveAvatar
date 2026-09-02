# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Self-developed tiny face detector (R1 M1) — a BlazeFace-lite re-implementation.

Design goals (docs/自研人脸检测与对齐方案_2026-08-31.md §4 M1):
- structure, training and inference code written from scratch; the only
  runtime dependencies are torch + numpy;
- ~0.5 MB parameter budget, single-level anchor head at stride 16 — the
  avatar-preparation domain has one large, mostly frontal face per frame,
  so WIDER-hard-style tiny-face recall is explicitly out of scope;
- anchor decoding and NMS implemented locally, no torchvision dependency.

Wire layout::

    image (1, 3, H, W), H/W multiples of 16
      → DSConv stem (s2) + 3 DSConv stages (s2 each) → stride-16 feature map
      → 3x3 conv head → per-cell: 3 anchor classes + 3 anchor boxes
    decode: cx = ax + dx * aw;  cy = ay + dy * ah;
            w  = aw * exp(dw);  h  = ah * exp(dh)      (normalized coords)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ANCHORS = 5
# Normalized to the image side. Five scales (was three 0.15/0.30/0.50):
# WIDER faces median 2.8% of the side never matched a 0.15 anchor under
# 0.5-IoU, so 96% of WIDER boxes got no positive supervision (see 方案 doc
# §Anchor 尺度适配检查). New trainings emit 5-anchor checkpoints that record
# "anchor_sizes"; legacy 3-anchor checkpoints load via the LEGACY default.
ANCHOR_SIZES = (0.05, 0.10, 0.15, 0.30, 0.50)
LEGACY_ANCHOR_SIZES = (0.15, 0.30, 0.50)
STRIDE = 16


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise + 1x1 pointwise, standard BlazeFace/MobileNet building block."""

    def __init__(self, ch_in: int, ch_out: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            ch_in, ch_in, 3, stride=stride, padding=1, groups=ch_in, bias=False
        )
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.pointwise = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        return self.act(self.bn2(self.pointwise(x)))


class TinyFaceDetector(nn.Module):
    """Stride-16 single-level anchor face detector (~0.2M params)."""

    def __init__(self, width: int = 48, anchor_sizes: tuple[float, ...] = ANCHOR_SIZES) -> None:
        super().__init__()
        self.anchor_sizes = tuple(anchor_sizes)
        num_anchors = len(self.anchor_sizes)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        chs = (16, 24, 32, width)
        blocks = []
        for ch_in, ch_out in zip(chs, chs[1:], strict=False):
            blocks.append(DepthwiseSeparableConv(ch_in, ch_out, stride=2))
        self.body = nn.Sequential(*blocks)
        self.head_conv = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(width, num_anchors, 1)
        self.box_head = nn.Conv2d(width, num_anchors * 4, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-anchor class logits (N, A, Hc*Wc) and box offsets (N, A*4, Hc*Wc)."""
        feat = self.head_conv(self.body(self.stem(x)))
        n, _, hc, wc = feat.shape
        a = len(self.anchor_sizes)
        cls = self.cls_head(feat).reshape(n, a, hc * wc)
        box = self.box_head(feat).reshape(n, a * 4, hc * wc)
        return cls, box

    @staticmethod
    def make_anchors(
        grid_h: int, grid_w: int, sizes: tuple[float, ...] = ANCHOR_SIZES
    ) -> torch.Tensor:
        """(A*grid_h*grid_w, 4) normalized anchors as (cx, cy, w, h)."""
        ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5) / grid_h
        xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5) / grid_w
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        anchors = []
        for size in sizes:
            anchors.append(
                torch.stack(
                    [cx.reshape(-1), cy.reshape(-1),
                     torch.full_like(cx.reshape(-1), size),
                     torch.full_like(cx.reshape(-1), size)],
                    dim=1,
                )
            )
        return torch.cat(anchors, dim=0)  # (A*gh*gw, 4)

    @staticmethod
    def decode_offsets(
        offsets: torch.Tensor, anchors: torch.Tensor, num_anchors: int = NUM_ANCHORS
    ) -> torch.Tensor:
        """(…, A*4, K) or (K, A*4) offsets → (…, K, 4) boxes (cx, cy, w, h)."""
        if offsets.dim() == 2:
            offsets = offsets.unsqueeze(0)
        n, a4, k = offsets.shape
        assert a4 == num_anchors * 4
        d = offsets.reshape(n, num_anchors, 4, k)
        ax = anchors[:, 0].view(1, num_anchors, 1, k)
        ay = anchors[:, 1].view(1, num_anchors, 1, k)
        aw = anchors[:, 2].view(1, num_anchors, 1, k)
        ah = anchors[:, 3].view(1, num_anchors, 1, k)
        # Keep every term at (n, A, 1, k) so broadcasting stays aligned.
        cx = ax + d[:, :, 0].unsqueeze(2) * aw
        cy = ay + d[:, :, 1].unsqueeze(2) * ah
        w = aw * torch.exp(d[:, :, 2].unsqueeze(2).clamp(max=4.0))
        h = ah * torch.exp(d[:, :, 3].unsqueeze(2).clamp(max=4.0))
        return torch.stack([cx, cy, w, h], dim=-1).reshape(n, num_anchors * k, 4)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.4) -> list[int]:
    """Vectorized greedy NMS on (N, 4) xyxy boxes. Self-written (no torchvision)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


@torch.no_grad()
def detect(
    model: TinyFaceDetector,
    image: np.ndarray,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.4,
    input_size: int = 256,
) -> list[tuple[list[float], float]]:
    """Run inference on one RGB uint8 image; return [(xyxy_pixel, score), ...].

    Boxes are in ORIGINAL-image pixel coordinates (x1, y1, x2, y2), despite
    the historical docstring claiming normalization — callers have always
    consumed pixel coords. ``input_size`` must match the resolution the
    model was trained on (multiple of STRIDE) so the anchor grid stays
    aligned.
    """
    model.eval()
    h, w = image.shape[:2]
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = F.interpolate(
        tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
    )
    cls, box = model(tensor)
    gh = gw = input_size // STRIDE
    sizes = getattr(model, "anchor_sizes", ANCHOR_SIZES)
    anchors = TinyFaceDetector.make_anchors(gh, gw, sizes)
    boxes = TinyFaceDetector.decode_offsets(
        box[0], anchors, len(sizes)
    )[0].numpy()  # (A*K, 4) cxcywh
    probs = torch.sigmoid(cls[0]).reshape(-1).numpy()  # (A*K,)
    mask = probs >= conf_threshold
    if not mask.any():
        return []
    xyxy = cxcywh_to_xyxy(boxes[mask])
    keep = nms(xyxy, probs[mask], iou_threshold)
    results: list[tuple[list[float], float]] = []
    for i in keep:
        cx, cy, bw, bh = boxes[mask][i]
        # back to input-image pixel coords
        results.append(
            (
                [
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                ],
                float(probs[mask][i]),
            )
        )
    return results


def detection_loss(
    cls_logits: torch.Tensor,
    box_offsets: torch.Tensor,
    anchors: torch.Tensor,
    gt_boxes: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Focal classification + smooth-L1 box loss with 0.5-IoU anchor matching.

    cls_logits: (N, A, K); box_offsets: (N, A*4, K); anchors: (A*K, 4) cxcywh;
    gt_boxes: per-sample (G_i, 4) normalized cxcywh (may be empty).
    Returns (loss, pos_mask) where pos_mask is (N, A, K) for diagnostics.
    """
    device = cls_logits.device
    n, a, k = cls_logits.shape
    anchors = anchors.to(device)
    targets_cls = torch.zeros(n, a, k, device=device)
    targets_box = torch.zeros(n, a, 4, k, device=device)
    pos_masks = torch.zeros(n, a, k, dtype=torch.bool, device=device)

    for i, gt in enumerate(gt_boxes):
        if gt.numel() == 0:
            continue
        gt = gt.to(device)
        # anchor/GT IoU in cxcywh → xyxy
        ax1 = anchors[:, 0] - anchors[:, 2] / 2
        ay1 = anchors[:, 1] - anchors[:, 3] / 2
        ax2 = anchors[:, 0] + anchors[:, 2] / 2
        ay2 = anchors[:, 1] + anchors[:, 3] / 2
        gx1 = gt[:, 0] - gt[:, 2] / 2
        gy1 = gt[:, 1] - gt[:, 3] / 2
        gx2 = gt[:, 0] + gt[:, 2] / 2
        gy2 = gt[:, 1] + gt[:, 3] / 2
        ix1 = torch.maximum(ax1[:, None], gx1[None, :])
        iy1 = torch.maximum(ay1[:, None], gy1[None, :])
        ix2 = torch.minimum(ax2[:, None], gx2[None, :])
        iy2 = torch.minimum(ay2[:, None], gy2[None, :])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_g = (gx2 - gx1) * (gy2 - gy1)
        iou = inter / (area_a[:, None] + area_g[None, :] - inter + 1e-9)
        pos = iou.max(dim=1).values >= 0.5  # (A*K,)
        # SSD-style guarantee: every GT keeps its best-matching anchor as a
        # positive even below the 0.5 threshold — small / odd-aspect faces
        # (WIDER median 0.028, w300 tight boxes) otherwise receive NO
        # supervision at all.
        if iou.shape[1] > 0:
            pos[iou.argmax(dim=0)] = True
        if not pos.any():
            continue  # ignore unmatched anchors (neither pos nor hard neg)
        best_gt = iou.argmax(dim=1)  # (A*K,)
        flat = pos.nonzero(as_tuple=True)[0]
        anchor_idx = flat // k
        cell_idx = flat % k
        targets_cls[i, anchor_idx, cell_idx] = 1.0
        pos_masks[i, anchor_idx, cell_idx] = True
        matched_gt = gt[best_gt[pos]]  # (P, 4)
        anc = anchors[pos]  # (P, 4)
        dx = (matched_gt[:, 0] - anc[:, 0]) / anc[:, 2]
        dy = (matched_gt[:, 1] - anc[:, 1]) / anc[:, 3]
        dw = torch.log(matched_gt[:, 2] / anc[:, 2] + 1e-9)
        dh = torch.log(matched_gt[:, 3] / anc[:, 3] + 1e-9)
        off = torch.stack([dx, dy, dw, dh], dim=1)  # (P, 4)
        for j, (ai, ci) in enumerate(zip(anchor_idx.tolist(), cell_idx.tolist(), strict=True)):
            targets_box[i, ai, :, ci] = off[j]

    # Focal loss on classification (ignore cells that are pos in *other*
    # samples' spirit — here simple binary focal over all cells).
    p = torch.sigmoid(cls_logits)
    ce = F.binary_cross_entropy_with_logits(
        cls_logits, targets_cls, reduction="none"
    )
    alpha, gamma = 0.75, 2.0
    p_t = targets_cls * p + (1 - targets_cls) * (1 - p)
    focal = alpha * (1 - p_t) ** gamma * ce
    cls_loss = focal.sum() / max(targets_cls.sum().item(), 1.0)

    # Smooth-L1 on positives only. (A, 4, K) → cell-major (A*K, 4) ordering
    # that matches the anchor layout; a raw reshape would scramble (4, K).
    pos_flat = pos_masks.reshape(n, a * k)
    box_flat = (
        box_offsets.reshape(n, a, 4, k).permute(0, 1, 3, 2).reshape(n, a * k, 4)
    )
    tgt_flat = targets_box.permute(0, 1, 3, 2).reshape(n, a * k, 4)
    if pos_flat.any():
        box_loss = F.smooth_l1_loss(box_flat[pos_flat], tgt_flat[pos_flat])
    else:
        box_loss = box_flat.sum() * 0.0
    return cls_loss + box_loss, pos_masks

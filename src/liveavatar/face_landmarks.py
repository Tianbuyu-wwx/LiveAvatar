"""Self-developed 5-point landmark student (R1 M2) — PIPNet-style.

Purpose (docs/自研人脸检测与对齐方案_2026-08-31.md §4 M2): replace the
MediaPipe FaceLandmarker 5-point output used by ``scripts/face_align.py``
with a <0.5M-parameter student trained on 300W ground truth + MediaPipe
teacher pseudo labels (training-time only).  Runtime deps: torch + numpy.

Design:
- input: a 128×128 face crop (``FaceCropTransform`` maps full-image
  normalized boxes/points into crop space);
- backbone: DSConv stem + 2 blocks → stride-4 feature map (32×32 cells);
- heads: 5 heatmap channels + 5×2 offset channels (PIPNet "heatmap+offset,
  single-step decoding", no post-processing);
- decode: per-channel argmax cell + offset residual → normalized coords;
- loss (``landmark_loss``): Gaussian-target heatmap MSE + offset L1 at the
  ground-truth cell + a small coordinate-distillation L2 (teacher point
  supervision) for fast convergence.

Offset semantics: ``x_norm = (cell_x + dx) / W`` with ``dx ∈ [0, 1)``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_POINTS = 5
LM_STRIDE = 4


class LandmarkNet5Self(nn.Module):
    """Tiny 5-point landmark student (~0.15M params at width 32).

    GroupNorm instead of BatchNorm: tiny student batches are 1–8 images and
    train/eval statistics must not drift (single-image overfit smoke tests
    rely on deterministic eval output).
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.decode_mode = "argmax"  # checkpoint may override ("soft")
        # 3×3 avg-pool on the heatmap before argmax (round-3 anti-jitter
        # measure): flattens single-cell argmax jumps between near-equal
        # neighbour peaks. Restored from the checkpoint / LANDMARK_POOL env.
        self.pool_before_argmax = False
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )
        self.ds1 = DepthwiseSeparable(16, width, stride=1)
        self.ds2 = DepthwiseSeparable(width, width, stride=2)  # → stride 4
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(4, width),
            nn.ReLU(inplace=True),
        )
        self.hm_head = nn.Conv2d(width, NUM_POINTS, 1)
        self.off_head = nn.Conv2d(width, NUM_POINTS * 2, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return heatmaps (N, 5, Hm, Wm) and offsets (N, 10, Hm, Wm)."""
        feat = self.head(self.ds2(self.ds1(self.stem(x))))
        return self.hm_head(feat), self.off_head(feat)

    @staticmethod
    def decode_landmarks(
        heatmaps: torch.Tensor,
        offsets: torch.Tensor,
        mode: str = "argmax",
        pool: bool = False,
    ) -> torch.Tensor:
        """(N, 5, Hm, Wm), (N, 10, Hm, Wm) → (N, 5, 2) normalized coords.

        mode="argmax": per-channel argmax cell + offset residual (PIPNet
        single-step decoding). Differentiable w.r.t. the offsets gathered at
        the (detached) argmax cell, so the coordinate loss trains the offset
        head directly. ``pool=True`` first smooths the heatmaps with a 3×3
        average pooling (stride 1), which stabilizes the argmax when two
        neighbouring cells carry near-equal logits; the offset is still
        gathered at the (possibly re-picked) argmax cell.

        mode="soft": differentiable DSNT-style decoding — sigmoid heatmaps
        normalized into a per-channel probability distribution, then the
        expected (x, y) is taken over the whole grid. Sub-pixel by
        construction — no argmax quantization, no keypoint jumping between
        cells; the offset head is unused at decode time. ``pool`` is ignored.
        """
        n, c, h, w = heatmaps.shape
        if mode == "soft":
            p = torch.sigmoid(heatmaps)
            p = p / p.sum(dim=(2, 3), keepdim=True).clamp(min=1e-12)
            xs = torch.arange(w, dtype=torch.float32, device=heatmaps.device)
            ys = torch.arange(h, dtype=torch.float32, device=heatmaps.device)
            ex = (p * xs.view(1, 1, 1, w)).sum(dim=(2, 3))  # (n, c)
            ey = (p * ys.view(1, 1, h, 1)).sum(dim=(2, 3))
            return torch.stack([ex / w, ey / h], dim=-1)
        hm = heatmaps
        if pool:
            hm = F.avg_pool2d(hm, kernel_size=3, stride=1, padding=1)
        flat = hm.reshape(n, c, h * w)
        idx = flat.argmax(dim=-1)  # (n, c) — detached by argmax
        cell_y = (idx // w).float()
        cell_x = (idx % w).float()
        off = offsets.reshape(n, c, 2, h * w)
        idx_e = idx.view(n, c, 1, 1).expand(n, c, 2, 1)  # (n, c, 2, 1)
        off_at = off.gather(3, idx_e).squeeze(3)  # (n, c, 2)
        x = (cell_x + off_at[..., 0]) / w
        y = (cell_y + off_at[..., 1]) / h
        return torch.stack([x, y], dim=-1)


class DepthwiseSeparable(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            ch_in, ch_in, 3, stride=stride, padding=1, groups=ch_in, bias=False
        )
        self.gn1 = nn.GroupNorm(4, ch_in)
        self.pointwise = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        self.gn2 = nn.GroupNorm(4, ch_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gn1(self.depthwise(x)))
        return self.act(self.gn2(self.pointwise(x)))


def gaussian_targets(
    points: torch.Tensor, grid_h: int, grid_w: int, sigma: float = 1.0
) -> torch.Tensor:
    """(N, 5, 2) normalized points → (N, 5, grid_h, grid_w) Gaussians."""
    n, c, _ = points.shape
    ys = torch.arange(grid_h, dtype=torch.float32, device=points.device).view(1, 1, grid_h, 1)
    xs = torch.arange(grid_w, dtype=torch.float32, device=points.device).view(1, 1, 1, grid_w)
    gx = (points[..., 0] * grid_w).view(n, c, 1, 1)
    gy = (points[..., 1] * grid_h).view(n, c, 1, 1)
    sq = (xs - gx) ** 2 + (ys - gy) ** 2
    return torch.exp(-sq / (2.0 * sigma * sigma))


def landmark_loss(
    heatmaps: torch.Tensor,
    offsets: torch.Tensor,
    gt_points: torch.Tensor,
    sigma: float = 1.5,
    coord_weight: float = 0.25,
    decode_mode: str = "argmax",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Focal heatmap loss (soft Gaussian labels) + offset L1 @ gt cell + coord L2.

    Plain BCE lets the net cheat by shifting all logits negative (targets are
    sparse), leaving the heatmap flat and the argmax stuck at center; the
    focal term suppresses that easy background mass (same recipe as the M1
    detector). gt_points: (N, 5, 2) crop-normalized.
    """
    n, c, h, w = heatmaps.shape
    gt_points = gt_points.to(heatmaps.device)
    target = gaussian_targets(gt_points, h, w, sigma)
    bce = F.binary_cross_entropy_with_logits(heatmaps, target, reduction="none")
    p = torch.sigmoid(heatmaps)
    p_t = target * p + (1 - target) * (1 - p)
    alpha_t = 0.75 * target + 0.25 * (1 - target)
    focal = alpha_t * (1 - p_t) ** 2 * bce
    hm_loss = focal.sum() / target.sum().clamp(min=1.0)

    # Offset supervision at the ground-truth cell: dx = x*W - floor(x*W).
    gx = gt_points[..., 0] * w
    gy = gt_points[..., 1] * h
    cell_x = gx.floor().clamp(max=w - 1)
    cell_y = gy.floor().clamp(max=h - 1)
    dx_t = (gx - cell_x).clamp(0.0, 1.0)
    dy_t = (gy - cell_y).clamp(0.0, 1.0)
    off = offsets.reshape(n, c, 2, h, w)
    b_idx = torch.arange(n, device=off.device).view(n, 1)
    c_idx = torch.arange(c, device=off.device).view(1, c)
    cx_i = cell_x.long()
    cy_i = cell_y.long()
    pred_dx = off[b_idx, c_idx, 0, cy_i, cx_i]  # (n, c)
    pred_dy = off[b_idx, c_idx, 1, cy_i, cx_i]
    off_loss = F.l1_loss(pred_dx, dx_t) + F.l1_loss(pred_dy, dy_t)

    parts = {"hm": float(hm_loss.detach()), "off": float(off_loss.detach())}
    loss = hm_loss + off_loss
    if coord_weight > 0:
        pts = LandmarkNet5Self.decode_landmarks(heatmaps, offsets, mode=decode_mode)
        coord_loss = F.l1_loss(pts, gt_points.to(heatmaps.device))
        loss = loss + coord_weight * coord_loss
        parts["coord"] = float(coord_loss.detach())
    return loss, parts


@torch.no_grad()
def landmarks5(
    model: LandmarkNet5Self,
    image_rgb: np.ndarray,
    face_box_xywh_norm: list[float],
    input_size: int = 128,
    decode_mode: str | None = None,
) -> np.ndarray:
    """Detect 5 landmarks for one face box; returns (5, 2) normalized to
    the *full image* (same convention as the manifest / MediaPipe teacher).

    ``decode_mode`` defaults to the mode recorded on the model (set from
    the checkpoint by ``face_backend.load_lm_model``).
    """
    model.eval()
    transform = FaceCropTransform(face_box_xywh_norm, out_size=input_size)
    crop = transform.crop(image_rgb)
    tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    hm, off = model(tensor)
    mode = decode_mode or str(getattr(model, "decode_mode", "argmax"))
    pool = bool(getattr(model, "pool_before_argmax", False))
    pts = LandmarkNet5Self.decode_landmarks(hm, off, mode=mode, pool=pool)[0].numpy()  # (5, 2) crop
    return transform.points_back(pts)


@dataclasses.dataclass
class FaceCropTransform:
    """Maps full-image normalized boxes/points into a square face crop.

    The box is expanded by ``margin`` on each side (relative to its size),
    clipped to the image, then resized to ``out_size``².
    """

    box_xywh_norm: list[float]
    out_size: int = 128
    margin: float = 0.25

    def __post_init__(self) -> None:
        x, y, w, h = self.box_xywh_norm
        mx, my = w * self.margin, h * self.margin
        self.x0 = x - mx
        self.y0 = y - my
        self.span = max(w + 2 * mx, h + 2 * my)  # square crop

    def crop(self, image_rgb: np.ndarray) -> np.ndarray:
        h_img, w_img = image_rgb.shape[:2]
        x1 = int(self.x0 * w_img)
        y1 = int(self.y0 * h_img)
        x2 = int((self.x0 + self.span) * w_img)
        y2 = int((self.y0 + self.span) * h_img)
        # Pad when the square crop exceeds the image bounds.
        top, bottom = max(0, -y1), max(0, y2 - h_img)
        left, right = max(0, -x1), max(0, x2 - w_img)
        canvas = np.pad(
            image_rgb,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        crop = canvas[
            top + y1 : top + y2, left + x1 : left + x2
        ]
        return cv2_resize(crop, self.out_size)

    def points(self, pts_full_norm: np.ndarray) -> np.ndarray:
        """Full-image normalized points → crop-normalized points."""
        h_img = w_img = 1.0  # inputs are already normalized
        px = pts_full_norm[..., 0] * w_img
        py = pts_full_norm[..., 1] * h_img
        return np.stack(
            [(px - self.x0) / self.span, (py - self.y0) / self.span], axis=-1
        ).clip(0.0, 1.0)

    def points_back(self, pts_crop_norm: np.ndarray) -> np.ndarray:
        """Crop-normalized points → full-image normalized points."""
        px = pts_crop_norm[..., 0] * self.span + self.x0
        py = pts_crop_norm[..., 1] * self.span + self.y0
        return np.stack([px, py], axis=-1)


def cv2_resize(image: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def nme(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean normalized L2 error over all points (lower is better)."""
    return float(np.linalg.norm(pred - gt, axis=-1).mean())

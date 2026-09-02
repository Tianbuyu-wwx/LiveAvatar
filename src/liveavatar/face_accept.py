# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""M4 acceptance metrics for the self-developed face backend (CPU only).

Pure-metric helpers backing ``scripts/accept_face_backend.py`` — the gates
come from docs/自研人脸检测与对齐方案_2026-08-31.md §M4:

- aligned eye/mouth landmark deviation ≤ 2 px (self vs legacy teacher)
- aligned-image SSIM ≥ 0.97 (self-written, no skimage)
- mask_coords IoU ≥ 0.95 (region-delta transport downstream unaffected)
- CPU speed ratio self/legacy ≤ 1.5
- latents cosine ≥ 0.99 (optional; needs the VAE latents files)

No model is loaded here: callers pass points/boxes/images, so every metric
is unit-testable with synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# 5-point layout: left eye, right eye, nose, mouth left, mouth right.
EYE_POINT_IDXS = (0, 1)
MOUTH_POINT_IDXS = (3, 4)

# M4 gates.
GATE_DEV_PX = 2.0
GATE_SSIM = 0.97
GATE_MASK_IOU = 0.95
GATE_SPEED_RATIO = 1.5
GATE_LATENTS_COSINE = 0.99


def to_gray_f32(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → float32 grayscale."""
    import cv2

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM between two float32 grayscale images (self-written).

    Standard 11×11 Gaussian-window formulation (σ=1.5), C1/C2 per Wang et
    al. 2004; implemented with cv2.GaussianBlur — no skimage dependency.
    """
    if a.shape != b.shape:
        raise ValueError(f"ssim_gray shape mismatch: {a.shape} vs {b.shape}")
    import cv2

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_aa
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_bb
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_aa + mu_bb + c1) * (sigma_a + sigma_b + c2)
    return float((num / den).mean())


def region_shift_px(
    gray_a: np.ndarray, gray_b: np.ndarray, box: tuple[int, int, int, int]
) -> float:
    """Dominant pixel shift between two aligned grayscale crops (px).

    Phase correlation (cv2.phaseCorrelate) on the cropped region; returns the
    shift magnitude. Flat crops (no structure) yield 0.0 by convention.
    """
    import cv2

    x1, y1, x2, y2 = box
    h, w = gray_a.shape
    # Trim to even width/height: odd-sized crops bias phaseCorrelate by
    # 0.5 px (asymmetric internal FFT padding).
    x2 = x1 + ((min(w, x2) - x1) & ~1)
    y2 = y1 + ((min(h, y2) - y1) & ~1)
    a = np.ascontiguousarray(gray_a[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)])
    b = np.ascontiguousarray(gray_b[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)])
    if a.size == 0 or a.shape != b.shape or min(a.shape) < 8:
        return 0.0
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0  # no structure → shift undefined
    result = cv2.phaseCorrelate(a, b)
    # OpenCV returns ((dx, dy), response) on some versions, (dx, dy, response)
    # on others — normalize here.
    first = result[0]
    if isinstance(first, (tuple, list, np.ndarray)):
        dx, dy = float(first[0]), float(first[1])
    else:  # older OpenCV: flat (dx, dy, response) tuple
        dx, dy = float(first[0]), float(result[1])
    return float(np.hypot(dx, dy))


def eye_mouth_dev_px(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> float:
    """Mean phase-correlated shift (px) over the eye/mouth region boxes.

    This is the M4 "眼/嘴区域平均像素偏差" gate metric: two correct
    alignments of the same face differ by ≈0 px, a misaligned backend shows
    up as a real pixel shift in the aligned output.
    """
    return float(np.mean([region_shift_px(gray_a, gray_b, box) for box in boxes]))


def eye_mouth_boxes(
    pts5_px: np.ndarray,
    image_w: int,
    image_h: int,
    scale: float = 0.55,
    min_half: float = 8.0,
) -> list[tuple[int, int, int, int]]:
    """Square analysis windows around the eye pair and mouth pair.

    Half-extent scales with the pair distance (interocular / mouth width) so
    the phase-correlation windows stay meaningful across alignment sizes.
    """
    boxes: list[tuple[int, int, int, int]] = []
    for idxs in (EYE_POINT_IDXS, MOUTH_POINT_IDXS):
        pts = pts5_px[list(idxs)]
        cx, cy = pts.mean(axis=0)
        span = float(np.linalg.norm(pts[1] - pts[0]))
        half = max(span * scale, min_half)
        boxes.append(
            (
                max(0, int(cx - half)),
                max(0, int(cy - half)),
                min(image_w, int(cx + half) + 1),
                min(image_h, int(cy + half) + 1),
            )
        )
    return boxes


def region_mean_abs_intensity(
    img_a: np.ndarray, img_b: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> float:
    """Mean |a−b| grayscale intensity within the union of pixel boxes."""
    ga, gb = to_gray_f32(img_a), to_gray_f32(img_b)
    h, w = ga.shape
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True
    if not mask.any():
        return 0.0
    return float(np.abs(ga - gb)[mask].mean())


def _is_empty_box(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x2 <= x1 or y2 <= y1


def box_iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    """IoU of two xyxy pixel boxes; two empty boxes count as identical (1.0)."""
    if _is_empty_box(a) and _is_empty_box(b):
        return 1.0
    if _is_empty_box(a) or _is_empty_box(b):
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def average_mask_iou(
    coords_a: list[tuple[int, int, int, int]],
    coords_b: list[tuple[int, int, int, int]],
) -> float:
    """Mean per-frame IoU over two mask_coords.pkl lists (must be same len)."""
    if len(coords_a) != len(coords_b):
        raise ValueError(
            f"mask coords length mismatch: {len(coords_a)} vs {len(coords_b)}"
        )
    if not coords_a:
        raise ValueError("no mask coords to compare")
    return float(np.mean([box_iou(a, b) for a, b in zip(coords_a, coords_b, strict=True)]))


def speed_ratio(t_legacy: float, t_self: float) -> float:
    """self / legacy per-frame time; gate is ratio ≤ 1.5."""
    if t_legacy <= 0:
        raise ValueError("legacy timing must be positive")
    return float(t_self / t_legacy)


def latents_cosine(path_a: str | Path, path_b: str | Path) -> float:
    """Mean cosine similarity between two latents.pt lists (MuseTalk gate).

    Requires torch (loads the [1, 8, 32, 32] tensors written by
    prepare_avatar); length mismatch or shape mismatch raises ValueError.
    """
    import torch

    # weights_only=True: latents.pt files are plain lists of tensors as
    # written by prepare_avatar — no pickle opcodes beyond tensors needed.
    a = torch.load(path_a, map_location="cpu", weights_only=True)
    b = torch.load(path_b, map_location="cpu", weights_only=True)
    if len(a) != len(b):
        raise ValueError(f"latents length mismatch: {len(a)} vs {len(b)}")
    cosines: list[float] = []
    for ta, tb in zip(a, b, strict=True):
        if ta.shape != tb.shape:
            raise ValueError(f"latent shape mismatch: {tuple(ta.shape)} vs {tuple(tb.shape)}")
        fa, fb = ta.reshape(-1).float(), tb.reshape(-1).float()
        denom = fa.norm() * fb.norm()
        if denom == 0:
            cosines.append(1.0 if fa.norm() == 0 and fb.norm() == 0 else 0.0)
        else:
            cosines.append(float((fa @ fb) / denom))
    return float(np.mean(cosines))


def evaluate_gates(report: dict) -> dict[str, str]:
    """Map each M4 gate to 'pass' / 'fail' / 'skipped' from a report dict.

    Expected keys: eye_mouth_dev_px, ssim, mask_coords_iou, speed_ratio,
    latents_cosine (optional). A gate whose metric is None is 'skipped'.
    """
    checks = [
        ("eye_mouth_dev_px<=2px", report.get("eye_mouth_dev_px"), lambda v: v <= GATE_DEV_PX),
        ("ssim>=0.97", report.get("ssim"), lambda v: v >= GATE_SSIM),
        ("mask_coords_iou>=0.95", report.get("mask_coords_iou"), lambda v: v >= GATE_MASK_IOU),
        ("speed_ratio<=1.5", report.get("speed_ratio"), lambda v: v <= GATE_SPEED_RATIO),
        ("latents_cosine>=0.99", report.get("latents_cosine"), lambda v: v >= GATE_LATENTS_COSINE),
    ]
    gates: dict[str, str] = {}
    for name, value, ok in checks:
        if value is None:
            gates[name] = "skipped"
        else:
            gates[name] = "pass" if ok(float(value)) else "fail"
    return gates


def overall_pass(gates: dict[str, str]) -> bool:
    """All gates pass; skipped gates are tolerated only for optional ones."""
    optional = {"latents_cosine>=0.99"}
    for name, status in gates.items():
        if status == "fail":
            return False
        if status == "skipped" and name not in optional:
            return False
    return True

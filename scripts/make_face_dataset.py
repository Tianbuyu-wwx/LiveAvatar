"""M0 (R1 face self-replacement): build a unified face-dataset manifest.

Assembles training data for the self-developed face detector (M1) and
5-point landmark student (M2) from three sources:

* WIDER FACE (``--wider-root``)      — bbox ground truth       → task "det"
* 300W / 300-VW (``--w300-root``)    — 68-point .pts ground truth
                                                               → task "pts"
* Own material (``--own``)           — MediaPipe teacher pseudo labels
  (images dir or video)                (478 landmarks → 5 points) → task "both"

MediaPipe is a *training-time-only* labeler (R1 plan §3): nothing here is
used at runtime.  Output is a JSONL manifest (``data/face_ds/manifest.jsonl``)
referencing the original image paths, plus optionally copied own-material
frames (``--copy-own``) so the manifest stays valid while real datasets are
downloaded later.

Manifest entry schema::

    {"id": str, "image": path, "width": int, "height": int,
     "source": "wider|w300|own", "pseudo": bool,
     "boxes": [[x, y, w, h], ...],          # normalized to [0, 1] (xywh)
     "points5": [[x, y], ...] | null}       # normalized, 5x2

Normalization keeps the manifest resolution-independent; the training
loader denormalizes with the image size.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

# 300W 68-point indices → 5-point template (ArcFace convention):
# eye *centroids* (stable under blink), nose tip, mouth corners.
_L_EYE = range(36, 42)
_R_EYE = range(42, 48)
_NOSE = 30
_MOUTH_L = 48
_MOUTH_R = 54
POINT5_SOURCES = (_L_EYE, _R_EYE, _NOSE, _MOUTH_L, _MOUTH_R)


def map68to5(pts68: np.ndarray) -> np.ndarray:
    """Map a (68, 2) landmark array to the (5, 2) alignment template."""
    if pts68.shape != (68, 2):
        raise ValueError(f"expected (68, 2) landmarks, got {pts68.shape}")
    pts5 = np.empty((5, 2), dtype=np.float32)
    pts5[0] = pts68[36:42].mean(axis=0)  # left eye centroid
    pts5[1] = pts68[42:48].mean(axis=0)  # right eye centroid
    pts5[2] = pts68[30]
    pts5[3] = pts68[48]
    pts5[4] = pts68[54]
    return pts5


def parse_pts(path: Path) -> np.ndarray:
    """Parse a 300W ``.pts`` file into a (68, 2) float array."""
    coords: list[list[float]] = []
    inside = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line == "{":
            inside = True
            continue
        if line == "}":
            break
        if inside:
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                coords.append([float(parts[0]), float(parts[1])])
    pts = np.asarray(coords, dtype=np.float32)
    if pts.shape[0] != 68:
        raise ValueError(f"{path}: expected 68 points, got {pts.shape[0]}")
    return pts


def parse_wider_annotations(path: Path) -> list[tuple[str, list[list[int]]]]:
    """Parse a WIDER FACE ``bbox_train.txt``-style annotation file.

    Line format: ``<rel path> <x>,<y>,<w>,<h>[,<lx>,<ly>,<blur>]...``.
    Invalid boxes (WIDER marks unusable faces with negative values) and the
    landmark triplets are ignored — we only need bboxes.
    """
    entries: list[tuple[str, list[list[int]]]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        items = line.split()
        if not items[0].endswith((".jpg", ".png", ".jpeg")):
            continue  # event header line ("0--Parade" totals row)
        rel = items[0]
        boxes: list[list[int]] = []
        for item in items[1:]:
            parts = item.split(",")
            if len(parts) < 4:
                continue
            x, y, w, h = (int(float(p)) for p in parts[:4])
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                continue  # WIDER "invalid face" marker
            boxes.append([x, y, w, h])
        if boxes:
            entries.append((rel, boxes))
    return entries


def _imread_safe(path: Path) -> np.ndarray | None:
    """Unicode-safe imread (cv2.imread fails on non-ASCII Windows paths)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def iter_own_frames(path: Path):
    """Yield ``(name, bgr_frame)`` from an image file/dir or a video."""
    if path.is_dir():
        for p in sorted(path.glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            data = np.fromfile(str(p), dtype=np.uint8)  # unicode-safe on Windows
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                yield p.name, frame
        return
    cap = cv2.VideoCapture(str(path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield f"{path.stem}_{idx:06d}.jpg", frame
        idx += 1
    cap.release()


def _bbox_from_points5(pts5: np.ndarray) -> list[float]:
    """Square bbox covering the 5 points with a 40% margin (teacher boxes
    are only used as a coarse prior for the detector student)."""
    lo = pts5.min(axis=0)
    hi = pts5.max(axis=0)
    center = (lo + hi) / 2
    size = float((hi - lo).max()) * 1.4
    half = size / 2
    return [
        float(center[0] - half),
        float(center[1] - half),
        size,
        size,
    ]


def label_own_with_mediapipe(frame: np.ndarray) -> tuple[list[float], list[list[float]]] | None:
    """Run the MediaPipe teacher on one BGR frame.

    Returns ``(bbox_xywh_norm, points5_norm)`` or ``None`` when no face is
    found.  Requires ``mediapipe`` (training-time dependency only).
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode as RunningMode,
        )
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "mediapipe is required to label own material (training-time only); "
            "install with: uv pip install --python <python> mediapipe"
        ) from exc

    model = Path("models/mediapipe/face_landmarker.task")
    if not model.exists():
        raise FileNotFoundError(
            f"{model} not found — run scripts/download_models.py first"
        )
    h, w = frame.shape[:2]
    landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
    )
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    finally:
        landmarker.close()
    if not result.face_landmarks:
        return None
    lm = np.array([[p.x, p.y] for p in result.face_landmarks[0]], dtype=np.float32)
    pts5 = map68to5(_mp478_to_68_like(lm))
    # Teacher landmarks are already normalized to [0, 1] by MediaPipe.
    return _bbox_from_points5(pts5), [[float(x), float(y)] for x, y in pts5]


def _mp478_to_68_like(lm: np.ndarray) -> np.ndarray:
    """Select the 68 300W-equivalent indices out of MediaPipe's 478 points.

    Only the 5 template sources are needed; the rest are unused.
    """
    # MediaPipe FaceMesh index map (canonical mesh): eye corners/contours,
    # nose tip, mouth corners.
    sel = {
        "l_eye": [33, 133, 160, 159, 158, 144],
        "r_eye": [362, 263, 387, 386, 385, 380],
        "nose": [1],
        "mouth_l": [61],
        "mouth_r": [291],
    }
    out = np.empty((68, 2), dtype=np.float32)
    for dst_idx, mp_idx in zip(_L_EYE, sel["l_eye"], strict=True):
        out[dst_idx] = lm[mp_idx]
    for dst_idx, mp_idx in zip(_R_EYE, sel["r_eye"], strict=True):
        out[dst_idx] = lm[mp_idx]
    out[30] = lm[sel["nose"][0]]
    out[48] = lm[sel["mouth_l"][0]]
    out[54] = lm[sel["mouth_r"][0]]
    # Fill unused slots with NaN so a misuse of the full array is loud.
    used = set(_L_EYE) | set(_R_EYE) | {30, 48, 54}
    for i in range(68):
        if i not in used:
            out[i] = np.nan
    return out


def _norm_boxes_xywh(boxes: list[list[float]], w: int, h: int) -> list[list[float]]:
    return [
        [max(0.0, b[0] / w), max(0.0, b[1] / h), min(1.0, b[2] / w), min(1.0, b[3] / h)]
        for b in boxes
    ]


def _norm_points(pts: np.ndarray, w: int, h: int) -> list[list[float]]:
    return [[float(x / w), float(y / h)] for x, y in pts]


def build_manifest(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    count = 0

    with manifest_path.open("w", encoding="utf-8") as fh:
        def emit(entry: dict) -> None:
            nonlocal count
            entry["id"] = uuid.uuid4().hex[:12]
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

        # --- WIDER FACE: detection ground truth ---------------------------
        if args.wider_root:
            ann = Path(args.wider_root) / "wider_face_split" / (
                "wider_face_train_bbx_gt.txt"
            )
            img_root = Path(args.wider_root) / "WIDER_train" / "images"
            for rel, boxes in parse_wider_annotations(ann):
                img = img_root / rel
                if not img.exists():
                    continue
                frame = _imread_safe(img)
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                emit(
                    {
                        "image": str(img),
                        "width": w,
                        "height": h,
                        "source": "wider",
                        "pseudo": False,
                        "boxes": _norm_boxes_xywh(boxes, w, h),
                        "points5": None,
                    }
                )
                if args.limit and count >= args.limit:
                    break

        # --- 300W: 5-point ground truth -----------------------------------
        w300_count = 0
        if args.w300_root:
            root = Path(args.w300_root)
            for pts_path in sorted(root.rglob("*.pts")):
                img_path = pts_path.with_suffix(".png")
                if not img_path.exists():
                    img_path = pts_path.with_suffix(".jpg")
                if not img_path.exists():
                    continue
                frame = _imread_safe(img_path)
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                pts5 = map68to5(parse_pts(pts_path))
                emit(
                    {
                        "image": str(img_path),
                        "width": w,
                        "height": h,
                        "source": "w300",
                        "pseudo": False,
                        "boxes": _norm_boxes_xywh([_bbox_from_points5(pts5)], w, h),
                        "points5": _norm_points(pts5, w, h),
                    }
                )
                w300_count += 1
                if args.limit and w300_count >= args.limit:
                    break

        # --- Own material: MediaPipe teacher pseudo labels ----------------
        if args.own:
            own_dir = out_dir / "own"
            own_dir.mkdir(exist_ok=True)
            own_count = 0
            for name, frame in iter_own_frames(Path(args.own)):
                labeled = label_own_with_mediapipe(frame)
                if labeled is None:
                    continue
                bbox, pts5 = labeled
                h, w = frame.shape[:2]
                dst = own_dir / name
                if args.copy_own:
                    cv2.imencode(".jpg", frame)[1].tofile(str(dst))
                    image_ref = str(dst)
                else:
                    own_root = Path(args.own)
                    image_ref = str(own_root / name if own_root.is_dir() else own_root)
                emit(
                    {
                        "image": image_ref,
                        "width": w,
                        "height": h,
                        "source": "own",
                        "pseudo": True,
                        "boxes": _norm_boxes_xywh([bbox], w, h),
                        "points5": pts5,
                    }
                )
                own_count += 1
                if args.limit and own_count >= args.limit:
                    break

    print(f"[face_ds] wrote {count} entries -> {manifest_path}")
    return count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wider-root", help="WIDER FACE root (wider_face_split/ + WIDER_train/)")
    ap.add_argument("--w300-root", help="300W root (rglob *.pts)")
    ap.add_argument("--own", help="own material: image dir, image file, or video")
    ap.add_argument("--out", default="data/face_ds", help="output dir (default: data/face_ds)")
    ap.add_argument("--copy-own", action="store_true", help="copy own-material frames into out dir")
    ap.add_argument("--limit", type=int, default=0, help="per-source entry limit (0 = all)")
    args = ap.parse_args(argv)
    if not (args.wider_root or args.w300_root or args.own):
        ap.error("at least one of --wider-root / --w300-root / --own is required")
    build_manifest(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""M4 acceptance harness: self vs legacy face-backend consistency (CPU only).

Runs the legacy (yunet/mediapipe) and self-developed backends over the same
frames and reports the M4 gates from
docs/自研人脸检测与对齐方案_2026-08-31.md:

- aligned eye/mouth landmark deviation ≤ 2 px
- aligned-image SSIM ≥ 0.97 (self-written SSIM)
- mask_coords IoU ≥ 0.95
- CPU per-frame speed ratio self/legacy ≤ 1.5
- latents cosine ≥ 0.99 (optional, via --latents-a/--latents-b)

Usage:
    python scripts/accept_face_backend.py --input data/video/yongen.mp4 \
        --det-ckpt weights/self/facedet_256.pt --lm-ckpt weights/self/landmarks5_128.pt \
        --out docs/face_accept_report.json

Exits 0 when all evaluated gates pass, 1 on gate failure, 2 when a backend
cannot run (missing weights / model files) — the scaffold is runnable
before real training and will report 'skipped' gates instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))  # for face_align template helpers

_SRC_DIR = _SCRIPTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))  # direct CLI run without editable install

from liveavatar import face_accept as fa  # noqa: E402
from liveavatar import face_backend as fb  # noqa: E402

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _imread_utf8(path: str) -> np.ndarray | None:
    """cv2.imread with unicode path support on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_frames(input_path: str, max_frames: int) -> list[np.ndarray]:
    """Load up to max_frames BGR frames from a video / image / directory."""
    p = Path(input_path)
    frames: list[np.ndarray] = []
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.*")))
        for f in files:
            if Path(f).suffix.lower() in _IMAGE_EXTS:
                img = _imread_utf8(f)
                if img is not None:
                    frames.append(img)
                if len(frames) >= max_frames:
                    break
    elif p.suffix.lower() in _VIDEO_EXTS:
        cap = cv2.VideoCapture(str(p))
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    elif p.suffix.lower() in _IMAGE_EXTS:
        img = _imread_utf8(str(p))
        if img is not None:
            frames.append(img)
    if not frames:
        raise RuntimeError(f"no readable frames from {input_path}")
    return frames


def _warp_to_template(
    frame: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, size: int
) -> np.ndarray | None:
    """Similar-transform the frame onto the 5-point template; return image."""
    m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if m is None:
        return None
    return cv2.warpAffine(
        frame,
        m,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _mask_coords_from_boxes(
    boxes_per_frame: list[list[fb.FaceBox]], frames: list[np.ndarray]
) -> list[tuple[int, int, int, int]]:
    """Largest face per frame → padded mask crop box (prepare_avatar logic)."""
    coords: list[tuple[int, int, int, int]] = []
    for frame, boxes in zip(frames, boxes_per_frame, strict=True):
        fh, fw = frame.shape[:2]
        if not boxes:
            coords.append((0, 0, 0, 0))
            continue
        best = max(boxes, key=lambda b: b.area)
        x1, y1 = max(0, int(round(best.x1))), max(0, int(round(best.y1)))
        x2, y2 = min(fw, int(round(best.x2))), min(fh, int(round(best.y2)))
        if x2 <= x1 or y2 <= y1:
            coords.append((0, 0, 0, 0))
            continue
        pad = int((x2 - x1) * 0.25)
        coords.append(
            (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(fw, x2 + pad),
                min(fh, y2 + pad),
            )
        )
    return coords


def run_acceptance(
    frames: list[np.ndarray],
    *,
    legacy_landmarks,
    self_landmarks,
    legacy_detect=None,
    self_detect=None,
    align_size: int = 768,
) -> dict:
    """Compare the two backends on the same frames; return the M4 report.

    ``*_landmarks(frame) -> (5, 2) pixel points | None``; ``*_detect(frame)``
    may be None to skip the mask-coords comparison.
    """
    from face_align import get_template_5

    dst_pts = get_template_5(align_size)
    # Eye/mouth region boxes in template space — the phase-correlation
    # windows for the ≤2 px deviation gate.
    region_boxes = fa.eye_mouth_boxes(dst_pts, align_size, align_size)
    devs: list[float] = []
    ssims: list[float] = []
    t_legacy = 0.0
    t_self = 0.0
    aligned_frames = 0

    for frame in frames:
        t0 = time.perf_counter()
        pts_a = legacy_landmarks(frame)
        t1 = time.perf_counter()
        pts_b = self_landmarks(frame)
        t2 = time.perf_counter()
        t_legacy += t1 - t0
        t_self += t2 - t1

        if pts_a is None or pts_b is None:
            continue
        aligned_a = _warp_to_template(frame, pts_a, dst_pts, align_size)
        aligned_b = _warp_to_template(frame, pts_b, dst_pts, align_size)
        if aligned_a is None or aligned_b is None:
            continue
        aligned_frames += 1
        gray_a = fa.to_gray_f32(aligned_a)
        gray_b = fa.to_gray_f32(aligned_b)
        devs.append(fa.eye_mouth_dev_px(gray_a, gray_b, region_boxes))
        ssims.append(fa.ssim_gray(gray_a, gray_b))

    report: dict = {
        "frames": len(frames),
        "aligned_frames": aligned_frames,
        "align_size": align_size,
        "eye_mouth_dev_px": float(np.mean(devs)) if devs else None,
        "ssim": float(np.mean(ssims)) if ssims else None,
    }

    if legacy_detect is not None and self_detect is not None:
        coords_a = _mask_coords_from_boxes(
            [legacy_detect(f) for f in frames], frames
        )
        coords_b = _mask_coords_from_boxes(
            [self_detect(f) for f in frames], frames
        )
        report["mask_coords_iou"] = fa.average_mask_iou(coords_a, coords_b)

    if len(frames) > 0 and t_legacy > 0:
        n = len(frames)
        report["legacy_ms_per_frame"] = round(t_legacy / n * 1000.0, 6)
        report["self_ms_per_frame"] = round(t_self / n * 1000.0, 6)
        report["speed_ratio"] = round(fa.speed_ratio(t_legacy, t_self), 3)

    report["gates"] = fa.evaluate_gates(report)
    report["pass"] = fa.overall_pass(report["gates"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="reference video/image/dir")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--align-size", type=int, default=768)
    ap.add_argument("--det-ckpt", default=fb.DEFAULT_DET_CKPT)
    ap.add_argument("--lm-ckpt", default=fb.DEFAULT_LM_CKPT)
    ap.add_argument("--yunet-model", default=fb.DEFAULT_YUNET_MODEL)
    ap.add_argument("--mp-model", default=fb.DEFAULT_MEDIAPIPE_MODEL)
    ap.add_argument("--latents-a", default=None, help="latents.pt from legacy backend")
    ap.add_argument("--latents-b", default=None, help="latents.pt from self backend")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()

    frames = load_frames(args.input, args.max_frames)
    print(f"[accept] loaded {len(frames)} frames from {args.input}")

    def legacy_landmarks(frame: np.ndarray) -> np.ndarray | None:
        h, w = frame.shape[:2]
        pts = fb.landmarks5(frame, backend="mediapipe", mp_model_path=args.mp_model)
        return None if pts is None else pts * np.array([w, h], np.float32)

    def self_landmarks(frame: np.ndarray) -> np.ndarray | None:
        h, w = frame.shape[:2]
        pts = fb.landmarks5(
            frame,
            backend="self",
            det_ckpt_path=args.det_ckpt,
            lm_ckpt_path=args.lm_ckpt,
        )
        return None if pts is None else pts * np.array([w, h], np.float32)

    def legacy_detect(frame: np.ndarray) -> list[fb.FaceBox]:
        return fb.detect_faces(frame, backend="yunet", yunet_model_path=args.yunet_model)

    def self_detect(frame: np.ndarray) -> list[fb.FaceBox]:
        return fb.detect_faces(frame, backend="self", det_ckpt_path=args.det_ckpt)

    try:
        report = run_acceptance(
            frames,
            legacy_landmarks=legacy_landmarks,
            self_landmarks=self_landmarks,
            legacy_detect=legacy_detect,
            self_detect=self_detect,
            align_size=args.align_size,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[accept] backend unavailable: {exc}")
        return 2

    if args.latents_a and args.latents_b:
        try:
            report["latents_cosine"] = fa.latents_cosine(args.latents_a, args.latents_b)
            report["gates"] = fa.evaluate_gates(report)
            report["pass"] = fa.overall_pass(report["gates"])
        except (ImportError, ValueError) as exc:
            print(f"[accept] latents comparison skipped: {exc}")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[accept] report → {out}")

    print(
        f"[accept] verdict: {'PASS' if report['pass'] else 'FAIL'} "
        f"({sum(1 for s in report['gates'].values() if s == 'pass')}/"
        f"{len(report['gates'])} gates passed)"
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""5-point face landmark alignment utilities.

Used by prepare_avatar.py to stabilize face geometry before MuseTalk
preprocessing (significantly improves lip-sync quality).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _ensure_mediapipe():
    """Import mediapipe lazily so the module can be imported without it."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for face alignment; "
            "install with: pip install mediapipe"
        ) from exc
    return mp


def get_landmarks_5(face_landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks to 5-point format (outer eyes, nose, mouth corners)."""
    idxs = [33, 263, 1, 61, 291]  # left eye, right eye, nose, mouth left, mouth right
    pts = []
    for idx in idxs:
        pt = face_landmarks[idx]
        pts.append([pt.x, pt.y])
    return np.array(pts, dtype=np.float32)


def get_template_5(image_size: int) -> np.ndarray:
    """Standard 5-point template for an aligned face on a square canvas."""
    left_eye = (0.34 * image_size, 0.36 * image_size)
    right_eye = (0.66 * image_size, 0.36 * image_size)
    nose = (0.50 * image_size, 0.53 * image_size)
    mouth_left = (0.38 * image_size, 0.73 * image_size)
    mouth_right = (0.62 * image_size, 0.73 * image_size)
    return np.array(
        [left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float32
    )


def mask_box_from_points5(
    pts5_px: np.ndarray, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Derive the mask crop box from 5-point landmarks (pixel coords).

    Replaces the det-box+25%-pad derivation (R1 M4 strategy): a landmark-
    derived region depends only on the points, so any two backends whose
    points agree produce near-identical mask boxes — the mask IoU gate no
    longer hinges on detection-box agreement.

    Order-agnostic by construction: eye midpoint / mouth midpoint / the
    interpupillary distance are all symmetric, so detector-specific point
    orderings (YuNet regresses right-eye-first) need no remapping.

    Extents are calibrated to the previous ~1.5x padded face box on typical
    front-face proportions: half-width 1.35d, half-height 1.30d where d is
    the outer-corner eye distance.
    """
    pts = np.asarray(pts5_px, dtype=np.float64)
    if pts.shape != (5, 2):
        raise ValueError(f"expected (5, 2) pixel points, got {pts.shape}")
    eye_mid = (pts[0] + pts[1]) / 2.0
    mouth_mid = (pts[3] + pts[4]) / 2.0
    cx = (eye_mid[0] + mouth_mid[0]) / 2.0
    cy = (eye_mid[1] + mouth_mid[1]) / 2.0
    d = float(np.linalg.norm(pts[1] - pts[0]))
    if d <= 0:
        return (0, 0, 0, 0)
    half_w = 1.35 * d
    half_h = 1.30 * d
    x1 = max(0, int(round(cx - half_w)))
    y1 = max(0, int(round(cy - half_h)))
    x2 = min(frame_w, int(round(cx + half_w)))
    y2 = min(frame_h, int(round(cy + half_h)))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)


def align_frame(
    frame: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, output_size: int
) -> np.ndarray:
    """Estimate similarity transform and warp frame."""
    M = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)[0]
    if M is None:
        raise RuntimeError("failed to estimate affine transform")
    return cv2.warpAffine(
        frame,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _create_landmarker(model_path: Path):
    """Create a MediaPipe FaceLandmarker for VIDEO mode."""
    _ensure_mediapipe()  # raises a clear error when mediapipe is missing
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode as RunningMode,
    )
    from mediapipe.tasks.python.vision.face_landmarker import (
        FaceLandmarker,
        FaceLandmarkerOptions,
    )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def _resolve_model_path(model_path: str | None = None) -> Path:
    """Resolve MediaPipe face landmarker model path.

    MediaPipe C++ backend cannot open paths with non-ASCII characters on
    Windows, so we copy the model to a pure-ASCII temp path.
    """
    src = Path("models/mediapipe/face_landmarker.task") if model_path is None else Path(model_path)
    if not src.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {src}")

    ascii_tmp = Path(tempfile.gettempdir()) / "face_landmarker.task"
    shutil.copy2(src, ascii_tmp)
    return ascii_tmp


def _import_face_backend():
    """Lazy import of the face_backend factory (liveavatar package)."""
    try:
        from liveavatar import face_backend
    except ImportError as exc:
        raise ImportError(
            "the face-backend switch requires the liveavatar package; "
            "install with `pip install -e .`"
        ) from exc
    return face_backend


def _resolve_landmark_backend(arg: str | None) -> str:
    fb = _import_face_backend()
    return fb.resolve_backend(
        arg,
        fb.LANDMARK_BACKEND_ENV,
        fb.DEFAULT_LANDMARK_BACKEND,
        fb.LANDMARK_BACKEND_CHOICES,
    )


def _self_landmarks5_px(
    frame: np.ndarray, det_ckpt: str | None, lm_ckpt: str | None
) -> np.ndarray | None:
    """Self-backend 5-point landmarks in *pixel* coords, or None if no face."""
    fb = _import_face_backend()
    pts = fb.landmarks5(
        frame,
        backend="self",
        det_ckpt_path=det_ckpt or fb.DEFAULT_DET_CKPT,
        lm_ckpt_path=lm_ckpt or fb.DEFAULT_LM_CKPT,
    )
    if pts is None:
        return None
    h, w = frame.shape[:2]
    return pts * np.array([w, h], dtype=np.float32)


def align_video(
    input_path: str,
    output_path: str,
    *,
    output_size: int = 768,
    max_frames: int | None = None,
    model_path: str | None = None,
    landmark_backend: str | None = None,
    det_ckpt: str | None = None,
    lm_ckpt: str | None = None,
) -> int:
    """Align faces in a video; return number of frames processed.

    ``landmark_backend``: "mediapipe" (default, VIDEO mode with tracking) or
    "self" (R1 self-developed detector + landmark student); may also come
    from the ``LANDMARK_BACKEND`` env var.
    """
    backend = _resolve_landmark_backend(landmark_backend)
    use_mp = backend == "mediapipe"
    if use_mp:
        mp = _ensure_mediapipe()
        landmarker = _create_landmarker(_resolve_model_path(model_path))
    else:
        mp = None
        landmarker = None

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_size, output_size))

    dst_pts = get_template_5(output_size)
    frame_count = 0
    failed = 0
    timestamp_ms = 0
    frame_duration_ms = 1000.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_count >= max_frames:
            break

        h, w = frame.shape[:2]
        if use_mp:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
            src_pts = None
            if result.face_landmarks:
                src_pts = get_landmarks_5(result.face_landmarks[0])
                src_pts[:, 0] *= w
                src_pts[:, 1] *= h
        else:
            src_pts = _self_landmarks5_px(frame, det_ckpt, lm_ckpt)

        if src_pts is not None:
            try:
                aligned = align_frame(frame, src_pts, dst_pts, output_size)
                writer.write(aligned)
            except Exception as e:
                print(f"\n[warn] frame {frame_count} alignment failed: {e}")
                failed += 1
                writer.write(np.zeros((output_size, output_size, 3), dtype=np.uint8))
        else:
            failed += 1
            writer.write(np.zeros((output_size, output_size, 3), dtype=np.uint8))

        frame_count += 1
        timestamp_ms += frame_duration_ms
        print(f"\r[align] {frame_count}/{total_frames} frames (failed: {failed})", end="")
        sys.stdout.flush()

    cap.release()
    writer.release()
    landmarker.close()
    print()
    if failed:
        print(f"[align] warning: {failed}/{frame_count} frames failed face detection")
    return frame_count


def align_image(
    input_path: str,
    output_path: str,
    *,
    output_size: int = 768,
    model_path: str | None = None,
    landmark_backend: str | None = None,
    det_ckpt: str | None = None,
    lm_ckpt: str | None = None,
) -> bool:
    """Align face in a single image; return True if a face was found."""
    backend = _resolve_landmark_backend(landmark_backend)

    frame = cv2.imread(input_path)
    if frame is None:
        # Try unicode-safe read.
        data = np.fromfile(input_path, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot read image: {input_path}")

    src_pts: np.ndarray | None
    if backend == "mediapipe":
        mp = _ensure_mediapipe()
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode as RunningMode,
        )
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_resolve_model_path(model_path))),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = FaceLandmarker.create_from_options(options)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        src_pts = None
        if result.face_landmarks:
            src_pts = get_landmarks_5(result.face_landmarks[0])
            src_pts[:, 0] *= w
            src_pts[:, 1] *= h
        landmarker.close()
    else:
        src_pts = _self_landmarks5_px(frame, det_ckpt, lm_ckpt)

    if src_pts is None:
        return False
    try:
        aligned = align_frame(frame, src_pts, get_template_5(output_size), output_size)
        cv2.imwrite(output_path, aligned)
        return True
    except Exception as e:
        print(f"[warn] image alignment failed: {e}")
        return False

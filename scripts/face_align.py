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


def align_video(
    input_path: str,
    output_path: str,
    *,
    output_size: int = 768,
    max_frames: int | None = None,
    model_path: str | None = None,
) -> int:
    """Align faces in a video; return number of frames processed."""
    mp = _ensure_mediapipe()
    landmarker = _create_landmarker(_resolve_model_path(model_path))

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if result.face_landmarks:
            src_pts = get_landmarks_5(result.face_landmarks[0])
            src_pts[:, 0] *= w
            src_pts[:, 1] *= h
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
) -> bool:
    """Align face in a single image; return True if a face was found."""
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

    frame = cv2.imread(input_path)
    if frame is None:
        # Try unicode-safe read.
        data = np.fromfile(input_path, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot read image: {input_path}")

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    ok = False
    if result.face_landmarks:
        src_pts = get_landmarks_5(result.face_landmarks[0])
        src_pts[:, 0] *= w
        src_pts[:, 1] *= h
        try:
            aligned = align_frame(frame, src_pts, get_template_5(output_size), output_size)
            cv2.imwrite(output_path, aligned)
            ok = True
        except Exception as e:
            print(f"[warn] image alignment failed: {e}")
    landmarker.close()
    return ok

"""Face backend factory (R1 M3): unified detection + 5-point landmark access.

Switches (priority: explicit argument > environment variable > default):

- detection:  ``FACE_BACKEND``      = ``yunet`` (default) | ``self``
- landmarks:  ``LANDMARK_BACKEND``  = ``mediapipe`` (default) | ``self``

The legacy backends (OpenCV YuNet / MediaPipe FaceLandmarker) stay the
default until the self-developed weights pass the M4 downstream-consistency
acceptance; they will be removed after a two-minor-version transition (same
convention as the codec switch). Neither is a runtime dependency:
- YuNet is a single ``.onnx`` resource file on top of the retained OpenCV;
- MediaPipe is imported lazily and is only needed as the *training-time
  teacher* (pseudo-labeling own material) or with the optional ``teacher``
  extra during the transition — ``pip install -e ".[teacher]"``.

The ``self`` backend is a pure ``torch + numpy`` re-implementation:
``face_self.TinyFaceDetector`` for detection and
``face_landmarks.LandmarkNet5Self`` for 5-point landmarks, loaded from the
checkpoints written by ``scripts/train_face_det.py`` /
``scripts/train_face_landmarks.py``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from liveavatar.face_landmarks import LandmarkNet5Self
    from liveavatar.face_self import TinyFaceDetector

FACE_BACKEND_ENV = "FACE_BACKEND"
LANDMARK_BACKEND_ENV = "LANDMARK_BACKEND"

FACE_BACKEND_CHOICES = ("yunet", "self")
LANDMARK_BACKEND_CHOICES = ("mediapipe", "self")

DEFAULT_FACE_BACKEND = "yunet"
DEFAULT_LANDMARK_BACKEND = "mediapipe"

# Checkpoints written by the R1 training scripts.
DEFAULT_DET_CKPT = "weights/self/facedet_256.pt"
DEFAULT_LM_CKPT = "weights/self/landmarks5_128.pt"

DEFAULT_YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
DEFAULT_MEDIAPIPE_MODEL = "models/mediapipe/face_landmarker.task"

try:  # torch is optional (light CI env); the self backend requires it
    import torch  # noqa: F401

    from liveavatar import face_landmarks as _face_landmarks
    from liveavatar import face_self as _face_self

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    _HAVE_TORCH = False


def resolve_backend(
    arg: str | None,
    env_var: str,
    default: str,
    choices: tuple[str, ...],
) -> str:
    """Explicit argument > env var > default; validate against choices."""
    name = arg or os.environ.get(env_var) or default
    if name not in choices:
        raise ValueError(
            f"unknown backend {name!r} for {env_var}; expected one of {list(choices)}"
        )
    return name


@dataclass
class FaceBox:
    """Detected face in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_xywh_norm(self, image_w: int, image_h: int) -> list[float]:
        """Normalized (x, y, w, h) as consumed by ``FaceCropTransform``."""
        return [
            self.x1 / image_w,
            self.y1 / image_h,
            (self.x2 - self.x1) / image_w,
            (self.y2 - self.y1) / image_h,
        ]


# ---------------------------------------------------------------------------
# Backend caches (per process; call reset_backend_caches() in tests)
# ---------------------------------------------------------------------------

_YUNET_CACHE: dict[tuple[str, int, int], Any] = {}
_MP_LANDMARKER_CACHE: dict[str, Any] = {}
_TORCH_CACHE: dict[tuple[str, str], tuple[Any, int]] = {}


def reset_backend_caches() -> None:
    """Drop all cached backend instances (used by tests)."""
    _YUNET_CACHE.clear()
    _MP_LANDMARKER_CACHE.clear()
    _TORCH_CACHE.clear()


# ---------------------------------------------------------------------------
# Checkpoint loading (self backend)
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    if not _HAVE_TORCH:  # pragma: no cover - exercised only without torch
        raise RuntimeError(
            "the 'self' face backend requires torch; install with "
            "`pip install torch` (CPU wheel is sufficient for inference)"
        )


def _load_torch_checkpoint(path: Path, kind: str) -> dict:
    import torch

    # weights_only=True: checkpoints are written by our own R1 training
    # scripts as plain dicts (state_dict tensors + int/str metadata), which
    # is exactly the payload class weights_only permits. This blocks
    # arbitrary code execution from tampered .pt files.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            f"malformed {kind} checkpoint {path}: expected a dict with "
            "'model' / 'width' / 'input_size' keys (as written by the "
            "R1 training scripts)"
        )
    return ckpt


def load_det_model(ckpt_path: str | Path) -> tuple[TinyFaceDetector, int]:
    """Load a TinyFaceDetector checkpoint → (model, input_size)."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"self face-detector checkpoint not found: {path}; train one with "
            "`python scripts/train_face_det.py` or download the released weights"
        )
    key = ("det", str(path))
    if key in _TORCH_CACHE:
        return _TORCH_CACHE[key]
    _require_torch()
    ckpt = _load_torch_checkpoint(path, "face-detector")
    model: TinyFaceDetector = _face_self.TinyFaceDetector(width=int(ckpt.get("width", 48)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    result = (model, int(ckpt.get("input_size", 256)))
    _TORCH_CACHE[key] = result
    return result


def load_lm_model(ckpt_path: str | Path) -> tuple[LandmarkNet5Self, int]:
    """Load a LandmarkNet5Self checkpoint → (model, input_size)."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"self landmark checkpoint not found: {path}; train one with "
            "`python scripts/train_face_landmarks.py` or download the "
            "released weights"
        )
    key = ("lm", str(path))
    if key in _TORCH_CACHE:
        return _TORCH_CACHE[key]
    _require_torch()
    ckpt = _load_torch_checkpoint(path, "landmark")
    model: LandmarkNet5Self = _face_landmarks.LandmarkNet5Self(width=int(ckpt.get("width", 32)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    result = (model, int(ckpt.get("input_size", 128)))
    _TORCH_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Unified detection
# ---------------------------------------------------------------------------


def detect_faces(
    image_bgr: np.ndarray,
    backend: str | None = None,
    *,
    yunet_model_path: str = DEFAULT_YUNET_MODEL,
    det_ckpt_path: str = DEFAULT_DET_CKPT,
    conf_threshold: float = 0.5,
) -> list[FaceBox]:
    """Detect faces in one BGR image; return pixel-space FaceBox list.

    ``backend`` is resolved via ``FACE_BACKEND`` (env) with fallback to
    ``yunet``.
    """
    name = resolve_backend(backend, FACE_BACKEND_ENV, DEFAULT_FACE_BACKEND, FACE_BACKEND_CHOICES)
    if name == "self":
        return _detect_faces_self(image_bgr, det_ckpt_path, conf_threshold)
    return _detect_faces_yunet(image_bgr, yunet_model_path, conf_threshold)


def _detect_faces_yunet(
    image_bgr: np.ndarray, model_path: str, conf_threshold: float
) -> list[FaceBox]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - cv2 is a dev extra
        raise RuntimeError(
            "the 'yunet' face backend requires OpenCV; install with "
            "`pip install opencv-python`"
        ) from exc

    h, w = image_bgr.shape[:2]
    key = (str(model_path), w, h)
    detector = _YUNET_CACHE.get(key)
    if detector is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}; fetch it with "
                "`python scripts/download_models.py`"
            )
        detector = cv2.FaceDetectorYN_create(  # type: ignore[attr-defined]
            model_path,
            "",
            (w, h),
            conf_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        _YUNET_CACHE[key] = detector
    else:
        detector.setInputSize((w, h))

    ok, faces = detector.detect(image_bgr)
    if not ok or faces is None or len(faces) == 0:
        return []
    return [
        FaceBox(
            x1=float(f[0]),
            y1=float(f[1]),
            x2=float(f[0] + f[2]),
            y2=float(f[1] + f[3]),
            score=float(f[-1]),
        )
        for f in faces
    ]


def _detect_faces_self(
    image_bgr: np.ndarray, ckpt_path: str, conf_threshold: float
) -> list[FaceBox]:
    model, input_size = load_det_model(ckpt_path)
    rgb = np.ascontiguousarray(image_bgr[..., ::-1])
    hits = _face_self.detect(model, rgb, conf_threshold=conf_threshold, input_size=input_size)
    return [
        FaceBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], score=score) for b, score in hits
    ]


# ---------------------------------------------------------------------------
# Unified 5-point landmarks
# ---------------------------------------------------------------------------

# MediaPipe landmark indices for the 5-point format (outer eyes, nose tip,
# mouth corners) — same mapping as scripts/face_align.get_landmarks_5.
_MP_5PT_IDXS = (33, 263, 1, 61, 291)


def landmarks5(
    image_bgr: np.ndarray,
    backend: str | None = None,
    *,
    mp_model_path: str = DEFAULT_MEDIAPIPE_MODEL,
    det_ckpt_path: str = DEFAULT_DET_CKPT,
    lm_ckpt_path: str = DEFAULT_LM_CKPT,
    det_conf_threshold: float = 0.5,
) -> np.ndarray | None:
    """5-point landmarks for the primary face in one BGR image.

    Returns (5, 2) coordinates normalized to the full image (same convention
    as the MediaPipe teacher / the R1 manifest), or None when no face is
    found. ``backend`` is resolved via ``LANDMARK_BACKEND`` (env) with
    fallback to ``mediapipe``.
    """
    name = resolve_backend(
        backend, LANDMARK_BACKEND_ENV, DEFAULT_LANDMARK_BACKEND, LANDMARK_BACKEND_CHOICES
    )
    if name == "self":
        return _landmarks5_self(image_bgr, det_ckpt_path, lm_ckpt_path, det_conf_threshold)
    return _landmarks5_mediapipe(image_bgr, mp_model_path)


def _landmarks5_mediapipe(image_bgr: np.ndarray, mp_model_path: str) -> np.ndarray | None:
    landmarker = _get_mp_landmarker(mp_model_path)
    h, w = image_bgr.shape[:2]
    rgb = np.ascontiguousarray(image_bgr[..., ::-1])

    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    return np.array(
        [[lm[i].x, lm[i].y] for i in _MP_5PT_IDXS], dtype=np.float32
    )  # already normalized


def _get_mp_landmarker(mp_model_path: str):
    """IMAGE-mode MediaPipe FaceLandmarker (lazy import, cached, ASCII path)."""
    key = str(mp_model_path)
    if key in _MP_LANDMARKER_CACHE:
        return _MP_LANDMARKER_CACHE[key]
    try:
        import mediapipe  # noqa: F401  (availability probe only)
    except ImportError as exc:
        raise RuntimeError(
            "the 'mediapipe' landmark backend requires mediapipe; install "
            "with `pip install mediapipe`"
        ) from exc

    src = Path(mp_model_path)
    if not src.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {src}; fetch it with "
            "`python scripts/download_models.py`"
        )
    # MediaPipe C++ cannot open non-ASCII Windows paths — copy to temp.
    ascii_tmp = Path(tempfile.gettempdir()) / "face_landmarker.task"
    shutil.copy2(src, ascii_tmp)

    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode as RunningMode,
    )
    from mediapipe.tasks.python.vision.face_landmarker import (
        FaceLandmarker,
        FaceLandmarkerOptions,
    )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ascii_tmp)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    _MP_LANDMARKER_CACHE[key] = landmarker
    return landmarker


def _landmarks5_self(
    image_bgr: np.ndarray,
    det_ckpt_path: str,
    lm_ckpt_path: str,
    det_conf_threshold: float,
) -> np.ndarray | None:
    _require_torch()
    det_model, det_size = load_det_model(det_ckpt_path)
    lm_model, lm_size = load_lm_model(lm_ckpt_path)

    rgb = np.ascontiguousarray(image_bgr[..., ::-1])
    hits = _face_self.detect(
        det_model, rgb, conf_threshold=det_conf_threshold, input_size=det_size
    )
    if not hits:
        return None

    h, w = image_bgr.shape[:2]
    best = max(
        hits,
        key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
    )
    x1, y1, x2, y2 = best[0]
    box_xywh_norm = [
        x1 / w,
        y1 / h,
        (x2 - x1) / w,
        (y2 - y1) / h,
    ]
    return _face_landmarks.landmarks5(
        lm_model, rgb, box_xywh_norm, input_size=lm_size
    )

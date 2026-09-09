# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Spike: can mediapipe FaceLandmarker measure openness on proxy faces?

Renders the mouth proxy at several openness levels and checks the
mediapipe-teacher extraction chain (inner-lip gap / face height). This
validates the P-B measurement path before the recorder is built.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mouth_proxy import render_face

_MODEL = os.path.join("models", "mediapipe", "face_landmarker.task")

# mediapipe dense indices: inner lips 13 (upper) / 14 (lower), corners
# 61 / 291; face-height normalization 10 (forehead) / 152 (chin).
_UP, _LOW, _C0, _C1, _TOP, _CHIN = 13, 14, 61, 291, 10, 152


def measure(bgr: np.ndarray) -> dict | None:
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    lm = _landmarker.detect(img)
    if not lm.face_landmarks:
        return None
    pts = lm.face_landmarks[0]
    p = lambda i: np.array([pts[i].x, pts[i].y])  # noqa: E731
    gap = float(np.linalg.norm(p(_UP) - p(_LOW)))
    width = float(np.linalg.norm(p(_C0) - p(_C1)))
    face_h = float(np.linalg.norm(p(_TOP) - p(_CHIN)))
    return {"gap": round(gap, 4), "width": round(width, 4),
            "face_h": round(face_h, 4),
            "openness_norm": round(gap / face_h, 4),
            "aspect": round(gap / width, 4) if width else None}


_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_MODEL), num_faces=1
    )
)

def _pixel_openness(bgr: np.ndarray, m: dict | None) -> float | None:
    """Mouth-cavity dark-pixel AREA fraction in the mouth ROI."""
    if m is None:
        return None
    lm_pts = _landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ).face_landmarks[0]
    mouth = np.array([lm_pts[13].x * bgr.shape[1], lm_pts[13].y * bgr.shape[0]])
    x0, x1 = int(mouth[0] - 60), int(mouth[0] + 60)
    y0, y1 = int(mouth[1] - 45), int(mouth[1] + 45)
    roi = bgr[y0:y1, x0:x1]
    v = roi.max(axis=2)
    return round(float((v < 100).mean()), 4)


if __name__ == "__main__":
    out_dir = os.path.join("data", "interruption_eval", "spike")
    os.makedirs(out_dir, exist_ok=True)
    print("openness | gap | width | face_h | gap/face_h | gap/width | pixel_h/face_h")
    for op in (0.02, 0.25, 0.5, 0.75, 0.95):
        img = render_face(op, t_s=1.0)
        path = os.path.join(out_dir, f"face_{int(op * 100):02d}.png")
        cv2.imwrite(path, img)
        m = measure(img)
        px = _pixel_openness(img, m)
        print(f"{op:5.2f} | {m} | {px}")

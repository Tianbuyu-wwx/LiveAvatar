# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Audio-driven mouth-proxy avatar worker (CPU, eval instrument).

Renders a stylized 256x256 talking face whose mouth openness tracks the
ENERGY of the TTS PCM it receives through the real star pipeline
(``_infer_batch(pcm)`` — the same fan-out path MuseTalk uses). When the
pipeline hard-cuts the stream (epoch advance), the mouth freezes at its
last openness and resumes from the new utterance's energy — the visual
artifact under study is therefore genuine pipeline behavior, not a
simulated overlay.

The face is deliberately simple: a flat-shaded cartoon head. The P-B
analyzer measures openness from PIXELS (mediapipe teacher, pixel-metric
fallback), never from this worker's internal state, so the measurement
path is the one the paper claims.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from liveavatar.runtime.transition import CLOSED_OPENNESS, alpha_schedule, crossfade_bgr
from liveavatar.worker import AvatarAssets, AvatarWorker

_WIDTH = 256
_HEIGHT = 256
_TARGET_FPS = 25
_BATCH_SIZE = 4

# RMS (fraction of full scale, dBFS) → openness logistic.
_MID_DBFS = -26.0
_SLOPE_DB = 7.0
# Per-frame one-pole openness easing at 25 fps (~120 ms time constant).
_FRAME_ALPHA = 0.18


def rms_to_openness(pcm_s16le: bytes) -> float:
    """Map a PCM chunk's RMS (dBFS) to a mouth-openness target 0..1."""
    if not pcm_s16le:
        return 0.0
    x = np.frombuffer(pcm_s16le[: len(pcm_s16le) // 2 * 2], dtype="<i2")
    if x.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms <= 0.0:
        return 0.0
    dbfs = 20.0 * math.log10(rms / 32767.0)
    return 1.0 / (1.0 + math.exp(-(dbfs - _MID_DBFS) / _SLOPE_DB))


def render_face(openness: float, t_s: float, w: int = _WIDTH, h: int = _HEIGHT) -> np.ndarray:
    """Render one BGR frame: stylized head with an openness-driven mouth.

    ``t_s`` drives a subtle head bob so normal playback has baseline
    inter-frame motion (transition-smoothness reference).
    """
    img = np.empty((h, w, 3), np.uint8)
    # Background: vertical gradient (region-encoder friendly: static).
    bg_top = np.array([70, 60, 50], np.float64)
    bg_bot = np.array([40, 34, 30], np.float64)
    ramp = np.linspace(0.0, 1.0, h)[:, None, None]
    img[:] = (bg_top * (1 - ramp) + bg_bot * ramp).astype(np.uint8)

    dx = 1.5 * math.sin(2 * math.pi * 0.31 * t_s)
    dy = 2.0 * math.sin(2 * math.pi * 0.50 * t_s)
    cx, cy = w / 2 + dx, h / 2 + 6 + dy

    def ellipse(center, axes, color, thickness=-1):
        cv2.ellipse(
            img, center, axes, 0, 0, 360, color, thickness, lineType=cv2.LINE_AA
        )

    # Hair (behind head), head, ears.
    ellipse((int(cx), int(cy - 26)), (84, 74), (60, 42, 34))
    ellipse((int(cx), int(cy)), (78, 96), (150, 190, 226))  # skin BGR
    ellipse((int(cx - 78), int(cy + 10)), (8, 14), (150, 190, 226))
    ellipse((int(cx + 78), int(cy + 10)), (8, 14), (150, 190, 226))
    # Fringe.
    ellipse((int(cx), int(cy - 62)), (80, 34), (60, 42, 34))

    # Eyes (whites + iris), pupils look slightly with the bob.
    for ex in (-32, 32):
        ellipse((int(cx + ex), int(cy - 22)), (16, 9), (255, 255, 255))
        ellipse((int(cx + ex + dx * 0.5), int(cy - 22)), (6, 6), (40, 40, 40))
    # Brows.
    cv2.line(
        img,
        (int(cx - 46), int(cy - 40)),
        (int(cx - 18), int(cy - 44)),
        (60, 42, 34),
        3,
        cv2.LINE_AA,
    )
    cv2.line(
        img,
        (int(cx + 18), int(cy - 44)),
        (int(cx + 46), int(cy - 40)),
        (60, 42, 34),
        3,
        cv2.LINE_AA,
    )
    # Nose.
    cv2.line(
        img,
        (int(cx), int(cy - 8)),
        (int(cx - 3), int(cy + 16)),
        (180, 160, 150),
        2,
        cv2.LINE_AA,
    )

    # Mouth: lips outline + inner cavity scaled by openness.
    my = int(cy + 44)
    inner_ry = 2 + int(24 * openness)
    inner_rx = 26 + int(7 * openness)
    ellipse((int(cx), my), (36, 14), (120, 90, 90), 2)  # lip outline
    ellipse((int(cx), my), (inner_rx, inner_ry), (70, 50, 60))  # cavity
    if openness > 0.5:  # upper teeth strip
        cv2.rectangle(
            img,
            (int(cx - inner_rx + 3), my - inner_ry),
            (int(cx + inner_rx - 3), my - inner_ry + 5),
            (235, 235, 235),
            -1,
        )
    return img


class MouthProxyWorker(AvatarWorker):
    """AvatarWorker whose mouth openness tracks received TTS PCM energy."""

    def __init__(self, avatar_id: str = "mouth_a") -> None:
        super().__init__(
            AvatarAssets(
                avatar_id=avatar_id,
                data_dir="nonexistent",
                full_imgs_dir="nonexistent",
                coords_path="nonexistent",
                latents_path="nonexistent",
                mask_dir="nonexistent",
                mask_coords_path="nonexistent",
            ),
            target_fps=_TARGET_FPS,
            width=_WIDTH,
            height=_HEIGHT,
            batch_size=_BATCH_SIZE,
        )
        self._t0 = time.perf_counter()
        self._openness = 0.05
        self._target = 0.05
        self._frame_idx = 0
        self._last_img: np.ndarray | None = None

    @property
    def last_openness(self) -> float:
        """Most recent rendered openness (debug/tests only)."""
        return self._openness

    def reset(self) -> None:
        """Reset mouth state between eval sessions (fresh idle start)."""
        self._t0 = time.perf_counter()
        self._openness = 0.05
        self._target = 0.05
        self._frame_idx = 0
        self._last_img = None

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        self._target = rms_to_openness(pcm_s16le)
        frames: list[tuple[bytes, bool]] = []
        for _ in range(self.batch_size):
            self._openness += _FRAME_ALPHA * (self._target - self._openness)
            t_s = (time.perf_counter() - self._t0) + self._frame_idx / _TARGET_FPS
            img = render_face(self._openness, t_s)
            self._last_img = img
            self._frame_idx += 1
            frames.append((img.tobytes(), True))
        return frames

    def render_transition_frames(
        self, num_frames: int = 3, mode: str = "openness"
    ) -> list[tuple[bytes, bool]]:
        """P-D eval backend: ease the mouth shut over ``num_frames``.

        Two renderings of the same α schedule
        (:func:`liveavatar.runtime.transition.alpha_schedule`):

        - ``mode="openness"`` (default) — the faithful CPU analog of the
          paper's latent interpolation: the MuseTalk UNet decodes
          ``z_i = (1-α_i)·z_now + α_i·z_closed`` into mouth GEOMETRY
          morphing shut, so the proxy re-renders the face at the
          interpolated openness ``op_i = (1-α_i)·op_now + α_i·closed``
          (head-bob phase keeps advancing). A dark-pixel openness
          extractor reads this back near-linearly, matching how the
          real latent path would behave.
        - ``mode="crossfade"`` — the paper's ghosting fallback: a pixel
          space dissolve of the last frame toward the closed face. The
          dark-cavity extractor reads the dissolve super-linearly (the
          first step swallows most of the range), so this mode is for
          A/B inspection, not for the primary eval.

        The internal state lands on the neutral openness so the new
        utterance's first batch continues from the closed mouth.
        ``False`` is_speaking: these frames are a visual bridge, not
        audio-driven.
        """
        if self._last_img is None:
            return []  # nothing rendered yet — hard cut
        alpha = alpha_schedule(num_frames)
        if alpha.size == 0:
            return []
        t0 = (time.perf_counter() - self._t0) + self._frame_idx / _TARGET_FPS
        if mode == "crossfade":
            closed = render_face(CLOSED_OPENNESS, t0)
            imgs = crossfade_bgr(self._last_img, closed, alpha)
        elif mode == "openness":
            op_now = self._openness
            imgs = [
                render_face((1.0 - a) * op_now + a * CLOSED_OPENNESS, t0 + i / _TARGET_FPS)
                for i, a in enumerate(alpha)
            ]
        else:
            raise ValueError(f"unknown transition mode: {mode!r}")
        self._openness = CLOSED_OPENNESS
        self._target = CLOSED_OPENNESS
        self._frame_idx += len(imgs)
        return [(img.tobytes(), False) for img in imgs]

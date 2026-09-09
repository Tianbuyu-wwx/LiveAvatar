# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Transition-frame synthesis for barge-in (P-D, paper §4.4).

A hard barge-in freezes the video on the last old-epoch frame (mouth
open mid-syllable) until the new utterance's first frame lands — the
"freeze-then-jump" artifact P-B quantified. This module synthesizes the
missing bridge: K frames (80-160 ms at 25 fps) that ease the avatar from
its current visual state to the closed-mouth neutral state, published as
the first frames of the interrupt epoch (epoch N+1). The first of them
is the epoch boundary keyframe, so old-epoch frames are invalidated and
the new utterance's first frame follows a closed, natural mouth.

Two backends share one alpha schedule:

- latent interpolation (the paper's method, MuseTalk): per-frame VAE
  latent ``z_i = (1 - α_i) · z_now + α_i · z_closed`` decoded by the
  VAE — training-free, inference only;
- pixel crossfade (CPU fallback / eval proxy): per-pixel blend of the
  last rendered frame and the neutral frame.

Pure functions over numpy arrays / floats — no asyncio, no torch
obligation — so the realtime adapter, the CPU eval proxy and the
offline tests share one implementation.
"""

from __future__ import annotations

import numpy as np

# Closed-mouth neutral openness used by the eval proxy (matches
# mouth_proxy's idle target; paper §4.4 "闭嘴中性帧").
CLOSED_OPENNESS = 0.05

# Legal transition lengths: 2-4 frames = 80-160 ms at 25 fps.
MIN_FRAMES, MAX_FRAMES = 2, 4


def clamp_frames(num_frames: int) -> int:
    """Clamp a requested transition length to the legal 2-4 frame range."""
    return int(max(0, min(MAX_FRAMES, max(0, num_frames))))


def alpha_schedule(num_frames: int) -> np.ndarray:
    """Uniform 0→1 blend schedule over ``num_frames`` frames.

    ``α_i = i / n`` for i = 1..n, so the LAST transition frame is the
    fully closed target (α = 1) — the mouth completes its close within
    the transition and the new utterance starts from the neutral state.
    Returns an empty array for ``num_frames <= 0`` (transition disabled).
    """
    n = clamp_frames(num_frames)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    return np.arange(1, n + 1, dtype=np.float64) / float(n)


def interpolate_openness(openness_now: float, alpha: np.ndarray) -> np.ndarray:
    """Per-frame openness easing current → closed under ``alpha``."""
    return (1.0 - alpha) * float(openness_now) + alpha * CLOSED_OPENNESS


def crossfade_bgr(
    frame_now: np.ndarray, frame_closed: np.ndarray, alpha: np.ndarray
) -> list[np.ndarray]:
    """Pixel-space fallback: per-alpha blend of two BGR frames.

    Both frames must share shape and uint8 dtype. Returns ``len(alpha)``
    frames; ``alpha = 1`` yields exactly ``frame_closed``.
    """
    if frame_now.shape != frame_closed.shape:
        raise ValueError(
            f"frame shape mismatch: {frame_now.shape} vs {frame_closed.shape}"
        )
    now = frame_now.astype(np.float64)
    closed = frame_closed.astype(np.float64)
    out: list[np.ndarray] = []
    for a in alpha:
        blended = (1.0 - a) * now + a * closed
        out.append(np.clip(blended + 0.5, 0, 255).astype(np.uint8))
    return out


def interpolate_latent_np(
    z_now: np.ndarray, z_closed: np.ndarray, alpha: np.ndarray
) -> list[np.ndarray]:
    """Numpy reference for the latent interpolation ``z_i = (1-α)·z_now + α·z_closed``.

    The MuseTalk worker runs the same math on torch tensors (tensor
    ``lerp``); this numpy form pins the semantics for CPU tests with
    fake latents. Returns one array per alpha, in alpha order.
    """
    if z_now.shape != z_closed.shape:
        raise ValueError(
            f"latent shape mismatch: {z_now.shape} vs {z_closed.shape}"
        )
    now = z_now.astype(np.float64)
    closed = z_closed.astype(np.float64)
    return [(1.0 - a) * now + a * closed for a in alpha]

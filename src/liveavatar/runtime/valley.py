# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Energy-valley cut-point selector for barge-in (P-C, paper §4.3).

A hard barge-in cuts the pending TTS audio at the current playback
position, chopping the open syllable mid-phoneme (the half-syllable
artifact). This module picks the nearest energy valley — a syllable gap
in the already-synthesized tail — so playback can roll forward to a
natural stop: audio before the valley plays out, audio after it is
dropped (谷前音频正常播放、谷后丢弃).

Pure functions over s16le PCM bytes — no asyncio, no torch — so the
realtime worker (``RealtimeWorker.advance_epoch``) and the offline
ablation replay (``scripts/interruption_eval/valley_ablation.py``) share
one implementation.

Selection semantics:
- The scan origin is the hard-cut point (sample 0 of the pending tail,
  i.e. the current playback position).
- The pending tail is framed into ``window_ms`` RMS windows; a window
  qualifies as a valley when its energy drops ``valley_depth_db`` below
  the span's loud level (90th percentile). The cut lands at the END of
  the first qualifying window — the quiet gap is played through, the
  next onset is dropped.
- ``prior_sample`` (tail-relative sample of an EOU/VAD endpoint prior
  mapped onto the TTS timeline) is evaluated first: when its window
  qualifies, it wins with ``reason="eou_prior"``; otherwise the scan
  falls back to the first valley.
- No qualifying valley inside ``search_ms`` → ``found=False`` (hard cut
  kept; half-syllable tolerated and masked later by the video
  transition of P-D).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValleyCut:
    """Selected cut point within the pending TTS tail.

    ``cut_sample`` is tail-relative (0 = the hard-cut point = current
    playback position); ``rollback_ms`` is how much extra audio plays
    past the hard-cut point before stopping.
    """

    cut_sample: int = 0
    found: bool = False
    reason: str = "none"  # "eou_prior" | "valley" | "none"
    rollback_ms: float = 0.0


def window_rms_db(pcm: bytes, sample_rate: int, window_ms: float) -> np.ndarray:
    """Fixed-window RMS energy in dBFS (last partial window dropped)."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
    n = int(sample_rate * window_ms / 1000.0)
    if n <= 0 or len(x) < n:
        return np.empty(0, dtype=np.float64)
    n_win = len(x) // n
    frames = x[: n_win * n].reshape(n_win, n)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-6))


def find_valley_cut(
    pcm: bytes,
    *,
    sample_rate: int = 16000,
    window_ms: float = 20.0,
    search_ms: float = 240.0,
    valley_depth_db: float = 6.0,
    prior_sample: int | None = None,
) -> ValleyCut:
    """Pick the nearest energy-valley cut in the pending TTS tail.

    ``pcm`` is the pending tail (s16le mono) starting at the hard-cut
    point. See the module docstring for semantics. The default depth
    (6 dB below the span's loud level) accepts syllable-onset boundaries
    — consonant dips and pauses alike — because the perceptual goal is
    "never chop a vowel nucleus", not "only stop in silence"; raise the
    threshold toward 12 dB to demand true silence gaps.
    """
    win = int(sample_rate * window_ms / 1000.0)
    energies = window_rms_db(pcm, sample_rate, window_ms)
    n_span = int(np.ceil(search_ms / window_ms))
    energies = energies[:n_span]
    if len(energies) == 0 or win <= 0:
        return ValleyCut()

    # A valley must sit well below the span's loud level so that the
    # quiet floor of continuous speech is not mistaken for a gap.
    loud = float(np.percentile(energies, 90))
    floor = loud - valley_depth_db

    def _take(idx: int, reason: str) -> ValleyCut:
        cut_sample = (idx + 1) * win
        return ValleyCut(
            cut_sample=cut_sample,
            found=True,
            reason=reason,
            rollback_ms=cut_sample / sample_rate * 1000.0,
        )

    # EOU/VAD endpoint prior mapped onto the tail: prefer it when the
    # window it points at already qualifies as a valley.
    if prior_sample is not None and prior_sample > 0:
        j = int(prior_sample) // win
        if 0 <= j < len(energies) and energies[j] <= floor:
            return _take(j, "eou_prior")

    for i, e in enumerate(energies):
        if e <= floor:
            return _take(i, "valley")
    return ValleyCut()

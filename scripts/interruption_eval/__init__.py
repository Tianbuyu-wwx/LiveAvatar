# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-B: interruption evaluation pipeline (CPU-only, paper metrics).

Modules:
- ``audio_synth`` — deterministic pseudo-speech with ground-truth phoneme
  schedule and mouth-openness curve (the metric ground truth).
- ``mouth_proxy`` — audio-driven mouth-proxy avatar worker (CPU): the
  rendered mouth openness tracks the TTS PCM energy through the real
  star pipeline, so hard-cut artifacts are genuine.
- ``record`` — media recorder: per-session JPEG frame sequence + mic/TTS
  WAV + meta.json (interrupt point, epoch arrivals, five-layer timeline).
- ``metrics`` — pure metric library: V-APT, viseme buckets, phoneme-
  viseme mismatch rate, half-syllable artifact rate, transition smoothness.
- ``analyze`` — offline analyzer: mediapipe-teacher mouth openness from
  recorded frames (+ pixel fallback), metric computation per sample.
- ``report`` — JSON/Markdown aggregation, blind-eval package, Spearman
  validation against human ratings.

Paper mapping: Section 4 (metric definitions) + Section 5.2 (validity:
Spearman rho >= 0.6 between human blind ratings and V-APT / mismatch).
"""

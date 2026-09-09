# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Deterministic pseudo-speech synthesizer with ground-truth phonetics.

Generates 16 kHz s16le mono PCM shaped like syllabic speech (consonant
onset + vowel nucleus + word pauses) together with:

- a phoneme event schedule ``[(phoneme, kind, start_s, end_s, openness)]``
  — the ground truth for the half-syllable artifact rate and the audio
  side of the phoneme-viseme mismatch rate;
- a per-sample mouth-openness curve (one-pole smoothed target) — the
  reference trajectory for V-APT.

Everything is seeded and deterministic: an interrupted session and its
no-interrupt reference play the SAME audio when they share a seed, so
the V-APT reference curve is exact.

Formant frequencies are rough vowel targets (F1/F2 in Hz); the render is
a lightweight formant + noise model, NOT intelligible speech — enough
for energy/phoneme-timing realism on CPU without TTS weights.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

_SR = 16000

# Vowel targets: (F1, F2) Hz and mouth-openness ground truth.
VOWELS: dict[str, tuple[int, int, float]] = {
    "a": (730, 1090, 0.90),
    "e": (530, 1840, 0.75),
    "o": (570, 840, 0.60),
    "i": (270, 2290, 0.35),
    "u": (300, 870, 0.22),
}

# Consonants: (openness, render kind).
CONSONANTS: dict[str, tuple[float, str]] = {
    "m": (0.02, "nasal"),
    "n": (0.05, "nasal"),
    "p": (0.02, "plosive"),
    "t": (0.08, "plosive"),
    "k": (0.10, "plosive"),
    "s": (0.12, "fricative"),
}

PAUSE_OPENNESS = 0.05
# One-pole smoothing time constant for the openness trajectory (seconds).
_OPENNESS_TAU_S = 0.025


@dataclass(frozen=True)
class PhonemeEvent:
    """One scheduled phoneme with its ground-truth mouth openness."""

    phoneme: str  # vowel key, consonant key, or "_" for pause
    kind: str  # "vowel" | "consonant" | "pause"
    start_s: float
    end_s: float
    openness: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Utterance:
    """Synthesized pseudo-speech plus its ground-truth schedules."""

    pcm_s16le: bytes
    sr: int
    events: list[PhonemeEvent] = field(default_factory=list)
    openness: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def openness_at(self, t_s: float) -> float:
        """Ground-truth openness at time ``t_s`` (clamped index)."""
        idx = int(t_s * self.sr)
        if idx < 0 or idx >= len(self.openness):
            return 0.0
        return float(self.openness[idx])

    def to_meta(self) -> dict:
        return {
            "sr": self.sr,
            "events": [
                {
                    "phoneme": e.phoneme,
                    "kind": e.kind,
                    "start_s": round(e.start_s, 4),
                    "end_s": round(e.end_s, 4),
                    "openness": e.openness,
                }
                for e in self.events
            ],
        }


def _render_vowel(n: int, f1: int, f2: int, rng: random.Random) -> np.ndarray:
    """Formant-pair vowel with attack/release envelope."""
    t = np.arange(n) / _SR
    env = np.ones(n)
    a = int(0.015 * _SR)
    r = int(0.040 * _SR)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[n - r:] = np.linspace(1.0, 0.0, r)
    wave = (
        0.62 * np.sin(2 * math.pi * f1 * t)
        + 0.30 * np.sin(2 * math.pi * f2 * t)
        + 0.08 * np.sin(2 * math.pi * (f1 + f2) * t)
    )
    jitter = rng.uniform(0.97, 1.03)
    return 0.35 * env * wave * jitter


def _render_consonant(n: int, kind: str, rng: random.Random) -> np.ndarray:
    if kind == "nasal":
        t = np.arange(n) / _SR
        env = np.ones(n)
        env[: n // 4] = np.linspace(0.0, 1.0, n // 4)
        return 0.14 * env * np.sin(2 * math.pi * 110 * t)
    if kind == "plosive":
        out = np.zeros(n)
        burst = max(n - int(0.010 * _SR), n // 3)
        noise = rng.gauss(0.0, 1.0, n - burst)
        decay = np.linspace(1.0, 0.2, len(noise))
        out[burst:] = 0.30 * noise * decay
        return out
    # fricative: shaped white noise
    noise = np.array([rng.gauss(0.0, 1.0) for _ in range(n)])
    return 0.13 * noise


def _smooth_openness(target: np.ndarray) -> np.ndarray:
    """One-pole lowpass of the per-sample openness target."""
    alpha = 1.0 - math.exp(-1.0 / (_OPENNESS_TAU_S * _SR))
    out = np.empty_like(target)
    acc = float(target[0])
    for i, v in enumerate(target):
        acc += alpha * (float(v) - acc)
        out[i] = acc
    return out


def synth_utterance(duration_s: float, seed: int, sr: int = _SR) -> Utterance:
    """Synthesize one deterministic pseudo-speech utterance.

    ``duration_s`` is the total PCM length; phonemes fill it minus a
    100 ms lead-in and 150 ms tail pause. Same ``seed`` → same audio,
    events, and openness curve (interrupt/reference pairing).
    """
    if sr != _SR:
        raise ValueError(f"audio_synth is fixed at {_SR} Hz, got {sr}")
    rng = random.Random(seed)
    n_total = int(duration_s * _SR)
    target = np.full(n_total, PAUSE_OPENNESS)
    events: list[PhonemeEvent] = []

    t = 0.10  # lead-in pause
    while t < duration_s - 0.30:
        if rng.random() < 0.65:
            c = rng.choice(list(CONSONANTS))
            dur = rng.uniform(0.04, 0.07)
            op, kind = CONSONANTS[c]
            events.append(PhonemeEvent(c, kind, t, t + dur, op))
            t += dur
        v = rng.choice(list(VOWELS))
        dur = rng.uniform(0.14, 0.26)
        f1, f2, op = VOWELS[v]
        events.append(PhonemeEvent(v, "vowel", t, t + dur, op))
        t += dur
        if rng.random() < 0.30:
            dur = rng.uniform(0.12, 0.20)
            events.append(PhonemeEvent("_", "pause", t, t + dur, PAUSE_OPENNESS))
            t += dur
    # Tail pause to the end of the buffer.
    events.append(PhonemeEvent("_", "pause", t, duration_s, PAUSE_OPENNESS))

    wave = np.zeros(n_total)
    for e in events:
        i0, i1 = int(e.start_s * _SR), min(int(e.end_s * _SR), n_total)
        n = i1 - i0
        if n <= 0:
            continue
        target[i0:i1] = e.openness
        if e.kind == "vowel":
            f1, f2, _ = VOWELS[e.phoneme]
            wave[i0:i1] += _render_vowel(n, f1, f2, rng)
        elif e.kind == "consonant":
            wave[i0:i1] += _render_consonant(n, e.kind, rng)

    pcm = np.clip(wave * 32767 * 0.8, -32768, 32767).astype("<i2")
    return Utterance(
        pcm_s16le=pcm.tobytes(),
        sr=sr,
        events=events,
        openness=_smooth_openness(target),
    )


if __name__ == "__main__":
    import os
    import sys
    import wave

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    utt = synth_utterance(6.0, seed)
    path = f"data/interruption_eval/demo_utt_seed{seed}.wav"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(utt.sr)
        w.writeframes(utt.pcm_s16le)
    vowels = [e.phoneme for e in utt.events if e.kind == "vowel"]
    print(f"wrote {path}: {len(utt.events)} phonemes, vowels={''.join(vowels)}")

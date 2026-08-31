"""Numpy-only Slaney mel filterbank — importable without torch.

Split out of ``whisper_encoder`` so the mel frontend can be unit-tested
(and reused) in light environments where torch is not installed.  The
values reproduce ``librosa.filters.mel(..., htk=False, norm='slaney')``
as used by the transformers Whisper feature extractor.
"""

from __future__ import annotations

import math

import numpy as np


def _hz_to_mel(freqs: np.ndarray) -> np.ndarray:
    """Slaney mel scale (librosa ``htk=False``)."""
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freqs - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(
        freqs >= min_log_hz,
        min_log_mel + np.log(np.maximum(freqs, min_log_hz) / min_log_hz) / logstep,
        mels,
    )


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(mels >= min_log_mel, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)


def mel_filterbank(sr: int = 16000, n_fft: int = 400, n_mels: int = 80, fmin: float = 0.0, fmax: float = 8000.0) -> np.ndarray:
    """Slaney-normalised triangular mel filterbank (librosa.filters.mel)."""
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0, sr // 2, n_freqs)

    m_min, m_max = _hz_to_mel(np.array([fmin])), _hz_to_mel(np.array([fmax]))
    mel_pts = _mel_to_hz(np.linspace(m_min[0], m_max[0], n_mels + 2))

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[None, :] - all_freqs[:, None]
    for i in range(n_mels):
        lower, upper = -ramps[:, i] / fdiff[i], ramps[:, i + 2] / fdiff[i + 1]
        fb[i] = np.clip(np.minimum(lower, upper), 0.0, None)
    # Slaney norm: scale each triangle by 2 / (upper - lower) frequency span.
    enorm = 2.0 / (mel_pts[2 : n_mels + 2] - mel_pts[:n_mels])
    return (fb * enorm[:, None]).astype(np.float32)

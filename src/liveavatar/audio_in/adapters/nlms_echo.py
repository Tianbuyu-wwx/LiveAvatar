"""NLMS (Normalized Least Mean Square) adaptive echo cancellation.

Cancels acoustic echo from the student microphone signal using the Tutor
audio as the far-end reference. The algorithm runs entirely in-process with
numpy (no torch dependency), as required by the Sprint 2 architecture.

Algorithm
---------
For each near-end sample ``d[n]``:

1. Retrieve the next far-end sample ``x[n]`` from a FIFO queue (or 0 if the
   queue is empty — no Tutor audio → no echo).
2. Update the far-end history buffer ``x_hist = [x[n], x[n-1], …, x[n-N+1]]``.
3. Estimate the echo: ``y_hat = w · x_hist``.
4. Compute the error (echo-cancelled output): ``e[n] = d[n] - y_hat``.
5. Update the filter weights (NLMS):
   ``w += μ · e[n] · x_hist / (‖x_hist‖² + ε)``.

The filter length ``N`` (default 3200 samples = 200 ms at 16 kHz) is large
enough to cover typical acoustic echo path delays (speaker→room→mic). The
NLMS weights automatically learn the echo path including the delay.

Integration
-----------
The worker calls :meth:`push_far_end` whenever Tutor audio is published,
and :meth:`process` for each student_mic frame. The output of ``process``
replaces the frame's ``pcm_s16le`` before VAD/EOU/ASR processing.

When headphones are detected (no acoustic echo), set ``enabled = False`` to
bypass the filter — frames pass through unchanged.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("realtime_audio.aec")

_EPS = 1e-10  # Regularization to avoid division by zero.


@dataclass
class AecStats:
    """Counters for the AEC filter."""

    near_end_samples: int = 0
    far_end_samples: int = 0
    frames_processed: int = 0
    frames_bypassed: int = 0
    # Running ERLE (Echo Return Loss Enhancement) in dB.
    erle_db: float = 0.0
    # Near-end and error power for ERLE computation.
    near_power_ema: float = 0.0
    error_power_ema: float = 0.0


class NlmsAec:
    """NLMS adaptive echo cancellation (numpy-only).

    Parameters
    ----------
    filter_length : int
        Number of filter taps. Default 3200 = 200 ms at 16 kHz, covering
        typical acoustic echo delays.
    step_size : float
        NLMS step size μ. Controls convergence speed vs. stability.
        Typical: 0.1–1.0. Default 0.2.
    sample_rate : int
        Audio sample rate. Default 16000.
    far_end_buffer_ms : int
        Maximum far-end buffer duration. Far-end samples beyond this are
        discarded to prevent unbounded memory growth. Default 500 ms.
    """

    def __init__(
        self,
        filter_length: int = 3200,
        step_size: float = 0.2,
        sample_rate: int = 16000,
        far_end_buffer_ms: int = 500,
    ) -> None:
        self._N = filter_length
        self._mu = step_size
        self._sample_rate = sample_rate
        self._enabled = True

        # Filter weights — learned echo path impulse response.
        self._weights = np.zeros(filter_length, dtype=np.float32)

        # Far-end history: the last N far-end samples, most recent first.
        # Updated sample-by-sample during process().
        self._far_end_history = np.zeros(filter_length, dtype=np.float32)

        # FIFO queue for incoming far-end samples not yet consumed.
        max_buffer = int(sample_rate * far_end_buffer_ms / 1000)
        self._far_end_queue: deque[float] = deque(maxlen=max_buffer)

        # EMA smoothing factor for ERLE computation.
        self._erle_alpha = 0.95

        self.stats = AecStats()

    @property
    def enabled(self) -> bool:
        """When False, ``process`` returns the input unchanged (headphone mode)."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def push_far_end(self, pcm: bytes) -> None:
        """Buffer far-end (Tutor) reference audio.

        Called whenever Tutor audio is published. The samples are queued and
        consumed one-by-one during :meth:`process` to maintain time alignment
        with the near-end (student_mic) signal.
        """
        if not pcm:
            return
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._far_end_queue.extend(samples.tolist())
        self.stats.far_end_samples += len(samples)

    def process(self, near_pcm: bytes) -> bytes:
        """Cancel echo from near-end PCM using the buffered far-end reference.

        Returns the echo-cancelled PCM (same format: S16LE). If the filter is
        disabled (headphone mode), the input is returned unchanged.
        """
        if not self._enabled or len(near_pcm) < 2:
            self.stats.frames_bypassed += 1
            return near_pcm

        near = np.frombuffer(near_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        output = np.empty_like(near)

        w = self._weights
        hist = self._far_end_history
        mu = self._mu

        near_power_sum = 0.0
        error_power_sum = 0.0

        for i in range(len(near)):
            # Retrieve next far-end sample (0 if queue is empty).
            if self._far_end_queue:
                x_new = self._far_end_queue.popleft()
            else:
                x_new = 0.0

            # Shift history: [x_new, x_old1, x_old2, ...]
            hist[1:] = hist[:-1]
            hist[0] = x_new

            # Estimate echo.
            y_hat = float(np.dot(w, hist))

            # Error = near - estimated echo.
            e = near[i] - y_hat
            output[i] = e

            # NLMS weight update.
            power = float(np.dot(hist, hist))
            if power > _EPS:
                w += mu * e * hist / (power + _EPS)

            # Accumulate power for ERLE.
            near_power_sum += float(near[i] * near[i])
            error_power_sum += float(e * e)

        self.stats.near_end_samples += len(near)
        self.stats.frames_processed += 1

        # Update ERLE (Echo Return Loss Enhancement) = 10*log10(near/error).
        n = max(1, len(near))
        near_power = near_power_sum / n
        error_power = error_power_sum / n
        self.stats.near_power_ema = (
            self._erle_alpha * self.stats.near_power_ema
            + (1 - self._erle_alpha) * near_power
        )
        self.stats.error_power_ema = (
            self._erle_alpha * self.stats.error_power_ema
            + (1 - self._erle_alpha) * error_power
        )
        if self.stats.error_power_ema > _EPS:
            erle = 10 * np.log10(
                max(_EPS, self.stats.near_power_ema) / self.stats.error_power_ema
            )
            self.stats.erle_db = float(erle)

        return (np.clip(output, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def reset(self) -> None:
        """Reset filter state (called on epoch advance)."""
        self._weights[:] = 0.0
        self._far_end_history[:] = 0.0
        self._far_end_queue.clear()

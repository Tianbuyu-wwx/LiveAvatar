"""Tests for the NLMS adaptive echo cancellation filter.

Covers:
- Basic contract (output format/length).
- Headphone bypass (``enabled = False``).
- No-far-end passthrough (no echo source → output ≈ input).
- Synthetic echo cancellation (the key convergence test): near-end is a
  delayed+attenuated copy of the far-end; after convergence the output
  power must drop significantly and ERLE must be positive.
- Stats accounting (near/far samples, frames processed/bypassed).
- ``reset()`` clears filter state.
- Edge cases: empty PCM, far-end buffer cap, idempotent far-end push.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False

from liveavatar.audio_in.adapters.nlms_echo import NlmsAec


def _pcm(samples: np.ndarray) -> bytes:
    """Convert float [-1, 1] samples to S16LE bytes."""
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _from_pcm(data: bytes) -> np.ndarray:
    """Convert S16LE bytes back to float samples."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


@unittest.skipUnless(_HAS_NUMPY, "numpy required for AEC tests")
class TestNlmsAecBasicContract(unittest.TestCase):
    """Verify the input/output contract of ``process``."""

    def test_process_returns_bytes_of_same_length(self):
        aec = NlmsAec(filter_length=64, step_size=0.2)
        near = _pcm(np.zeros(320, dtype=np.float32))
        aec.push_far_end(_pcm(np.zeros(320, dtype=np.float32)))
        out = aec.process(near)
        self.assertIsInstance(out, bytes)
        self.assertEqual(len(out), len(near))

    def test_process_empty_pcm_is_bypassed(self):
        aec = NlmsAec(filter_length=64)
        out = aec.process(b"")
        self.assertEqual(out, b"")
        self.assertEqual(aec.stats.frames_bypassed, 1)

    def test_process_single_byte_pcm_is_bypassed(self):
        # < 2 bytes → cannot form one int16 sample → bypassed.
        aec = NlmsAec(filter_length=64)
        out = aec.process(b"\x01")
        self.assertEqual(out, b"\x01")
        self.assertEqual(aec.stats.frames_bypassed, 1)


@unittest.skipUnless(_HAS_NUMPY, "numpy required for AEC tests")
class TestNlmsAecHeadphoneBypass(unittest.TestCase):
    """``enabled = False`` (headphone mode) must pass input through."""

    def test_disabled_returns_input_unchanged(self):
        aec = NlmsAec(filter_length=64)
        aec.enabled = False
        near_samples = np.random.RandomState(42).randn(320).astype(np.float32) * 0.3
        near = _pcm(near_samples)
        aec.push_far_end(_pcm(np.ones(320, dtype=np.float32) * 0.5))
        out = aec.process(near)
        # Output identical to input (passthrough).
        self.assertEqual(out, near)
        self.assertEqual(aec.stats.frames_bypassed, 1)
        self.assertEqual(aec.stats.frames_processed, 0)

    def test_enabled_property_round_trip(self):
        aec = NlmsAec(filter_length=64)
        self.assertTrue(aec.enabled)
        aec.enabled = False
        self.assertFalse(aec.enabled)
        aec.enabled = True
        self.assertTrue(aec.enabled)


@unittest.skipUnless(_HAS_NUMPY, "numpy required for AEC tests")
class TestNlmsAecNoFarEnd(unittest.TestCase):
    """With no far-end reference queued, output tracks the near-end input."""

    def test_no_far_end_output_approximates_input(self):
        # When the far-end queue is empty, x_new = 0 every sample, so the
        # filter estimates zero echo → output == near-end (modulo float round).
        aec = NlmsAec(filter_length=64, step_size=0.2)
        near_samples = (np.random.RandomState(7).randn(320).astype(np.float32) * 0.2)
        near = _pcm(near_samples)
        out = aec.process(near)
        out_samples = _from_pcm(out)
        # Output should be very close to the input (no echo to subtract).
        np.testing.assert_allclose(out_samples, near_samples, atol=2e-3)
        self.assertEqual(aec.stats.frames_processed, 1)
        self.assertEqual(aec.stats.near_end_samples, 320)
        self.assertEqual(aec.stats.far_end_samples, 0)


@unittest.skipUnless(_HAS_NUMPY, "numpy required for AEC tests")
class TestNlmsAecSyntheticEcho(unittest.TestCase):
    """The key convergence test: cancel a known synthetic echo path."""

    def test_cancels_pure_echo_after_convergence(self):
        """Near-end = delayed+attenuated far-end (pure echo, no near speech).

        After the filter converges, the output (error) power must be far
        smaller than the near-end (input) power → ERLE strongly positive.

        Far-end is pushed frame-by-frame (matching real usage where TTS audio
        arrives incrementally). Pushing it all at once would overflow the
        500 ms far-end buffer and break time alignment.
        """
        rng = np.random.RandomState(123)
        sample_rate = 16000
        # Far-end: band-limited noise (rich spectral content converges faster
        # than a pure tone which has rank-1 autocorrelation).
        far = rng.randn(sample_rate * 2).astype(np.float32) * 0.3
        # Smooth slightly to make it band-limited.
        kernel = np.ones(5, dtype=np.float32) / 5.0
        far = np.convolve(far, kernel, mode="same").astype(np.float32)

        # Echo path: delay 120 samples (~7.5 ms) + attenuation 0.6.
        delay = 120
        attenuation = 0.6
        near = np.zeros_like(far)
        near[delay:] = attenuation * far[:-delay]

        # Use a filter long enough to cover the echo delay.
        aec = NlmsAec(filter_length=512, step_size=0.5, sample_rate=sample_rate)

        frame_size = 320  # 20 ms
        n_frames = len(far) // frame_size

        # Push far-end and process near-end frame-by-frame in lockstep. This
        # mirrors real usage: the Tutor publishes a TTS frame, then the mic
        # frame containing its echo arrives and is processed. The far-end
        # queue thus never overflows the 500 ms buffer cap.
        output_frames: list[np.ndarray] = []
        for i in range(n_frames):
            far_chunk = far[i * frame_size : (i + 1) * frame_size]
            near_chunk = near[i * frame_size : (i + 1) * frame_size]
            aec.push_far_end(_pcm(far_chunk))
            out = aec.process(_pcm(near_chunk))
            output_frames.append(_from_pcm(out))

        output = np.concatenate(output_frames)

        # Compare the LAST second (post-convergence) against the last second
        # of near-end input.
        tail = sample_rate  # last 1 second
        near_power = float(np.mean(near[-tail:] ** 2))
        error_power = float(np.mean(output[-tail:] ** 2))

        self.assertGreater(near_power, 1e-6, "near-end power too low to test")
        erle_db = 10 * np.log10(near_power / max(error_power, 1e-12))
        # Expect a meaningful echo reduction (>= 10 dB) after convergence.
        self.assertGreater(
            erle_db,
            10.0,
            f"ERLE {erle_db:.1f} dB < 10 dB; AEC failed to converge",
        )
        # Stats reflect the work done.
        self.assertEqual(aec.stats.frames_processed, n_frames)
        self.assertEqual(aec.stats.near_end_samples, n_frames * frame_size)

    def test_erle_becomes_positive_when_echo_cancelled(self):
        """``stats.erle_db`` must become positive once echo is suppressed."""
        rng = np.random.RandomState(99)
        far = rng.randn(16000).astype(np.float32) * 0.25
        delay = 80
        near = np.zeros_like(far)
        near[delay:] = 0.5 * far[:-delay]

        aec = NlmsAec(filter_length=256, step_size=0.5)

        frame_size = 320
        for i in range(len(near) // frame_size):
            far_chunk = far[i * frame_size : (i + 1) * frame_size]
            near_chunk = near[i * frame_size : (i + 1) * frame_size]
            aec.push_far_end(_pcm(far_chunk))
            aec.process(_pcm(near_chunk))

        self.assertGreater(aec.stats.erle_db, 5.0)

    def test_preserves_near_end_speech_with_echo(self):
        """When near-end has speech + echo, AEC should reduce echo but keep
        the near-end speech (output power < input power, but not silenced)."""
        rng = np.random.RandomState(2024)
        far = rng.randn(16000).astype(np.float32) * 0.3
        delay = 100
        echo = np.zeros_like(far)
        echo[delay:] = 0.5 * far[:-delay]
        # Near-end speech: a low-amplitude sinusoid.
        t = np.arange(16000) / 16000.0
        near_speech = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        near = echo + near_speech

        aec = NlmsAec(filter_length=256, step_size=0.5)

        frame_size = 320
        outputs = []
        for i in range(len(near) // frame_size):
            far_chunk = far[i * frame_size : (i + 1) * frame_size]
            near_chunk = near[i * frame_size : (i + 1) * frame_size]
            aec.push_far_end(_pcm(far_chunk))
            outputs.append(_from_pcm(aec.process(_pcm(near_chunk))))
        output = np.concatenate(outputs)

        tail = 8000  # last 0.5 s (post-convergence)
        near_power = float(np.mean(near[-tail:] ** 2))
        error_power = float(np.mean(output[-tail:] ** 2))
        # Echo reduced (error < near)...
        self.assertLess(error_power, near_power * 0.5)
        # ...but near-end speech still present (output not silenced).
        self.assertGreater(error_power, 1e-7)


@unittest.skipUnless(_HAS_NUMPY, "numpy required for AEC tests")
class TestNlmsAecStats(unittest.TestCase):
    """Verify ``AecStats`` accounting."""

    def test_stats_track_near_and_far_samples(self):
        aec = NlmsAec(filter_length=64)
        far = _pcm(np.ones(160, dtype=np.float32) * 0.2)
        near = _pcm(np.ones(160, dtype=np.float32) * 0.1)
        aec.push_far_end(far)
        aec.process(near)
        self.assertEqual(aec.stats.far_end_samples, 160)
        self.assertEqual(aec.stats.near_end_samples, 160)
        self.assertEqual(aec.stats.frames_processed, 1)
        self.assertEqual(aec.stats.frames_bypassed, 0)

    def test_bypassed_frames_counted_when_disabled(self):
        aec = NlmsAec(filter_length=64)
        aec.enabled = False
        near = _pcm(np.zeros(160, dtype=np.float32))
        aec.push_far_end(_pcm(np.ones(160, dtype=np.float32)))
        aec.process(near)
        aec.process(near)
        self.assertEqual(aec.stats.frames_bypassed, 2)
        self.assertEqual(aec.stats.frames_processed, 0)


@unittest.skipUnless(_HAS_NUMPY, "numpy required for AEC tests")
class TestNlmsAecReset(unittest.TestCase):
    """``reset()`` must clear filter state and buffers."""

    def test_reset_clears_weights_and_history(self):
        aec = NlmsAec(filter_length=128, step_size=0.5)
        # Train the filter briefly so weights become non-zero.
        far = _pcm(np.random.RandomState(1).randn(320).astype(np.float32) * 0.3)
        near = _pcm(np.random.RandomState(2).randn(320).astype(np.float32) * 0.3)
        aec.push_far_end(far)
        aec.process(near)
        # At least some weights should be non-zero after one frame.
        self.assertGreater(float(np.sum(np.abs(aec._weights))), 0.0)

        aec.reset()
        np.testing.assert_array_equal(aec._weights, np.zeros(128, dtype=np.float32))
        np.testing.assert_array_equal(
            aec._far_end_history, np.zeros(128, dtype=np.float32)
        )
        self.assertEqual(len(aec._far_end_queue), 0)

    def test_reset_allows_clean_restart(self):
        """After reset, processing with no far-end gives near-end passthrough."""
        aec = NlmsAec(filter_length=64, step_size=0.5)
        far = _pcm(np.random.RandomState(5).randn(320).astype(np.float32) * 0.3)
        near = _pcm(np.random.RandomState(6).randn(320).astype(np.float32) * 0.3)
        aec.push_far_end(far)
        aec.process(near)
        aec.reset()
        # No far-end queued now → output ≈ input.
        near2 = (np.random.RandomState(7).randn(320).astype(np.float32) * 0.2)
        out = _from_pcm(aec.process(_pcm(near2)))
        np.testing.assert_allclose(out, near2, atol=2e-3)


@unittest.skipUnless(_HAS_NUMPY, "numpy required for AEC tests")
class TestNlmsAecFarEndBuffer(unittest.TestCase):
    """Far-end buffering edge cases."""

    def test_push_empty_far_end_is_noop(self):
        aec = NlmsAec(filter_length=64)
        aec.push_far_end(b"")
        self.assertEqual(aec.stats.far_end_samples, 0)
        self.assertEqual(len(aec._far_end_queue), 0)

    def test_far_end_buffer_capped_at_maxlen(self):
        # far_end_buffer_ms=500 at 16 kHz → 8000 samples max.
        aec = NlmsAec(filter_length=64, sample_rate=16000, far_end_buffer_ms=500)
        # Push 16000 samples (1 second) — exceeds the 8000 cap.
        big = _pcm(np.ones(16000, dtype=np.float32) * 0.1)
        aec.push_far_end(big)
        self.assertLessEqual(len(aec._far_end_queue), 8000)
        self.assertEqual(aec.stats.far_end_samples, 16000)  # counted on input

    def test_far_end_consumed_by_process(self):
        aec = NlmsAec(filter_length=64)
        far = _pcm(np.ones(320, dtype=np.float32) * 0.3)
        near = _pcm(np.zeros(320, dtype=np.float32))
        aec.push_far_end(far)
        self.assertEqual(len(aec._far_end_queue), 320)
        aec.process(near)
        # All 320 far-end samples consumed during process.
        self.assertEqual(len(aec._far_end_queue), 0)


@unittest.skipUnless(_HAS_NUMPY, "numpy required for AEC tests")
class TestNlmsAecOutputClipping(unittest.TestCase):
    """Output must be valid S16LE (clamped to int16 range)."""

    def test_output_never_exceeds_int16_range(self):
        aec = NlmsAec(filter_length=64, step_size=1.0)
        # Large far-end + large near-end can produce large error early on.
        far = _pcm(np.ones(320, dtype=np.float32) * 0.9)
        near = _pcm(np.ones(320, dtype=np.float32) * 0.9)
        aec.push_far_end(far)
        out = aec.process(near)
        samples = np.frombuffer(out, dtype=np.int16)
        self.assertTrue(np.all(samples >= -32768))
        self.assertTrue(np.all(samples <= 32767))


if __name__ == "__main__":
    unittest.main()

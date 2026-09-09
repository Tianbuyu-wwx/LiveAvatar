# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-D transition synthesis pure-function tests (paper §4.4).

Covers the shared alpha schedule and both transition backends' math:
pixel crossfade (CPU eval proxy) and the numpy latent-interpolation
reference the MuseTalk worker mirrors on torch tensors.
"""

from __future__ import annotations

import unittest

import numpy as np

from liveavatar.runtime.transition import (
    CLOSED_OPENNESS,
    MAX_FRAMES,
    MIN_FRAMES,
    alpha_schedule,
    clamp_frames,
    crossfade_bgr,
    interpolate_latent_np,
    interpolate_openness,
)


class TestClampFrames(unittest.TestCase):
    def test_legal_range_unchanged(self) -> None:
        for n in range(MIN_FRAMES, MAX_FRAMES + 1):
            self.assertEqual(clamp_frames(n), n)

    def test_out_of_range_clamped(self) -> None:
        self.assertEqual(clamp_frames(0), 0)
        self.assertEqual(clamp_frames(-3), 0)
        self.assertEqual(clamp_frames(1), 1)  # below MIN still legal (disabled=0 only)
        self.assertEqual(clamp_frames(9), MAX_FRAMES)


class TestAlphaSchedule(unittest.TestCase):
    def test_uniform_rise_last_frame_is_one(self) -> None:
        alpha = alpha_schedule(3)
        np.testing.assert_allclose(alpha, [1 / 3, 2 / 3, 1.0])

    def test_single_frame_is_full_close(self) -> None:
        np.testing.assert_allclose(alpha_schedule(1), [1.0])

    def test_disabled_yields_empty(self) -> None:
        self.assertEqual(alpha_schedule(0).size, 0)
        self.assertEqual(alpha_schedule(-2).size, 0)

    def test_over_max_clamped(self) -> None:
        self.assertEqual(alpha_schedule(99).size, MAX_FRAMES)

    def test_monotone_non_decreasing(self) -> None:
        alpha = alpha_schedule(4)
        self.assertTrue(np.all(np.diff(alpha) > 0))


class TestInterpolateOpenness(unittest.TestCase):
    def test_eases_current_to_closed(self) -> None:
        out = interpolate_openness(0.85, alpha_schedule(4))
        self.assertEqual(out.size, 4)
        self.assertAlmostEqual(out[0], 0.75 * 0.85 + 0.25 * CLOSED_OPENNESS)
        self.assertAlmostEqual(float(out[-1]), CLOSED_OPENNESS)
        self.assertTrue(np.all(np.diff(out) < 0))  # monotone closing


class TestCrossfadeBgr(unittest.TestCase):
    def _frames(self) -> tuple[np.ndarray, np.ndarray]:
        now = np.zeros((4, 4, 3), dtype=np.uint8)
        closed = np.full((4, 4, 3), 200, dtype=np.uint8)
        return now, closed

    def test_alpha_one_is_exactly_closed(self) -> None:
        now, closed = self._frames()
        out = crossfade_bgr(now, closed, np.array([1.0]))
        self.assertEqual(len(out), 1)
        np.testing.assert_array_equal(out[0], closed)

    def test_monotone_approach_to_closed(self) -> None:
        now, closed = self._frames()
        out = crossfade_bgr(now, closed, alpha_schedule(4))
        dists = [float(np.linalg.norm(f.astype(np.float64) - closed)) for f in out]
        self.assertEqual(len(dists), 4)
        for a, b in zip(dists, dists[1:], strict=False):
            self.assertLess(b, a)

    def test_output_dtype_and_range(self) -> None:
        now, closed = self._frames()
        for f in crossfade_bgr(now, closed, alpha_schedule(3)):
            self.assertEqual(f.dtype, np.uint8)

    def test_shape_mismatch_raises(self) -> None:
        now, _ = self._frames()
        other = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            crossfade_bgr(now, other, alpha_schedule(2))

    def test_empty_alpha_yields_no_frames(self) -> None:
        now, closed = self._frames()
        self.assertEqual(crossfade_bgr(now, closed, np.empty(0)), [])


class TestInterpolateLatentNp(unittest.TestCase):
    def test_known_blend_values(self) -> None:
        z_now = np.full((1, 2, 2, 2), 1.0)
        z_closed = np.full((1, 2, 2, 2), 3.0)
        out = interpolate_latent_np(z_now, z_closed, np.array([0.25, 1.0]))
        self.assertEqual(len(out), 2)
        np.testing.assert_allclose(out[0], np.full_like(z_now, 1.5))
        np.testing.assert_allclose(out[1], z_closed)  # α=1 → exactly closed

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            interpolate_latent_np(
                np.zeros((1, 2, 2, 2)), np.zeros((1, 2, 2)), np.array([0.5])
            )

    def test_empty_alpha(self) -> None:
        z = np.zeros((1, 2, 2, 2))
        self.assertEqual(interpolate_latent_np(z, z, np.empty(0)), [])


if __name__ == "__main__":
    unittest.main()

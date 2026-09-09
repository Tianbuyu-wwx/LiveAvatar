# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-B metric library unit tests (pure numpy, no media needed).

Pins the paper metric definitions on synthetic trajectories:
- transition_smoothness: mutation across the epoch boundary vs baseline,
  stale-frame handling, freeze counting, zero-baseline guard.
- v_apt: perturbation area ≈ offset × span, recovery detection, no-
  recovery span capping.
- viseme_buckets: quantile edges and shared bucketing across series.
- phoneme_viseme_mismatch: full-mismatch / full-match windows.
- half_syllable_flag: strict-inside-vowel semantics.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "interruption_eval"
)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import metrics as m  # noqa: E402


class TestTransitionSmoothness:
    def test_mutation_dominates_baseline(self) -> None:
        t = np.arange(120) / 25.0
        base = 0.4 + 0.01 * np.sin(2 * np.pi * 1.0 * t)  # |Δ| ≈ 0.0025
        op = base.copy()
        op[100:] = 0.9 + 0.01 * np.sin(2 * np.pi * 1.0 * t[100:])
        epochs = np.where(np.arange(120) < 100, 1, 2)
        r = m.transition_smoothness(op, epochs, cut_idx=100)
        assert r.baseline_p50 > 0
        assert r.mutation > 0.4
        assert r.ratio is not None and r.ratio > 100
        assert r.freeze_frames == 0

    def test_stale_frame_and_freeze_counting(self) -> None:
        op = np.concatenate([np.full(100, 0.4), np.full(4, 0.4), [0.8], np.zeros(5)])
        epochs = np.concatenate([np.full(101, 1), np.full(19, 2)]).astype(int)
        # stale old-epoch frame at idx 100; first new-epoch frame at 104
        r = m.transition_smoothness(op, epochs, cut_idx=101, old_epoch=1)
        assert r.mutation > 0.39
        assert r.freeze_frames == 3  # idx 101..103 dropped between old and new

    def test_zero_baseline_gives_none_ratio(self) -> None:
        op = np.concatenate([np.full(50, 0.4), [0.8], np.zeros(10)])
        epochs = np.concatenate([np.full(50, 1), np.full(11, 2)]).astype(int)
        r = m.transition_smoothness(op, epochs, cut_idx=50)
        assert r.baseline_p50 == 0.0
        assert r.ratio is None


class TestVapt:
    def test_offset_area_and_recovery(self) -> None:
        t = np.arange(200) / 25.0
        ref = np.full(200, 0.4)
        ref[70:] = 0.4 + 0.2 * np.sin(2 * np.pi * 3.0 * (t[70:] - t[70]))
        op = ref.copy()
        op[50:70] = 0.8  # +0.4 for 20 frames, then rejoins exactly
        r = m.v_apt(op, ref, cut_idx=50, frame_period_ms=40.0)
        assert abs(r.area - 0.4 * 20) < 1e-6
        assert abs(r.apt_ms - 0.4 * 20 * 40.0) < 1e-6
        assert r.recover_idx == 74  # 5-frame hold after rejoin at 70
        assert r.span_frames == 24

    def test_no_recovery_spans_to_max(self) -> None:
        ref = np.zeros(100)
        op = np.full(100, 0.5)
        r = m.v_apt(op, ref, cut_idx=10, frame_period_ms=40.0, max_span=50)
        assert r.recover_idx is None
        assert r.span_frames == 50
        assert abs(r.area - 25.0) < 1e-6

    def test_partial_recovery_then_rejoin(self) -> None:
        t = np.arange(100) / 25.0
        ref = 0.5 + 0.45 * np.sin(2 * np.pi * 1.5 * t)  # speech-like motion
        op = ref.copy()
        op[20:30] = ref[20:30] + 0.5  # 10 perturbed frames
        op[30:] = ref[30:] + 0.05  # below tolerance 0.08 → recovers
        r = m.v_apt(op, ref, cut_idx=20, frame_period_ms=40.0)
        assert r.recover_idx == 34  # 5-frame hold ends at 34
        assert abs(r.area - 5.0) < 0.05  # 10 frames × 0.5

    def test_frozen_series_never_recovers_when_reference_moves_away(self) -> None:
        # False-recovery regression: a frozen screen (ZOH, Δ = 0) that
        # coincides with a reference pause is value-synced but NOT
        # recovered — the area must keep accumulating once the
        # reference resumes speaking.
        ref = np.full(100, 0.4)
        ref[40:] = np.linspace(0.4, 0.9, 60)
        op = np.full(100, 0.4)  # frozen at the pre-cut value
        r = m.v_apt(op, ref, cut_idx=0, frame_period_ms=40.0)
        assert r.recover_idx is None
        assert r.area > 2.0  # post-dwell divergence counts

    def test_recovery_waits_for_motion_after_frozen_match(self) -> None:
        # Freeze matching a reference pause is visually invisible
        # (≈ zero penalty), but recovery is only stamped once real
        # motion appears — never during the frozen dwell.
        t = np.arange(80) / 25.0
        ref = np.full(80, 0.4)
        ref[40:] = 0.4 + 0.3 * np.sin(2 * np.pi * 2.0 * (t[40:] - t[40]))
        op = ref.copy()
        op[:40] = 0.4  # frozen during the reference's pause (equal values)
        r = m.v_apt(op, ref, cut_idx=0, frame_period_ms=40.0)
        assert r.recover_idx == 41  # run spans 0..; first motion frame +1
        assert r.area < 1.0  # matched the pause → almost no penalty


class TestVisemeBuckets:
    def test_quantile_edges_partition(self) -> None:
        op = np.linspace(0.0, 1.0, 400)
        b = m.viseme_buckets(op, n_buckets=4)
        assert len(b.edges) == 5
        counts = np.bincount(b.labels, minlength=4)
        assert counts.min() >= 90  # roughly balanced quantile buckets
        assert b.bucket_of(0.05) == 0
        assert b.bucket_of(0.95) == 3

    def test_shared_bucketing_across_series(self) -> None:
        ref = np.linspace(0.0, 1.0, 100)
        b = m.viseme_buckets(ref, n_buckets=4)
        interrupted = np.array([0.1, 0.45, 0.6, 0.95])
        labels = [b.bucket_of(v) for v in interrupted]
        assert labels == [0, 1, 2, 3]


class TestPhonemeVisemeMismatch:
    EVENTS = [
        {"phoneme": "a", "kind": "vowel", "start_s": 0.0, "end_s": 0.5, "openness": 0.9},
        {"phoneme": "_", "kind": "pause", "start_s": 0.5, "end_s": 1.0, "openness": 0.05},
    ]

    def _labels_for(self, openness_values: list[float], ref: np.ndarray) -> np.ndarray:
        b = m.viseme_buckets(ref, n_buckets=4)
        return np.array([b.bucket_of(v) for v in openness_values])

    def test_full_mismatch_window(self) -> None:
        ref = np.linspace(0.0, 1.0, 100)
        b = m.viseme_buckets(ref, n_buckets=4)
        # frames at 2.00..2.44s; utterance time = frame - tts_start (2.0)
        times = np.arange(2.0, 2.44, 0.04)
        wrong = self._labels_for([0.05] * len(times), ref)  # closed-mouth bucket
        rate = m.phoneme_viseme_mismatch(b, wrong, times, self.EVENTS, 2.0, 0.4, 2.0)
        assert rate == 1.0

    def test_full_match_window(self) -> None:
        ref = np.linspace(0.0, 1.0, 100)
        b = m.viseme_buckets(ref, n_buckets=4)
        times = np.arange(2.0, 2.44, 0.04)
        right = self._labels_for([0.9] * len(times), ref)
        rate = m.phoneme_viseme_mismatch(b, right, times, self.EVENTS, 2.0, 0.4, 2.0)
        assert rate == 0.0

    def test_empty_window_is_zero(self) -> None:
        ref = np.linspace(0.0, 1.0, 100)
        b = m.viseme_buckets(ref, n_buckets=4)
        times = np.array([0.0, 0.1])
        labels = self._labels_for([0.5, 0.5], ref)
        rate = m.phoneme_viseme_mismatch(b, labels, times, self.EVENTS, 5.0, 0.4, 2.0)
        assert rate == 0.0


class TestHalfSyllableFlag:
    EVENTS = [
        {"phoneme": "m", "kind": "consonant", "start_s": 0.0, "end_s": 0.05, "openness": 0.02},
        {"phoneme": "a", "kind": "vowel", "start_s": 0.05, "end_s": 0.30, "openness": 0.9},
        {"phoneme": "_", "kind": "pause", "start_s": 0.30, "end_s": 0.50, "openness": 0.05},
    ]

    def test_cut_inside_vowel(self) -> None:
        assert m.half_syllable_flag(0.15, self.EVENTS) == 1.0

    def test_cut_inside_consonant(self) -> None:
        assert m.half_syllable_flag(0.02, self.EVENTS) == 1.0

    def test_cut_inside_pause(self) -> None:
        assert m.half_syllable_flag(0.40, self.EVENTS) == 0.0

    def test_cut_on_boundary_is_clean(self) -> None:
        assert m.half_syllable_flag(0.05, self.EVENTS) == 0.0
        assert m.half_syllable_flag(0.30, self.EVENTS) == 0.0

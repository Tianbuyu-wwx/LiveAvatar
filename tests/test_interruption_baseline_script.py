# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-A5: recorder frame-difference analysis tests (CPU-only, no service).

Covers the pure analysis helpers of ``scripts/record_interruption_baseline.py``:
- ``_frame_diff_stats``: mutation diff (last old-epoch frame vs first
  new-epoch frame) vs normal inter-frame diffs inside the pre-interrupt
  window; ratio, freeze gap, and the None edge cases.
- ``_percentile`` / ``_fd_summary``: aggregation correctness.
- ``_speech_pcm``: chunk cadence matches the session pacing constants.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import record_interruption_baseline as rec  # noqa: E402


def _jpg(values: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", values)
    assert ok
    return buf.tobytes()


def _synth_obs() -> dict:
    """Old epoch: brightness ramps 100→104 (MAD 1/frame); new epoch: jump to 180."""
    h, w = 32, 32
    frames: list[tuple[float, int, bytes]] = []
    for i, ts in enumerate((0.0, 0.1, 0.2, 0.3, 0.4)):
        img = np.full((h, w), 100 + i, np.uint8)
        frames.append((ts, 2, _jpg(img)))
    frames.append((1.2, 3, _jpg(np.full((h, w), 180, np.uint8))))
    return {"frames": frames}


class TestFrameDiffStats(unittest.TestCase):
    """P-A5 evidence math on synthetic JPEG frames."""

    def test_mutation_dominates_normal(self) -> None:
        fd = rec._frame_diff_stats(
            _synth_obs(), pre_epoch=2, interrupt_sent=0.5, window_s=2.0
        )
        self.assertIsNotNone(fd)
        self.assertEqual(fd["normal_pairs"], 4)
        self.assertAlmostEqual(fd["normal_p50"], 1.0, delta=0.3)
        self.assertGreater(fd["mutation"], 50.0)
        self.assertGreater(fd["ratio_vs_p50"], 10.0)
        self.assertAlmostEqual(fd["freeze_gap_ms"], 800.0, delta=1.0)

    def test_window_excludes_old_pairs(self) -> None:
        fd = rec._frame_diff_stats(
            _synth_obs(), pre_epoch=2, interrupt_sent=0.5, window_s=0.05
        )
        self.assertIsNone(fd)  # no normal pairs inside the tiny window

    def test_no_new_epoch_frames(self) -> None:
        obs = _synth_obs()
        obs["frames"] = [f for f in obs["frames"] if f[1] == 2]
        self.assertIsNone(rec._frame_diff_stats(obs, 2, 0.5))

    def test_no_old_epoch_frames(self) -> None:
        obs = _synth_obs()
        obs["frames"] = [f for f in obs["frames"] if f[1] > 2]
        self.assertIsNone(rec._frame_diff_stats(obs, 2, 0.5))


class TestAggregation(unittest.TestCase):
    def test_percentile_interpolation(self) -> None:
        self.assertEqual(rec._percentile([1.0, 2.0, 3.0], 50), 2.0)
        self.assertAlmostEqual(rec._percentile([0.0, 10.0], 50), 5.0)
        self.assertEqual(rec._percentile([4.0], 95), 4.0)
        self.assertIsNone(rec._percentile([], 50))

    def test_fd_summary_picks_rows(self) -> None:
        rows = [
            {"avatar_id": "a", "frame_diff": {"mutation": 2.0, "normal_p50": 0.5,
                                              "ratio_vs_p50": 4.0,
                                              "freeze_gap_ms": 400.0}},
            {"avatar_id": "a", "frame_diff": {"mutation": 4.0, "normal_p50": 1.0,
                                              "ratio_vs_p50": 4.0,
                                              "freeze_gap_ms": 600.0}},
            {"avatar_id": "a"},  # no frame_diff → ignored
        ]
        s = rec._fd_summary(rows)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["mutation_p50"], 3.0)
        self.assertEqual(s["normal_p50_ms"], 0.75)
        self.assertEqual(s["ratio_p50"], 4.0)
        self.assertEqual(s["freeze_gap_p50_ms"], 500.0)


class TestSpeechPcm(unittest.TestCase):
    def test_chunk_cadence(self) -> None:
        chunks = rec._speech_pcm(rec._SESSION_SECONDS)
        self.assertEqual(len(chunks), int(rec._SESSION_SECONDS / rec._CHUNK_S))
        for c in chunks:
            self.assertEqual(len(c), rec._CHUNK_BYTES)


if __name__ == "__main__":
    unittest.main()

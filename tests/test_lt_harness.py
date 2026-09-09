# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-E unit tests: LiveTalking harness grid + extractor pure functions."""

from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "interruption_eval"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

lt_harness = importlib.import_module("lt_harness")
lt_extract = importlib.import_module("lt_extract")


class TestInterruptGrid:
    def test_grid_matches_record_py_seeds(self) -> None:
        """Same 20-point grid as the LiveAvatar recorder (dataset dir names)."""
        grid = lt_harness._interrupt_grid(20)
        assert len(grid) == 20
        assert grid[0] == 0.5
        assert grid[-1] == 4.5
        # locked against the existing recorded dataset directory names
        # (interrupt_mouth_a_seed10_t2.40s etc.) — same format string.
        names = [f"t{t:.2f}s" for t in grid]
        assert names[0] == "t0.50s"
        assert names[9] == "t2.40s"
        assert names[19] == "t4.50s"

    def test_grid_monotonic(self) -> None:
        grid = lt_harness._interrupt_grid(20)
        assert all(b > a for a, b in zip(grid, grid[1:], strict=False))


class TestPrepareWavs:
    def test_write_wav_roundtrip(self, tmp_path: Path) -> None:
        from audio_synth import synth_utterance

        utt = synth_utterance(0.5, 7)
        out = tmp_path / "utt.wav"
        lt_harness.write_wav(out, utt)
        with wave.open(str(out), "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == utt.sr
            data = w.readframes(w.getnframes())
        assert data == utt.pcm_s16le


class TestXcorrAnchor:
    def test_finds_inserted_wav_offset(self) -> None:
        from audio_synth import synth_utterance

        rng = np.random.default_rng(0)
        utt = synth_utterance(1.0, 3)
        wav = lt_extract._pcm_float(utt)
        offset = 12345
        recorded = rng.normal(0.0, 800.0, offset + len(wav) + 4000)
        recorded[offset : offset + len(wav)] += wav * 20.0
        got = lt_extract.xcorr_anchor(recorded, wav)
        assert got == offset

    def test_forward_lags_only(self) -> None:
        from audio_synth import synth_utterance

        utt = synth_utterance(0.5, 11)
        wav = lt_extract._pcm_float(utt)
        recorded = np.concatenate([np.zeros(500), wav, np.zeros(500)])
        assert lt_extract.xcorr_anchor(recorded, wav) == 500


class TestLastAudible:
    def test_energy_threshold(self) -> None:
        pcm = np.zeros(1000)
        pcm[100:200] = 500.0
        pcm[400:410] = 200.0  # below threshold
        assert lt_extract._last_audible(pcm, 0, 1000) == 199
        assert lt_extract._last_audible(pcm, 200, 1000) is None


class TestMouthRoi:
    def test_geometric_roi_within_face_box(self) -> None:
        x0, y0, x1, y1 = lt_extract._mouth_roi((100, 50, 200, 200))
        assert 100 + 200 <= (x0 + x1) / 2 * 2  # sanity: derived from box
        assert 100 <= x0 < x1 <= 300
        assert 50 <= y0 < y1 <= 250

    def test_openness_series_calibration(self) -> None:
        """Closed vs open synthetic mouth → separated openness values."""
        box = (0, 0, 200, 200)
        x0, y0, x1, y1 = lt_extract._mouth_roi(box)
        closed = np.full((200, 200, 3), 220.0, dtype=np.uint8)
        open_img = np.full((200, 200, 3), 220.0, dtype=np.uint8)
        open_img[y0:y1, x0:x1] = 30  # dark cavity fills the whole ROI
        op = lt_extract.openness_series([closed, open_img] * 5, box=box)
        assert op[0::2].max() < 0.1  # closed frames
        assert op[1::2].min() > 0.9  # open frames


class TestLtHarnessWiring:
    def test_utt_constants_match_record_py(self) -> None:
        assert lt_harness._UTT_S == 6.0
        assert lt_harness._UTT2_S == 2.5

    def test_prepare_writes_events(self, tmp_path: Path) -> None:
        ns = argparse_ns(tmp_path)
        assert lt_harness.cmd_prepare(ns) == 0
        events = (tmp_path / "events_seed1.json").read_text(encoding="utf-8")
        assert '"utt1"' in events and '"events"' in events
        assert (tmp_path / "utt1_seed1.wav").exists()
        assert (tmp_path / "utt2_seed1.wav").exists()


def argparse_ns(tmp_path: Path) -> object:
    import argparse as _ap

    ns = _ap.Namespace(wav_dir=str(tmp_path), seeds=2)
    return ns


@pytest.mark.parametrize("seed,dur,offset", [(1, 6.0, 0.0)])
def test_synth_reference_utts_deterministic(seed: int, dur: float, offset: float) -> None:
    from audio_synth import synth_utterance

    a = synth_utterance(dur, seed + int(offset) + 1000)
    b = synth_utterance(dur, seed + int(offset) + 1000)
    assert a.pcm_s16le == b.pcm_s16le

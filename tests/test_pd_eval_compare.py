# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-D4 acceptance tests: transition-aware anchors and ablation compare.

Covers the recorder's driven-frame anchor, the analyzer's transition
time mapping, and the paired hard-vs-transition comparison with its
≥ 50 % mutation-improvement gate.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "interruption_eval"
)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover - light CI envs
    _HAS_CV2 = False

_NEED_CV2 = pytest.mark.skipif(not _HAS_CV2, reason="cv2 required")


# ────────────────────────────────────── recorder: driven epoch-2 anchor


class TestFirstDrivenEpoch2:
    def test_no_transition_degenerates_to_first(self) -> None:
        from record import _first_driven_epoch2_frame_ts, _first_epoch2_frame_ts

        obs = {"frames": [{"epoch": 1, "ts": 0.5}, {"epoch": 2, "ts": 1.0},
                          {"epoch": 2, "ts": 1.1}]}
        assert _first_epoch2_frame_ts(obs) == 1.0
        assert _first_driven_epoch2_frame_ts(obs, 0) == 1.0

    def test_skips_transition_bridge_frames(self) -> None:
        from record import _first_driven_epoch2_frame_ts

        obs = {"frames": [
            {"epoch": 1, "ts": 0.5},
            {"epoch": 2, "ts": 1.0},   # transition frame 0
            {"epoch": 2, "ts": 1.1},   # transition frame 1
            {"epoch": 2, "ts": 1.2},   # transition frame 2
            {"epoch": 2, "ts": 1.8},   # first driven frame
            {"epoch": 2, "ts": 1.9},
        ]}
        assert _first_driven_epoch2_frame_ts(obs, 3) == 1.8

    def test_only_bridge_frames_yields_none(self) -> None:
        from record import _first_driven_epoch2_frame_ts

        obs = {"frames": [{"epoch": 1, "ts": 0.5},
                          {"epoch": 2, "ts": 1.0}]}
        assert _first_driven_epoch2_frame_ts(obs, 1) is None
        assert _first_driven_epoch2_frame_ts(obs, 3) is None


# ────────────────────────────── analyzer: transition-aware time mapping


def _write_jpg(path: str, openness: float) -> None:
    from mouth_proxy import render_face

    ok, buf = cv2.imencode(".jpg", render_face(openness, t_s=0.3))
    assert ok
    buf.tofile(path)  # non-ASCII-path-safe write


def _write_session(root: str, *, transition_frames: int = 0,
                   with_driven: bool = True) -> str:
    """Fabricate a recorded interrupted session (meta + jpg frames).

    4 epoch-1 frames (openness 0.8 — mid-speech), then 3 transition
    frames (0.5/0.25/0.05) when ``transition_frames=3``, then 4 driven
    epoch-2 frames (0.6 …). Timeline ts values mirror the recorder:
    epoch-2 anchors are on the client wall clock, while the analyzer's
    ``u`` is the deterministic utterance clock.
    """
    sess = os.path.join(root, "interrupt_a_seed1")
    os.makedirs(os.path.join(sess, "frames"), exist_ok=True)

    if transition_frames:
        bridge = [0.5, 0.25, 0.05]
    else:
        bridge = []
    driven = [0.6, 0.3, 0.7, 0.1]
    timeline: list[dict] = []
    idx = 0
    for op in [0.8] * 4:  # epoch-1, ts_rel 0.50..0.62
        fname = f"f{idx + 1:05d}.jpg"
        _write_jpg(os.path.join(sess, "frames", fname), op)
        timeline.append({"ts_rel": round(0.5 + idx * 0.04, 4), "epoch": 1,
                         "boundary": False, "file": fname})
        idx += 1
    for j, op in enumerate(bridge):  # transition, client ts 1.06..1.14
        fname = f"f{idx + 1:05d}.jpg"
        _write_jpg(os.path.join(sess, "frames", fname), op)
        timeline.append({"ts_rel": round(1.06 + j * 0.04, 4),
                         "epoch": 2, "boundary": j == 0,
                         "file": fname})
        idx += 1
    for j, op in enumerate(driven):  # driven utt2, client ts 1.80..1.92
        fname = f"f{idx + 1:05d}.jpg"
        _write_jpg(os.path.join(sess, "frames", fname), op)
        timeline.append({"ts_rel": round(1.8 + j * 0.04, 4),
                         "epoch": 2, "boundary": False, "file": fname})
        idx += 1

    meta: dict = {
        "kind": "interrupt",
        "avatar_id": "a",
        "seed": 1,
        "frame_period_s": 0.04,
        "interrupt_sent_s": 1.02,
        "cut_in_utterance_s": 1.0,
        "utt2_start_s": 1.06 if transition_frames else 1.8,
        "transition_frames": transition_frames,
        "frame_timeline": timeline,
        "utterance": {"events": []},
    }
    if with_driven:
        meta["utt2_driven_start_s"] = 1.8
    with open(os.path.join(sess, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return sess


@_NEED_CV2
class TestLoadSeriesTransitionMapping:
    def test_transition_frames_map_to_cut_clock(self, tmp_path) -> None:
        import analyze as az

        sess = _write_session(str(tmp_path), transition_frames=3)
        s = az.load_series(sess)
        period = s["period"]
        # Epoch-1: deterministic index clock.
        np.testing.assert_allclose(s["u"][:4], [0.0, 0.04, 0.08, 0.12])
        # Transition frames i: cut_u + (i+1)·period → 1.04, 1.08, 1.12.
        np.testing.assert_allclose(s["u"][4:7], [1.04, 1.08, 1.12])
        # Driven frames: u2_0 = 1.8 − 0.5 = 1.3, then +k·period.
        np.testing.assert_allclose(
            s["u"][7:], [1.3, 1.34, 1.38, 1.42], atol=1e-9
        )
        assert period == pytest.approx(0.04)

    def test_transition_openness_eases_shut_in_u_order(self, tmp_path) -> None:
        import analyze as az

        sess = _write_session(str(tmp_path), transition_frames=3)
        s = az.load_series(sess)
        # The bridge eases the mouth shut right after the cut, then the
        # driven utterance resumes: openness dips to ~closed in slot 3.
        ops = s["op"]
        assert ops[3] > 0.5          # last pre-cut frame still speaking
        assert ops[6] < 0.15         # α=1 → closed neutral target
        assert ops[7] > 0.3          # driven utt-2 speech resumes

    def test_hard_cut_backward_compatible(self, tmp_path) -> None:
        import analyze as az

        sess = _write_session(str(tmp_path), transition_frames=0)
        s = az.load_series(sess)
        # No bridge: epoch-2 maps from utt2_start_s (P-B semantics).
        np.testing.assert_allclose(s["u"][4:], [1.3, 1.34, 1.38, 1.42],
                                   atol=1e-9)

    def test_legacy_meta_without_driven_field(self, tmp_path) -> None:
        import analyze as az

        sess = _write_session(str(tmp_path), transition_frames=0,
                              with_driven=False)
        s = az.load_series(sess)
        np.testing.assert_allclose(s["u"][4:], [1.3, 1.34, 1.38, 1.42],
                                   atol=1e-9)


# ───────────────────────────────────────────── compare: paired ablation


def _write_metrics(root: str, avatar_id: str, seed: int, *,
                   mutation: float, v_apt: float, mismatch: float,
                   half_syl: bool, resume_ms: float) -> None:
    sess = os.path.join(root, f"interrupt_{avatar_id}_seed{seed}")
    os.makedirs(sess, exist_ok=True)
    with open(os.path.join(sess, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "kind": "interrupt",
            "avatar_id": avatar_id,
            "seed": seed,
            "v_apt": {"apt_ms": v_apt},
            "smoothness": {"mutation": mutation, "ratio": 9.9,
                           "freeze_frames": 0},
            "mismatch": {"rate": mismatch},
            "half_syllable": {"flag": half_syl},
            "latency": {"resume_ms": resume_ms},
        }, f)


def _datasets(tmp_path, *, mutation_t: float) -> tuple[str, str]:
    hard = os.path.join(str(tmp_path), "hard")
    trans = os.path.join(str(tmp_path), "trans")
    for i in range(1, 4):
        _write_metrics(hard, "a", i, mutation=1.0, v_apt=200.0,
                       mismatch=0.6, half_syl=bool(i % 2), resume_ms=500.0)
        _write_metrics(trans, "a", i, mutation=mutation_t, v_apt=80.0,
                       mismatch=0.2, half_syl=bool(i % 2), resume_ms=490.0)
    _write_metrics(hard, "b", 9, mutation=2.0, v_apt=300.0,
                   mismatch=0.8, half_syl=True, resume_ms=600.0)
    return hard, trans


class TestPairedCompare:
    def test_gate_passes_on_big_improvement(self, tmp_path) -> None:
        from report import paired_compare

        hard, trans = _datasets(tmp_path, mutation_t=0.4)
        rep = paired_compare(hard, trans)
        assert rep["n_paired_sessions"] == 3
        assert rep["only_in_hard"] == [["b", 9]]
        g = rep["gates"]
        assert g["plan_gate_mutation"]["value"] == 60.0
        assert g["plan_gate_mutation"]["pass"] is True
        assert g["semantic_gate_mismatch"]["pass"] is True
        m = rep["metrics"]
        assert m["mutation"]["hard"]["p50"] == 1.0
        assert m["mutation"]["trans"]["p50"] == 0.4
        assert m["v_apt_ms"]["improve_p50_pct"] == 60.0

    def test_gate_fails_on_small_improvement(self, tmp_path) -> None:
        from report import paired_compare

        hard, trans = _datasets(tmp_path, mutation_t=0.9)
        assert paired_compare(hard, trans)["gates"][
            "plan_gate_mutation"]["pass"] is False

    def test_half_syllable_determinism(self, tmp_path) -> None:
        from report import paired_compare

        hard, trans = _datasets(tmp_path, mutation_t=0.4)
        rep = paired_compare(hard, trans)
        # Same cut points ⇒ identical flags across arms (incl. the
        # unpaired b/9 session, which is excluded from pairing).
        assert rep["half_syllable_determinism"]["matching"] == 3
        # Booleans carry no sign test.
        assert "sign_test_p" not in rep["metrics"]["half_syllable"]

    def test_sign_test_extreme_and_ties(self) -> None:
        from report import _sign_test_p

        # All 6 pairs improved → exact two-sided p = 2·(1/64) = 0.03125.
        assert _sign_test_p([5, 4, 3, 2, 1.2, 1.5], [1, 1, 1, 1, 1, 1]) == 0.0312
        # Ties are excluded: the two (1,1) pairs shrink n to 4 wins → 2/16.
        assert _sign_test_p([5, 4, 3, 2, 1, 1], [1, 1, 1, 1, 1, 1]) == 0.125
        # All ties → no evidence.
        assert _sign_test_p([1, 1], [1, 1]) is None


class TestCompareCli:
    def test_end_to_end_writes_report(self, tmp_path) -> None:
        from report import main

        hard, trans = _datasets(tmp_path, mutation_t=0.3)
        out = os.path.join(str(tmp_path), "cmp")
        rc = main(["compare", "--hard", hard, "--trans", trans,
                   "--out", out])
        assert rc == 0  # gate passed
        assert os.path.isfile(os.path.join(out, "compare.json"))
        md = open(os.path.join(out, "compare.md"), encoding="utf-8").read()
        assert "PASS" in md
        assert "V-APT" in md

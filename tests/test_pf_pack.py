# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-F unit tests: slot picking, survey-page generation, rating analysis."""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "interruption_eval"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

pf_pack = importlib.import_module("pf_pack")
pf_survey = importlib.import_module("pf_survey")
pf_analyze = importlib.import_module("pf_analyze")

# half-syllable flags mirroring the recorded yongen r1 distribution
# (seed -> flag); t follows the recorder grid seed s -> 0.5 + (s-1)*0.21.
_HALF = {s: s not in (3, 4, 6, 8, 12, 19, 20) for s in range(1, 21)}


def _make_dataset(root: Path) -> None:
    """Fake three-arm r1 dataset: metrics.json with half_syllable only."""
    for sub in ("pe_real_hard_r1", "pe_real_trans_r1", "lt_yongen"):
        for seed, half in _HALF.items():
            t = 0.5 + (seed - 1) * 0.21
            if sub == "lt_yongen":
                name = f"interrupt_yongen_seed{seed}_t{t:.2f}s_r1"
            else:
                name = f"interrupt_yongen_seed{seed}_t{t:.2f}s"
            d = root / "data" / sub / name
            d.mkdir(parents=True, exist_ok=True)
            flag = 1.0 if half else 0.0
            (d / "metrics.json").write_text(
                json.dumps({"half_syllable": {"flag": flag}}), encoding="utf-8")


def _three_arm_candidates(root: Path) -> dict:
    cands: dict = {}
    for arm in ("hard", "trans"):
        cands.update(pf_pack.self_dirs(root / "data", arm))
    cands.update(pf_pack.lt_dirs(root / "data"))
    return cands


class TestPickSlots:
    def test_deterministic_and_in_range(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path)
        hard = pf_pack.self_dirs(tmp_path / "data", "hard")
        lt = pf_pack.lt_dirs(tmp_path / "data")
        assert len(hard) == 20 and len(lt) == 20
        cands = _three_arm_candidates(tmp_path)
        assert len(cands) == 20  # same (seed, t) keys across the three arms
        assert set(hard) == set(lt) == set(cands)

        s1 = pf_pack.pick_slots(cands)
        s2 = pf_pack.pick_slots(cands)
        assert s1 == s2
        assert len({s["name"] for s in s1}) == 4
        assert len({s["seed"] for s in s1}) == 4
        for slot in s1:
            lo, hi = next(
                sl["t_range"] for sl in pf_pack.SLOTS
                if sl["name"] == slot["name"])
            assert lo <= float(slot["t"]) < hi
            assert slot["half_syllable"] == _HALF[slot["seed"]]

    def test_expected_picks(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path)
        picks = {s["name"]: s["seed"]
                 for s in pf_pack.pick_slots(_three_arm_candidates(tmp_path))}
        # median-t session within each slot (see pick_slots med logic)
        assert picks == {"early_half": 2, "mid_half": 9,
                         "late_half": 15, "mid_nohalf": 8}


class TestSurveyPage:
    def test_manifest_embedded_and_clips_copied(self, tmp_path: Path) -> None:
        pack = tmp_path / "pack"
        (pack / "clips").mkdir(parents=True)
        manifest = {"avatar": "yongen", "scale": "1-5",
                    "dimensions": ["naturalness", "comfort"],
                    "groups": [{"id": "g11", "clips": ["c01.mp4", "c02.mp4"]},
                               {"id": "g12", "clips": ["c03.mp4", "c04.mp4"]}]}
        for g in manifest["groups"]:
            for c in g["clips"]:
                (pack / "clips" / c).write_bytes(b"fake")
        pf_survey.write_survey_page(pack / "survey", manifest)
        html = (pack / "survey" / "index.html").read_text(encoding="utf-8")
        assert '"g11"' in html and '"c01.mp4"' in html
        for g in manifest["groups"]:
            for c in g["clips"]:
                assert (pack / "survey" / "clips" / c).exists()


def _make_pack_and_ratings(tmp_path: Path, n_subjects: int = 20) -> Path:
    """pack_key.json + synthetic ratings with a +1 trans-over-hard effect."""
    comps = ["hard_vs_trans", "hard_vs_lt", "trans_vs_lt"]
    slots = ["early_half", "mid_half", "late_half", "mid_nohalf"]
    ids = [f"c{k:02d}" for k in range(1, 25)]
    pairs = []
    clip2meta: dict[str, dict] = {}
    cid = 0
    for s_i, slot in enumerate(slots, 1):
        for c_i, comp in enumerate(comps, 1):
            a, b = comp.split("_vs_")
            group = f"g{s_i}{c_i}"
            arms = {}
            for arm in (a, b):
                clip = ids[cid]
                cid += 1
                arms[arm] = {"clip": clip, "dir": f"fake/{arm}"}
                clip2meta[clip] = {"group": group, "arm": arm,
                                   "comparison": comp, "slot": slot}
            pairs.append({"group_id": group, "slot": slot, "seed": s_i,
                          "t": f"{s_i}.11", "comparison": comp, "arms": arms})
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "pack_key.json").write_text(
        json.dumps({"seed": 1, "slots": [], "pairs": pairs}),
        encoding="utf-8")

    rng = random.Random(7)
    ratings_dir = tmp_path / "ratings"
    ratings_dir.mkdir()
    for p in range(1, n_subjects + 1):
        trials = []
        for clip, meta in sorted(clip2meta.items()):
            if meta["comparison"] == "hard_vs_trans":
                base = 3 if meta["arm"] == "hard" else 4
            else:
                base = rng.randint(1, 5)
            trials.append({"group": meta["group"], "clip": clip,
                           "shown_as": "A", "naturalness": base,
                           "comfort": base})
        (ratings_dir / f"pf_ratings_P{p:02d}.json").write_text(
            json.dumps({"participant_id": f"P{p:02d}",
                        "finished_at": "2026-09-17T00:00:00Z",
                        "trials": trials}),
            encoding="utf-8")
    return pack_dir


class TestAnalyze:
    def test_trans_effect_detected(self, tmp_path: Path) -> None:
        pack_dir = _make_pack_and_ratings(tmp_path)
        out = tmp_path / "out"
        rc = pf_analyze.analyze(
            tmp_path / "ratings", pack_dir / "pack_key.json", out)
        assert rc == 0
        stats = json.loads((out / "pf_stats.json").read_text(encoding="utf-8"))
        nat = stats["dimensions"]["naturalness"]["hard_vs_trans"]
        assert nat["n_pairs"] == 80  # 20 subjects x 4 slots
        # diff = hard(3) - trans(4) = -1: negative => trans (arm_b) favoured
        assert nat["median_diff"] == -1.0
        assert nat["p_holm"] < 0.05
        assert nat["direction_matches"] is True
        com = stats["dimensions"]["comfort"]["hard_vs_trans"]
        assert com["p_holm"] < 0.05 and com["direction_matches"] is True
        assert (out / "pf_report.md").exists()

    def test_duplicate_participant_rejected(self, tmp_path: Path) -> None:
        pack_dir = _make_pack_and_ratings(tmp_path, n_subjects=2)
        ratings = tmp_path / "ratings"
        (ratings / "pf_ratings_P03.json").write_text(
            (ratings / "pf_ratings_P01.json").read_text(encoding="utf-8"),
            encoding="utf-8")
        with pytest.raises(SystemExit, match="duplicate participant"):
            pf_analyze.analyze(ratings, pack_dir / "pack_key.json",
                               tmp_path / "out2")

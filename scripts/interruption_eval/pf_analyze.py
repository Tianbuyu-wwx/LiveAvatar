# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-F3: aggregate paired survey ratings + Wilcoxon significance (P-F gate).

Input: ``pack_key.json`` (the experimenter-private clip identity mapping
written by pf_pack.py) and the collected ``pf_ratings_P*.json`` files from
the survey page. Output: per-dimension two-sided Wilcoxon signed-rank tests
(Holm-corrected within each dimension's 3 comparisons), rank-biserial effect
sizes, and a direction-consistency check against the P-E5 objective metrics
(plan gate: p < 0.05 AND the effect direction agrees with the objective
table).

Pairing unit: (participant x slot) — each comparison pools its 4 slots'
paired A/B ratings across participants (n = 4 x n_subjects per comparison).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lt_compare import holm, rank_biserial  # noqa: E402

DIMS = ("naturalness", "comfort")

# Direction expectations from the P-E5 objective main table
# (data/interruption_eval/pe5_main), all lower-is-better metrics:
# mismatch / resume / V-APT. "strong" = decisive objective gap,
# "weak" = mixed or near-tied objective evidence.
EXPECTED: dict[str, dict[str, dict[str, str]]] = {
    "naturalness": {
        "hard_vs_trans": {"favors": "trans", "via": "mismatch 0.71 vs 0.11, "
                          "resume 602 vs 109 ms", "strength": "strong"},
        "hard_vs_lt": {"favors": "hard", "via": "mismatch ~tied (0.71/0.73), "
                       "resume 602 vs 1948 ms, LT audio tail 2.6 s",
                       "strength": "weak"},
        "trans_vs_lt": {"favors": "trans", "via": "mismatch 0.11 vs 0.73, "
                        "resume 109 vs 1948 ms", "strength": "strong"},
    },
    "comfort": {
        "hard_vs_trans": {"favors": "trans", "via": "resume 602 vs 109 ms, "
                          "bridge eases the mouth shut", "strength": "strong"},
        "hard_vs_lt": {"favors": "hard", "via": "LT reacts ~2.6 s later "
                       "(resume + audio tail)", "strength": "strong"},
        "trans_vs_lt": {"favors": "trans", "via": "mismatch 0.11 vs 0.73, "
                        "resume 109 vs 1948 ms", "strength": "strong"},
    },
}


def load_pack(path: Path) -> dict[str, Any]:
    pack = json.loads(path.read_text(encoding="utf-8"))
    clip2meta: dict[str, dict[str, Any]] = {}
    for p in pack["pairs"]:
        for arm, info in p["arms"].items():
            clip2meta[info["clip"]] = {
                "group": p["group_id"], "arm": arm,
                "comparison": p["comparison"], "slot": p["slot"],
                "seed": p["seed"], "t": p["t"],
            }
    return {"pack": pack, "clip2meta": clip2meta}


def load_ratings(paths: list[Path]) -> dict[str, dict[tuple[str, str], dict]]:
    """participant_id -> {(group, clip): {naturalness, comfort}}."""
    subjects: dict[str, dict[tuple[str, str], dict]] = {}
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        pid = data["participant_id"]
        if pid in subjects:
            raise SystemExit(f"duplicate participant id: {pid} ({p.name})")
        rows: dict[tuple[str, str], dict] = {}
        for t in data["trials"]:
            key = (t["group"], t["clip"])
            if key in rows:
                raise SystemExit(f"{pid}: duplicate rating for {key}")
            rows[key] = {"naturalness": t["naturalness"],
                         "comfort": t["comfort"]}
        subjects[pid] = rows
    return subjects


def validate(subjects: dict, clip2meta: dict[str, dict]) -> None:
    n_clips = len(clip2meta)
    for pid, rows in subjects.items():
        bad = [k for k in rows if k[1] not in clip2meta]
        if bad:
            raise SystemExit(f"{pid}: unknown clips {bad[:3]}")
        for clip, meta in clip2meta.items():
            key = (meta["group"], clip)
            if key not in rows:
                raise SystemExit(f"{pid}: missing rating for {key}")
            for dim in DIMS:
                v = rows[key][dim]
                if not isinstance(v, (int, float)) or not 1 <= v <= 5:
                    raise SystemExit(f"{pid}: bad {dim} for {key}: {v}")
        if len(rows) != n_clips:
            raise SystemExit(f"{pid}: {len(rows)} ratings, expected {n_clips}")


def paired_scores(
    subjects: dict, clip2meta: dict, comparison: str, dim: str
) -> list[tuple[float, float]]:
    """(arm_a, arm_b) rating pairs over (participant x slot)."""
    arm_a, arm_b = comparison.split("_vs_")
    out: list[tuple[float, float]] = []
    for rows in subjects.values():
        by_comp: dict[str, dict[str, float]] = {}
        for (_group, clip), r in rows.items():
            meta = clip2meta[clip]
            if meta["comparison"] != comparison:
                continue
            by_comp.setdefault(meta["slot"], {})[meta["arm"]] = r[dim]
        for slot, arms in sorted(by_comp.items()):
            if len(arms) != 2:
                raise SystemExit(
                    f"{comparison}/{slot}: incomplete pair {sorted(arms)}")
            out.append((arms[arm_a], arms[arm_b]))
    return out


def analyze(ratings_dir: Path, pack_path: Path, out_dir: Path) -> int:
    loaded = load_pack(pack_path)
    clip2meta = loaded["clip2meta"]
    paths = sorted(ratings_dir.glob("pf_ratings_*.json"))
    if not paths:
        raise SystemExit(f"no pf_ratings_*.json under {ratings_dir}")
    subjects = load_ratings(paths)
    validate(subjects, clip2meta)
    print(f"subjects: {len(subjects)}; clips: {len(clip2meta)}")

    comparisons = sorted({m["comparison"] for m in clip2meta.values()})
    report: dict[str, Any] = {
        "n_subjects": len(subjects),
        "n_pairs_per_comparison": len(subjects) * 4,
        "dimensions": {},
    }
    lines = [
        "# P-F 用户研究：配对评分统计（Wilcoxon 符号秩）",
        "",
        f"- 被试 {len(subjects)} 人 × 12 组 × 2 片段；配对单元 = 被试 × 槽位"
        f"（每比较 n={len(subjects) * 4}）。",
        "- 每维度内 3 个比较做 Holm-Bonferroni 校正；双侧检验，"
        "zero_method='wilcox'；效应量为配对秩双列 r。",
        "- 方向参照来自 P-E5 客观主表（mismatch/resume/V-APT，均为低优）。",
        "",
    ]
    for dim in DIMS:
        fam: dict[str, dict] = {}
        pvals: list[float] = []
        for comp in comparisons:
            pairs = paired_scores(subjects, clip2meta, comp, dim)
            xs = np.array([p[0] for p in pairs], dtype=float)
            ys = np.array([p[1] for p in pairs], dtype=float)
            diff = xs - ys
            res = stats.wilcoxon(xs, ys, zero_method="wilcox",
                                 alternative="two-sided")
            arm_a, arm_b = comp.split("_vs_")
            fam[comp] = {
                "n_pairs": len(pairs),
                "mean_a": round(float(xs.mean()), 3),
                "mean_b": round(float(ys.mean()), 3),
                "median_diff": round(float(np.median(diff)), 3),
                "statistic": float(res.statistic),
                "p": float(res.pvalue),
                "rank_biserial": round(rank_biserial(diff), 3),
                "expected": EXPECTED[dim][comp],
            }
            pvals.append(fam[comp]["p"])
        adj = holm(pvals)
        for comp, a in zip(comparisons, adj, strict=True):
            fam[comp]["p_holm"] = a
            fam[comp]["significant_0.05"] = bool(a < 0.05)
            exp = fam[comp]["expected"]
            a_name, b_name = comp.split("_vs_")
            fav = a_name if fam[comp]["median_diff"] > 0 else b_name
            fam[comp]["direction_matches"] = bool(fav == exp["favors"])
        report["dimensions"][dim] = fam

        lines.append(f"## {dim}（{'自然度' if dim == 'naturalness' else '打断舒适度'}）")
        lines.append("")
        lines.append("| 比较 | 前者 mean | 后者 mean | 中位差(前−后) | W | p | "
                     "p(Holm) | r | 预期方向 | 方向一致 | 显著 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for comp in comparisons:
            r = fam[comp]
            a_name, b_name = comp.split("_vs_")
            lines.append(
                f"| {comp} | {r['mean_a']:.2f}（{a_name}） "
                f"| {r['mean_b']:.2f}（{b_name}） "
                f"| {r['median_diff']:+.2f} | {r['statistic']:.0f} "
                f"| {r['p']:.2e} | {r['p_holm']:.2e} | {r['rank_biserial']:+.2f} "
                f"| {r['expected']['favors']}（{r['expected']['strength']}） "
                f"| {'是' if r['direction_matches'] else '否'} "
                f"| {'是' if r['significant_0.05'] else '否'} |")
        lines.append("")
        for comp in comparisons:
            r = fam[comp]
            lines.append(f"- {comp}：客观参照（{r['expected']['via']}）。")
        lines.append("")

    gate_ok = all(
        r["significant_0.05"] and r["direction_matches"]
        for dim in DIMS for r in report["dimensions"][dim].values()
        if r["expected"]["strength"] == "strong"
    )
    report["gate_strong_expectations"] = "PASS" if gate_ok else "FAIL"
    lines.append(f"门禁（强预期比较全部显著且方向一致）："
                 f"**{report['gate_strong_expectations']}**")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pf_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "pf_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_dir / 'pf_report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratings", default="data/interruption_eval/pf_pack/ratings",
                    help="dir of collected pf_ratings_P*.json files")
    ap.add_argument("--pack",
                    default="data/interruption_eval/pf_pack/pack_key.json")
    ap.add_argument("--out", default="data/interruption_eval/pf_pack/pf_stats")
    args = ap.parse_args(argv)
    return analyze(Path(args.ratings), Path(args.pack), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())

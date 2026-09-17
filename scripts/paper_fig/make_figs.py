# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-G paper figures: render every figure from REAL experiment artifacts.

Sources (all produced by scripts/interruption_eval, never hand-edited):
- P-A baseline        data/interruption_eval/dataset/report.json
- P-C valley ablation data/interruption_eval/dataset/valley_ablation.json
- P-D hard/trans      data/interruption_eval/pd_compare/compare.json
- P-E main study      data/interruption_eval/pe5_main/pe5_stats.json

Outputs PDF (paper) + PNG (preview) into docs/paper/figs/ by default.
Figure text is English (camera-ready convention); Chinese captions live
in docs/paper/*.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interruption_eval"

ARM_COLORS = {"hard": "#d95f02", "trans": "#1b9e77", "lt": "#7570b3"}
ARM_LABELS = {"hard": "Hard-cut (ours)", "trans": "Transition (ours)",
              "lt": "LiveTalking"}


def _load(rel: str) -> dict:
    return json.loads((DATA / rel).read_text(encoding="utf-8"))


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_pa_baseline(out: Path) -> None:
    """P-A: hard-cut artifact baseline (60 sessions, mouth_proxy avatars)."""
    r = _load("dataset/report.json")["overall"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    axes[0].bar(["p50", "p95"],
                [r["v_apt_ms"]["p50"], r["v_apt_ms"]["p95"]],
                color="#d95f02")
    axes[0].set_ylabel("V-APT (ms)")
    axes[0].set_title("(a) Visual articulation perturbation")
    axes[1].bar(["p50", "p95"],
                [r["mutation"]["p50"], r["mutation"]["p95"]],
                color="#7570b3")
    axes[1].set_ylabel("|Δopenness|")
    axes[1].set_title(f"(b) First jump (ratio p50 = {r['ratio']['p50']:.1f}x)")
    axes[2].bar(["mismatch", "half-syll"],
                [r["mismatch_rate"]["mean"], r["half_syllable_rate"]],
                color="#1b9e77")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("rate")
    axes[2].set_title("(c) Semantic damage of hard cuts")
    fig.suptitle("P-A baseline: hard-cut artifacts (n=60, 0 errors)",
                 y=1.04)
    _save(fig, out, "fig_pa_baseline")


def fig_valley(out: Path) -> None:
    """P-C: energy-valley cut ablation (60 sessions)."""
    a = _load("dataset/valley_ablation.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
    sweep = a["sweep"]
    xs = [s["search_ms"] for s in sweep]
    ys = [s["hit_rate"] * 100 for s in sweep]
    axes[0].plot(xs, ys, "o-", color="#1b9e77")
    axes[0].axhline(a["gate"] * 100, ls="--", c="gray", lw=1)
    axes[0].text(xs[-1], a["gate"] * 100 - 6, "gate 80%", ha="right",
                 color="gray", fontsize=9)
    axes[0].set_xlabel("search window (ms)")
    axes[0].set_ylabel("valley hit rate (%)")
    axes[0].set_title(f"(a) Sweep (default {a['search_ms']:.0f} ms)")
    methods = ["valley", "hard", "random"]
    cats = ["pause", "soft", "vowel"]
    width = 0.26
    for i, m in enumerate(methods):
        src = a[f"{m}_landing"] if m == "valley" else a[f"{m}_cut_landing"]
        vals = [src[c] * 100 for c in cats]
        axes[1].bar([x + (i - 1) * width for x in range(len(cats))],
                    vals, width, label=m, color=list(ARM_COLORS.values())[i])
    axes[1].set_xticks(range(len(cats)), cats)
    axes[1].set_ylabel("landing rate (%)")
    axes[1].set_title("(b) Cut-point landing distribution")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, out, "fig_valley")


def fig_pd_compare(out: Path) -> None:
    """P-D: hard-cut vs transition ablation (60 paired sessions)."""
    m = _load("pd_compare/compare.json")["metrics"]
    panels = [
        ("v_apt_ms", "V-APT (ms)", "p50"),
        ("mutation", "first jump |Δopenness|", "p50"),
        ("mismatch_rate", "viseme mismatch rate", "mean"),
        ("resume_ms", "visual resume (ms)", "p50"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.7))
    for ax, (key, label, stat) in zip(axes, panels, strict=True):
        hard = m[key]["hard"][stat]
        trans = m[key]["trans"][stat]
        ax.bar(["hard", "trans"], [hard, trans],
               color=[ARM_COLORS["hard"], ARM_COLORS["trans"]])
        ax.set_title(f"{label} ({stat})", fontsize=9)
        for x, v in enumerate([hard, trans]):
            ax.text(x, v, f"{v:.3g}" if v < 10 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.18 * max(hard, trans))
    fig.suptitle(
        "P-D ablation: hard cut vs transition bridge (n=60 pairs)", y=1.05)
    _save(fig, out, "fig_pd_compare")


def fig_main(out: Path) -> None:
    """P-E: main study, 4 metrics x 3 arms, Wilcoxon-Holm vs LiveTalking."""
    s = _load("pe5_main/pe5_stats.json")
    desc, wil = s["descriptives"], s["arms"]
    panels = [
        ("v_apt_ms", "V-APT (ms)"),
        ("resume_ms", "visual resume (ms)"),
        ("mutation", "first jump |Δopenness|"),
        ("mismatch", "viseme mismatch rate"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
    for ax, (key, label) in zip(axes, panels, strict=True):
        d = desc[key]
        arms = ["hard", "trans", "lt"]
        means = [d[a]["mean"] for a in arms]
        stds = [d[a]["std"] for a in arms]
        ax.bar(arms, means, yerr=stds, capsize=3,
               color=[ARM_COLORS[a] for a in arms])
        # annotate Holm-adjusted significance of each ours-arm vs LT
        ymax = max(m + s for m, s in zip(means, stds, strict=True))
        for i, arm in enumerate(("hard", "trans")):
            w = wil[arm]["wilcoxon"][key]
            if w["significant_0.05"]:
                p = w["p_holm"]
                star = "***" if p < 1e-4 else ("**" if p < 1e-3 else "*")
                ax.text(i, means[i] + stds[i] + 0.04 * ymax, star,
                        ha="center", fontsize=9)
        ax.set_ylim(0, 1.18 * ymax)
        ax.set_xticks(range(3), ["Hard", "Trans", "LT"], fontsize=8)
        ax.set_title(label, fontsize=10)
    fig.suptitle(
        "P-E main study: ours vs LiveTalking "
        "(n=120 pairs/arm, Wilcoxon-Holm; * p<.05, ** p<.001, *** p<1e-4)",
        y=1.06)
    _save(fig, out, "fig_main")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/paper/figs")
    args = ap.parse_args(argv)
    out = Path(args.out)
    fig_pa_baseline(out)
    fig_valley(out)
    fig_pd_compare(out)
    fig_main(out)
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

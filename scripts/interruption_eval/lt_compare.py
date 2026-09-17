# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-E5: main comparison table + Wilcoxon signed-rank significance.

Pairs interrupt sessions by (avatar, seed, interrupt point, repeat) across
three system arms recorded under the SAME Design-B protocol, the same
deterministic utterances (``audio_synth`` seeds), the same 20-point
interrupt grid, and the shared ``metrics.py`` library:

  - LiveAvatar hard-cut arm   : ``pe_real_hard_r{k}``  (P-E2b, real GPU)
  - LiveAvatar transition arm : ``pe_real_trans_r{k}`` (P-D transition)
  - LiveTalking baseline      : ``lt_{avatar}``        (P-E4, same machine)

Per-arm sample: 2 avatars x 20 seeds x 3 repeats = 120 paired sessions.

Statistics: two-sided Wilcoxon signed-rank on paired per-session
differences (zero_method='wilcox'), Holm-Bonferroni correction within each
comparison family (4 cadence-robust metrics), matched-pairs rank-biserial effect size.
Self-side arm minus LiveTalking: negative differences favour LiveAvatar
for every metric here (all are lower-is-better).

Metric semantics (documented for the paper):
- ``v_apt_ms``      V-APT perturbation area, 1.2 s transition window, ZOH grid
                    (both arms; the P-B7 Spearman-validated 口径).
- ``resume_ms``     cut → first utterance-2-driven visual frame. Self side:
                    client wall clock (``meta.resume_ms``); LT side: offline
                    recording-derived (``visual_resume_ms``, xcorr-anchored).
- ``mutation``      freeze-aware first-jump |Δopenness| vs pre-cut baseline.
- ``ratio``         mutation / pre-cut baseline.
- ``freeze_frames`` frames held at the stale frame before the first jump.
- ``mismatch``      phoneme-vs-viseme bucket disagreement in [cut, cut+0.6 s]
                    against the ACTUAL post-cut audio model (self: pause then
                    utterance-2; LT: in-flight utterance-1 backlog renders out
                    — the system difference the table is measuring).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

AVATARS = ("yongen", "sun")
REPEATS = (1, 2, 3)
SEEDS = range(1, 21)

# metric key -> (label, self accessor, LT accessor); all lower-is-better.
# The family covers the four cadence-robust metrics: v_apt / resume /
# mutation / mismatch are computed on wall-clock grids or as openness
# magnitudes. ``ratio`` (jump/baseline) and ``freeze_frames`` are excluded
# from cross-system tests: the self-side recorder publishes frames in
# per-batch bursts (4 frames per 160 ms chunk, no frames in the gaps), so
# its pre-cut |Δ| baseline degenerates to 0 (ratio=None) and its
# "frames before first jump" counts published bursts, not wall time —
# while LiveTalking records at a steady 25 fps. They are artifacts of
# recording cadence, not of system behavior.
METRICS: list[tuple[str, str, Callable[[dict], float], Callable[[dict], float]]] = [
    ("v_apt_ms", "V-APT (ms)",
     lambda m: m["v_apt"]["apt_ms"], lambda m: m["v_apt_ms"]),
    ("resume_ms", "Visual resume (ms)",
     lambda m: m["latency"]["resume_ms"], lambda m: m["visual_resume_ms"]),
    ("mutation", "First jump |Δopenness|",
     lambda m: m["smoothness"]["mutation"], lambda m: m["mutation"]),
    ("mismatch", "Viseme mismatch rate",
     lambda m: m["mismatch"]["rate"], lambda m: m["mismatch_rate"]),
]


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def load_arm_rows(data_root: Path) -> dict[tuple, dict]:
    """LT interrupt rows keyed by (avatar, seed, t_str, repeat)."""
    rows: dict[tuple, dict] = {}
    for avatar in AVATARS:
        root = data_root / f"lt_{avatar}"
        for d in sorted(root.glob(f"interrupt_{avatar}_seed*_r*")):
            name = d.name  # interrupt_{avatar}_seed{S}_t{T}s_r{k}
            body = name.removeprefix(f"interrupt_{avatar}_seed")
            seed_str, rest = body.split("_", 1)
            t_str, r_str = rest.rsplit("_r", 1)
            t_str = t_str.removeprefix("t").removesuffix("s")
            key = (avatar, int(seed_str), t_str, int(r_str))
            rows[key] = _load_json(d / "metrics.json")
    return rows


def load_self_rows(data_root: Path, arm: str) -> dict[tuple, dict]:
    """pe_real_{arm}_r{k} rows keyed by (avatar, seed, t_str, repeat)."""
    rows: dict[tuple, dict] = {}
    for k in REPEATS:
        root = data_root / f"pe_real_{arm}_r{k}"
        for d in sorted(root.glob("interrupt_*_seed*")):
            body = d.name.removeprefix("interrupt_")  # {avatar}_seed{S}_t{T}s
            avatar, rest = body.split("_seed", 1)
            seed_str, t_part = rest.split("_t", 1)
            t_str = t_part.removesuffix("s")
            if avatar not in AVATARS:
                continue
            rows[(avatar, int(seed_str), t_str, k)] = _load_json(d / "metrics.json")
    return rows


def pair_rows(lt: dict[tuple, dict], self_arm: dict[tuple, dict]) -> list[tuple]:
    """Intersect keys; fail loudly when pairing is incomplete."""
    missing_lt = sorted(set(self_arm) - set(lt))
    missing_self = sorted(set(lt) - set(self_arm))
    if missing_lt or missing_self:
        raise SystemExit(
            f"pairing incomplete: {len(missing_lt)} missing on LT side, "
            f"{len(missing_self)} missing on self side; "
            f"first few: {missing_lt[:3] + missing_self[:3]}")
    return sorted(set(lt) & set(self_arm))


def describe(values: list[float]) -> dict[str, float]:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone enforcement)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(x: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for non-zero differences."""
    d = x[x != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def wilcoxon_family(
    pairs: list[tuple], lt: dict[tuple, dict], self_arm: dict[tuple, dict]
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    pvals: list[float] = []
    per_metric: dict[str, dict] = {}
    for key, _label, self_get, lt_get in METRICS:
        xs = np.array([self_get(self_arm[k]) for k in pairs], dtype=float)
        ys = np.array([lt_get(lt[k]) for k in pairs], dtype=float)
        finite = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[finite], ys[finite]
        diff = xs - ys
        res = stats.wilcoxon(xs, ys, zero_method="wilcox",
                             alternative="two-sided")
        row = {
            "n_pairs": len(pairs),
            "n_nonzero": int(np.count_nonzero(diff)),
            "median_diff_self_minus_lt": float(np.median(diff)),
            "mean_diff_self_minus_lt": float(diff.mean()),
            "statistic": float(res.statistic),
            "p": float(res.pvalue),
            "rank_biserial": rank_biserial(diff),
        }
        per_metric[key] = row
        pvals.append(row["p"])
    adj = holm(pvals)
    for (key, _l, _s, _t), a in zip(METRICS, adj):
        per_metric[key]["p_holm"] = a
        per_metric[key]["significant_0.05"] = bool(a < 0.05)
    out.update(per_metric)
    return out


def fmt_desc(d: dict[str, float]) -> str:
    return f"{d['mean']:.2f}±{d['std']:.2f}"


def fmt_med(d: dict[str, float]) -> str:
    return f"{d['median']:.2f} [{d['q25']:.2f}, {d['q75']:.2f}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/interruption_eval")
    ap.add_argument("--out", default="data/interruption_eval/pe5_main")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lt_rows = load_arm_rows(data_root)
    arms = {
        "hard": load_self_rows(data_root, "hard"),
        "trans": load_self_rows(data_root, "trans"),
    }
    print(f"LT rows: {len(lt_rows)}; self hard: {len(arms['hard'])}; "
          f"self trans: {len(arms['trans'])}")

    report: dict[str, Any] = {"n_expected_per_arm": 120, "arms": {}}
    lines: list[str] = []
    lines.append("# P-E5 主对比表（LiveAvatar vs LiveTalking）")
    lines.append("")
    lines.append("协议：Design-B，同一确定性语音（seed 1-20）、同一 20 点打断"
                 "网格（0.50-4.50 s）、同一台机器；每臂 2 avatar × 20 seed × "
                 "3 repeat = 120 会话。V-APT 口径：1.2 s 转换窗 + ZOH 网格。")
    lines.append("")

    # ── descriptive main table ──
    desc: dict[str, dict[str, dict[str, float]]] = {}
    for key, label, self_get, lt_get in METRICS:
        desc[key] = {
            "hard": describe([self_get(m) for m in arms["hard"].values()]),
            "trans": describe([self_get(m) for m in arms["trans"].values()]),
            "lt": describe([lt_get(m) for m in lt_rows.values()]),
        }
        _ = label  # label used in md below
    lines.append("## 主表：描述统计（mean±std；median [IQR]）")
    lines.append("")
    lines.append("| 指标 | LiveAvatar-hard | LiveAvatar-trans | LiveTalking |")
    lines.append("|---|---|---|---|")
    for key, label, _s, _t in METRICS:
        d = desc[key]
        lines.append(
            f"| {label} | {fmt_desc(d['hard'])} ({fmt_med(d['hard'])}) "
            f"| {fmt_desc(d['trans'])} ({fmt_med(d['trans'])}) "
            f"| {fmt_desc(d['lt'])} ({fmt_med(d['lt'])}) |")
    lines.append("")

    # ── Wilcoxon per comparison ──
    wilcox: dict[str, dict[str, dict]] = {}
    for arm in ("hard", "trans"):
        pairs = pair_rows(lt_rows, arms[arm])
        fam = wilcoxon_family(pairs, lt_rows, arms[arm])
        wilcox[arm] = fam
        report["arms"][arm] = {"n_pairs": len(pairs), "wilcoxon": fam}
        lines.append(f"## Wilcoxon 符号秩：LiveAvatar-{arm} vs LiveTalking"
                     f"（n={len(pairs)} 配对，双侧，Holm 校正 ×{len(METRICS)}）")
        lines.append("")
        lines.append("| 指标 | 中位差（self−LT） | W | p（原始） | p（Holm） | "
                     "秩双列 r | 显著 0.05 |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, label, _s, _t in METRICS:
            r = fam[key]
            lines.append(
                f"| {label} | {r['median_diff_self_minus_lt']:+.3f} "
                f"| {r['statistic']:.1f} | {r['p']:.2e} "
                f"| {r['p_holm']:.2e} | {r['rank_biserial']:+.2f} "
                f"| {'是' if r['significant_0.05'] else '否'} |")
        lines.append("")

    # ── half-syllable rates + LT audio decomposition ──
    def half_rate(rows: dict[tuple, dict]) -> float:
        vals = [bool(m["half_syllable"]["flag"]) for m in rows.values()
                if m.get("half_syllable")]
        return sum(vals) / len(vals) if vals else float("nan")

    lt_tails = [m["audio_tail_after_cut_ms"] for m in lt_rows.values()
                if m.get("audio_tail_after_cut_ms") is not None]
    lines.append("## 附加分解")
    lines.append("")
    lines.append(f"- 半音节切断率（打断落在非停顿音素内）："
                 f"hard {half_rate(arms['hard']):.1%}, "
                 f"trans {half_rate(arms['trans']):.1%}, "
                 f"LT {half_rate(lt_rows):.1%}")
    lines.append(f"- LT 音频拖尾（cut→最后一个 utt1 可闻样本，flush_talk 只清"
                 f"输入队列、在途音频继续播出）：median "
                 f"{np.median(lt_tails):.0f} ms, mean {np.mean(lt_tails):.0f} ms"
                 f"（n={len(lt_tails)}）")
    lines.append("- 自研 hard 臂音频在 cut 即刻冲刷（detect→audio flush "
                 "0 ms，见 server_timeline）；trans 臂以过渡桥接帧缓合口型。")
    lines.append("")

    # per-avatar medians (appendix)
    lines.append("## 口径说明")
    lines.append("")
    lines.append("- V-APT / viseme mismatch 在墙钟 ZOH 网格上计算，visual "
                 "resume 为墙钟时长，first jump 为 openness 幅值——四者均"
                 "与录制帧节奏无关，跨系统可比。")
    lines.append("- ratio（跳变/基线）与 freeze_frames 不参与跨系统检验："
                 "自研侧录制器按批次突发发布帧（每 160 ms chunk 4 帧，间隙"
                 "无帧），pre-cut 基线退化为 0（ratio=None），且\"首跳前帧"
                 "数\"统计的是发布批次而非墙钟时间；LiveTalking 为匀速 25 fps"
                 "录制。二者是录制节奏伪影，不反映系统行为差异。")
    lines.append("- 开口度提取器两侧同源（YuNet 人脸框 + 几何嘴部 ROI + "
                 "暗腔占比 + 会话内 p2/p99 自校准），但暗腔亮度阈值不同："
                 "LT 臂（MuseTalk 真人脸录制）用 dark<90，自研渲染 avatar"
                 "（512×512）用 dark<150——渲染 avatar 口腔亮度远高于真人"
                 "录制，dark<90 下 sun 的腔体占比全零（阈值扫描见会话诊断），"
                 "150 对 yongen/sun 均保有完整 p2→p99 动态范围。")
    lines.append("- pe_real 数据集仅录制 yongen 的 20 条无打断参考会话；"
                 "sun 打断会话复用同 seed 的 yongen 参考作为 V-APT 基线。"
                 "两侧由同一确定性音频驱动、开口度目标调度与 avatar 无关，"
                 "且提取器按会话内自校准，该跨 avatar 近似的主要误差在渲染"
                 "响应差异，解读时需注意。")
    lines.append("- LT 的 flush_talk 只清输入队列，在途音频继续播出，故其"
                 "音频拖尾单独列出（自研 hard 臂在 cut 即刻冲刷音频）。")
    lines.append("")
    lines.append("## 附录：分 avatar 中位数")
    lines.append("")
    lines.append("| 指标 | avatar | hard | trans | LT |")
    lines.append("|---|---|---|---|---|")
    for key, label, self_get, lt_get in METRICS:
        for avatar in AVATARS:
            hk = [k for k in arms["hard"] if k[0] == avatar]
            tk = [k for k in arms["trans"] if k[0] == avatar]
            lk = [k for k in lt_rows if k[0] == avatar]
            med = lambda vals: float(np.median(
                [v for v in vals if v is not None and np.isfinite(v)]))
            lines.append(
                f"| {label} | {avatar} "
                f"| {med([self_get(arms['hard'][k]) for k in hk]):.2f} "
                f"| {med([self_get(arms['trans'][k]) for k in tk]):.2f} "
                f"| {med([lt_get(lt_rows[k]) for k in lk]):.2f} |")
    lines.append("")

    report["descriptives"] = desc
    report["half_syllable_rate"] = {
        "hard": half_rate(arms["hard"]),
        "trans": half_rate(arms["trans"]),
        "lt": half_rate(lt_rows),
    }
    report["lt_audio_tail_ms"] = {
        "median": float(np.median(lt_tails)), "mean": float(np.mean(lt_tails)),
        "n": len(lt_tails),
    }

    (out_dir / "pe5_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "main_table.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out_dir / 'main_table.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-C acceptance: replay the recorded interruption dataset through the
energy-valley cut selector (paper §4.3 + §5.3 ablation).

For every interrupted session (3 avatars x 20 scripted cut points) the
pending TTS tail at the cut is reconstructed by re-synthesizing
utterance 1 from its seed (``audio_synth.synth_utterance`` is fully
deterministic — byte-identical to the recorded reference wavs) and
slicing at ``cut_in_utterance_s``. The recorded ``audio.wav`` cannot be
used: after the hard cut it contains epoch-2 (utterance-2) audio, not
the un-played utterance-1 tail. The selector picks the nearest energy
valley; the ground-truth phoneme schedule (``meta.utterance.events``)
then grades the landing point:

- hit           — a valley/prior was found within the search span
- pause-landed  — the selected cut lands inside a true pause phoneme
- soft-landed   — pause OR consonant (no vowel chopped mid-nucleus)
- rollback_ms   — extra audio played past the hard-cut point

Ablations for the paper's comparison table: the plain hard cut
(rollback = 0) and a uniform-random cut within the same span. Gate:
valley hit rate >= ``--gate`` (default 80%), exit 0 when passed, 2 when
not (CI-visible).

Usage:
    python scripts/interruption_eval/valley_ablation.py \
        --data data/interruption_eval/dataset [--search-ms 240] [--sweep]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
    ),
)

from audio_synth import synth_utterance  # noqa: E402

from liveavatar.runtime.valley import find_valley_cut  # noqa: E402

_SR = 16000


def _load_sessions(data_dir: str) -> list[dict]:
    """All recorded interrupted sessions with schema-2 meta + utt1 PCM."""
    sessions: list[dict] = []
    for name in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, name)
        meta_path = os.path.join(d, "meta.json")
        if not (os.path.isdir(d) and os.path.isfile(meta_path)):
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("kind") != "interrupt" or meta.get("schema") != 2:
            continue
        if meta.get("cut_in_utterance_s") is None:
            continue
        # Rebuild utterance 1 deterministically (the recorded audio.wav is
        # utt1-prefix + epoch-2 audio, unusable past the cut).
        utt = synth_utterance(meta["utterance_duration_s"], meta["seed"])
        sessions.append({"dir": name, "meta": meta, "pcm": utt.pcm_s16le})
    return sessions


_TOL_S = 0.02


def _grade(meta: dict, cut_u_s: float) -> str:
    """Landing grade of a cut at utterance time ``cut_u_s``.

    - "pause"  — inside (or adjacent to) a true silence gap
    - "soft"   — boundary touch within 20 ms of a phoneme edge (the
      nucleus is essentially complete / barely started) or a consonant
      chop (short, low-energy — mild)
    - "vowel"  — the vowel nucleus is audibly truncated (the artifact)
    """
    for ev in meta["utterance"]["events"]:
        if ev["start_s"] <= cut_u_s < ev["end_s"]:
            kind = ev["kind"]
            if kind == "pause":
                return "pause"
            if cut_u_s < ev["start_s"] + _TOL_S or cut_u_s >= ev["end_s"] - _TOL_S:
                return "soft"
            return "vowel" if kind == "vowel" else "soft"
    return "pause"


def replay(
    pcm: bytes, cut_sample: int, search_ms: float, seed: int, depth_db: float
) -> dict:
    """One session: valley / hard / random cut outcomes.

    ``pcm`` is s16le bytes, so sample offsets are doubled for slicing.
    """
    n = int(search_ms / 1000.0 * _SR)
    tail = pcm[cut_sample * 2: (cut_sample + n) * 2]
    vc = find_valley_cut(tail, sample_rate=_SR, search_ms=search_ms,
                         valley_depth_db=depth_db)
    rng = random.Random(seed)
    rand_ms = rng.uniform(0.0, search_ms)
    return {
        "valley_found": vc.found,
        "valley_reason": vc.reason,
        "valley_rollback_ms": vc.rollback_ms,
        "random_rollback_ms": rand_ms,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/interruption_eval/dataset")
    ap.add_argument("--search-ms", type=float, default=240.0)
    ap.add_argument("--depth-db", type=float, default=6.0)
    ap.add_argument("--gate", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="also report hit rate across search budgets 80..400 ms",
    )
    args = ap.parse_args(argv)

    sessions = _load_sessions(args.data)
    if not sessions:
        print(f"no interrupted schema-2 sessions under {args.data}")
        return 1

    rows: list[dict] = []
    for i, s in enumerate(sessions):
        meta = s["meta"]
        cut_u = meta.get("cut_in_utterance_s")
        if cut_u is None:
            continue
        cut_sample = int(round(float(cut_u) * _SR))
        out = replay(
            s["pcm"], cut_sample, args.search_ms, args.seed + i, args.depth_db
        )
        cut_u_s = float(cut_u)
        row = {
            "dir": s["dir"],
            "hard": _grade(meta, cut_u_s),
            "pause": _grade(meta, cut_u_s + out["valley_rollback_ms"] / 1000.0)
            if out["valley_found"]
            else _grade(meta, cut_u_s),
            "random_pause": _grade(
                meta, cut_u_s + out["random_rollback_ms"] / 1000.0
            ),
            **out,
        }
        rows.append(row)

    n = len(rows)
    hits = sum(1 for r in rows if r["valley_found"])
    hit_rate = hits / n

    def _rates(key: str) -> dict[str, float]:
        counts = {"pause": 0, "soft": 0, "vowel": 0}
        for r in rows:
            counts[r[key]] += 1
        return {k: v / n for k, v in counts.items()}

    valley_rates = _rates("pause")
    hard_rates = _rates("hard")
    random_rates = _rates("random_pause")
    rollbacks = sorted(r["valley_rollback_ms"] for r in rows if r["valley_found"])

    def _pct(vals: list[float], q: float) -> float:
        return float(np.percentile(vals, q)) if vals else 0.0

    report = {
        "sessions": n,
        "search_ms": args.search_ms,
        "valley_hit_rate": hit_rate,
        "gate": args.gate,
        "gate_passed": hit_rate >= args.gate,
        "valley_landing": valley_rates,
        "hard_cut_landing": hard_rates,
        "random_cut_landing": random_rates,
        "pause_landed_rate": valley_rates["pause"],
        "hard_pause_landed_rate": hard_rates["pause"],
        "random_pause_landed_rate": random_rates["pause"],
        "rollback_ms_p50": _pct(rollbacks, 50),
        "rollback_ms_p95": _pct(rollbacks, 95),
        "reasons": {
            "valley": sum(1 for r in rows if r["valley_reason"] == "valley"),
            "eou_prior": sum(1 for r in rows if r["valley_reason"] == "eou_prior"),
            "none": sum(1 for r in rows if not r["valley_found"]),
        },
    }

    if args.sweep:
        sweep = []
        for ms in (80.0, 120.0, 160.0, 200.0, 240.0, 320.0, 400.0):
            hits_ms = 0
            rbs: list[float] = []
            for i, s in enumerate(sessions):
                meta = s["meta"]
                cut_u = meta.get("cut_in_utterance_s")
                if cut_u is None:
                    continue
                out = replay(
                    s["pcm"],
                    int(round(float(cut_u) * _SR)),
                    float(ms),
                    args.seed + i,
                    args.depth_db,
                )
                hits_ms += 1 if out["valley_found"] else 0
                if out["valley_found"]:
                    rbs.append(out["valley_rollback_ms"])
            sweep.append(
                {
                    "search_ms": ms,
                    "hit_rate": hits_ms / n,
                    "rollback_p50": _pct(rbs, 50),
                    "rollback_p95": _pct(rbs, 95),
                }
            )
        report["sweep"] = sweep

    out_json = os.path.join(args.data, "valley_ablation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# P-C 能量谷截断消融报告",
        "",
        f"- 会话数：{n}（打断会话，schema 2）",
        f"- 搜索窗：{args.search_ms:.0f} ms（深度阈值 {args.depth_db:.0f} dB / 20 ms 窗）",
        f"- **能量谷命中率：{hit_rate:.1%}**（门禁 ≥ {args.gate:.0%}，"
        f"{'通过' if report['gate_passed'] else '未通过'}）",
        f"- 谷点回退时长：p50 {_pct(rollbacks, 50):.0f} ms / "
        f"p95 {_pct(rollbacks, 95):.0f} ms",
        "",
        "## 落点对比（论文 §5.3 消融）",
        "",
        "| 策略 | 落入真实停顿 | 停顿+辅音（软着陆） | 切在元音内 |",
        "| --- | --- | --- | --- |",
        f"| 硬切（现状） | {hard_rates['pause']:.1%} "
        f"| {hard_rates['pause'] + hard_rates['soft']:.1%} "
        f"| {hard_rates['vowel']:.1%} |",
        f"| 随机回退 | {random_rates['pause']:.1%} "
        f"| {random_rates['pause'] + random_rates['soft']:.1%} "
        f"| {random_rates['vowel']:.1%} |",
        f"| 能量谷选择器 | {valley_rates['pause']:.1%} "
        f"| {valley_rates['pause'] + valley_rates['soft']:.1%} "
        f"| {valley_rates['vowel']:.1%} |",
        "",
        "详细数据：`valley_ablation.json`",
    ]
    if "sweep" in report:
        lines += [
            "",
            "## 搜索窗预算扫描",
            "",
            "| 搜索窗 (ms) | 命中率 | 回退 p50 (ms) | 回退 p95 (ms) |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {s['search_ms']:.0f} | {s['hit_rate']:.1%} "
            f"| {s['rollback_p50']:.0f} | {s['rollback_p95']:.0f} |"
            for s in report["sweep"]
        ]
    out_md = os.path.join(args.data, "valley_ablation.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"sessions={n} hit_rate={hit_rate:.1%} "
          f"pause_landed={valley_rates['pause']:.1%} "
          f"(hard {hard_rates['pause']:.1%}, random {random_rates['pause']:.1%})")
    print(f"wrote {out_json} and {out_md}")
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

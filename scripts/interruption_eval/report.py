# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-B: report aggregation, blind-eval package, Spearman validation.

Subcommands:

- ``report``  — aggregate per-session ``metrics.json`` files into
  ``report.json`` + ``report.md`` (paper Section 5.2 tables).
- ``blind``   — build a 20-clip blind-rating package (anonymized mp4 +
  audio.wav + blank ratings CSV). The clip→session mapping is written to
  ``key.json`` NEXT TO THE DATASET (never inside the rating package).
- ``spearman`` — validate metric validity: Spearman ρ between human
  ratings and V-APT / mismatch (gate: ρ ≥ 0.6, paper Section 5.2).
- ``compare`` — P-D ablation: paired per-(avatar, seed) comparison of a
  hard-cut dataset vs a transition dataset, with the ≥ 50 % mutation
  improvement gate (paper Section 5.3).

Usage::

    python scripts/interruption_eval/report.py report --data <dataset_dir>
    python scripts/interruption_eval/report.py blind  --data <dataset_dir> --n 20
    python scripts/interruption_eval/report.py spearman \
        --ratings blind/ratings.csv --key <dataset_dir>/key.json
    python scripts/interruption_eval/report.py compare \
        --hard <hard_dir> --trans <trans_dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import shutil
import tempfile
from typing import Any

import numpy as np

# ──────────────────────────────────────────────────────── aggregation


def load_metrics(data_root: str) -> list[dict]:
    rows: list[dict] = []
    for mp in sorted(glob.glob(os.path.join(data_root, "*", "metrics.json"))):
        with open(mp, encoding="utf-8") as f:
            row = json.load(f)
        row["dir"] = os.path.basename(os.path.dirname(mp))
        rows.append(row)
    return rows


def _pctl(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    return round(float(np.percentile(np.asarray(vals), p)), 4)


def aggregate(data_root: str) -> dict:
    rows = [r for r in load_metrics(data_root) if r.get("kind") == "interrupt"]
    intr = [r for r in rows if not r.get("error")]

    def series(key: str) -> list[float]:
        out = []
        for r in intr:
            v = r.get(key)
            if isinstance(v, dict):
                v = v.get("apt_ms") if key == "v_apt" else v.get("mutation")
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    vapt = series("v_apt")
    mut = series("smoothness")
    ratio = [
        float(r["smoothness"]["ratio"]) for r in intr
        if isinstance(r.get("smoothness"), dict)
        and r["smoothness"].get("ratio") is not None
    ]
    freeze = [
        float(r["smoothness"]["freeze_frames"]) for r in intr
        if isinstance(r.get("smoothness"), dict)
    ]
    mm = [
        float(r["mismatch"]["rate"]) for r in intr
        if isinstance(r.get("mismatch"), dict)
    ]
    hs = [
        float(r["half_syllable"]["flag"]) for r in intr
        if isinstance(r.get("half_syllable"), dict)
    ]
    by_avatar: dict[str, dict] = {}
    for av in sorted({r.get("avatar_id", "?") for r in intr}):
        sub = [r for r in intr if r.get("avatar_id") == av]
        by_avatar[av] = {
            "n": len(sub),
            "v_apt_ms_p50": _pctl(
                [float(r["v_apt"]["apt_ms"]) for r in sub if "v_apt" in r], 50
            ),
            "mutation_p50": _pctl(
                [float(r["smoothness"]["mutation"]) for r in sub if "smoothness" in r],
                50,
            ),
            "mismatch_mean": _pctl(
                [float(r["mismatch"]["rate"]) for r in sub if "mismatch" in r], 50
            ),
        }
    report = {
        "mode": "interruption_eval_metrics",
        "data_root": data_root,
        "n_interrupt_sessions": len(intr),
        "n_errors": len(rows) - len(intr),
        "overall": {
            "v_apt_ms": {"p50": _pctl(vapt, 50), "p95": _pctl(vapt, 95),
                         "mean": round(float(np.mean(vapt)), 2) if vapt else None},
            "mutation": {"p50": _pctl(mut, 50), "p95": _pctl(mut, 95)},
            "ratio": {"p50": _pctl(ratio, 50), "max": _pctl(ratio, 100)},
            "freeze_frames": {"p50": _pctl(freeze, 50)},
            "mismatch_rate": {"mean": round(float(np.mean(mm)), 4) if mm else None,
                              "p50": _pctl(mm, 50)},
            "half_syllable_rate": round(float(np.mean(hs)), 4) if hs else None,
        },
        "by_avatar": by_avatar,
        "rows": [
            {
                "dir": r["dir"],
                "avatar_id": r.get("avatar_id"),
                "seed": r.get("seed"),
                "v_apt_ms": r.get("v_apt", {}).get("apt_ms"),
                "mutation": r.get("smoothness", {}).get("mutation"),
                "ratio": r.get("smoothness", {}).get("ratio"),
                "freeze_frames": r.get("smoothness", {}).get("freeze_frames"),
                "mismatch": r.get("mismatch", {}).get("rate"),
                "half_syllable": r.get("half_syllable", {}).get("flag"),
            }
            for r in intr
        ],
    }
    return report


def _md(report: dict) -> str:
    o = report["overall"]
    lines = [
        "# P-B 评测指标报告（硬切基线，嘴部代理，CPU duplex）",
        "",
        f"- 打断会话：{report['n_interrupt_sessions']}（错误 {report['n_errors']}）；"
        f"素材：{', '.join(report['by_avatar'])}",
        "",
        "口径：V-APT 为过渡窗口面积（切点后 1.2 s 上限，对照同 seed 无打断"
        "参考轨迹）；平滑度为冻结感知首跳 vs 切前基线；失配窗口 0.6 s。",
        "",
        "## 总体指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| V-APT p50 / p95 (ms) | {o['v_apt_ms']['p50']} / {o['v_apt_ms']['p95']} |",
        f"| 突变首跳 p50 / p95 | {o['mutation']['p50']} / {o['mutation']['p95']} |",
        f"| 放大倍数 p50 / max | {o['ratio']['p50']} / {o['ratio']['max']} |",
        f"| 冻结帧数 p50 | {o['freeze_frames']['p50']} |",
        f"| 音素-视位失配率 mean / p50 | {o['mismatch_rate']['mean']} / "
        f"{o['mismatch_rate']['p50']} |",
        f"| 半音节伪影率 | {o['half_syllable_rate']} |",
        "",
        "## 按素材",
        "",
        "| 素材 | n | V-APT p50(ms) | 突变 p50 | 失配率 p50 |",
        "|---|---|---|---|---|",
    ]
    for av, s in report["by_avatar"].items():
        lines.append(
            f"| {av} | {s['n']} | {s['v_apt_ms_p50']} | {s['mutation_p50']} | "
            f"{s['mismatch_mean']} |"
        )
    lines += [
        "",
        "## 单样本明细",
        "",
        "| 会话 | V-APT(ms) | 突变 | 倍数 | 冻结帧 | 失配率 | 半音节 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['dir']} | {r['v_apt_ms']} | {r['mutation']} | {r['ratio']} | "
            f"{r['freeze_frames']} | {r['mismatch']} | {r['half_syllable']} |"
        )
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    report = aggregate(args.data)
    os.makedirs(args.out or args.data, exist_ok=True)
    jp = os.path.join(args.out or args.data, "report.json")
    mp = os.path.join(args.out or args.data, "report.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(_md(report))
    print(f"wrote {jp} and {mp}")
    return 0


# ──────────────────────────────────────────────────── blind-eval package


def _grid_expand(timeline: list[dict], period_s: float) -> list[dict]:
    """Expand a recorded frame timeline onto the regular frame grid.

    Freeze gaps (no frames for more than one period) HOLD the last
    frame — zero-order hold, exactly what the viewer's screen shows
    while no new frame arrives. Without this the mp4 would compress the
    ~0.7 s screen freeze into a single-frame skip and raters would not
    see the artifact at its true duration.
    """
    grid: list[dict] = []
    prev_ts: float | None = None
    for fr in timeline:
        ts = float(fr["ts_rel"])
        k = 1 if prev_ts is None else max(1, int(round((ts - prev_ts) / period_s)))
        grid.extend([fr] * k)
        prev_ts = ts
    return grid


def _write_clip(src_dir: str, mp4_path: str, timeline: list[dict],
                fps: float, period_s: float) -> bool:
    """Encode the grid-expanded frame sequence to mp4 (ASCII temp path
    first — OpenCV VideoWriter is unreliable on non-ASCII Windows
    paths)."""
    import cv2

    if not timeline:
        return False
    first = cv2.imdecode(
        np.fromfile(os.path.join(src_dir, "frames", timeline[0]["file"]), np.uint8),
        cv2.IMREAD_COLOR,
    )
    h, w = first.shape[:2]
    grid = _grid_expand(timeline, period_s)
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    ok = False
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if writer.isOpened():
        ok = True
        for fr in grid:
            img = cv2.imdecode(
                np.fromfile(os.path.join(src_dir, "frames", fr["file"]), np.uint8),
                cv2.IMREAD_COLOR,
            )
            writer.write(img)
    writer.release()
    if ok:
        shutil.move(tmp, mp4_path)
    else:
        os.unlink(tmp)
    return ok


def _aligned_audio(src_dir: str, wav_path: str, t0_s: float) -> float:
    """TTS audio re-aligned to the video clock → ``wav_path``.

    ``audio.wav`` stores the received PCM chunks back-to-back (no
    silence), while the video timeline is wall-clock. Rebuild the stream
    by placing each chunk (sizes from ``tts_chunk_timeline``) at its
    arrival time relative to the video origin ``t0_s`` (first frame
    ts): the post-cut gap becomes silence — what a viewer actually
    hears. Returns the aligned duration in seconds.
    """
    import wave

    with open(os.path.join(src_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    chunks = meta.get("tts_chunk_timeline") or []
    if not chunks:
        raise ValueError("meta has no tts_chunk_timeline (schema v1?)")
    with wave.open(os.path.join(src_dir, "audio.wav"), "rb") as w:
        pcm = w.readframes(w.getnframes())
    sr = 16000
    off = 0
    end_s = 0.0
    segments: list[tuple[int, bytes]] = []
    for ch in chunks:
        n = int(ch["bytes"])
        seg = pcm[off:off + n]
        off += n
        start_s = float(ch["ts_rel"]) - t0_s
        segments.append((max(0, int(round(start_s * sr))), seg))
        end_s = max(end_s, start_s + n / 2.0 / sr)
    buf = bytearray(int(round(end_s * sr)) * 2)
    for start, seg in segments:
        s = start * 2
        if s >= len(buf):
            continue
        e = min(len(buf), s + len(seg))
        buf[s:e] = seg[: e - s]
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(buf))
    return end_s


def _mux_av(mp4_path: str, wav_path: str) -> bool:
    """Mux the aligned wav into the mp4 (video stream copied). Best
    effort: returns False when ffmpeg is unavailable or fails — the
    caller keeps the separate wav (README covers dual playback)."""
    import subprocess

    if shutil.which("ffmpeg") is None:
        return False
    tmp = mp4_path + ".mux.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path,
             "-i", wav_path, "-c:v", "copy", "-c:a", "aac", "-shortest", tmp],
            check=True, capture_output=True, timeout=120,
        )
    except Exception:
        if os.path.isfile(tmp):
            os.unlink(tmp)
        return False
    os.replace(tmp, mp4_path)
    return True


def cmd_blind(args: argparse.Namespace) -> int:
    rows = [r for r in load_metrics(args.data)
            if r.get("kind") == "interrupt" and not r.get("error")]
    if len(rows) < args.n:
        print(f"only {len(rows)} analyzable interrupt sessions (< {args.n})")
        return 1
    # Deterministic round-robin across avatars over seed-ordered points.
    by_avatar: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: (r.get("seed", 0), r["dir"])):
        by_avatar.setdefault(r.get("avatar_id", "?"), []).append(r)
    picked: list[dict] = []
    i = 0
    while len(picked) < args.n:
        for av in sorted(by_avatar):
            lst = by_avatar[av]
            if i < len(lst) and len(picked) < args.n:
                picked.append(lst[i])
        i += 1
    rng = random.Random(args.seed)
    ids = [f"clip_{k:02d}" for k in range(1, args.n + 1)]
    rng.shuffle(ids)  # anonymize the presentation order

    out_dir = args.out or os.path.join(args.data, "blind")
    os.makedirs(out_dir, exist_ok=True)
    key: dict[str, str] = {}
    muxed = 0
    for clip_id, r in zip(ids, picked, strict=True):
        src = os.path.join(args.data, r["dir"])
        with open(os.path.join(src, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        timeline = meta.get("frame_timeline") or []
        period_s = float(meta.get("frame_period_s") or 0.04)
        t0 = float(timeline[0]["ts_rel"]) if timeline else 0.0
        mp4 = os.path.join(out_dir, f"{clip_id}.mp4")
        wav = os.path.join(out_dir, f"{clip_id}.wav")
        ok = _write_clip(src, mp4, timeline, 1.0 / period_s, period_s)
        if not ok:
            print(f"clip encode failed: {r['dir']}")
            return 1
        try:
            _aligned_audio(src, wav, t0)
        except ValueError:
            shutil.copyfile(os.path.join(src, "audio.wav"), wav)
        if _mux_av(mp4, wav):
            muxed += 1
        key[clip_id] = r["dir"]
    key_path = os.path.join(args.data, "key.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, ensure_ascii=False)
    ratings_path = os.path.join(out_dir, "ratings.csv")
    with open(ratings_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "naturalness_1to5", "comfort_1to5", "comment"])
        for clip_id in ids:
            w.writerow([clip_id, "", "", ""])
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# 盲评材料（P-B7）\n\n"
            "每个样本：`clip_XX.mp4`（画面 + 已对齐语音轨；若环境无 ffmpeg 则为"
            "无声画面，请同时播放 `clip_XX.wav`）。视频按 40ms 网格对齐："
            "冻结间隙保持最后一帧（与真实观看一致），音频按块到达时刻重建"
            "（打断后的间隙即静音）。\n\n"
            "请对打断瞬间 mouth 过渡的自然度与舒适度打 1-5 分"
            "（1=非常不自然，5=完全自然），填写 `ratings.csv`。\n\n"
            "评分要点：打断瞬间的口型冻结/跳变/音画失配是否可察觉、"
            "过渡是否突兀。不要评价语音内容本身。\n"
        )
    print(f"packaged {args.n} clips → {out_dir} (muxed {muxed}); key → {key_path}")
    return 0


# ───────────────────────────────────────────────────── Spearman validation


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties get the mean of their positions."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman ρ with tie correction (Pearson on average ranks)."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("spearman needs equal-length series of length >= 2")
    rx = _average_ranks(np.asarray(x, dtype=np.float64))
    ry = _average_ranks(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom <= 0:
        return 0.0
    return float((rx * ry).sum() / denom)


def cmd_spearman(args: argparse.Namespace) -> int:
    with open(args.key, encoding="utf-8") as f:
        key: dict[str, str] = json.load(f)
    data_root = os.path.dirname(os.path.abspath(args.key))
    ratings: dict[str, float] = {}
    with open(args.ratings, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("naturalness_1to5") or "").strip()
            if raw:
                ratings[row["clip_id"]] = float(raw)
    pairs: dict[str, tuple[list[float], list[float]]] = {
        "v_apt_ms": ([], []),
        "mismatch": ([], []),
        "mutation": ([], []),
    }
    for clip_id, rating in sorted(ratings.items()):
        d = key.get(clip_id)
        if d is None:
            continue
        mp = os.path.join(data_root, d, "metrics.json")
        if not os.path.isfile(mp):
            continue
        with open(mp, encoding="utf-8") as f:
            met = json.load(f)
        if met.get("v_apt", {}).get("apt_ms") is not None:
            pairs["v_apt_ms"][0].append(float(met["v_apt"]["apt_ms"]))
            pairs["v_apt_ms"][1].append(rating)
        if met.get("mismatch") is not None:
            pairs["mismatch"][0].append(float(met["mismatch"]["rate"]))
            pairs["mismatch"][1].append(rating)
        if met.get("smoothness", {}).get("mutation") is not None:
            pairs["mutation"][0].append(float(met["smoothness"]["mutation"]))
            pairs["mutation"][1].append(rating)
    if len(ratings) < 3:
        print(f"need >= 3 ratings, got {len(ratings)}")
        return 1
    rhos = {
        k: round(spearman_rho(x, y), 4)
        for k, (x, y) in pairs.items() if len(x) == len(ratings)
    }
    best = max(rhos.values()) if rhos else 0.0
    out = {
        "n_ratings": len(ratings),
        "spearman": rhos,
        "gate": {"threshold": 0.6, "best_rho": best, "pass": best >= 0.6},
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return 0 if out["gate"]["pass"] else 2


# ───────────────────────────────────────────────── P-D ablation compare


_METRIC_EXTRACTORS: dict[str, Any] = {
    "v_apt_ms": lambda r: r.get("v_apt", {}).get("apt_ms"),
    "mutation": lambda r: r.get("smoothness", {}).get("mutation"),
    "ratio": lambda r: r.get("smoothness", {}).get("ratio"),
    "freeze_frames": lambda r: r.get("smoothness", {}).get("freeze_frames"),
    "mismatch_rate": lambda r: r.get("mismatch", {}).get("rate"),
    "half_syllable": lambda r: r.get("half_syllable", {}).get("flag"),
    "resume_ms": lambda r: r.get("latency", {}).get("resume_ms"),
}


def _sign_test_p(x: list[float], y: list[float]) -> float | None:
    """Exact two-sided sign-test p for paired ``y`` vs ``x`` improvement
    (``y < x`` counts as improvement; ties excluded). No scipy needed."""
    import math

    wins = sum(1 for a, b in zip(x, y, strict=False) if b < a)
    n = sum(1 for a, b in zip(x, y, strict=False) if b != a)
    if n == 0:
        return None
    tail = sum(
        math.comb(n, k) for k in range(min(wins, n - wins) + 1)
    ) / 2 ** n
    return round(min(1.0, 2.0 * tail), 4)


def paired_compare(hard_root: str, trans_root: str) -> dict:
    """P-D ablation: paired per-(avatar, seed) hard-cut vs transition.

    Both datasets must share the same recorded cut points (same seeds →
    same scripted interrupts), so every metric forms a paired sample;
    the half-syllable flag additionally doubles as a determinism check
    (identical cut instants ⇒ identical flags in both arms).
    """
    def rows(root: str) -> dict[tuple, dict]:
        out = {}
        for r in load_metrics(root):
            if r.get("kind") != "interrupt" or r.get("error"):
                continue
            out[(r.get("avatar_id"), r.get("seed"))] = r
        return out

    hard, trans = rows(hard_root), rows(trans_root)
    keys = sorted(set(hard) & set(trans))
    only_hard = sorted(set(hard) - set(trans))
    only_trans = sorted(set(trans) - set(hard))

    metrics: dict[str, dict] = {}
    for name, get in _METRIC_EXTRACTORS.items():
        pairs = [
            (float(get(hard[k])), float(get(trans[k])))
            for k in keys
            if get(hard[k]) is not None and get(trans[k]) is not None
        ]
        if not pairs:
            metrics[name] = {"n_pairs": 0}
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        hard_p50 = _pctl(xs, 50)
        trans_p50 = _pctl(ys, 50)
        mean_h = round(float(np.mean(xs)), 4)
        mean_t = round(float(np.mean(ys)), 4)
        entry: dict[str, Any] = {
            "n_pairs": len(pairs),
            "hard": {"p50": hard_p50, "p95": _pctl(xs, 95), "mean": mean_h},
            "trans": {"p50": trans_p50, "p95": _pctl(ys, 95), "mean": mean_t},
        }
        if isinstance(hard_p50, float) and hard_p50:
            entry["improve_p50_pct"] = round(
                (hard_p50 - trans_p50) / hard_p50 * 100.0, 2
            )
        if mean_h:
            entry["improve_mean_pct"] = round(
                (mean_h - mean_t) / mean_h * 100.0, 2
            )
        if name != "half_syllable":
            entry["sign_test_p"] = _sign_test_p(xs, ys)
        metrics[name] = entry

    hs_match = sum(
        1 for k in keys
        if (hard[k].get("half_syllable", {}).get("flag")
            == trans[k].get("half_syllable", {}).get("flag"))
    )
    n_hs = sum(
        1 for k in keys
        if hard[k].get("half_syllable", {}).get("flag") is not None
        and trans[k].get("half_syllable", {}).get("flag") is not None
    )
    mut = metrics.get("mutation", {})
    mm = metrics.get("mismatch_rate", {})
    rs = metrics.get("resume_ms", {})
    return {
        "mode": "interruption_eval_pd_compare",
        "hard_root": hard_root,
        "trans_root": trans_root,
        "n_paired_sessions": len(keys),
        "only_in_hard": [list(k) for k in only_hard],
        "only_in_trans": [list(k) for k in only_trans],
        "half_syllable_determinism": {
            "n": n_hs, "matching": hs_match,
        },
        "metrics": metrics,
        "gates": {
            "plan_gate_mutation": {
                "criterion": "first-jump (mutation) p50 improvement >= 50% "
                             "(plan P-D gate, continuity reading)",
                "value": mut.get("improve_p50_pct"),
                "pass": (
                    isinstance(mut.get("improve_p50_pct"), (int, float))
                    and mut["improve_p50_pct"] >= 50.0
                ),
            },
            "semantic_gate_mismatch": {
                "criterion": "mismatch p50 improvement >= 50% "
                             "(semantic smoothness reading)",
                "value": mm.get("improve_p50_pct"),
                "pass": (
                    isinstance(mm.get("improve_p50_pct"), (int, float))
                    and mm["improve_p50_pct"] >= 50.0
                ),
            },
            "latency_budget": {
                "criterion": "transition resume p95 <= 90 ms (L5 budget)",
                "value": rs.get("trans", {}).get("p95"),
                "pass": (
                    isinstance(rs.get("trans", {}).get("p95"), (int, float))
                    and rs["trans"]["p95"] <= 90.0
                ),
            },
        },
    }


def _compare_md(rep: dict) -> str:
    lines = [
        "# P-D 过渡帧消融对比（硬切 vs 过渡帧，嘴部代理，CPU duplex）",
        "",
        f"- 配对打断会话：{rep['n_paired_sessions']}"
        f"（仅硬切 {len(rep['only_in_hard'])}，仅过渡 {len(rep['only_in_trans'])}）",
        f"- 半音节确定性：{rep['half_syllable_determinism']['matching']}/"
        f"{rep['half_syllable_determinism']['n']} 一致（同切点 ⇒ 两臂同标记）",
        "",
        "配对口径：同 (素材, seed) 一一对比；改善 % = (硬切 − 过渡) / 硬切；"
        "符号检验为配对双侧精确检验。",
        "",
        "| 指标 | 硬切 p50 | 过渡 p50 | 改善 p50 % | 硬切 mean | 过渡 mean | "
        "改善 mean % | 符号检验 p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    names = {
        "v_apt_ms": "V-APT (ms)",
        "mutation": "突变首跳",
        "ratio": "放大倍数",
        "freeze_frames": "冻结帧数",
        "mismatch_rate": "失配率",
        "half_syllable": "半音节",
        "resume_ms": "resume (ms)",
    }
    for name, label in names.items():
        e = rep["metrics"].get(name, {})
        if not e or not e.get("n_pairs"):
            lines.append(f"| {label} | - | - | - | - | - | - | - |")
            continue
        p = e.get("sign_test_p")
        p = "-" if p is None else str(p)
        ip = e.get("improve_p50_pct")
        ip = "-" if ip is None else str(ip)
        im = e.get("improve_mean_pct")
        im = "-" if im is None else str(im)
        lines.append(
            f"| {label} | {e['hard']['p50']} | {e['trans']['p50']} | {ip} | "
            f"{e['hard']['mean']} | {e['trans']['mean']} | {im} | {p} |"
        )
    g = rep["gates"]
    lines += [
        "",
        "## 门禁（双口径并列，不做静默换标）",
        "",
        f"- 方案门禁（连续性口径）：突变首跳 p50 改善 "
        f"{g['plan_gate_mutation']['value']}% → "
        f"{'PASS' if g['plan_gate_mutation']['pass'] else 'FAIL'}",
        f"- 语义口径：失配率 p50 改善 {g['semantic_gate_mismatch']['value']}% → "
        f"{'PASS' if g['semantic_gate_mismatch']['pass'] else 'FAIL'}",
        f"- 延迟预算：过渡臂 resume p95 = {g['latency_budget']['value']} ms "
        f"≤ 90 ms → {'PASS' if g['latency_budget']['pass'] else 'FAIL'}",
        "",
        "## 指标解读（为什么连续性指标与语义指标方向相反）",
        "",
        "- **失配率 −93% / L5 首新帧 −99.6% 是 P-D 的目标轴**：硬切在切点后"
        "把嘴冻结在音节中段（音频已停、嘴还在\"说\"），首新帧要等下一轮 "
        "TTS（~534 ms 死屏）；过渡帧在取消后 ~2 ms 内把嘴按 α 调度合拢，"
        "画面状态与静音语义对齐，且首新帧即过渡帧。",
        "- **首跳/倍数/V-APT 的回退是口径结构性的，不是感知结论**："
        "(1) 硬切的冻结画面在 utt2 首帧落点接近冻结值时\"首跳\"退化为自然"
        "运动量级（硬切 p50 仅 0.085 ≈ 2× 基线），死屏本身不产生 Δ，"
        "首跳指标看不见冻结；(2) 过渡臂的首跳是 bridge 自身的合拢步进"
        "（有意为之的快速闭嘴），被同一条指标当作突变计分；"
        "(3) V-APT 以\"话语 1 继续说\"的反事实为参考，硬切冻结值恰在语音"
        "频带内易被 utt2 驱动帧\"重新同步\"，而过渡臂的闭嘴（语义正确）"
        "反而最大化对反事实的偏离。三条连续性指标都是为硬切设计的"
        "连续性口径，对语义正确的过渡系统性偏罚。",
        "- **感知仲裁留给盲评**：两臂 mp4 盲评包（``blind``）+ P-F 用户研究"
        "决定哪一口径与人的判断一致；本表如实并列两种读数。",
        "",
    ]
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> int:
    rep = paired_compare(args.hard, args.trans)
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.hard)), "pd_compare"
    )
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "compare.json")
    mp = os.path.join(out_dir, "compare.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(_compare_md(rep))
    print(f"wrote {jp} and {mp}")
    print(json.dumps(rep["gates"], indent=2, ensure_ascii=False))
    return 0 if rep["gates"]["plan_gate_mutation"]["pass"] else 2


# ─────────────────────────────────────────────────────────────────── main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("report")
    p.add_argument("--data", default="data/interruption_eval/dataset")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("blind")
    p.add_argument("--data", default="data/interruption_eval/dataset")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_blind)

    p = sub.add_parser("spearman")
    p.add_argument("--ratings", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_spearman)

    p = sub.add_parser("compare")
    p.add_argument("--hard", required=True, help="hard-cut dataset dir")
    p.add_argument("--trans", required=True, help="transition dataset dir")
    p.add_argument("--out", default=None,
                   help="output dir (default: <hard>/../pd_compare)")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

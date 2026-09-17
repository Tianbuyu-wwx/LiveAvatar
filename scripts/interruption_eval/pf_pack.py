# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-F1: three-arm paired blind-rating pack builder (P-F user study).

Builds the paired comparison material for the P-F perceptual study:
12 pairs = 3 system pairings (hard vs trans, hard vs LT, trans vs LT)
x 4 deterministic (seed, t) slots. Slots cover early/mid/late half-syllable
cuts plus one non-half cut; within a slot the (seed, t) whose cut point is
closest to the slot's median t is picked (deterministic, no cherry-picking).
All clips use the yongen avatar, repeat 1.

Clip construction:
- self arms (pe_real): the FULL session is rebuilt from the frame timeline
  via report.py's grid expansion (freeze gaps hold the last frame — what
  the viewer's screen actually shows) with arrival-aligned audio; encoded
  in ONE H.264 pass from the source JPGs (crf 16), resolution-independent
  (512x512 legacy or 768x768 HD recuts).
- LT arm: ffmpeg-cut from ``recording.mp4`` (the actual WebRTC output),
  anchored at the xcorr-derived rec-clock offsets stored in
  ``metrics.json`` (utt1_start_rec / utt2_start_rec).

Anonymization: clip files get random ids (c01..c24); the mapping to
(session dir, system arm) lives ONLY in ``pack_key.json``. The survey page
embeds ``manifest.json`` (neutral group ids + clip filenames) — no arm
identity ever reaches the participant. NOTE for the paper: the LT yongen
avatar is the MuseTalk material face while the self arms show the rendered
yongen, so cross-system pairs are cross-appearance by necessity; rating is
within-pair (dual stimulus), and the questionnaire discloses that the two
clips of a pair may show different virtual characters.

Outputs (default ``data/interruption_eval/pf_pack``):
- ``clips/c##.mp4``          24 anonymous clips
- ``manifest.json``          survey-page manifest (neutral ids)
- ``pack_key.json``          experimenter-private identity mapping
- ``survey/index.html``      self-contained survey page (manifest embedded)
- ``README.md``              protocol notes
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import report as rp  # noqa: E402  (reuses _grid_expand / _aligned_audio / _mux_av)

AVATAR = "yongen"
REPEAT = 1
UTT2_S = 2.5
LT_LEAD_S = 0.2  # lead-in before utt1 audio start in the LT cut
LT_TAIL_S = 0.4  # tail after utt2 audio end in the LT cut

# 4 deterministic (seed, t) slots: t range + required half-syllable flag.
SLOTS: list[dict[str, Any]] = [
    {"name": "early_half", "t_range": (0.0, 1.5), "half": True},
    {"name": "mid_half", "t_range": (1.5, 2.9), "half": True},
    {"name": "late_half", "t_range": (2.9, 99.0), "half": True},
    {"name": "mid_nohalf", "t_range": (1.5, 3.5), "half": False},
]
PAIRINGS = [("hard", "trans"), ("hard", "lt"), ("trans", "lt")]


def _t_of(name: str) -> str:
    """interrupt_yongen_seed{S}_t{T}s -> "{T}"."""
    body = name.removeprefix(f"interrupt_{AVATAR}_seed")
    _seed, rest = body.split("_t", 1)
    return rest.removesuffix("s")


def self_dirs(data_root: Path, arm: str,
              prefix: str = "pe_real") -> dict[tuple[int, str], Path]:
    """{prefix}_{arm}_r{REPEAT} yongen interrupt sessions keyed by (seed, t)."""
    root = data_root / f"{prefix}_{arm}_r{REPEAT}"
    out: dict[tuple[int, str], Path] = {}
    for d in sorted(root.glob(f"interrupt_{AVATAR}_seed*_t*")):
        body = d.name.removeprefix(f"interrupt_{AVATAR}_seed")
        seed_str, _rest = body.split("_t", 1)
        out[(int(seed_str), _t_of(d.name))] = d
    return out


def lt_dirs(data_root: Path) -> dict[tuple[int, str], Path]:
    """lt_{avatar} yongen interrupt sessions (repeat 1) keyed by (seed, t)."""
    root = data_root / f"lt_{AVATAR}"
    out: dict[tuple[int, str], Path] = {}
    for d in sorted(root.glob(f"interrupt_{AVATAR}_seed*_r{REPEAT}")):
        body = d.name.removeprefix(f"interrupt_{AVATAR}_seed")
        seed_str, rest = body.split("_", 1)
        t_str = rest.rsplit("_r", 1)[0].removeprefix("t").removesuffix("s")
        out[(int(seed_str), t_str)] = d
    return out


def _half_flag(session: Path) -> bool:
    m = json.loads((session / "metrics.json").read_text(encoding="utf-8"))
    flag = m.get("half_syllable", {}).get("flag")
    if flag is None:
        raise SystemExit(f"session has no half_syllable flag: {session}")
    return bool(flag)


def pick_slots(candidates: dict[tuple[int, str], Path]) -> list[dict[str, Any]]:
    """One (seed, t) per slot: closest t to the slot's median candidate t."""
    picks: list[dict[str, Any]] = []
    for slot in SLOTS:
        lo, hi = slot["t_range"]
        pool = [
            (seed, t, d) for (seed, t), d in candidates.items()
            if lo <= float(t) < hi and _half_flag(d) == slot["half"]
        ]
        if not pool:
            raise SystemExit(f"slot {slot['name']}: no candidate session")
        med = sorted(float(t) for _s, t, _d in pool)[len(pool) // 2 - 1] \
            if len(pool) % 2 == 0 else \
            sorted(float(t) for _s, t, _d in pool)[len(pool) // 2]
        seed, t, _d = min(pool, key=lambda r: (abs(float(r[1]) - med), r[0]))
        picks.append({"name": slot["name"], "seed": seed, "t": t,
                      "half_syllable": slot["half"]})
    return picks


def _write_clip_h264(src_dir: str, mp4_path: str, timeline: list[dict],
                     fps: float, period_s: float) -> bool:
    """Encode the grid-expanded JPEG frames straight to H.264 (crf 16) via
    an ffmpeg rawvideo pipe. The old path (cv2 mp4v → libx264 recode) lost
    detail twice: cv2's default mp4v bitrate is ~0.65 Mbps and the second
    transcode compounds it — the P-E 512x512 clips came out visibly muddy.
    One encode from the source JPGs keeps the 768x768 HD recuts sharp."""
    import cv2

    if not timeline:
        return False
    first = cv2.imdecode(
        np.fromfile(os.path.join(src_dir, "frames", timeline[0]["file"]),
                    np.uint8),
        cv2.IMREAD_COLOR,
    )
    if first is None:
        return False
    h, w = first.shape[:2]
    grid = rp._grid_expand(timeline, period_s)
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ok = proc.stdin is not None
    try:
        if ok:
            for fr in grid:
                img = cv2.imdecode(
                    np.fromfile(os.path.join(src_dir, "frames", fr["file"]),
                                np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if img is None:
                    ok = False
                    break
                proc.stdin.write(img.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        code = proc.wait()
    if ok and code == 0:
        shutil.move(tmp, mp4_path)
        return True
    if os.path.isfile(tmp):
        os.unlink(tmp)
    return False


def build_self_clip(session: Path, out_mp4: Path) -> None:
    """Rebuild one pe_real session as a clip (grid-expanded + aligned audio)."""
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    timeline = meta.get("frame_timeline") or []
    period = float(meta.get("frame_period_s") or 0.04)
    t0 = float(timeline[0]["ts_rel"]) if timeline else 0.0
    if not _write_clip_h264(str(session), str(out_mp4), timeline,
                            1.0 / period, period):
        raise SystemExit(f"clip encode failed: {session}")
    wav = out_mp4.with_suffix(".tmp.wav")
    try:
        rp._aligned_audio(str(session), str(wav), t0)
        if not rp._mux_av(str(out_mp4), str(wav)):
            raise SystemExit(f"mux failed: {session}")
    finally:
        wav.unlink(missing_ok=True)


def build_lt_clip(session: Path, out_mp4: Path) -> None:
    """Cut one LT session from recording.mp4 (re-encode: frame-accurate)."""
    m = json.loads((session / "metrics.json").read_text(encoding="utf-8"))
    a1 = m.get("utt1_start_rec")
    a2 = m.get("utt2_start_rec")
    if a1 is None or a2 is None:
        raise SystemExit(f"LT metrics missing rec anchors: {session}")
    start = max(float(a1) - LT_LEAD_S, 0.0)
    end = float(a2) + UTT2_S + LT_TAIL_S
    tmp = out_mp4.with_suffix(".tmp.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(session / "recording.mp4"),
         "-c:v", "libx264", "-preset", "medium", "-crf", "16",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
         str(tmp)],
        check=True,
    )
    shutil.move(tmp, out_mp4)


BUILDERS = {
    "hard": build_self_clip,
    "trans": build_self_clip,
    "lt": build_lt_clip,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/interruption_eval")
    ap.add_argument("--out", default="data/interruption_eval/pf_pack")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--self-prefix", default="pe_real",
                    help="self-arm dataset prefix (pe_real_hd = the "
                         "768x768 HD recuts)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse existing clip files (idempotent re-runs)")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    arms = {"hard": self_dirs(data_root, "hard", args.self_prefix),
            "trans": self_dirs(data_root, "trans", args.self_prefix),
            "lt": lt_dirs(data_root)}
    common = set(arms["hard"]) & set(arms["trans"]) & set(arms["lt"])
    if len(common) < len(SLOTS):
        raise SystemExit(f"only {len(common)} common (seed, t) sessions")
    candidates = {key: arms["hard"][key] for key in common}
    slots = pick_slots(candidates)
    print("slots:")
    for s in slots:
        print(f"  {s['name']}: seed={s['seed']} t={s['t']}s "
              f"half={s['half_syllable']}")

    # 12 pairs x 2 clips = 24 anonymous ids
    rng = random.Random(args.seed)
    ids = [f"c{k:02d}" for k in range(1, 25)]
    rng.shuffle(ids)

    pairs: list[dict[str, Any]] = []
    manifest_groups: list[dict[str, Any]] = []
    id_i = 0
    for s_i, slot in enumerate(slots, 1):
        for arm_a, arm_b in PAIRINGS:
            group_id = f"g{s_i}{PAIRINGS.index((arm_a, arm_b)) + 1}"
            clips: dict[str, dict[str, str]] = {}
            for arm in (arm_a, arm_b):
                clip_id = ids[id_i]
                id_i += 1
                clip_file = f"{clip_id}.mp4"
                dst = clips_dir / clip_file
                if not (args.skip_existing and dst.exists()):
                    BUILDERS[arm](arms[arm][(slot["seed"], slot["t"])], dst)
                clips[arm] = {"clip": clip_file,
                              "dir": str(arms[arm][(slot["seed"], slot["t"])]
                                         .relative_to(data_root))}
            pair_key = {
                "group_id": group_id,
                "slot": slot["name"],
                "seed": slot["seed"],
                "t": slot["t"],
                "comparison": f"{arm_a}_vs_{arm_b}",
                "arms": clips,
            }
            pairs.append(pair_key)
            manifest_groups.append({
                "id": group_id,
                "clips": [clips[arm_a]["clip"], clips[arm_b]["clip"]],
            })
            print(f"  {group_id} ({arm_a} vs {arm_b}, seed={slot['seed']} "
                  f"t={slot['t']}s): {clips[arm_a]['clip']} / "
                  f"{clips[arm_b]['clip']}")

    manifest = {"avatar": AVATAR, "scale": "1-5",
                "dimensions": ["naturalness", "comfort"],
                "groups": manifest_groups}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "pack_key.json").write_text(
        json.dumps({"seed": args.seed, "slots": slots, "pairs": pairs},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    from pf_survey import write_survey_page

    write_survey_page(out_dir / "survey", manifest)
    (out_dir / "README.md").write_text(
        "# P-F 盲评材料\n\n"
        "12 组 × 2 片段 = 24 个匿名片段（clips/），组结构见 pack_key.json"
        "（仅实验员持有，勿发给被试）。问卷页：survey/index.html（被试完成后"
        "导出 JSON 收回）。\n\n"
        "设计：三臂两两配对（hard/trans/LT 各 4 组），固定 yongen、repeat 1，"
        "同组两片段同 seed 同打断点。半音节槽位 3 个 + 非半音节 1 个，覆盖"
        "早/中/晚打断点。\n\n"
        "注意：跨系统配对（*-vs-lt）两侧形象不同（自研渲染 yongen vs MuseTalk"
        " 素材脸）——属跨系统盲评的必然，评分为组内双刺激比较，问卷已向被试"
        "说明两片段可能来自不同虚拟形象。\n",
        encoding="utf-8")
    print(f"\npack → {out_dir} (24 clips, 12 groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

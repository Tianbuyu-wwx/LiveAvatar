# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-E: metric extraction for LiveTalking baseline recordings (py312 side).

Input: one session dir from lt_harness (``recording.mp4`` + ``meta.json``)
plus the prepared wavs/events dir. Output: ``metrics.json`` with the same
five-metric row schema as the LiveAvatar arms, plus LiveTalking-specific
latency decompositions.

Measurement pipeline
--------------------
1. Demux the mp4 audio to 16 kHz mono PCM (ffmpeg) and decode frames (cv2).
2. Anchor the recording clock sample-accurately by normalized
   cross-correlation of the recorded PCM against the known utterance wavs.
3. Mouth openness from the real-face video: YuNet face box on a mid frame →
   fixed geometric mouth ROI (the MuseTalk head is static) → dark-cavity
   fraction → per-session min-max calibration (closed/idle = p2, widest
   speech aperture = p99).
4. Metrics computed with the shared library in ``metrics.py`` so the LT arm
   is directly comparable with the LiveAvatar arms.

LiveTalking-specific latency semantics (documented for the paper): the
baseline's ``flush_talk`` clears only its input queue — in-flight pipeline
audio renders out. We therefore report ``audio_tail_after_cut_ms`` (cut →
last utt1 audio sample actually played) and ``visual_resume_ms`` (cut →
first utterance-2-driven frame) alongside the five standard metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from audio_synth import synth_utterance  # noqa: E402
from metrics import (  # noqa: E402
    half_syllable_flag,
    phoneme_viseme_mismatch,
    transition_smoothness,
    v_apt,
    viseme_buckets,
)

_SR = 16000
_ENERGY_THR = 300.0  # s16 amplitude threshold for "audible" samples
_FRAME_FPS = 25.0
_VAPT_WINDOW_S = 1.2  # transition-window 口径, mirrors analyze.py (P-B7 gate)
_YUNET = Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet_2023mar.onnx"


def _zoh_grid(u: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Zero-order hold (same semantics as analyze._zoh_grid): grid[i] holds
    the latest sample ≤ grid[i]; before the first sample hold values[0]."""
    idx = np.searchsorted(u, grid, side="right") - 1
    out = np.empty_like(grid)
    if len(u):
        ok = idx >= 0
        out[~ok] = values[0]
        out[ok] = values[idx[ok]]
    else:  # pragma: no cover - caller guards empty series
        out[:] = 0.0
    return out


def _yunet_ascii_path() -> str:
    """OpenCV's ONNX importer can fopen() non-ASCII absolute paths only when
    the ANSI code page encodes them; on some systems it fails (repo lives at
    E:\\项目\\). Copy the model once to an ASCII temp path and load from
    there."""
    import shutil
    import tempfile

    cached = Path(tempfile.gettempdir()) / "lt_extract_yunet.onnx"
    if not cached.exists() or cached.stat().st_size != _YUNET.stat().st_size:
        shutil.copyfile(_YUNET, cached)
    return str(cached)


def _pcm_float(utt: Any) -> np.ndarray:
    """Utterance.pcm_s16le bytes → float64 PCM for xcorr/energy analysis."""
    return np.frombuffer(utt.pcm_s16le, dtype=np.int16).astype(np.float64)


def demux_audio_pcm(mp4: Path) -> np.ndarray:
    """Decode the mp4 audio track to int16 mono 16 kHz (s16le via ffmpeg)."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(mp4),
         "-f", "s16le", "-ac", "1", "-ar", str(_SR), "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.int16).astype(np.float64)


def read_frames(mp4: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def xcorr_anchor(recorded: np.ndarray, wav: np.ndarray) -> int:
    """Sample offset of ``wav`` inside ``recorded`` (peak normalized xcorr).

    Both are float64 mono PCM. Uses FFT correlation; the recorded track is
    much longer than the wav so the search spans the whole session.
    """
    r = recorded - recorded.mean()
    w = wav - wav.mean()
    n = 1 << int(np.ceil(np.log2(len(r) + len(w) - 1)))
    corr = np.fft.irfft(np.fft.rfft(r, n) * np.conj(np.fft.rfft(w, n)), n)
    # only forward lags make sense (wav starts after recording start)
    fwd = corr[: len(r) - len(w) + 1]
    energy = np.sum(w * w)
    score = fwd / (energy + 1e-9)
    return int(np.argmax(score))


def _face_box(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """YuNet face box on one frame (returns x, y, w, h of the face rect)."""
    if not _YUNET.exists():
        raise FileNotFoundError(f"yunet model missing: {_YUNET}")
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(_yunet_ascii_path(), "", (w, h), 0.6)
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    faces = sorted(faces.tolist(), key=lambda f: -f[2] * f[3])
    x, y, fw, fh = faces[0][:4]
    return int(x), int(y), int(fw), int(fh)


def _mouth_roi(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Geometric mouth ROI inside the face box (static-head convention)."""
    x, y, fw, fh = box
    x0 = x + int(0.34 * fw)
    x1 = x + int(0.66 * fw)
    y0 = y + int(0.74 * fh)
    y1 = y + int(0.92 * fh)
    return x0, y0, x1, y1


def openness_series(
    frames: list[np.ndarray],
    box: tuple[int, int, int, int] | None = None,
    dark_v: float = 90.0,
) -> np.ndarray:
    """Per-frame mouth openness via dark-cavity fraction + self-calibration.

    ``box`` overrides the YuNet mid-frame face detection (used by tests to
    inject a synthetic geometry). ``dark_v`` is the cavity brightness
    threshold: MuseTalk's real-face recordings have genuinely dark oral
    cavities (90 works), while the 512x512 rendered avatars (P-E2b) render
    far brighter mouths — 150 keeps their p2→p99 dynamic range alive (the
    sun avatar at 90 degenerates to an all-zero series; documented in the
    pe5_main 口径说明).
    """
    if not frames:
        return np.zeros(0)
    if box is None:
        n = len(frames)
        mid = n // 2
        # Transition first-frames can be blurry enough for YuNet to miss;
        # retry outwards from the mid frame before giving up.
        offsets = [0] + [s * ((k + 1) // 2)
                         for k in range(1, n) for s in (1, -1)][: n - 1]
        for off in offsets:
            idx = min(max(mid + off, 0), n - 1)
            box = _face_box(frames[idx])
            if box is not None:
                break
    if box is None:
        raise RuntimeError("no face detected in any frame")
    x0, y0, x1, y1 = _mouth_roi(box)
    raw = np.array([
        float((f[y0:y1, x0:x1].max(axis=2) < dark_v).mean()) for f in frames
    ])
    lo, hi = np.percentile(raw, 2), np.percentile(raw, 99)
    if hi - lo < 1e-4:
        return np.zeros(len(raw))
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def _last_audible(pcm: np.ndarray, lo: int, hi: int) -> int | None:
    """Last sample index in [lo, hi) whose |amplitude| exceeds the threshold."""
    seg = np.abs(pcm[lo:hi])
    idx = np.nonzero(seg > _ENERGY_THR)[0]
    return int(idx[-1]) + lo if idx.size else None


def _pctl(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def analyze_lt_session(session_dir: str, wav_dir: str) -> dict[str, Any]:
    d = Path(session_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    mp4 = d / "recording.mp4"
    seed = int(meta["seed"])
    utt1 = synth_utterance(6.0, seed)
    utt2 = synth_utterance(2.5, seed + 1000)
    events = json.loads((Path(wav_dir) / f"events_seed{seed}.json").read_text(
        encoding="utf-8"))

    pcm = demux_audio_pcm(mp4)
    frames = read_frames(mp4)
    op = openness_series(frames)
    frame_ts = np.arange(len(op)) / _FRAME_FPS  # recording clock, seconds

    a1 = xcorr_anchor(pcm, _pcm_float(utt1))
    row: dict[str, Any] = {
        "kind": meta.get("kind"),
        "system": "livetalking",
        "avatar_id": meta.get("avatar_id"),
        "seed": seed,
        "repeat": meta.get("repeat"),
        "interrupt_at_s": meta.get("interrupt_at_s"),
        "n_frames": len(frames),
        "utt1_start_rec": a1 / _SR,
    }

    if meta.get("kind") == "reference":
        row["openness_p50"] = _pctl(list(op), 50)
        (d / "metrics.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        return row

    utt2_anchor = xcorr_anchor(pcm, _pcm_float(utt2))
    u2_start_rec = utt2_anchor / _SR
    clock = meta.get("perf_clock", {})
    utt1_start_perf = clock.get("utt1_start")
    interrupt_perf = clock.get("interrupt_sent")
    t_int_rec = None
    if utt1_start_perf is not None and interrupt_perf is not None:
        t_int_rec = (a1 + (interrupt_perf - utt1_start_perf) * _SR) / _SR

    # audio tail: last audible utt1 sample before utt2 audio starts
    tail_ms = None
    last = _last_audible(pcm, a1, max(utt2_anchor, a1 + 1))
    if last is not None and t_int_rec is not None:
        tail_ms = (last / _SR - t_int_rec) * 1000.0

    # visual resume: first frame after utt2 audio start with real motion
    resume_ms = None
    u2_frame = int(u2_start_rec * _FRAME_FPS)
    for i in range(max(u2_frame - 2, 1), len(op)):
        if abs(op[i] - op[i - 1]) >= 0.05:
            resume_ms = (frame_ts[i] - t_int_rec) * 1000.0 if t_int_rec is not None else None
            break

    row.update({
        "utt2_start_rec": u2_start_rec,
        "t_int_rec": t_int_rec,
        "audio_tail_after_cut_ms": tail_ms,
        "visual_resume_ms": resume_ms,
    })

    # ── shared five-metric schema (LiveAvatar-comparable) ──
    cut_idx = int(t_int_rec * _FRAME_FPS) if t_int_rec is not None else 0
    cut_idx = min(max(cut_idx, 1), len(op) - 1) if len(op) > 1 else 0
    epochs = np.zeros(len(op), dtype=np.int64)  # no epoch concept in LT
    sm = transition_smoothness(op, epochs, cut_idx)
    row["mutation"] = round(sm.mutation, 4)
    row["ratio"] = round(sm.ratio, 4) if sm.ratio is not None else None
    row["freeze_frames"] = int(sm.freeze_frames)
    row["first_jump"] = round(sm.mutation, 4)  # alias for report compat

    ref_dir = Path(session_dir).parent / f"reference_{meta.get('avatar_id')}_seed{seed}"
    if ref_dir.exists() and (ref_dir / "recording.mp4").exists():
        ref_pcm = demux_audio_pcm(ref_dir / "recording.mp4")
        ref_frames = read_frames(ref_dir / "recording.mp4")
        ref_op = openness_series(ref_frames)
        ref_a1 = xcorr_anchor(ref_pcm, _pcm_float(utt1))
        # align both series on the utterance-1 anchor (utterance clock),
        # zero-order hold — the browser keeps showing the last rendered
        # frame, so ZOH is the honest resampling (mirrors analyze.py)
        grid = np.arange(0.0, 6.0, 1.0 / _FRAME_FPS)
        obs_u = frame_ts - a1 / _SR
        ref_u = np.arange(len(ref_op)) / _FRAME_FPS - ref_a1 / _SR
        ref_on_grid = _zoh_grid(ref_u, ref_op, grid)
        obs_on_grid = _zoh_grid(obs_u, op, grid)
        cut_u = t_int_rec - a1 / _SR if t_int_rec is not None else 0.0
        cut_g = int(round(cut_u * _FRAME_FPS))
        if 0 < cut_g < len(grid):
            va = v_apt(
                obs_on_grid, ref_on_grid, cut_g, 1000.0 / _FRAME_FPS,
                max_span=int(np.ceil(_VAPT_WINDOW_S * _FRAME_FPS)),
            )
            row["v_apt_ms"] = round(va.apt_ms, 2)
            row["v_apt_recover_s"] = (
                round(grid[va.recover_idx], 3) if va.recover_idx is not None else None
            )
        # phoneme-viseme mismatch in [cut, cut+0.6s] (utterance-1 clock — the
        # pipeline backlog keeps playing utt1, so utt1 ground truth applies)
        buckets = viseme_buckets(ref_on_grid)
        tt = frame_ts
        row["mismatch_rate"] = round(
            phoneme_viseme_mismatch(
                buckets,
                np.array([buckets.bucket_of(v) for v in op]),
                tt,
                events["utt1"]["events"],
                t_int_rec if t_int_rec is not None else 0.0,
                0.6,
                a1 / _SR,
            ),
            4,
        )
    row["half_syllable"] = {
        "flag": half_syllable_flag(
            meta.get("interrupt_at_s") or 0.0, events["utt1"]["events"]
        )
        if meta.get("interrupt_at_s") is not None else None,
        "note": "flag is grid-inherited; LT audio is not truncated (see tail)",
    }
    (d / "metrics.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", help="one session dir, or a dataset root")
    ap.add_argument("--wav-dir", default="data/interruption_eval/lt_wavs")
    args = ap.parse_args(argv)

    root = Path(args.session)
    if (root / "meta.json").exists():
        rows = [analyze_lt_session(str(root), args.wav_dir)]
    else:
        rows = []
        for d in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
            try:
                rows.append(analyze_lt_session(str(d), args.wav_dir))
            except Exception as exc:  # noqa: BLE001
                rows.append({"dir": d.name, "error": str(exc)})
        (root / "lt_metrics.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in rows if not r.get("error"))
    print(f"analyzed {ok}/{len(rows)} sessions")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

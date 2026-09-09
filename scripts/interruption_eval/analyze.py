# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-B: offline analyzer over recorded eval sessions.

Measurement path (paper Section 4.5): mouth openness is extracted from
RECORDED PIXELS — the dark mouth-cavity area fraction in a fixed mouth
ROI, calibrated to the proxy's openness scale with two reference frames
(``render_face`` at 0.02 / 0.95). The spike (``spike_mediapipe.py``)
showed the pixel-area metric is linear and monotone on proxy faces
while mediapipe's inner-lip landmarks are not, so pixels are the primary
extractor; mediapipe stays available for real-face datasets later.

Time model (recorder design B): the TTS streams 160 ms chunks = exactly
one avatar batch (4 frames @ 25 fps), so frame ``k`` of an epoch sits at
``k × frame_period`` on that epoch's utterance clock. The analyzer maps

- epoch-1 frames to utterance-1 time ``u = k1 × period``;
- epoch-2 frames to ``u = u2_0 + k2 × period``, where ``u2_0`` is the
  epoch-2 anchor (arrival difference between the first epoch-2 frame and
  the first epoch-1 frame — both generated from their utterance's first
  TTS chunk);
- P-D transition frames (the first ``transition_frames`` epoch-2
  frames, published at cancel time) map to ``cut_u + (i+1) × period``:
  they ease the mouth shut right after the cut, so they belong to the
  utterance-1 clock, not to the utterance-2 schedule;
- the screen-freeze gap between the cut and the epoch-2 anchor holds the
  last rendered openness (zero-order hold): with no new frames the
  browser keeps showing the last one, which IS the hard-cut artifact.

Per interrupted session the analyzer computes (see ``metrics.py``):

- ``smoothness`` — freeze-aware first-jump |Δopenness| vs pre-cut
  baseline, on the real frame sequence;
- ``v_apt`` — perturbation area vs the no-interrupt reference
  trajectory, TRANSITION-WINDOW 口径: the area integrates at most
  ``vapt_window_s`` (default 1.2 s) after the cut on a zero-order-hold
  grid (freeze included). Rationale: after a barge-in the agent stops
  and then answers the follow-up, so a full-utterance reference diff
  would be dominated by the intentional content change, not by the
  artifact. The window isolates freeze + jump + mismatched-motion
  transition, which is what raters judge; the P-B7 Spearman gate
  validates this choice.
- ``half_syllable`` — whether the cut lands inside a non-pause phoneme;
- ``mismatch`` — phoneme-vs-viseme bucket disagreement in the first
  ``delta_s`` (0.6 s) after the cut. The audio side is the ACTUAL
  post-cut audio: silence (pause) until the epoch-2 anchor, then the
  utterance-2 phoneme schedule.

Writes ``metrics.json`` next to each session's ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import metrics as m  # noqa: E402

# Mouth ROI (relative to the 256x256 proxy face): mouth center sits at
# (w/2 ± 2, h/2 + 50 ± 2) with the head bob; cavity half-axes ≤ 33/26.
_ROI_DX = 45
_ROI_DY = 35
_MOUTH_CY_OFF = 50
_DARK_V = 100  # cavity BGR max-channel threshold (cavity ≈ 70, skin ≈ 226)

_CAL_OPEN_LO, _CAL_OPEN_HI = 0.02, 0.95
_CAL: tuple[float, float] | None = None  # (raw_closed, raw_open)


def _raw_cavity_fraction(bgr: np.ndarray) -> float:
    """Dark-pixel (mouth cavity) area fraction inside the mouth ROI."""
    h, w = bgr.shape[:2]
    cx = w // 2
    cy = h // 2 + _MOUTH_CY_OFF
    x0, x1 = max(cx - _ROI_DX, 0), min(cx + _ROI_DX, w)
    y0, y1 = max(cy - _ROI_DY, 0), min(cy + _ROI_DY, h)
    roi = bgr[y0:y1, x0:x1]
    return float((roi.max(axis=2) < _DARK_V).mean())


def _calibration() -> tuple[float, float]:
    """Two-point calibration from the proxy renderer itself (offline)."""
    global _CAL
    if _CAL is None:
        from mouth_proxy import render_face

        lo = _raw_cavity_fraction(render_face(_CAL_OPEN_LO, t_s=0.0))
        hi = _raw_cavity_fraction(render_face(_CAL_OPEN_HI, t_s=1.0))
        if hi - lo < 1e-4:  # pragma: no cover - renderer regression guard
            raise RuntimeError("mouth cavity calibration is degenerate")
        _CAL = (lo, hi)
    return _CAL


def extract_openness(bgr: np.ndarray) -> float:
    """Calibrated mouth openness [0, 1] from one recorded BGR frame."""
    lo, hi = _calibration()
    raw = _raw_cavity_fraction(bgr)
    op = _CAL_OPEN_LO + (raw - lo) / (hi - lo) * (_CAL_OPEN_HI - _CAL_OPEN_LO)
    return float(np.clip(op, 0.0, 1.0))


# ─────────────────────────────────────────────────────── session loading


def load_series(session_dir: str) -> dict[str, Any]:
    """meta.json + per-frame (ts_rel, epoch, u, openness) arrays.

    ``u`` is the utterance-1 wall clock (seconds since utterance 1
    started): epoch-1 frames at ``k1 × period``, epoch-2 frames at
    ``u2_0 + k2 × period`` (see module docstring). P-D transition
    frames (the first ``transition_frames`` epoch-2 frames, published
    at cancel time) sit BETWEEN the two clocks: they ease the mouth
    shut right after the cut, so frame ``i`` maps to
    ``cut_u + (i+1) × period`` on the utterance-1 clock; the driven
    utterance-2 frames anchor at ``utt2_driven_start_s`` (P-B datasets
    without that field fall back to ``utt2_start_s``, which is then
    the same instant).
    """
    import cv2

    with open(os.path.join(session_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    timeline = meta.get("frame_timeline") or []
    ts = np.array([f["ts_rel"] for f in timeline], dtype=np.float64)
    epochs = np.array([f["epoch"] for f in timeline], dtype=np.int64)
    boundary = np.array([f["boundary"] for f in timeline], dtype=bool)
    op = np.empty(len(timeline), dtype=np.float64)
    for i, f in enumerate(timeline):
        path = os.path.join(session_dir, "frames", f["file"])
        # cv2.imread fails on non-ASCII Windows paths — decode from bytes.
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        op[i] = extract_openness(img)

    period = float(meta.get("frame_period_s") or 0.04)
    n_trans = int(meta.get("transition_frames") or 0)
    cut_u = meta.get("cut_in_utterance_s")
    driven = meta.get("utt2_driven_start_s")
    u2_start = meta.get("utt2_start_s")
    anchor = driven if driven is not None else u2_start
    u2_0: float | None = None
    if anchor is not None and len(timeline):
        u2_0 = float(anchor) - float(ts[0])
    u = np.empty(len(timeline), dtype=np.float64)
    if len(timeline):
        k1 = k2 = 0
        n_seen = 0
        for i, e in enumerate(epochs):
            if e > 1 and u2_0 is not None:
                if n_seen < n_trans and cut_u is not None:
                    # Transition bridge frame i (α-schedule): rendered at
                    # the cut, one frame slot later on the utterance-1
                    # clock — the ZOH grid at the cut instant still holds
                    # the last pre-cut frame.
                    u[i] = float(cut_u) + (n_seen + 1) * period
                    n_seen += 1
                else:
                    u[i] = u2_0 + k2 * period
                    k2 += 1
            else:
                u[i] = k1 * period
                k1 += 1
    return {"dir": session_dir, "meta": meta, "ts": ts, "epochs": epochs,
            "boundary": boundary, "op": op, "u": u, "period": period}


def _zoh_grid(u: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Zero-order hold: grid[i] = value of the latest sample ≤ grid[i]."""
    idx = np.searchsorted(u, grid, side="right") - 1
    out = np.empty_like(grid)
    if len(u):
        first = float(values[0])
        ok = idx >= 0
        out[~ok] = first
        out[ok] = values[idx[ok]]
    else:  # pragma: no cover - caller guards empty series
        out[:] = 0.0
    return out


# ────────────────────────────────────────────────────────── per-session


def analyze_session(
    session_dir: str,
    ref_dir: str | None = None,
    *,
    delta_s: float = 0.6,
    vapt_window_s: float = 1.2,
) -> dict:
    """Compute paper metrics for one recorded session; write metrics.json."""
    s = load_series(session_dir)
    meta = s["meta"]
    out: dict[str, Any] = {
        "kind": meta.get("kind"),
        "avatar_id": meta.get("avatar_id"),
        "seed": meta.get("seed"),
        "frames": int(len(s["ts"])),
        "frame_period_ms": round(s["period"] * 1000.0, 2),
    }
    if not len(s["ts"]):
        out["error"] = "no frames"
        _write(session_dir, out)
        return out

    if meta.get("kind") != "interrupt" or meta.get("interrupt_sent_s") is None:
        out["note"] = "reference session — no cut metrics"
        _write(session_dir, out)
        return out

    cut_idx = int(np.searchsorted(s["ts"], meta["interrupt_sent_s"]))
    if cut_idx <= 0 or cut_idx > len(s["ts"]):
        out["error"] = f"cut_idx {cut_idx} outside frame timeline"
        _write(session_dir, out)
        return out
    cut_u = meta.get("cut_in_utterance_s")
    if cut_u is None:
        out["error"] = "meta missing cut_in_utterance_s"
        _write(session_dir, out)
        return out
    cut_u = float(cut_u)

    # ── transition smoothness: real frame sequence, freeze-aware ──
    pre_epoch = int(s["epochs"][:cut_idx].max()) if cut_idx else 0
    sm = m.transition_smoothness(
        s["op"], s["epochs"], cut_idx, old_epoch=int(pre_epoch)
    )
    out["smoothness"] = {
        "baseline_p50": round(sm.baseline_p50, 5),
        "mutation": round(sm.mutation, 5),
        "ratio": round(sm.ratio, 2) if sm.ratio is not None else None,
        "freeze_frames": sm.freeze_frames,
    }

    # ── V-APT on the zero-order-hold grid (freeze gap included) ──
    period = s["period"]
    grid_end = cut_u + vapt_window_s
    grid = np.arange(0.0, grid_end + 1e-9, period)
    grid_op = _zoh_grid(s["u"], s["op"], grid)

    if ref_dir is not None and os.path.isdir(ref_dir):
        ref = load_series(ref_dir)
        if len(ref["u"]):
            ref_grid = _zoh_grid(ref["u"], ref["op"], grid)
            cut_grid_idx = int(round(cut_u / period))
            max_span = int(np.ceil(vapt_window_s / period))
            v = m.v_apt(
                grid_op, ref_grid, cut_grid_idx, period * 1000.0,
                max_span=max_span,
            )
            out["v_apt"] = {
                "area": round(v.area, 4),
                "apt_ms": round(v.apt_ms, 1),
                "recover_idx": v.recover_idx,
                "span_frames": v.span_frames,
                "window_frames": max_span,
                "ref_dir": os.path.basename(ref_dir),
            }

            # ── phoneme-viseme mismatch on the ACTUAL post-cut audio ──
            # Audio in [cut, cut+delta]: pause (silence) until the
            # utterance-2 DRIVEN anchor (transition bridge frames arrive
            # before any utterance-2 audio), then the utterance-2
            # schedule.
            anchor = meta.get("utt2_driven_start_s")
            if anchor is None:
                anchor = meta.get("utt2_start_s")
            u2_0 = float(anchor) - float(s["ts"][0]) if anchor is not None \
                else None
            merged: list[dict] = []
            if u2_0 is not None:
                merged.append({
                    "phoneme": "_", "kind": "pause",
                    "start_s": 0.0, "end_s": round(u2_0, 4), "openness": 0.05,
                })
                for e in (meta.get("utterance_2") or {}).get("events", []):
                    merged.append({
                        "phoneme": e["phoneme"], "kind": e["kind"],
                        "start_s": round(u2_0 + e["start_s"], 4),
                        "end_s": round(u2_0 + e["end_s"], 4),
                        "openness": e["openness"],
                    })
            else:
                merged.append({
                    "phoneme": "_", "kind": "pause",
                    "start_s": 0.0, "end_s": 1e9, "openness": 0.05,
                })
            buckets = m.viseme_buckets(ref_grid, n_buckets=4)
            labels = np.array(
                [buckets.bucket_of(v_) for v_ in grid_op], dtype=np.int64
            )
            mm = m.phoneme_viseme_mismatch(
                buckets, labels, grid, merged,
                t_cut_s=cut_u, delta_s=delta_s, tts_start_s=0.0,
            )
            out["mismatch"] = {
                "rate": round(mm, 4),
                "delta_s": delta_s,
                "audio": "pause+utt2" if u2_0 is not None else "pause",
            }

    cut_in_utt = meta.get("cut_in_utterance_s")
    if cut_in_utt is not None:
        flag = m.half_syllable_flag(float(cut_in_utt), meta["utterance"]["events"])
        out["half_syllable"] = {"flag": flag, "cut_in_utterance_s": cut_in_utt}

    # Client + server latency rows (audit / cross-check vs P-A).
    for key in ("stale_tail_ms", "resume_ms", "client_e2e_ms"):
        if meta.get(key) is not None:
            out.setdefault("latency", {})[key] = meta[key]
    if meta.get("server_timeline"):
        out["server_timeline"] = meta["server_timeline"]

    _write(session_dir, out)
    return out


def _write(session_dir: str, out: dict) -> None:
    with open(os.path.join(session_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────── main


def find_ref_dir(data_root: str, seed: int, avatar_hint: str | None = None) -> str | None:
    """Reference session for ``seed`` (recorded once per seed)."""
    import glob

    pattern = os.path.join(data_root, f"reference_*_seed{seed}")
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def analyze_all(data_root: str, **kwargs: Any) -> list[dict]:
    """Analyze every interrupt_* session under ``data_root``."""
    import glob

    rows: list[dict] = []
    for d in sorted(glob.glob(os.path.join(data_root, "interrupt_*"))):
        meta_path = os.path.join(d, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path, encoding="utf-8") as f:
            seed = json.load(f).get("seed")
        ref = find_ref_dir(data_root, seed)
        try:
            rows.append(analyze_session(d, ref, **kwargs))
            err = rows[-1].get("error")
            print(f"[{os.path.basename(d)}] "
                  f"vapt={rows[-1].get('v_apt', {}).get('apt_ms')} "
                  f"mut={rows[-1].get('smoothness', {}).get('mutation')} "
                  f"{'ERR ' + str(err) if err else ''}",
                  flush=True)
        except Exception as exc:  # keep going; report collects errors
            rows.append({"dir": d, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{os.path.basename(d)}] ERROR {exc}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str,
                        default="data/interruption_eval/dataset")
    parser.add_argument("--delta-s", type=float, default=0.6)
    parser.add_argument("--vapt-window-s", type=float, default=1.2)
    args = parser.parse_args(argv)

    rows = analyze_all(
        args.data, delta_s=args.delta_s, vapt_window_s=args.vapt_window_s
    )
    bad = [r for r in rows if r.get("error")]
    print(f"analyzed {len(rows)} sessions ({len(bad)} errors)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

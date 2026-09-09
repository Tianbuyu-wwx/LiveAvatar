# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-B metric library — pure numpy functions over measurement series.

Metrics (paper Section 4; definitions pinned in code before writing):

1. ``transition_smoothness`` — abruptness of the mouth trajectory at the
   cut: the first significant |Δopenness| jump after the cut (freeze-
   aware) vs the normal-playback baseline (paper 4.6 过渡平滑度).
2. ``v_apt`` — Visual Average Penalty Time: perturbation area between the
   interrupted mouth-openness trajectory and the no-interrupt reference,
   normalized by frame period; recovery = |diff| below tolerance for a
   hold window AND real motion inside that window (a frozen screen that
   happens to match a reference pause is NOT recovered) (paper 4.4).
3. ``viseme_buckets`` — quantile clustering of the openness trajectory
   into viseme buckets (paper 4.5 视位桶).
4. ``phoneme_viseme_mismatch`` — fraction of frames in the interrupt
   window whose measured viseme bucket differs from the bucket implied
   by the ground-truth audio phoneme at that instant.
5. ``half_syllable_flag`` — 1.0 when the cut lands strictly inside a
   non-pause phoneme (paper 4.5 半音节伪影率; dataset-level rate =
   mean of flags).

All functions take plain floats/arrays so they are unit-testable on
synthetic trajectories without any media dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ───────────────────────────────────────────── 1. transition smoothness


@dataclass(frozen=True)
class SmoothnessResult:
    """Amplification of mouth-trajectory abruptness at the cut."""

    baseline_p50: float  # median |Δopenness| in the pre-cut window
    mutation: float  # first significant |Δopenness| jump after the cut
    ratio: float | None  # mutation / baseline_p50 (None when baseline is 0)
    freeze_frames: int  # frozen frames between last old frame and the jump


def transition_smoothness(
    openness: np.ndarray,
    epochs: np.ndarray,
    cut_idx: int,
    *,
    baseline_window: int = 40,
    peak_window: int = 25,
    old_epoch: int | None = None,
) -> SmoothnessResult:
    """First significant |Δopenness| jump after ``cut_idx`` vs baseline.

    The hard-cut artifact is freeze-then-jump: the mouth freezes at its
    last openness for a few frames (stale tail), then jumps when the new
    utterance's first frames land. The metric therefore reports the
    FIRST jump whose |Δ| reaches 25% of the window peak — not the peak
    itself, which later normal motion (e.g. mouth closing) can dominate.

    ``openness``/``epochs`` are per-frame series; ``epochs`` carries the
    pipeline epoch each frame belongs to. ``cut_idx`` is the first frame
    index at/after the interrupt instant.
    """
    n = len(openness)
    lo = max(0, cut_idx - baseline_window)
    pre = openness[lo:cut_idx]
    deltas = np.abs(np.diff(pre)) if len(pre) >= 2 else np.array([0.0])
    baseline = float(np.median(deltas))

    if old_epoch is None:
        old_epoch = int(epochs[cut_idx - 1]) if cut_idx > 0 else (int(epochs[0]) if n else 0)
    old_mask = (epochs[:cut_idx] == old_epoch) if cut_idx else np.zeros(0, bool)
    last_old = int(np.max(np.nonzero(old_mask)[0])) if old_mask.any() else cut_idx - 1

    hi = min(n, cut_idx + peak_window + 1)
    start = max(0, last_old)
    post = np.abs(np.diff(openness[start:hi]))  # rel r → frame start+1+r
    if post.size == 0:
        return SmoothnessResult(baseline, 0.0, None, 0)

    peak = float(post.max())
    threshold = 0.25 * peak
    rel = int(np.argmax(post >= threshold))
    mutation = float(post[rel])
    ratio = mutation / baseline if baseline > 1e-9 else None
    return SmoothnessResult(baseline, mutation, ratio, rel)


# ──────────────────────────────────────────────────────────── 2. V-APT


@dataclass(frozen=True)
class VaptResult:
    """Perturbation area between interrupted and reference trajectories."""

    area: float  # Σ|openness - reference| over the perturbed span (openness·frames)
    apt_ms: float  # area × frame period × 1000 (ms of full-penalty time)
    recover_idx: int | None  # first recovery frame (None = no recovery)
    span_frames: int  # frames from cut to recovery (or end of series)


def v_apt(
    openness: np.ndarray,
    reference: np.ndarray,
    cut_idx: int,
    frame_period_ms: float,
    *,
    recover_tol: float = 0.08,
    recover_hold: int = 5,
    alive_tol: float = 0.02,
    max_span: int = 10000,
) -> VaptResult:
    """Perturbation area vs the no-interrupt reference after ``cut_idx``.

    Recovery requires BOTH (a) value sync — |openness - reference| below
    ``recover_tol`` for ``recover_hold`` consecutive frames — and (b)
    aliveness — the interrupted trajectory actually MOVES (|Δopenness| ≥
    ``alive_tol``) somewhere inside that confirming run. Requirement (b)
    blocks the false recovery where a frozen screen (zero-order hold,
    Δ = 0) coincides with a reference pause: the artifact is still on
    screen and the area must keep accumulating once the reference moves
    away. ``recover_idx`` is the END of the confirming run; the area
    integrates the absolute deviation up to the START of that run.
    ``apt_ms`` is the equivalent time at full penalty (area × period).
    """
    n = min(len(openness), len(reference))
    span_end = min(n, cut_idx + max_span)
    diff = np.abs(openness[cut_idx:span_end] - reference[cut_idx:span_end])
    recover_idx: int | None = None
    area_end = len(diff)
    run = 0
    for i, d in enumerate(diff):
        run = run + 1 if d < recover_tol else 0
        if run >= recover_hold:
            s0 = i - run + 1
            seg = openness[cut_idx + s0: cut_idx + i + 1]
            alive = bool(np.any(np.abs(np.diff(seg)) >= alive_tol))
            if alive:
                recover_idx = cut_idx + i  # end of the confirming run
                area_end = s0  # start of the run
                break
    area = float(np.sum(diff[:area_end]))
    return VaptResult(
        area=area,
        apt_ms=area * frame_period_ms,
        recover_idx=recover_idx,
        span_frames=(recover_idx - cut_idx) if recover_idx is not None else len(diff),
    )


# ─────────────────────────────────────────────── 3. viseme buckets


@dataclass(frozen=True)
class VisemeBuckets:
    """Quantile viseme clustering of an openness trajectory."""

    edges: np.ndarray  # bucket edges on the openness axis, len = n+1
    labels: np.ndarray  # per-frame bucket index 0..n-1

    def bucket_of(self, openness_value: float) -> int:
        return int(
            np.clip(
                np.searchsorted(self.edges, openness_value, "right") - 1,
                0, len(self.edges) - 2,
            )
        )


def viseme_buckets(openness: np.ndarray, n_buckets: int = 4) -> VisemeBuckets:
    """Bucket frames by openness quantiles (closed→open mouth classes).

    Quantile edges are computed over the WHOLE series so interrupted and
    reference trajectories are bucketed identically when the caller
    passes the concatenation (or the reference) — pass the same edges by
    constructing buckets on the reference series and using ``bucket_of``
    for the interrupted one.
    """
    q = np.quantile(openness, np.linspace(0.0, 1.0, n_buckets + 1)[1:-1])
    edges = np.concatenate(([0.0], q, [1.0 + 1e-9]))
    edges = np.maximum.accumulate(edges)  # guard duplicate quantiles
    labels = np.clip(np.searchsorted(edges, openness, "right") - 1, 0, n_buckets - 1)
    return VisemeBuckets(edges=edges, labels=labels)


# ──────────────────────── 4. phoneme-viseme mismatch in the cut window


def phoneme_viseme_mismatch(
    buckets: VisemeBuckets,
    measured_labels: np.ndarray,
    frame_times_s: np.ndarray,
    phoneme_events: list[dict],
    t_cut_s: float,
    delta_s: float,
    tts_start_s: float,
) -> float:
    """Fraction of mismatching frames in ``[t_cut, t_cut + delta]``.

    ``phoneme_events`` is the ground-truth audio schedule (list of dicts
    with ``start_s``/``end_s``/``openness`` in utterance time); utterance
    time maps to session time via ``tts_start_s`` (when the TTS stream
    began). Expected bucket per frame = bucket_of(phoneme openness at
    that utterance instant); a frame matches when its measured label
    equals the expected label.
    """
    lo = t_cut_s
    hi = t_cut_s + delta_s
    idx = np.where((frame_times_s >= lo) & (frame_times_s < hi))[0]
    if idx.size == 0:
        return 0.0
    mismatched = 0
    for i in idx:
        ut = float(frame_times_s[i]) - tts_start_s
        op = _openness_at(phoneme_events, ut)
        expected = buckets.bucket_of(op)
        if int(measured_labels[i]) != expected:
            mismatched += 1
    return mismatched / float(idx.size)


def _openness_at(phoneme_events: list[dict], ut: float) -> float:
    for e in phoneme_events:
        if e["start_s"] <= ut < e["end_s"]:
            return float(e["openness"])
    return float(phoneme_events[-1]["openness"]) if phoneme_events else 0.0


# ─────────────────────────────────────── 5. half-syllable artifact flag


def half_syllable_flag(cut_in_utterance_s: float, phoneme_events: list[dict]) -> float:
    """1.0 when the cut lands strictly inside a non-pause phoneme."""
    for e in phoneme_events:
        if e["kind"] == "pause":
            continue
        if e["start_s"] < cut_in_utterance_s < e["end_s"]:
            return 1.0
    return 0.0

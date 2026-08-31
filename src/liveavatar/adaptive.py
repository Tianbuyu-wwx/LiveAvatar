"""Adaptive quality for the self-developed transport (R2 M5).

Video-WS clients periodically report congestion signals (sequence gaps =
frames the server dropped for them, received bitrate). A per-sink
:class:`FeedbackAggregator` smooths them with an EWMA and a
:class:`QualityController` maps the signals onto one of five quality
tiers, trading JPEG quality and keyframe cadence for throughput:

    tier 0 (excellent) → quality 80, keyframe every 1.0 s
    tier 4 (poor)      → quality 40, keyframe every 0.5 s  (faster base
                         refresh so a recovered client resyncs quickly)

"降画质不冻结" (degrade quality, never freeze): the tier never blocks
publishing — it only changes encoder parameters. Degradation keys on the
EWMA **confirmed by a congested current window** (a healthy report never
degrades, so the EWMA tail of a healed link cannot stall recovery);
recovery keys on N **consecutive raw healthy reports**. Both paths are
streak/hold-protected against flapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityTier:
    """One rung of the quality ladder."""

    name: str
    quality: int  # JPEG quality (1-100)
    keyframe_interval_us: int  # base-image refresh cadence


TIERS: list[QualityTier] = [
    QualityTier("excellent", 80, 1_000_000),
    QualityTier("good", 70, 1_000_000),
    QualityTier("fair", 62, 750_000),
    QualityTier("poor", 52, 500_000),
    QualityTier("bad", 40, 500_000),
]


@dataclass
class FeedbackSignals:
    """One client report (already computed client-side)."""

    seq_gap_rate: float  # dropped frames per delivered frame (0 = good)
    kbps: float = 0.0  # received bitrate (informational)
    fps: float = 0.0  # rendered fps (informational)


@dataclass
class _Ewma:
    """Exponentially weighted moving average (bias-corrected warmup)."""

    alpha: float = 0.3
    value: float | None = None
    _n: int = 0

    def update(self, sample: float) -> float:
        self._n += 1
        if self.value is None:
            self.value = sample
        else:
            # Warmup correction keeps early samples responsive.
            a = max(self.alpha, 1.0 / self._n)
            self.value = a * sample + (1 - a) * self.value
        return self.value


@dataclass
class FeedbackAggregator:
    """EWMA aggregate of all connected clients' congestion signals."""

    alpha: float = 0.3
    gap_rate: _Ewma = field(default_factory=lambda: _Ewma(0.3))
    reports: int = 0
    # Raw most-recent report, kept alongside the EWMA: recovery keys on the
    # fresh per-window signal, degradation on the smoothed one.
    last_gap_rate: float = 0.0

    def update(self, signals: FeedbackSignals) -> None:
        self.reports += 1
        self.last_gap_rate = max(0.0, signals.seq_gap_rate)
        self.gap_rate.update(self.last_gap_rate)

    @property
    def smoothed_gap_rate(self) -> float:
        return self.gap_rate.value or 0.0


class QualityController:
    """Tier state machine: degrade fast, recover verified ("2s 回满").

    Degrade: one EWMA threshold crossing after a short hold drops one tier
    — but only while the current window is itself congested, so the EWMA
    tail of an already-healed link can neither degrade further nor stall
    recovery. Recover: only N **consecutive raw healthy reports**
    (per-window gap below the threshold) restore tier 0 in a single step;
    requiring the streak (not a single report) keeps the hysteresis
    against flapping.
    """

    # Tier changes when the smoothed gap rate crosses these thresholds.
    DEGRADE_GAP = 0.02  # >2% of frames dropped → worse tier
    RECOVER_GAP = 0.005  # <0.5% sustained → better tier
    MIN_HOLD_REPORTS = 4  # hysteresis: stay ≥4 reports before degrading
    RECOVER_STREAK = 3  # consecutive healthy reports → full recovery

    def __init__(self, tier_index: int = 0) -> None:
        self.tier_index = min(tier_index, len(TIERS) - 1)
        self._since_change = 0
        self._healthy_streak = 0

    @property
    def tier(self) -> QualityTier:
        return TIERS[self.tier_index]

    def update(self, agg: FeedbackAggregator) -> bool:
        """Feed aggregated signals; returns True when the tier changed."""
        self._since_change += 1
        gap = agg.smoothed_gap_rate
        changed = False
        if (
            gap > self.DEGRADE_GAP
            and agg.last_gap_rate >= self.RECOVER_GAP
            and self.tier_index < len(TIERS) - 1
        ):
            self._healthy_streak = 0
            if self._since_change >= self.MIN_HOLD_REPORTS // 2:
                self.tier_index += 1  # degrade fast
                changed = True
        elif (
            agg.last_gap_rate < self.RECOVER_GAP and self.tier_index > 0
        ):
            self._healthy_streak += 1
            if self._healthy_streak >= self.RECOVER_STREAK:
                # Verified healthy: jump straight back to full quality.
                self.tier_index = 0
                changed = True
        else:
            self._healthy_streak = 0
        if changed:
            self._since_change = 0
            self._healthy_streak = 0
        return changed

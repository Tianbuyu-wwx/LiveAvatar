# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for adaptive quality + feedback (R2 M5) — pure CPU.

Covers: EWMA convergence, tier state machine (degrade fast / recover
slowly with hysteresis), WebSocketSink.apply_feedback integration, and
the /video WS feedback round trip.
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from liveavatar.adaptive import (
    TIERS,
    FeedbackAggregator,
    FeedbackSignals,
    QualityController,
)
from liveavatar.ws_sink import WebSocketSink
from tests.test_ws_sink import _FakeEncoder, _frame


class EwmaTests(unittest.TestCase):
    def test_first_sample_is_identity(self) -> None:
        agg = FeedbackAggregator()
        agg.update(FeedbackSignals(seq_gap_rate=0.10))
        self.assertAlmostEqual(agg.smoothed_gap_rate, 0.10)

    def test_converges_toward_constant(self) -> None:
        agg = FeedbackAggregator()
        for _ in range(60):
            agg.update(FeedbackSignals(seq_gap_rate=0.05))
        self.assertLess(abs(agg.smoothed_gap_rate - 0.05), 0.005)

    def test_zero_reports_stay_zero(self) -> None:
        agg = FeedbackAggregator()
        agg.update(FeedbackSignals(seq_gap_rate=0.0))
        agg.update(FeedbackSignals(seq_gap_rate=0.0))
        self.assertEqual(agg.smoothed_gap_rate, 0.0)


class QualityControllerTests(unittest.TestCase):
    def _agg_with(self, gap: float) -> FeedbackAggregator:
        agg = FeedbackAggregator()
        agg.update(FeedbackSignals(seq_gap_rate=gap))
        return agg

    def test_degrades_fast_on_congestion(self) -> None:
        ctrl = QualityController()
        agg = self._agg_with(0.5)
        changed = [ctrl.update(agg) for _ in range(6)]
        # Degrade quickly (half-hold) — 6 reports should drop ≥2 tiers.
        self.assertGreaterEqual(ctrl.tier_index, 2)
        self.assertTrue(any(changed))

    def test_recovers_slowly_with_hysteresis(self) -> None:
        ctrl = QualityController(tier_index=2)
        agg = self._agg_with(0.0)
        results = [ctrl.update(agg) for _ in range(20)]
        self.assertLess(ctrl.tier_index, 2, "should climb back down")
        # Hysteresis: not every report changes the tier.
        self.assertLess(sum(results), 5)
        self.assertEqual(ctrl.tier_index, 0)  # fully recovered eventually

    def test_recovery_is_verified_then_full(self) -> None:
        """3 consecutive healthy reports → straight back to tier 0 (≤2 s
        at the client's 500 ms reporting interval), not gradual climbing."""
        ctrl = QualityController(tier_index=4)
        agg = self._agg_with(0.0)
        results = [ctrl.update(agg) for _ in range(5)]
        self.assertEqual(results[:2], [False, False])  # verification window
        self.assertTrue(results[2])
        self.assertEqual(ctrl.tier_index, 0)
        self.assertEqual(sum(results), 1)  # one decisive change, no flapping

    def test_flapping_signals_do_not_recover(self) -> None:
        """A single healthy report never triggers recovery."""
        ctrl = QualityController(tier_index=2)
        for i in range(20):
            gap = 0.0 if i % 3 == 0 else 0.01  # healthy / mediocre mix
            ctrl.update(self._agg_with(gap))
        self.assertEqual(ctrl.tier_index, 2)

    def test_recovery_ignores_stale_average(self) -> None:
        """The EWMA tail of a lossy period must not delay the 2 s
        recovery: the streak keys on raw per-window reports."""
        ctrl = QualityController(tier_index=3)
        agg = FeedbackAggregator()
        agg.update(FeedbackSignals(seq_gap_rate=0.5))  # lossy history
        for _ in range(3):
            agg.update(FeedbackSignals(seq_gap_rate=0.0))
            ctrl.update(agg)
        self.assertEqual(ctrl.tier_index, 0)

    def test_healthy_window_never_degrades(self) -> None:
        """Degrade requires the current window to be congested, so the
        EWMA tail of an already-healed link can't push the tier down."""
        ctrl = QualityController(tier_index=1)
        agg = FeedbackAggregator()
        agg.update(FeedbackSignals(seq_gap_rate=0.5))  # smoothed > threshold
        for _ in range(6):
            agg.update(FeedbackSignals(seq_gap_rate=0.0))
            ctrl.update(agg)
        self.assertEqual(ctrl.tier_index, 0)  # recovered, never degraded

    def test_healthy_signals_never_degrade(self) -> None:
        ctrl = QualityController()
        agg = self._agg_with(0.001)
        for _ in range(30):
            self.assertFalse(ctrl.update(agg))
        self.assertEqual(ctrl.tier_index, 0)

    def test_tiers_are_ordered(self) -> None:
        qualities = [t.quality for t in TIERS]
        self.assertEqual(qualities, sorted(qualities, reverse=True))


class SinkFeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_feedback_degrades_quality_and_requests_keyframe(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        self.assertEqual(sink.quality, 80)
        for _ in range(6):
            sink.apply_feedback(
                client,
                {"type": "feedback", "seq_gaps": 50, "frames": 100},
            )
        self.assertLess(sink.quality, 80)
        self.assertEqual(sink.stats()["tier"], TIERS[sink.controller.tier_index].name)
        self.assertTrue(client.wants_keyframe)  # resync at the new tier

    async def test_feedback_recovers_quality(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        for _ in range(8):
            sink.apply_feedback(
                client, {"seq_gaps": 50, "frames": 100}
            )
        degraded = sink.quality
        self.assertLess(degraded, 80)
        for _ in range(30):
            sink.apply_feedback(client, {"seq_gaps": 0, "frames": 100})
        self.assertEqual(sink.quality, 80)

    async def test_feedback_with_zero_frames_is_safe(self) -> None:
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        sink.apply_feedback(client, {"seq_gaps": 3, "frames": 0})
        self.assertEqual(sink.stats()["smoothed_gap_rate"], 0.0)

    async def test_quality_flows_into_wire_frames(self) -> None:
        """After degradation the next keyframe carries the new quality."""
        sink = WebSocketSink(encoder=_FakeEncoder())
        client = sink.add_client()
        await sink.publish_frame(_frame(pts_us=0), epoch=0)
        for _ in range(8):
            sink.apply_feedback(client, {"seq_gaps": 50, "frames": 100})
        await sink.publish_frame(_frame(pts_us=40_000), epoch=0)
        from liveavatar.video_protocol import unpack_video_frame

        header, _ = unpack_video_frame(client.queue.get_nowait())
        header2, _ = unpack_video_frame(client.queue.get_nowait())
        self.assertEqual(header.quality, 80)
        self.assertEqual(header2.quality, sink.quality)
        self.assertLess(header2.quality, 80)


class VideoWsFeedbackTests(unittest.TestCase):
    """/video WS feedback round trip (TestClient, CPU)."""

    def setUp(self) -> None:
        from liveavatar.publish import app
        from tests.test_video_ws import _configure_ws_transport_mode

        _configure_ws_transport_mode()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        from liveavatar.publish import state

        state.pipeline = None

    def test_feedback_over_ws_changes_tier(self) -> None:

        resp = self.client.post("/v1/sessions", json={"avatar_id": "yongen"})
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()["session_id"]
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video"
        ) as video:
            self.assertEqual(video.receive_json()["type"], "ready")
            for _ in range(10):
                video.send_json(
                    {"type": "feedback", "seq_gaps": 50, "frames": 100}
                )
            stats = self.client.get(
                f"/v1/sessions/{session_id}/stats"
            ).json()
            self.assertNotEqual(stats["publisher"]["tier"], "excellent")
            self.assertLess(stats["publisher"]["quality"], 80)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

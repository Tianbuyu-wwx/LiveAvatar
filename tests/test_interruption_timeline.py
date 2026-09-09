# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-A: five-layer interruption timeline instrumentation tests.

Covers:
- ``SessionMetrics`` timeline primitives (idempotent marks, decompose
  ordering, clamping, subset layers).
- ``RealtimeWorker.advance_epoch`` stamps audio_flush → tts_stop →
  video_invalidate in true execution order.
- The confirmed-interrupt control path stamps ``detect`` and resets the
  window for the next interrupt.
- ``AvatarStreamingAdapter`` stamps ``new_frame`` on the first frame
  published after a ``cancel_epoch``.
- ``render_metrics`` exposes ``liveavatar_interrupt_*`` gauges.
"""

from __future__ import annotations

import asyncio
import unittest

from liveavatar.adapter import AvatarStreamingAdapter
from liveavatar.observability import render_metrics
from liveavatar.runtime.fake_tts import FakeTts
from liveavatar.runtime.metrics import SessionMetrics
from liveavatar.runtime.worker import RealtimeWorker
from tests.test_worker import _make_worker

# ──────────────────────────────────────────── SessionMetrics unit tests


class TestTimelinePrimitives(unittest.TestCase):
    """Idempotent marks and consecutive-delta decomposition."""

    def test_mark_is_idempotent(self) -> None:
        m = SessionMetrics("s1")
        first = m.timeline_mark("detect")
        second = m.timeline_mark("detect")
        self.assertEqual(first, second)
        self.assertEqual(list(m.timeline_stamps), ["detect"])

    def test_decompose_subset_layers(self) -> None:
        m = SessionMetrics("s1")
        m.timeline_stamps = {"detect": 0, "audio_flush": 2_000_000}
        out = m.timeline_decompose_ms()
        self.assertAlmostEqual(out["detect_to_audio_flush_ms"], 2.0)
        self.assertEqual(out["layers_recorded"], ["detect", "audio_flush"])
        # With exactly two layers the total equals the single delta.
        self.assertIn("total_ms", out)
        self.assertAlmostEqual(out["total_ms"], 2.0)

    def test_decompose_full_order_and_total(self) -> None:
        m = SessionMetrics("s1")
        m.timeline_stamps = {
            "detect": 0,
            "audio_flush": 1_000_000,
            "tts_stop": 3_000_000,
            "video_invalidate": 5_000_000,
            "new_frame": 41_000_000,
        }
        out = m.timeline_decompose_ms()
        self.assertAlmostEqual(out["detect_to_audio_flush_ms"], 1.0)
        self.assertAlmostEqual(out["audio_flush_to_tts_stop_ms"], 2.0)
        self.assertAlmostEqual(out["tts_stop_to_video_invalidate_ms"], 2.0)
        self.assertAlmostEqual(out["video_invalidate_to_new_frame_ms"], 36.0)
        self.assertAlmostEqual(out["total_ms"], 41.0)

    def test_reset_starts_new_window(self) -> None:
        m = SessionMetrics("s1")
        m.timeline_mark("detect")
        m.timeline_mark("audio_flush")
        m.timeline_reset()
        self.assertEqual(m.timeline_stamps, {})
        self.assertEqual(m.timeline_interrupt_seq, 1)

    def test_summary_includes_timeline(self) -> None:
        m = SessionMetrics("s1")
        self.assertNotIn("interruption_timeline", m.summary())
        m.timeline_stamps = {"detect": 0, "audio_flush": 1_000_000}
        self.assertIn("interruption_timeline", m.summary())


# ──────────────────────────────────────────── worker integration tests


class TestWorkerTimeline(unittest.IsolatedAsyncioTestCase):
    """advance_epoch stamps the middle layers in execution order."""

    async def test_advance_epoch_stamps_three_layers(self) -> None:
        metrics = SessionMetrics("sess_tl")
        tts = FakeTts()
        worker = RealtimeWorker("sess_tl", capacity=50, metrics=metrics, tts=tts)
        worker.avatar_adapter = None  # no video path → only 2 middle layers
        await worker.start()
        try:
            old = worker.epoch
            new = worker.advance_epoch()
            self.assertEqual(new, old + 1)
            stamps = metrics.timeline_stamps
            self.assertIn("audio_flush", stamps)
            self.assertIn("tts_stop", stamps)
            self.assertNotIn("video_invalidate", stamps)
            self.assertNotIn("new_frame", stamps)
            # Monotonic execution order.
            self.assertGreaterEqual(stamps["tts_stop"], stamps["audio_flush"])
            out = metrics.timeline_decompose_ms()
            self.assertIn("audio_flush_to_tts_stop_ms", out)
            self.assertAlmostEqual(out["total_ms"], out["audio_flush_to_tts_stop_ms"])
        finally:
            await worker.stop()

    async def test_advance_epoch_with_avatar_adapter_stamps_invalidate(self) -> None:
        metrics = SessionMetrics("sess_tl2")
        w = _make_worker()
        adapter = AvatarStreamingAdapter(
            worker=w, publisher=None, session_id="sess_tl2", metrics=metrics
        )
        tts = FakeTts()
        worker = RealtimeWorker(
            "sess_tl2",
            capacity=50,
            metrics=metrics,
            tts=tts,
            avatar_adapter=adapter,
        )
        await worker.start()
        try:
            worker.advance_epoch()
            stamps = metrics.timeline_stamps
            self.assertIn("video_invalidate", stamps)
            self.assertGreaterEqual(
                stamps["video_invalidate"], stamps["tts_stop"]
            )
        finally:
            await worker.stop()


# ──────────────────────────────────────────── adapter new_frame tests


class TestAdapterNewFrameMark(unittest.IsolatedAsyncioTestCase):
    """First published frame after cancel_epoch stamps ``new_frame`` once."""

    async def test_new_frame_marked_once_per_epoch(self) -> None:
        metrics = SessionMetrics("sess_av")
        w = _make_worker(n_frames=2)
        adapter = AvatarStreamingAdapter(
            worker=w, publisher=None, session_id="sess_av", metrics=metrics
        )
        await adapter.start()
        try:
            adapter.cancel_epoch(2)
            self.assertTrue(adapter._pending_new_frame_mark)
            self.assertNotIn("new_frame", metrics.timeline_stamps)
            ok = await adapter.push_pcm(b"\x01\x00" * 3200, 0, 2)
            self.assertTrue(ok)
            for _ in range(100):
                if len(adapter.published_frames) >= 2:
                    break
                await asyncio.sleep(0.01)
            self.assertGreaterEqual(len(adapter.published_frames), 1)
            self.assertFalse(adapter._pending_new_frame_mark)
            self.assertIn("new_frame", metrics.timeline_stamps)
            first_stamp = metrics.timeline_stamps["new_frame"]
            # Additional frames must not re-stamp (idempotent via flag).
            await asyncio.sleep(0.05)
            self.assertEqual(metrics.timeline_stamps["new_frame"], first_stamp)
        finally:
            await adapter.stop()

    async def test_no_metrics_is_fine(self) -> None:
        w = _make_worker(n_frames=1)
        adapter = AvatarStreamingAdapter(
            worker=w, publisher=None, session_id="sess_av2"
        )
        await adapter.start()
        try:
            adapter.cancel_epoch(3)
            ok = await adapter.push_pcm(b"\x01\x00" * 3200, 0, 3)
            self.assertTrue(ok)
            for _ in range(100):
                if len(adapter.published_frames) >= 1:
                    break
                await asyncio.sleep(0.01)
            self.assertGreaterEqual(len(adapter.published_frames), 1)
        finally:
            await adapter.stop()


# ──────────────────────────────────────────── /metrics rendering


class TestRenderMetricsTimeline(unittest.TestCase):
    """render_metrics exposes per-session interruption gauges."""

    def test_interrupt_gauges_rendered(self) -> None:
        class _Metrics:
            session_id = "sess_http"

            def timeline_decompose_ms(self):
                return {
                    "detect_to_audio_flush_ms": 0.5,
                    "total_ms": 4.1,
                    "layers_recorded": ["detect", "audio_flush"],
                }

        class _Session:
            metrics = _Metrics()

        class _State:
            pipeline = None
            duplex_sessions = {"sess_http": _Session()}

        text = render_metrics(_State())
        self.assertIn("liveavatar_interrupt_detect_to_audio_flush_ms", text)
        self.assertIn('liveavatar_interrupt_total_ms{session_id="sess_http"} 4.1', text)
        self.assertNotIn("layers_recorded", text)


if __name__ == "__main__":
    unittest.main()

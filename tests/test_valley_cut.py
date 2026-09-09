# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-C: energy-valley cut selector tests.

Covers the pure selector (valley/prior/none semantics), the FakeTts
pending-tail protocol, the worker integration (valley decision recorded
in metrics and carried on the flush control event), and the graceful
degradation when the TTS adapter exposes no pending tail.
"""

from __future__ import annotations

import asyncio
import math
import unittest

import numpy as np

from liveavatar.runtime.fake_tts import FakeTts, FakeTtsSegment
from liveavatar.runtime.metrics import SessionMetrics
from liveavatar.runtime.valley import find_valley_cut, window_rms_db
from liveavatar.runtime.worker import RealtimeWorker

_SR = 16000


def _build_pcm(parts: list[tuple[float, float]], freq: float = 220.0) -> bytes:
    """Concatenated s16le mono: (duration_s, peak_amplitude) sine parts."""
    out = bytearray()
    for dur_s, amp in parts:
        n = int(dur_s * _SR)
        t = np.arange(n) / _SR
        wave = (np.sin(2 * math.pi * freq * t) * amp * 32767).astype("<i2")
        out.extend(wave.tobytes())
    return bytes(out)


# ────────────────────────────────────────────── selector unit tests


class TestFindValleyCut(unittest.TestCase):
    def test_finds_pause_valley(self) -> None:
        # 100 ms loud + 120 ms silence + 300 ms loud: the cut lands at
        # the end of the first silent window (~120 ms), playing through
        # the gap instead of chopping the loud onset.
        pcm = _build_pcm([(0.10, 0.5), (0.12, 0.0), (0.30, 0.5)])
        cut = find_valley_cut(pcm, sample_rate=_SR)
        self.assertTrue(cut.found)
        self.assertEqual(cut.reason, "valley")
        cut_s = cut.cut_sample / _SR
        self.assertGreaterEqual(cut_s, 0.09)
        self.assertLessEqual(cut_s, 0.14)
        self.assertAlmostEqual(cut.rollback_ms, cut.cut_sample / _SR * 1000.0, places=6)

    def test_no_valley_in_continuous_speech(self) -> None:
        pcm = _build_pcm([(0.40, 0.5)])
        cut = find_valley_cut(pcm, sample_rate=_SR)
        self.assertFalse(cut.found)
        self.assertEqual(cut.reason, "none")
        self.assertEqual(cut.cut_sample, 0)
        self.assertEqual(cut.rollback_ms, 0.0)

    def test_search_span_limits_scan(self) -> None:
        # Valley exists at 300 ms but the span is only 100 ms.
        pcm = _build_pcm([(0.30, 0.5), (0.12, 0.0), (0.30, 0.5)])
        cut = find_valley_cut(pcm, sample_rate=_SR, search_ms=100.0)
        self.assertFalse(cut.found)

    def test_short_and_empty_input(self) -> None:
        for pcm in (b"", _build_pcm([(0.01, 0.5)])):
            cut = find_valley_cut(pcm, sample_rate=_SR)
            self.assertFalse(cut.found)

    def test_valley_at_scan_start(self) -> None:
        # Playback position already sits in a gap → immediate tiny rollback.
        pcm = _build_pcm([(0.06, 0.0), (0.30, 0.5)])
        cut = find_valley_cut(pcm, sample_rate=_SR)
        self.assertTrue(cut.found)
        self.assertLessEqual(cut.rollback_ms, 40.0)

    def test_window_rms_db_shapes(self) -> None:
        loud = window_rms_db(_build_pcm([(0.10, 0.5)]), _SR, 20.0)
        quiet = window_rms_db(_build_pcm([(0.10, 0.0)]), _SR, 20.0)
        self.assertEqual(len(loud), 5)
        self.assertGreater(loud.mean(), -30.0)
        self.assertLess(quiet.mean(), -100.0)


class TestPriorFusion(unittest.TestCase):
    def test_prior_in_valley_wins(self) -> None:
        # 100 ms loud + 120 ms silence + 300 ms loud; the EOU prior sits
        # inside the silence (within the 240 ms search span).
        pcm = _build_pcm([(0.10, 0.5), (0.12, 0.0), (0.30, 0.5)])
        cut = find_valley_cut(pcm, sample_rate=_SR, prior_sample=int(0.14 * _SR))
        self.assertTrue(cut.found)
        self.assertEqual(cut.reason, "eou_prior")
        # Cut at the end of the prior's window (~160 ms).
        self.assertGreaterEqual(cut.cut_sample / _SR, 0.14)

    def test_prior_in_loud_region_falls_back(self) -> None:
        pcm = _build_pcm([(0.10, 0.5), (0.12, 0.0), (0.30, 0.5)])
        cut = find_valley_cut(pcm, sample_rate=_SR, prior_sample=int(0.04 * _SR))
        self.assertEqual(cut.reason, "valley")  # first valley, not the prior


# ────────────────────────────────────────── FakeTts pending-tail tests


class TestFakeTtsPendingTail(unittest.TestCase):
    def _tts_with_two_segments(self) -> FakeTts:
        tts = FakeTts(sample_rate=_SR)
        tts.synthesize("a", epoch=0, pts_us=0)  # spans 0..200 ms
        tts.synthesize("b", epoch=0, pts_us=200_000)  # spans 200..400 ms
        return tts

    def test_tail_trims_played_prefix_and_concatenates(self) -> None:
        tts = self._tts_with_two_segments()
        # Consumed 150 ms: seg1 (0..200 ms) trimmed to 50 ms, seg2 full.
        tail = tts.pending_audio_tail(150_000)
        self.assertIsNotNone(tail)
        self.assertEqual(len(tail), (50 + 200) * _SR // 1000 * 2)

    def test_tail_from_zero_returns_everything(self) -> None:
        tts = self._tts_with_two_segments()
        self.assertEqual(len(tts.pending_audio_tail(0)), 400 * _SR // 1000 * 2)

    def test_nothing_pending_is_none(self) -> None:
        tts = self._tts_with_two_segments()
        self.assertIsNone(tts.pending_audio_tail(400_000))


# ───────────────────────────────────────── worker integration tests


class _SpeechStubTts(FakeTts):
    """FakeTts whose segments carry a real energy valley (150/100/150)."""

    def synthesize(self, text: str, epoch: int, pts_us: int) -> list[FakeTtsSegment]:
        self.segment_counter += 1
        seg = FakeTtsSegment(
            segment_seq=self.segment_counter,
            text=text,
            epoch=epoch,
            pcm_s16le=_build_pcm([(0.15, 0.5), (0.10, 0.0), (0.15, 0.5)]),
            pts_us=pts_us,
            duration_us=400_000,
        )
        self.active_segments.append(seg)
        return [seg]


class _BareTts(FakeTts):
    """TTS without the P-C pending-tail protocol (degraded path)."""

    pending_audio_tail = None  # type: ignore[assignment]


class TestWorkerValleyCut(unittest.IsolatedAsyncioTestCase):
    async def _flush_event(self, worker: RealtimeWorker) -> dict:
        for _ in range(200):
            ev = worker.output_queue.try_dequeue()
            if ev is not None and ev.get("event_type") == "control":
                return ev["payload"]["control_event"]
            await asyncio.sleep(0.01)
        raise AssertionError("flush control event not observed")

    async def test_valley_recorded_and_attached_to_flush(self) -> None:
        metrics = SessionMetrics("sess_vc")
        worker = RealtimeWorker(
            "sess_vc", capacity=50, metrics=metrics, tts=_SpeechStubTts()
        )
        await worker.start()
        try:
            worker.tts.synthesize("hi", epoch=0, pts_us=0)
            await worker.push_control(
                {"intent": "playback_ack", "consumed_pts_us": 100_000}
            )
            await worker.push_control({"intent": "cancel_and_flush"})
            payload = await self._flush_event(worker)
            self.assertEqual(payload["intent"], "flush")
            valley = payload["metadata"]["valley"]
            self.assertTrue(valley["found"])
            self.assertIn(valley["reason"], ("valley", "eou_prior"))
            # Playback at 100 ms; the valley sits within the 240 ms scan.
            self.assertGreater(valley["rollback_ms"], 0.0)
            self.assertEqual(valley["cut_pts_us"], 100_000 + int(valley["rollback_ms"] * 1000))
            self.assertEqual(metrics.valley_cut_count, 1)
            self.assertEqual(metrics.valley_hit_count, 1)
            summary = metrics.summary()
            self.assertEqual(summary["valley_hit_rate"], 1.0)
        finally:
            await worker.stop()

    async def test_chirp_tts_records_miss(self) -> None:
        # Plain FakeTts chirps have constant energy → no valley → the
        # decision is recorded as a miss and the hard cut is kept.
        metrics = SessionMetrics("sess_vc2")
        worker = RealtimeWorker("sess_vc2", capacity=50, metrics=metrics)
        await worker.start()
        try:
            worker.tts.synthesize("hi", epoch=0, pts_us=0)
            await worker.push_control({"intent": "cancel_and_flush"})
            payload = await self._flush_event(worker)
            self.assertFalse(payload["metadata"]["valley"]["found"])
            self.assertEqual(metrics.valley_cut_count, 1)
            self.assertEqual(metrics.valley_hit_count, 0)
        finally:
            await worker.stop()

    async def test_no_pending_tail_degrades_to_hard_cut(self) -> None:
        metrics = SessionMetrics("sess_vc3")
        worker = RealtimeWorker(
            "sess_vc3", capacity=50, metrics=metrics, tts=_BareTts()
        )
        await worker.start()
        try:
            worker.tts.synthesize("hi", epoch=0, pts_us=0)
            await worker.push_control({"intent": "cancel_and_flush"})
            payload = await self._flush_event(worker)
            self.assertNotIn("valley", payload["metadata"])
            self.assertEqual(metrics.valley_cut_count, 0)
        finally:
            await worker.stop()


if __name__ == "__main__":
    unittest.main()

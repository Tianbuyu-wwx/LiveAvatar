# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Step 4 integration tests: RealtimeWorker streaming-TTS background path.

Verifies that when a TTS backend exposing ``synthesize_stream`` is injected,
the worker spawns a background task that consumes the async generator and
emits ``tts_audio`` events incrementally — without blocking the event loop.

Coverage:
- Incremental emission (segments appear as they are produced, not batched).
- Event-loop responsiveness while TTS streams (concurrent frame processing).
- Epoch-advance cancels the in-flight stream promptly.
- stop() cancels in-flight tasks.
- Sync FakeTts path still works (backward compat).
- Segment field correctness.
- Multiple ASR finals spawn independent tasks.
- Streaming errors surface as ``error`` events.
"""

from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from liveavatar.audio_in.frame import PCMFrame
from liveavatar.runtime.worker import RealtimeWorker

# ──────────────────────────────────────── fakes

@dataclass
class _FakeSeg:
    """TtsSegment-compatible stub."""
    segment_seq: int
    text: str
    epoch: int
    pcm_s16le: bytes
    pts_us: int
    duration_us: int


class _StreamingFakeTts:
    """TTS backend with an async ``synthesize_stream``.

    Yields ``n_chunks`` segments with ``chunk_delay`` between each so tests
    can observe incremental emission and concurrency. ``cancel_epoch`` flips
    a flag the generator checks between chunks (mirrors NvcStreamingTtsAdapter).
    """

    def __init__(
        self,
        *,
        n_chunks: int = 3,
        chunk_delay: float = 0.0,
        chunk_samples: int = 320,
        sample_rate: int = 16000,
        fail_on_chunk: int | None = None,
    ) -> None:
        self.n_chunks = n_chunks
        self.chunk_delay = chunk_delay
        self.chunk_samples = chunk_samples
        self.sample_rate = sample_rate
        self.fail_on_chunk = fail_on_chunk
        self._current_epoch = 0
        self._cancel_flag = False
        self.stream_invocations = 0

    async def synthesize_stream(
        self, text: str, epoch: int, pts_us: int
    ) -> AsyncGenerator[_FakeSeg, None]:
        self.stream_invocations += 1
        if epoch < self._current_epoch:
            return
        current_pts = pts_us
        duration_us = self.chunk_samples * 1_000_000 // self.sample_rate
        pcm = b"\x00\x10" * self.chunk_samples  # deterministic non-empty PCM
        for i in range(self.n_chunks):
            if self._cancel_flag:
                break
            if self.chunk_delay > 0:
                await asyncio.sleep(self.chunk_delay)
            if self.fail_on_chunk is not None and i == self.fail_on_chunk:
                raise RuntimeError(f"tts_stream: streaming failed at chunk {i}")
            yield _FakeSeg(
                segment_seq=i + 1,
                text=text,
                epoch=epoch,
                pcm_s16le=pcm,
                pts_us=current_pts,
                duration_us=duration_us,
            )
            current_pts += duration_us

    def cancel_epoch(self, epoch: int) -> int:
        self._current_epoch = epoch
        self._cancel_flag = True
        return 0

    def pop_played_segments(self, consumed_pts_us: int) -> list[Any]:
        return []


class _FinalOnPushAsr:
    """ASR stub that emits one ``final`` event per push_frame call.

    Each call yields exactly one final event whose text encodes the call
    index, so tests can distinguish multiple finals.
    """

    def __init__(self) -> None:
        self._call = 0

    def push_frame(self, frame: Any) -> list[dict[str, Any]]:
        self._call += 1
        return [
            {
                "phase": "final",
                "text": f"utt{self._call}",
                "stability": 1.0,
                "revision": self._call,
                "words": [],
            }
        ]

    def advance_epoch(self, epoch: int) -> None:
        pass


def _frame(samples: list[int], seq: int = 1, pts_us: int = 0) -> PCMFrame:
    return PCMFrame.from_int16_array(
        samples, "sess_stream", 0, seq, pts_us, pts_us + 1_000_000
    )


def _drain_events(queue: Any) -> list[dict[str, Any]]:
    """Synchronously drain all currently-queued events."""
    events: list[dict[str, Any]] = []
    while True:
        item = queue.try_dequeue()
        if item is None:
            break
        events.append(item)
    return events


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    """Extract nested payload from a RealtimeContracts Envelope."""
    event_type = event.get("event_type", "")
    payload = event.get("payload", {})
    if isinstance(payload, dict):
        return payload.get(f"{event_type}_event", payload)
    return event


# ──────────────────────────────────────── tests

class TestStreamingTtsEmission(unittest.IsolatedAsyncioTestCase):
    """The streaming path emits tts_audio events incrementally."""

    async def asyncSetUp(self) -> None:
        self.tts = _StreamingFakeTts(n_chunks=3, chunk_delay=0.0)
        self.asr = _FinalOnPushAsr()
        self.worker = RealtimeWorker(
            "sess_stream", capacity=50, asr=self.asr, tts=self.tts
        )
        await self.worker.start()

    async def asyncTearDown(self) -> None:
        await self.worker.stop()

    async def test_emits_one_event_per_chunk(self):
        await self.worker.push_frame(_frame([30000] * 320, seq=1, pts_us=0))
        # Wait for the background stream task to finish.
        for _ in range(100):
            if self.worker.stats.tts_segments >= 3:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.worker.stats.tts_segments, 3)

    async def test_tts_audio_events_have_correct_fields(self):
        await self.worker.push_frame(_frame([30000] * 320, seq=1, pts_us=0))
        for _ in range(100):
            if self.worker.stats.tts_segments >= 3:
                break
            await asyncio.sleep(0.01)
        events = [
            e for e in _drain_events(self.worker.output_queue)
            if e.get("event_type") == "tts_audio"
        ]
        self.assertEqual(len(events), 3)
        payloads = [_payload(e) for e in events]
        # Segment seqs are 1, 2, 3.
        seqs = [p["segment_seq"] for p in payloads]
        self.assertEqual(seqs, [1, 2, 3])
        # PTS advances by duration_us each chunk (320 samples / 16000 = 20000us).
        self.assertEqual(payloads[0]["pts_us"], 0)
        self.assertEqual(payloads[1]["pts_us"], 20000)
        self.assertEqual(payloads[2]["pts_us"], 40000)
        self.assertEqual(payloads[0]["duration_us"], 20000)
        # pcm_s16le is a hex string of the PCM.
        self.assertTrue(all(isinstance(p["pcm_s16le"], str) for p in payloads))
        self.assertGreater(len(payloads[0]["pcm_s16le"]), 0)

    async def test_incremental_emission_not_batched(self):
        """With chunk_delay, segments appear one-by-one over time."""
        self.tts.chunk_delay = 0.05
        await self.worker.push_frame(_frame([30000] * 320, seq=1, pts_us=0))
        # After ~30ms (less than 2*50ms=100ms for 2 chunks) at most 1 segment.
        await asyncio.sleep(0.03)
        self.assertLessEqual(self.worker.stats.tts_segments, 1)
        # After all chunks done, exactly 3.
        for _ in range(100):
            if self.worker.stats.tts_segments >= 3:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.worker.stats.tts_segments, 3)


class TestStreamingDoesNotBlockLoop(unittest.IsolatedAsyncioTestCase):
    """The event loop stays responsive while TTS streams."""

    async def asyncSetUp(self) -> None:
        self.tts = _StreamingFakeTts(n_chunks=5, chunk_delay=0.04)
        self.asr = _FinalOnPushAsr()
        self.worker = RealtimeWorker(
            "sess_concurrent", capacity=50, asr=self.asr, tts=self.tts
        )
        await self.worker.start()

    async def asyncTearDown(self) -> None:
        await self.worker.stop()

    async def test_frame_processed_while_tts_streams(self):
        # First frame triggers a 5-chunk stream (5 * 40ms = 200ms total).
        await self.worker.push_frame(_frame([30000] * 320, seq=1, pts_us=0))
        # Push a second frame shortly after — it should be processed
        # (input_frames incremented) well before the first stream finishes.
        await asyncio.sleep(0.02)
        await self.worker.push_frame(_frame([30000] * 320, seq=2, pts_us=40000))
        # Give the loop a chance to process the second frame.
        await asyncio.sleep(0.02)
        # Second frame was consumed (input_frames >= 2) even though the
        # first stream is still running (only ~40ms elapsed of 200ms).
        self.assertGreaterEqual(self.worker.stats.input_frames, 2)
        # Let streams finish for clean teardown.
        for _ in range(200):
            if not self.worker._tts_tasks:
                break
            await asyncio.sleep(0.01)


class TestEpochAdvanceCancelsStream(unittest.IsolatedAsyncioTestCase):
    """advance_epoch cancels the in-flight streaming task promptly."""

    async def asyncSetUp(self) -> None:
        self.tts = _StreamingFakeTts(n_chunks=20, chunk_delay=0.03)
        self.asr = _FinalOnPushAsr()
        self.worker = RealtimeWorker(
            "sess_cancel", capacity=50, asr=self.asr, tts=self.tts
        )
        await self.worker.start()

    async def asyncTearDown(self) -> None:
        await self.worker.stop()

    async def test_advance_epoch_stops_emission(self):
        await self.worker.push_frame(_frame([30000] * 320, seq=1, pts_us=0))
        # Wait for at least one chunk to emit (robust to scheduling jitter),
        # then advance epoch before all 20 chunks complete.
        segments_before = 0
        for _ in range(100):
            if self.worker.stats.tts_segments >= 1:
                segments_before = self.worker.stats.tts_segments
                break
            await asyncio.sleep(0.01)
        self.assertGreaterEqual(segments_before, 1)
        self.assertLess(segments_before, 20)

        self.worker.advance_epoch()

        # The task should be cancelled and no longer tracked.
        for _ in range(50):
            if not self.worker._tts_tasks:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(len(self.worker._tts_tasks), 0)

        # Wait a bit and confirm no further segments are emitted.
        segments_after = self.worker.stats.tts_segments
        await asyncio.sleep(0.1)
        self.assertEqual(self.worker.stats.tts_segments, segments_after)


class TestStopCancelsInflightTasks(unittest.IsolatedAsyncioTestCase):
    """stop() cancels in-flight TTS tasks so they don't outlive the worker."""

    async def test_stop_clears_tts_tasks(self):
        tts = _StreamingFakeTts(n_chunks=20, chunk_delay=0.03)
        asr = _FinalOnPushAsr()
        worker = RealtimeWorker("sess_stop", capacity=50, asr=asr, tts=tts)
        await worker.start()
        try:
            await worker.push_frame(_frame([30000] * 320, seq=1, pts_us=0))
            await asyncio.sleep(0.05)
            self.assertGreaterEqual(len(worker._tts_tasks), 1)
        finally:
            await worker.stop()
        # After stop, no tasks remain.
        self.assertEqual(len(worker._tts_tasks), 0)


class TestSyncFakeTtsBackwardCompat(unittest.IsolatedAsyncioTestCase):
    """Default FakeTts still uses the sync path (no background task)."""

    async def asyncSetUp(self) -> None:
        # No tts= injected → default FakeTts (sync path).
        self.asr = _FinalOnPushAsr()
        self.worker = RealtimeWorker("sess_sync", capacity=50, asr=self.asr)
        await self.worker.start()

    async def asyncTearDown(self) -> None:
        await self.worker.stop()

    async def test_sync_path_emits_segments(self):
        await self.worker.push_frame(_frame([30000] * 320, seq=1, pts_us=0))
        for _ in range(50):
            if self.worker.stats.tts_segments > 0:
                break
            await asyncio.sleep(0.01)
        self.assertGreater(self.worker.stats.tts_segments, 0)
        # Sync path never spawns a background task.
        self.assertEqual(len(self.worker._tts_tasks), 0)


class TestMultipleFinalsSpawnTasks(unittest.IsolatedAsyncioTestCase):
    """Two ASR finals spawn two independent streaming tasks."""

    async def asyncSetUp(self) -> None:
        self.tts = _StreamingFakeTts(n_chunks=2, chunk_delay=0.0)
        self.asr = _FinalOnPushAsr()
        self.worker = RealtimeWorker(
            "sess_multi", capacity=50, asr=self.asr, tts=self.tts
        )
        await self.worker.start()

    async def asyncTearDown(self) -> None:
        await self.worker.stop()

    async def test_two_finals_emit_four_segments(self):
        await self.worker.push_frame(_frame([30000] * 320, seq=1, pts_us=0))
        await self.worker.push_frame(_frame([30000] * 320, seq=2, pts_us=40000))
        for _ in range(100):
            if self.worker.stats.tts_segments >= 4:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.worker.stats.tts_segments, 4)
        self.assertEqual(self.tts.stream_invocations, 2)


class TestStreamingErrorHandling(unittest.IsolatedAsyncioTestCase):
    """A streaming error surfaces as an ``error`` event, not a crash."""

    async def test_error_event_emitted_on_stream_failure(self):
        tts = _StreamingFakeTts(n_chunks=3, chunk_delay=0.0, fail_on_chunk=1)
        asr = _FinalOnPushAsr()
        worker = RealtimeWorker("sess_err", capacity=50, asr=asr, tts=tts)
        await worker.start()
        try:
            await worker.push_frame(_frame([30000] * 320, seq=1, pts_us=0))
            # Wait for the error event to land in the output queue.
            error_event = None
            for _ in range(100):
                for e in _drain_events(worker.output_queue):
                    if e.get("event_type") == "error":
                        error_event = e
                        break
                if error_event is not None:
                    break
                await asyncio.sleep(0.01)
            self.assertIsNotNone(error_event, "no error event emitted")
            self.assertIn("tts_stream", _payload(error_event).get("message", ""))
        finally:
            await worker.stop()


if __name__ == "__main__":
    unittest.main()

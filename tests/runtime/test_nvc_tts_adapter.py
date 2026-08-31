"""Tests for NvcStreamingTtsAdapter.

Uses a self-contained FakeNvcWorker (no torch/voice_pool dependency)
to test both the sync (FakeTts-compat) and async (streaming) interfaces.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from collections.abc import AsyncGenerator, Generator

import numpy as np

from liveavatar.tts import NvcStreamingTtsAdapter, TtsSegment

# ──────────────────────────────────────── Fake NvcWorker

class _FakeTts:
    """Synchronous TTS generator stub."""

    def __init__(
        self,
        n_chunks: int = 3,
        sample_rate: int = 16000,
        chunk_samples: int = 320,
        delay: float = 0.0,
    ) -> None:
        self.n_chunks = n_chunks
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.delay = delay

    def run(self, req: dict) -> Generator[tuple[int, np.ndarray], None, None]:
        for i in range(self.n_chunks):
            if self.delay > 0:
                time.sleep(self.delay)
            audio = np.full(
                self.chunk_samples, 0.5 if i % 2 == 0 else -0.5, dtype=np.float32
            )
            yield (self.sample_rate, audio)


class FakeNvcWorker:
    """Minimal NvcWorker stub for adapter testing.

    Implements the duck-typed interface that NvcStreamingTtsAdapter
    accesses: ``_build_request``, ``_to_canonical_pcm``,
    ``synthesize_stream``, ``_tts.run``, and ``char_id``.
    """

    def __init__(
        self,
        char_id: str = "nahida",
        n_chunks: int = 3,
        sample_rate: int = 16000,
        chunk_samples: int = 320,
        tts_delay: float = 0.0,
    ) -> None:
        self.char_id = char_id
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self._tts = _FakeTts(n_chunks, sample_rate, chunk_samples, tts_delay)
        self.stream_call_count = 0

    def _build_request(
        self, *, text, text_lang, speed_factor,
        top_k, top_p, temperature, repetition_penalty,
    ) -> dict:
        return {
            "text": text,
            "text_lang": text_lang,
            "streaming_mode": True,
        }

    def _to_canonical_pcm(self, audio_np: np.ndarray, sr: int) -> bytes:
        audio = np.asarray(audio_np, dtype=np.float32).reshape(-1)
        if sr != 16000:
            n_out = int(len(audio) * 16000 / sr)
            if n_out > 0:
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n_out),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16).tobytes()

    async def synthesize_stream(
        self,
        text: str,
        *,
        cancel_token=None,
        text_lang: str = "zh",
        speed_factor: float = 1.0,
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        self.stream_call_count += 1
        # Import CancelToken from the adapter module (handles fallback stub)
        from liveavatar.tts import CancelToken as _CT

        token = cancel_token or _CT()
        for i in range(self._tts.n_chunks):
            if token.cancelled:
                break
            if self._tts.delay > 0:
                await asyncio.sleep(self._tts.delay)
            audio = np.full(
                self._tts.chunk_samples,
                0.5 if i % 2 == 0 else -0.5,
                dtype=np.float32,
            )
            yield self._to_canonical_pcm(audio, self._tts.sample_rate)


# ──────────────────────────────────────── Fake VoicePool

class FakeVoicePool:
    """Minimal VoicePool stub for pool-mode testing."""

    def __init__(self, worker: FakeNvcWorker) -> None:
        self._worker = worker
        self.acquire_count = 0
        self.release_count = 0

    async def acquire(self, session_id: str, char_id: str, **kwargs):
        self.acquire_count += 1
        # Return a lease-like object.
        class _FakeLease:
            def __init__(self, worker):
                self.worker = worker
        return _FakeLease(self._worker)

    async def release_async(self, session_id: str) -> bool:
        self.release_count += 1
        return True


# ──────────────────────────────────────── helpers

def _make_adapter(
    *,
    n_chunks: int = 3,
    chunk_samples: int = 320,
    tts_delay: float = 0.0,
    char_id: str = "nahida",
) -> tuple[NvcStreamingTtsAdapter, FakeNvcWorker]:
    """Create an adapter in direct mode with a fake worker."""
    worker = FakeNvcWorker(
        char_id=char_id,
        n_chunks=n_chunks,
        chunk_samples=chunk_samples,
        tts_delay=tts_delay,
    )
    adapter = NvcStreamingTtsAdapter(worker=worker)
    return adapter, worker


# ──────────────────────────────────────── sync interface tests

class TestSyncSynthesize(unittest.TestCase):
    def test_returns_list_of_segments(self):
        adapter, _ = _make_adapter(n_chunks=3)
        segments = adapter.synthesize("你好", epoch=0, pts_us=0)
        self.assertEqual(len(segments), 3)
        for seg in segments:
            self.assertIsInstance(seg, TtsSegment)

    def test_segment_fields_correct(self):
        adapter, _ = _make_adapter(n_chunks=1, chunk_samples=320)
        segments = adapter.synthesize("test", epoch=5, pts_us=1000)
        seg = segments[0]
        self.assertEqual(seg.text, "test")
        self.assertEqual(seg.epoch, 5)
        self.assertEqual(seg.pts_us, 1000)
        self.assertGreater(len(seg.pcm_s16le), 0)
        # 320 samples × 2 bytes = 640 bytes
        self.assertEqual(len(seg.pcm_s16le), 640)
        # duration = 320 samples / 16000 Hz = 20ms = 20000us
        self.assertEqual(seg.duration_us, 20000)
        self.assertEqual(seg.segment_seq, 1)

    def test_pts_advances_across_chunks(self):
        adapter, _ = _make_adapter(n_chunks=3, chunk_samples=320)
        segments = adapter.synthesize("hello", epoch=0, pts_us=0)
        self.assertEqual(segments[0].pts_us, 0)
        self.assertEqual(segments[1].pts_us, 20000)
        self.assertEqual(segments[2].pts_us, 40000)

    def test_segment_seq_monotonic(self):
        adapter, _ = _make_adapter(n_chunks=3)
        segs1 = adapter.synthesize("first", epoch=0, pts_us=0)
        segs2 = adapter.synthesize("second", epoch=0, pts_us=60000)
        seqs = [s.segment_seq for s in segs1 + segs2]
        self.assertEqual(seqs, [1, 2, 3, 4, 5, 6])

    def test_pcm_is_s16le_bytes(self):
        adapter, _ = _make_adapter(n_chunks=1, chunk_samples=100)
        segments = adapter.synthesize("x", epoch=0, pts_us=0)
        pcm = segments[0].pcm_s16le
        self.assertIsInstance(pcm, bytes)
        # 100 samples × 2 bytes = 200 bytes
        self.assertEqual(len(pcm), 200)

    def test_old_epoch_skipped(self):
        adapter, _ = _make_adapter()
        adapter._current_epoch = 5
        segments = adapter.synthesize("old", epoch=3, pts_us=0)
        self.assertEqual(len(segments), 0)

    def test_segments_added_to_active_list(self):
        adapter, _ = _make_adapter(n_chunks=3)
        adapter.synthesize("test", epoch=0, pts_us=0)
        self.assertEqual(adapter.active_segment_count, 3)


class TestCancelEpoch(unittest.TestCase):
    def test_cancel_removes_old_segments(self):
        adapter, _ = _make_adapter(n_chunks=3)
        adapter.synthesize("first", epoch=1, pts_us=0)
        adapter.synthesize("second", epoch=2, pts_us=60000)
        self.assertEqual(adapter.active_segment_count, 6)

        removed = adapter.cancel_epoch(2)
        self.assertEqual(removed, 3)  # epoch=1 segments removed
        self.assertEqual(adapter.active_segment_count, 3)

    def test_cancel_sets_cancel_token(self):
        adapter, _ = _make_adapter()
        adapter.synthesize("text", epoch=0, pts_us=0)
        self.assertFalse(adapter._cancel_token.cancelled)
        adapter.cancel_epoch(1)
        self.assertTrue(adapter._cancel_token.cancelled)

    def test_cancel_updates_current_epoch(self):
        adapter, _ = _make_adapter()
        adapter.cancel_epoch(5)
        self.assertEqual(adapter._current_epoch, 5)

    def test_cancel_zero_when_no_old_segments(self):
        adapter, _ = _make_adapter(n_chunks=2)
        adapter.synthesize("text", epoch=3, pts_us=0)
        removed = adapter.cancel_epoch(3)
        self.assertEqual(removed, 0)


class TestPopPlayedSegments(unittest.TestCase):
    def test_pop_returns_played_segments(self):
        adapter, _ = _make_adapter(n_chunks=3, chunk_samples=320)
        adapter.synthesize("test", epoch=0, pts_us=0)
        # Each segment is 20ms. At consumed=25000us, first segment (0-20000) is played.
        played = adapter.pop_played_segments(25000)
        self.assertEqual(len(played), 1)
        self.assertEqual(played[0].pts_us, 0)
        self.assertEqual(adapter.active_segment_count, 2)

    def test_pop_all_when_consumed_past_end(self):
        adapter, _ = _make_adapter(n_chunks=3, chunk_samples=320)
        adapter.synthesize("test", epoch=0, pts_us=0)
        played = adapter.pop_played_segments(100000)
        self.assertEqual(len(played), 3)
        self.assertEqual(adapter.active_segment_count, 0)

    def test_pop_none_when_nothing_played(self):
        adapter, _ = _make_adapter(n_chunks=2, chunk_samples=320)
        adapter.synthesize("test", epoch=0, pts_us=0)
        played = adapter.pop_played_segments(5000)
        self.assertEqual(len(played), 0)


# ──────────────────────────────────────── async streaming tests

class TestAsyncSynthesizeStream(unittest.IsolatedAsyncioTestCase):
    async def test_yields_segments_incrementally(self):
        adapter, _ = _make_adapter(n_chunks=3, chunk_samples=320)
        segments = []
        async for seg in adapter.synthesize_stream("hello", epoch=0, pts_us=0):
            segments.append(seg)
        self.assertEqual(len(segments), 3)

    async def test_stream_segment_fields(self):
        adapter, _ = _make_adapter(n_chunks=1, chunk_samples=320)
        async for seg in adapter.synthesize_stream("test", epoch=2, pts_us=100):
            self.assertEqual(seg.text, "test")
            self.assertEqual(seg.epoch, 2)
            self.assertEqual(seg.pts_us, 100)
            self.assertEqual(len(seg.pcm_s16le), 640)
            self.assertEqual(seg.duration_us, 20000)
            self.assertEqual(seg.segment_seq, 1)

    async def test_stream_pts_advances(self):
        adapter, _ = _make_adapter(n_chunks=3, chunk_samples=320)
        segments = []
        async for seg in adapter.synthesize_stream("x", epoch=0, pts_us=0):
            segments.append(seg)
        self.assertEqual(segments[0].pts_us, 0)
        self.assertEqual(segments[1].pts_us, 20000)
        self.assertEqual(segments[2].pts_us, 40000)

    async def test_stream_old_epoch_skipped(self):
        adapter, _ = _make_adapter()
        adapter._current_epoch = 5
        segments = []
        async for _ in adapter.synthesize_stream("old", epoch=3, pts_us=0):
            segments.append(_)
        self.assertEqual(len(segments), 0)

    async def test_stream_cancelled_by_epoch_advance(self):
        """Cancel epoch mid-stream stops the generator."""
        adapter, worker = _make_adapter(
            n_chunks=10, chunk_samples=320, tts_delay=0.03
        )

        async def _cancel_after_delay():
            await asyncio.sleep(0.05)
            adapter.cancel_epoch(1)

        cancel_task = asyncio.create_task(_cancel_after_delay())
        segments = []
        async for seg in adapter.synthesize_stream("test", epoch=0, pts_us=0):
            segments.append(seg)
        await cancel_task

        # Should have stopped before all 10 chunks.
        self.assertLess(len(segments), 10)

    async def test_stream_uses_worker_synthesize_stream(self):
        """The async path delegates to worker.synthesize_stream()."""
        adapter, worker = _make_adapter(n_chunks=2)
        async for _ in adapter.synthesize_stream("x", epoch=0, pts_us=0):
            pass
        self.assertEqual(worker.stream_call_count, 1)

    async def test_stream_segments_added_to_active_list(self):
        adapter, _ = _make_adapter(n_chunks=3)
        async for _ in adapter.synthesize_stream("test", epoch=0, pts_us=0):
            pass
        self.assertEqual(adapter.active_segment_count, 3)


# ──────────────────────────────────────── pool mode tests

class TestPoolMode(unittest.IsolatedAsyncioTestCase):
    async def test_acquire_sets_worker(self):
        worker = FakeNvcWorker(char_id="nahida")
        pool = FakeVoicePool(worker)
        adapter = NvcStreamingTtsAdapter(
            pool=pool, session_id="s1", char_id="nahida"
        )
        self.assertFalse(adapter.has_worker)
        await adapter.acquire()
        self.assertTrue(adapter.has_worker)
        self.assertEqual(pool.acquire_count, 1)

    async def test_release_clears_lease(self):
        worker = FakeNvcWorker()
        pool = FakeVoicePool(worker)
        adapter = NvcStreamingTtsAdapter(
            pool=pool, session_id="s1", char_id="nahida"
        )
        await adapter.acquire()
        await adapter.release()
        self.assertEqual(pool.release_count, 1)
        # Worker is still accessible for reference.
        self.assertTrue(adapter.has_worker)

    async def test_synthesize_after_acquire(self):
        worker = FakeNvcWorker(n_chunks=2)
        pool = FakeVoicePool(worker)
        adapter = NvcStreamingTtsAdapter(
            pool=pool, session_id="s1", char_id="nahida"
        )
        await adapter.acquire()
        segments = adapter.synthesize("hello", epoch=0, pts_us=0)
        self.assertEqual(len(segments), 2)

    async def test_synthesize_without_acquire_raises(self):
        pool = FakeVoicePool(FakeNvcWorker())
        adapter = NvcStreamingTtsAdapter(
            pool=pool, session_id="s1", char_id="nahida"
        )
        with self.assertRaises(RuntimeError):
            adapter.synthesize("hello", epoch=0, pts_us=0)

    async def test_stream_after_acquire(self):
        worker = FakeNvcWorker(n_chunks=2)
        pool = FakeVoicePool(worker)
        adapter = NvcStreamingTtsAdapter(
            pool=pool, session_id="s1", char_id="nahida"
        )
        await adapter.acquire()
        segments = []
        async for seg in adapter.synthesize_stream("hi", epoch=0, pts_us=0):
            segments.append(seg)
        self.assertEqual(len(segments), 2)


# ──────────────────────────────────────── error cases

class TestErrorCases(unittest.TestCase):
    def test_constructor_without_worker_or_pool_raises(self):
        with self.assertRaises(ValueError):
            NvcStreamingTtsAdapter()

    def test_synthesize_without_worker_raises(self):
        adapter = NvcStreamingTtsAdapter(pool=FakeVoicePool(FakeNvcWorker()))
        with self.assertRaises(RuntimeError):
            adapter.synthesize("hello", epoch=0, pts_us=0)

    def test_char_id_returns_worker_char_id(self):
        worker = FakeNvcWorker(char_id="jingyuan")
        adapter = NvcStreamingTtsAdapter(worker=worker)
        self.assertEqual(adapter.char_id, "jingyuan")

    def test_char_id_returns_configured_when_no_worker(self):
        adapter = NvcStreamingTtsAdapter(
            pool=FakeVoicePool(FakeNvcWorker()),
            session_id="s1",
            char_id="nahida",
        )
        self.assertEqual(adapter.char_id, "nahida")

    def test_to_dict_snapshot(self):
        adapter, _ = _make_adapter()
        d = adapter.to_dict()
        self.assertIn("char_id", d)
        self.assertIn("has_worker", d)
        self.assertIn("current_epoch", d)
        self.assertTrue(d["has_worker"])


# ──────────────────────────────────────── FakeTts compat

class TestFakeTtsCompat(unittest.TestCase):
    """Verify the adapter is a drop-in for FakeTts."""

    def test_same_method_signatures(self):
        """Adapter has synthesize, cancel_epoch, pop_played_segments."""
        adapter, _ = _make_adapter()
        self.assertTrue(callable(getattr(adapter, "synthesize", None)))
        self.assertTrue(callable(getattr(adapter, "cancel_epoch", None)))
        self.assertTrue(callable(getattr(adapter, "pop_played_segments", None)))

    def test_synthesize_returns_list(self):
        adapter, _ = _make_adapter(n_chunks=2)
        result = adapter.synthesize("test", epoch=0, pts_us=0)
        self.assertIsInstance(result, list)

    def test_cancel_epoch_returns_int(self):
        adapter, _ = _make_adapter()
        adapter.synthesize("test", epoch=0, pts_us=0)
        result = adapter.cancel_epoch(1)
        self.assertIsInstance(result, int)

    def test_pop_played_returns_list(self):
        adapter, _ = _make_adapter()
        adapter.synthesize("test", epoch=0, pts_us=0)
        result = adapter.pop_played_segments(100000)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for NvcWorker with a fake TTS engine (no torch dependency)."""

from __future__ import annotations

import asyncio
import time
import unittest
from collections.abc import Generator

import numpy as np

from liveavatar.voice.lease import CancelToken
from liveavatar.voice.worker import CharacterAssets, NvcWorker

# ── Fake TTS engine ──

class FakeTts:
    """Deterministic TTS that yields canned float32 audio chunks.

    Mimics the GPT-SoVITS ``TTS.run()`` interface:
    ``run(inputs: dict) -> Generator[tuple[int, np.ndarray]]``
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_chunks: int = 3,
        chunk_samples: int = 1600,
        delay: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_chunks = n_chunks
        self.chunk_samples = chunk_samples
        self.delay = delay  # seconds to sleep between chunks
        self.last_request: dict | None = None
        self.run_count: int = 0

    def run(self, inputs: dict) -> Generator[tuple[int, np.ndarray], None, None]:
        self.run_count += 1
        self.last_request = inputs
        for i in range(self.n_chunks):
            if self.delay > 0:
                time.sleep(self.delay)
            # Sine wave at 440 Hz, amplitude 0.5.
            t = np.arange(self.chunk_samples) / self.sample_rate
            freq = 440 + i * 100
            audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            yield (self.sample_rate, audio)


class FailingTts:
    """TTS that raises after the first chunk."""

    def __init__(self, sample_rate: int = 32000) -> None:
        self.sample_rate = sample_rate

    def run(self, inputs: dict) -> Generator[tuple[int, np.ndarray], None, None]:
        yield (self.sample_rate, np.zeros(100, dtype=np.float32))
        raise RuntimeError("simulated TTS failure")


# ── helpers ──

def _make_assets(char_id: str = "nahida") -> CharacterAssets:
    return CharacterAssets(
        char_id=char_id,
        gpt_path=f"weights/{char_id}/model.ckpt",
        sovits_path=f"weights/{char_id}/model.pth",
        ref_audio_path=f"weights/{char_id}/ref.wav",
        ref_text="你好世界",
    )


def _make_worker(
    *,
    tts: FakeTts | None = None,
    target_sr: int = 16000,
) -> tuple[NvcWorker, FakeTts]:
    tts = tts or FakeTts()
    worker = NvcWorker(_make_assets(), tts, target_sample_rate=target_sr)
    return worker, tts


# ── tests ──

class TestNvcWorkerBasics(unittest.TestCase):
    def test_char_id_matches_assets(self):
        worker, _ = _make_worker()
        self.assertEqual(worker.char_id, "nahida")

    def test_idle_when_not_synthesizing(self):
        worker, _ = _make_worker()
        self.assertTrue(worker.idle)
        self.assertFalse(worker.busy)

    def test_uptime_is_positive(self):
        worker, _ = _make_worker()
        # Windows clock granularity can quantize short sleeps: poll until
        # the monotonic clock advances past the worker's creation stamp.
        deadline = time.monotonic() + 1.0
        while worker.uptime_s <= 0.0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreater(worker.uptime_s, 0.0)


class TestSynthesizeStream(unittest.IsolatedAsyncioTestCase):
    async def test_yields_all_chunks(self):
        worker, tts = _make_worker(tts=FakeTts(n_chunks=3))
        chunks = []
        async for pcm in worker.synthesize_stream("hello"):
            chunks.append(pcm)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(worker.stats.chunks_emitted, 3)
        self.assertEqual(worker.stats.syntheses_completed, 1)
        self.assertEqual(worker.stats.syntheses_cancelled, 0)

    async def test_each_chunk_is_s16le_bytes(self):
        worker, _ = _make_worker(tts=FakeTts(n_chunks=1, chunk_samples=3200))
        async for pcm in worker.synthesize_stream("test"):
            # 3200 samples at 32kHz → resampled to 16kHz = 1600 samples
            # × 2 bytes (int16) = 3200 bytes
            self.assertIsInstance(pcm, bytes)
            self.assertEqual(len(pcm), 3200)
            break

    async def test_resampling_32k_to_16k(self):
        """Output should be at target_sample_rate (16kHz)."""
        worker, _ = _make_worker(
            tts=FakeTts(sample_rate=32000, n_chunks=1, chunk_samples=3200),
            target_sr=16000,
        )
        async for pcm in worker.synthesize_stream("test"):
            # 3200 samples @ 32kHz → 1600 samples @ 16kHz → 3200 bytes
            self.assertEqual(len(pcm), 3200)
            break

    async def test_no_resample_when_sr_matches(self):
        """If TTS output is already at target_sr, no resampling."""
        worker, _ = _make_worker(
            tts=FakeTts(sample_rate=16000, n_chunks=1, chunk_samples=1600),
            target_sr=16000,
        )
        async for pcm in worker.synthesize_stream("test"):
            # 1600 samples @ 16kHz → 3200 bytes
            self.assertEqual(len(pcm), 3200)
            break

    async def test_busy_flag_during_synthesis(self):
        worker, tts = _make_worker(tts=FakeTts(n_chunks=5))

        async def _check_busy():
            while not worker.busy:
                await asyncio.sleep(0.001)
            self.assertTrue(worker.busy)

        check_task = asyncio.create_task(_check_busy())
        async for _ in worker.synthesize_stream("test"):
            pass
        await check_task
        self.assertFalse(worker.busy)
        self.assertTrue(worker.idle)

    async def test_cancellation_stops_early(self):
        # Use a delay between chunks so cancellation can fire mid-stream.
        worker, _ = _make_worker(tts=FakeTts(n_chunks=10, delay=0.03))
        token = CancelToken()
        chunks = []

        async def _cancel_after_first():
            await asyncio.sleep(0.05)
            token.cancel()

        cancel_task = asyncio.create_task(_cancel_after_first())
        async for pcm in worker.synthesize_stream("test", cancel_token=token):
            chunks.append(pcm)
        await cancel_task

        # Should have stopped before all 10 chunks.
        self.assertLess(len(chunks), 10)
        self.assertEqual(worker.stats.syntheses_cancelled, 1)
        self.assertEqual(worker.stats.syntheses_completed, 0)

    async def test_pre_cancelled_token_yields_nothing(self):
        worker, _ = _make_worker(tts=FakeTts(n_chunks=5))
        token = CancelToken()
        token.cancel()
        chunks = []
        async for pcm in worker.synthesize_stream("test", cancel_token=token):
            chunks.append(pcm)
        self.assertEqual(len(chunks), 0)
        self.assertEqual(worker.stats.syntheses_cancelled, 1)

    async def test_failure_increments_error_count(self):
        worker, _ = _make_worker(tts=FailingTts())  # type: ignore
        with self.assertRaises(RuntimeError):
            async for _ in worker.synthesize_stream("test"):
                pass
        self.assertEqual(worker.stats.syntheses_failed, 1)
        self.assertEqual(worker.stats.syntheses_completed, 0)
        self.assertFalse(worker.busy)

    async def test_inference_lock_serializes_calls(self):
        """Two concurrent synthesize_stream calls run sequentially."""
        worker, _ = _make_worker(tts=FakeTts(n_chunks=3))

        async def _synth():
            chunks = []
            async for pcm in worker.synthesize_stream("x"):
                chunks.append(pcm)
            return len(chunks)

        results = await asyncio.gather(_synth(), _synth())
        self.assertEqual(results, [3, 3])
        self.assertEqual(worker.stats.syntheses_completed, 2)

    async def test_request_uses_streaming_mode(self):
        worker, tts = _make_worker()
        async for _ in worker.synthesize_stream("hello"):
            break
        self.assertIsNotNone(tts.last_request)
        self.assertTrue(tts.last_request["streaming_mode"])
        self.assertTrue(tts.last_request["fixed_length_chunk"])
        self.assertEqual(tts.last_request["text"], "hello")

    async def test_request_includes_character_assets(self):
        worker, tts = _make_worker()
        async for _ in worker.synthesize_stream("hi"):
            break
        req = tts.last_request
        self.assertEqual(req["ref_audio_path"], "weights/nahida/ref.wav")
        self.assertEqual(req["prompt_text"], "你好世界")
        self.assertEqual(req["prompt_lang"], "zh")


class TestPcmConversion(unittest.TestCase):
    def test_float32_to_int16_range(self):
        """Out-of-range audio is peak-normalized to 0.95 before int16 clipping."""
        worker, _ = _make_worker(target_sr=16000)
        # Create audio exceeding [-1, 1] range.
        loud = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
        pcm = worker._to_canonical_pcm(loud, 16000)
        samples = np.frombuffer(pcm, dtype=np.int16)
        self.assertEqual(len(samples), 4)
        # peak=2.0 → normalize to 0.95 → ±31128; 0.5/2.0*0.95 → ±7782
        self.assertAlmostEqual(abs(int(samples[0])), int(0.95 * 32767), delta=2)
        self.assertAlmostEqual(abs(int(samples[1])), int(0.95 * 32767), delta=2)
        self.assertAlmostEqual(abs(int(samples[2])), int(0.2375 * 32767), delta=2)
        self.assertAlmostEqual(abs(int(samples[3])), int(0.2375 * 32767), delta=2)

    def test_empty_audio_yields_empty_bytes(self):
        worker, _ = _make_worker()
        pcm = worker._to_canonical_pcm(np.array([], dtype=np.float32), 16000)
        self.assertEqual(len(pcm), 0)

    def test_stereo_flattened_to_mono(self):
        """2-D array (channels, samples) is flattened to 1-D."""
        worker, _ = _make_worker(target_sr=16000)
        stereo = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
        pcm = worker._to_canonical_pcm(stereo, 16000)
        samples = np.frombuffer(pcm, dtype=np.int16)
        # Flattened to 6 samples (2 channels × 3 samples).
        self.assertEqual(len(samples), 6)

    def test_resample_halves_samples(self):
        """32kHz → 16kHz should roughly halve the sample count."""
        worker, _ = _make_worker(target_sr=16000)
        audio = np.ones(3200, dtype=np.float32)
        pcm = worker._to_canonical_pcm(audio, 32000)
        samples = np.frombuffer(pcm, dtype=np.int16)
        self.assertEqual(len(samples), 1600)


class TestWorkerStats(unittest.TestCase):
    def test_to_dict_contains_all_fields(self):
        worker, _ = _make_worker()
        d = worker.stats.to_dict()
        for key in (
            "syntheses_started",
            "syntheses_completed",
            "syntheses_cancelled",
            "syntheses_failed",
            "chunks_emitted",
            "bytes_emitted",
            "total_inference_s",
        ):
            self.assertIn(key, d)

    def test_to_dict_includes_assets(self):
        worker, _ = _make_worker()
        d = worker.to_dict()
        self.assertIn("char_id", d)
        self.assertIn("busy", d)
        self.assertIn("stats", d)
        self.assertIn("assets", d)
        self.assertEqual(d["assets"]["ref_text"], "你好世界")


if __name__ == "__main__":
    unittest.main()

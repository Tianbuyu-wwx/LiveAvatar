# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for DuplexSession (full-duplex star topology over WS, audio plane).

Covers: PCM re-chunking, the mic → VAD/EOU/ASR → TTS loop with the
reference adapters + FakeTts (CPU-only), barge-in epoch advance, spoke
description and stats. No GPU, no network.
"""

from __future__ import annotations

import asyncio
import unittest

from liveavatar.duplex import DuplexSession, DuplexSettings
from liveavatar.runtime.fake_tts import FakeTts

LOUD_CHUNK = b"\x00\x40" * 3200  # 100ms loud @ 16kHz s16le mono
SILENT_CHUNK = b"\x00\x00" * 3200  # 100ms silence


class TestDuplexSettings(unittest.TestCase):
    def test_from_env_defaults(self):
        settings = DuplexSettings.from_env()
        self.assertEqual(settings.asr_url, "")
        self.assertFalse(settings.enable_aec)
        self.assertEqual(settings.char_id, "")
        self.assertFalse(settings.with_avatar)
        self.assertEqual(
            settings.describe(),
            {
                "tts": "fake",
                "asr": "reference",
                "llm": "echo",
                "aec": "off",
                "avatar": "audio_only",
            },
        )

    def test_describe_with_spokes(self):
        settings = DuplexSettings(
            asr_url="ws://asr",
            enable_aec=True,
            char_id="char1",
            llm_base_url="http://llm/v1",
            llm_model="m1",
            with_avatar=True,
        )
        self.assertEqual(
            settings.describe(),
            {
                "tts": "voice_pool",
                "asr": "remote",
                "llm": "openai-chat",
                "aec": "nlms",
                "avatar": "on",
            },
        )


class _SessionHarness(unittest.IsolatedAsyncioTestCase):
    """Base: a started duplex session + output collector helpers."""

    def _make_session(self, session_id: str = "dup1", **kwargs) -> DuplexSession:
        return DuplexSession(
            session_id,
            "yongen",
            settings=DuplexSettings(),
            **kwargs,
        )

    async def _collect(self, session, kind, count, timeout=5.0):
        out = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while len(out) < count:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            item = session.out_queue.try_dequeue()
            if item is None:
                await asyncio.sleep(0.01)
                continue
            if kind is None or item["kind"] == kind:
                out.append(item)
        return out

    async def _wait_for_event(self, session, event_type, timeout=5.0):
        """Drain out_queue until an event of ``event_type`` shows up."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        seen = []
        while loop.time() < deadline:
            item = session.out_queue.try_dequeue()
            if item is None:
                await asyncio.sleep(0.01)
                continue
            if item["kind"] == "event":
                if item["event_type"] == event_type:
                    return item, seen
                seen.append(item["event_type"])
        raise AssertionError(
            f"no {event_type!r} event within {timeout}s; saw {seen}"
        )

    async def _speak_utterance(self, session) -> None:
        """10 loud frames (partial) + 30 silent frames (final) of mic audio."""
        await session.push_pcm(LOUD_CHUNK)
        await asyncio.sleep(0.05)
        await session.push_pcm(SILENT_CHUNK * 3)


class TestDuplexSessionAudioLoop(_SessionHarness):
    async def test_full_loop_mic_to_tts_audio(self):
        session = self._make_session()
        await session.start()
        try:
            await self._speak_utterance(session)

            # An ASR final must have surfaced on the downlink.
            await self._wait_for_event(session, "asr")

            pcm_items = await self._collect(session, "pcm", 1)
            self.assertEqual(len(pcm_items), 1)
            self.assertGreater(len(pcm_items[0]["data"]), 0)
            self.assertEqual(pcm_items[0]["epoch"], 0)

            # FakeTts synthesized one segment from the ASR-final text.
            self.assertEqual(len(session.worker.tts.active_segments), 1)
            self.assertEqual(session.worker.stats.input_frames > 0, True)
        finally:
            await session.stop()

    async def test_pcm_rechunked_to_canonical_frames(self):
        session = self._make_session()
        await session.start()
        try:
            pushed = await session.push_pcm(LOUD_CHUNK)  # 200ms → 10 frames
            self.assertEqual(pushed, 10)
            # Whole chunks were consumed — nothing stays buffered.
            self.assertEqual(len(session._pcm_buf), 0)
            self.assertEqual(session._pts_us, 200_000)
        finally:
            await session.stop()

    async def test_barge_in_advances_epoch_and_cancels_tts(self):
        session = self._make_session()
        await session.start()
        try:
            await self._speak_utterance(session)
            await self._collect(session, "pcm", 1)

            old_epoch = session.worker.epoch
            new_epoch = session.cancel_epoch()
            self.assertEqual(new_epoch, old_epoch + 1)
            self.assertEqual(session.worker.epoch, new_epoch)

            # Old-epoch TTS segments were reaped by cancel_epoch.
            self.assertEqual(session.worker.tts.active_segments, [])
        finally:
            await session.stop()

    async def test_stats_shape(self):
        session = self._make_session()
        await session.start()
        try:
            stats = session.stats()
            self.assertEqual(stats["mode"], "duplex")
            self.assertEqual(stats["avatar_id"], "yongen")
            self.assertEqual(stats["spokes"]["tts"], "fake")
            self.assertIn("worker", stats)
        finally:
            await session.stop()

    async def test_stop_is_idempotent(self):
        session = self._make_session()
        await session.start()
        await session.stop()
        await session.stop()  # no raise
        self.assertFalse(session._running)


class TestDuplexSessionWiring(_SessionHarness):
    async def test_injected_worker_and_tts_are_used(self):
        tts = FakeTts()
        from liveavatar.runtime.worker import RealtimeWorker

        worker = RealtimeWorker("dup2", tts=tts)
        session = self._make_session("dup2", worker=worker)
        await session.start()
        try:
            self.assertIs(session.worker, worker)
            self.assertIs(session.worker.tts, tts)
        finally:
            await session.stop()

    async def test_llm_spoke_wired_from_settings(self):
        from liveavatar.text_source import OpenAIChatTextSource

        session = DuplexSession(
            "dup3",
            "yongen",
            settings=DuplexSettings(
                llm_base_url="http://llm/v1",
                llm_model="test-model",
                llm_system_prompt="hi",
            ),
        )
        try:
            self.assertIsInstance(session.worker.text_source, OpenAIChatTextSource)
        finally:
            # Not started — nothing to tear down beyond the worker.
            await session.worker.stop()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

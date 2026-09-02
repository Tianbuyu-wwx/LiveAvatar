# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for the plugin registry (ASR/TTS plugin interface)."""

from __future__ import annotations

import unittest

from liveavatar import plugins
from liveavatar.plugins import (
    ASRPlugin,
    TTSPlugin,
    available,
    clear,
    get,
    register,
)


class _FakeASR:
    name = "fake_asr"

    async def transcribe(self, pcm_s16le: bytes, sample_rate: int) -> str:
        return "hello"


class _FakeTTS:
    name = "fake_tts"

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        return b"\x00\x00" * 160, 16000


class TestRegistry(unittest.TestCase):
    def setUp(self) -> None:
        clear()

    def tearDown(self) -> None:
        clear()

    def test_register_and_get(self):
        asr = _FakeASR()
        register("asr", asr)
        self.assertIs(get("asr"), asr)
        self.assertEqual(available(), ["asr"])

    def test_get_unknown_returns_none(self):
        self.assertIsNone(get("ghost"))

    def test_last_registration_wins(self):
        register("p", _FakeASR())
        tts = _FakeTTS()
        register("p", tts)
        self.assertIs(get("p"), tts)

    def test_protocol_conformance(self):
        self.assertIsInstance(_FakeASR(), ASRPlugin)
        self.assertIsInstance(_FakeTTS(), TTSPlugin)

    def test_discover_entry_points_no_entries(self):
        # No installed package provides liveavatar.plugins — returns [].
        self.assertEqual(plugins.discover_entry_points(), [])
        # Second call is a no-op (discovered once per process).
        self.assertEqual(plugins.discover_entry_points(), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

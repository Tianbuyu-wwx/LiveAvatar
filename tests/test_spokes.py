"""Tests for shared spoke assembly (A2): resolve/build helpers."""

from __future__ import annotations

import logging
import unittest
from unittest import mock

from liveavatar import spokes
from liveavatar.spokes import (
    build_tts_adapter,
    resolve_aec,
    resolve_avatar_adapter,
    resolve_avatar_pool,
    resolve_remote_asr,
    resolve_text_source,
    resolve_voice_pool,
    static_fallback_worker,
)

logger = logging.getLogger("test.spokes")


class TestResolveRemoteAsr(unittest.TestCase):
    def test_no_url_returns_none(self):
        self.assertIsNone(resolve_remote_asr("", "s1", logger=logger))
        self.assertIsNone(resolve_remote_asr(None, "s1", logger=logger))

    @unittest.skipUnless(spokes.HAS_REMOTE, "remote adapters not installed")
    def test_builds_shared_client_trio(self):
        result = resolve_remote_asr("ws://x/asr", "s1", logger=logger)
        self.assertIsNotNone(result)
        self.assertIs(result.vad._client, result.client)
        self.assertIs(result.eou._client, result.client)
        self.assertIs(result.asr._client, result.client)

    def test_degrades_when_unavailable(self):
        with mock.patch.object(spokes, "HAS_REMOTE", False):
            with self.assertLogs(logger, level="WARNING"):
                result = resolve_remote_asr("ws://x/asr", "s1", logger=logger)
        self.assertIsNone(result)


class TestResolveAec(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(resolve_aec(False, logger=logger))

    @unittest.skipUnless(spokes.HAS_AEC, "NlmsAec not installed")
    def test_enabled_builds_nlms(self):
        aec = resolve_aec(True, logger=logger)
        self.assertIsNotNone(aec)

    def test_degrades_when_unavailable(self):
        with mock.patch.object(spokes, "HAS_AEC", False):
            with self.assertLogs(logger, level="WARNING"):
                self.assertIsNone(resolve_aec(True, logger=logger))


class TestResolveTextSource(unittest.TestCase):
    def test_incomplete_config_returns_none(self):
        self.assertIsNone(
            resolve_text_source(base_url="", api_key="k", model="m",
                                system_prompt="", logger=logger)
        )
        self.assertIsNone(
            resolve_text_source(base_url="http://x", api_key="k", model="",
                                system_prompt="", logger=logger)
        )

    def test_degrades_when_unavailable(self):
        with mock.patch.object(spokes, "HAS_TEXT_SOURCE", False):
            with self.assertLogs(logger, level="WARNING"):
                self.assertIsNone(
                    resolve_text_source(base_url="http://x", api_key="k",
                                        model="m", system_prompt="p",
                                        logger=logger)
                )


class TestResolveVoicePool(unittest.TestCase):
    def test_external_pool_wins(self):
        pool = object()
        got, owns = resolve_voice_pool(pool, object(), None)
        self.assertIs(got, pool)
        self.assertFalse(owns)

    def test_no_pool_no_config(self):
        got, owns = resolve_voice_pool(None, None, None)
        self.assertIsNone(got)
        self.assertFalse(owns)

    def test_config_ignored_when_extra_missing(self):
        with mock.patch.object(spokes, "HAS_VOICE_POOL", False):
            got, owns = resolve_voice_pool(None, object(), None)
        self.assertIsNone(got)
        self.assertFalse(owns)


class TestBuildTtsAdapter(unittest.TestCase):
    def test_none_cases(self):
        self.assertIsNone(build_tts_adapter(None, "s1", "char"))
        self.assertIsNone(build_tts_adapter(object(), "s1", ""))
        with mock.patch.object(spokes, "HAS_VOICE_POOL", False):
            self.assertIsNone(build_tts_adapter(object(), "s1", "char"))


class TestResolveAvatarPool(unittest.TestCase):
    def test_external_pool_wins(self):
        pool = object()
        got, owns = resolve_avatar_pool(pool, object(), None)
        self.assertIs(got, pool)
        self.assertFalse(owns)

    def test_no_pool_no_config(self):
        got, owns = resolve_avatar_pool(None, None, None)
        self.assertIsNone(got)
        self.assertFalse(owns)

    def test_config_ignored_when_extra_missing(self):
        with mock.patch.object(spokes, "HAS_AVATAR", False):
            got, owns = resolve_avatar_pool(None, object(), None)
        self.assertIsNone(got)
        self.assertFalse(owns)


class TestResolveAvatarAdapter(unittest.TestCase):
    def test_none_when_no_pool_or_publisher(self):
        self.assertIsNone(resolve_avatar_adapter(None, "s1", "a", object()))
        self.assertIsNone(resolve_avatar_adapter(object(), "s1", "a", None))

    def test_none_when_extra_missing(self):
        with mock.patch.object(spokes, "HAS_AVATAR", False):
            self.assertIsNone(
                resolve_avatar_adapter(object(), "s1", "a", object())
            )

    @unittest.skipUnless(spokes.HAS_AVATAR, "avatar extra not installed")
    def test_builds_adapter_with_kwargs(self):
        pool, publisher = object(), object()
        adapter = resolve_avatar_adapter(
            pool, "s1", "a", publisher,
            fallback_worker=object(),
            degrade_after_errors=5,
        )
        self.assertIsNotNone(adapter)
        self.assertIs(adapter._pool, pool)
        self.assertIs(adapter._publisher, publisher)
        self.assertEqual(adapter._avatar_id, "a")
        self.assertEqual(adapter._degrade_after_errors, 5)


class TestStaticFallbackWorker(unittest.TestCase):
    def test_none_when_extra_missing(self):
        with mock.patch.object(spokes, "HAS_AVATAR", False):
            self.assertIsNone(static_fallback_worker("a"))

    @unittest.skipUnless(spokes.HAS_AVATAR, "avatar extra not installed")
    def test_empty_assets(self):
        worker = static_fallback_worker("a")
        self.assertEqual(worker.assets.avatar_id, "a")
        self.assertEqual(worker.assets.data_dir, "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

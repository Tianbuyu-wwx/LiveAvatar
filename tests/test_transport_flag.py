"""Tests for the LIVEAVATAR_TRANSPORT switch (R2 M3).

- default (no env) → ws: sessions get video_ws, no LiveKit token/url
- transport=ws with LiveKit configured → still ws (no room join)
- transport=livekit without config → 503
- invalid transport env → ValueError at from_env
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from liveavatar.publish import PublishSettings, app, state
from tests.test_publish import _configure_capture_mode


class TransportFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        _configure_capture_mode()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        state.settings.transport = "ws"
        state.settings.livekit_url = ""
        state.pipeline = None

    def test_default_transport_is_ws(self) -> None:
        self.assertEqual(state.settings.transport, "ws")
        resp = self.client.post("/v1/sessions", json={"avatar_id": "yongen"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["transport"], "ws")
        self.assertFalse(body["livekit"])
        self.assertTrue(body["video_ws"].endswith("/video"))
        self.assertNotIn("token", body)
        self.assertNotIn("url", body)

    def test_ws_transport_wins_over_livekit_config(self) -> None:
        """Explicit ws + full LiveKit env → no room join, ws response."""
        state.settings.transport = "ws"
        state.settings.livekit_url = "wss://example"
        state.settings.livekit_api_key = "k"
        state.settings.livekit_api_secret = "s"
        resp = self.client.post("/v1/sessions", json={"avatar_id": "yongen"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["transport"], "ws")
        self.assertFalse(body["livekit"])
        self.assertNotIn("token", body)

    def test_livekit_transport_without_config_503(self) -> None:
        state.settings.transport = "livekit"
        resp = self.client.post("/v1/sessions", json={"avatar_id": "yongen"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("LIVEKIT_URL", resp.json()["error"])

    def test_livekit_transport_without_sdk_503(self) -> None:
        state.settings.transport = "livekit"
        state.settings.livekit_url = "wss://example"
        state.settings.livekit_api_key = "k"
        state.settings.livekit_api_secret = "s"
        if "livekit" in __import__("sys").modules:
            self.skipTest("livekit SDK installed; cannot test missing-SDK path")
        resp = self.client.post("/v1/sessions", json={"avatar_id": "yongen"})
        self.assertEqual(resp.status_code, 503)

    def test_invalid_transport_env_raises(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(
            os.environ, {"LIVEAVATAR_TRANSPORT": "grpc"}, clear=False
        ):
            with self.assertRaises(ValueError):
                PublishSettings.from_env()

    def test_from_env_reads_transport(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(
            os.environ, {"LIVEAVATAR_TRANSPORT": "LIVEKIT"}, clear=False
        ):
            settings = PublishSettings.from_env()
        self.assertEqual(settings.transport, "livekit")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

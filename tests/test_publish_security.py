"""Security tests for the publish service: auth, limits, validation, static."""

from __future__ import annotations

import base64
import json
import time
import unittest

from liveavatar.publish import state
from tests.test_publish import PublishTestCase, _pcm


class TestApiAuth(PublishTestCase):
    """When api_key is set, REST and WS endpoints require it."""

    def setUp(self) -> None:
        super().setUp()
        state.settings.api_key = "secret1"

    def tearDown(self) -> None:
        state.settings.api_key = ""
        super().tearDown()

    def test_create_session_unauthorized(self):
        resp = self.client.post("/v1/sessions", json={})
        self.assertEqual(resp.status_code, 401)

    def test_delete_session_unauthorized(self):
        resp = self.client.delete("/v1/sessions/s1")
        self.assertEqual(resp.status_code, 401)

    def test_stats_unauthorized(self):
        resp = self.client.get("/v1/sessions/s1/stats")
        self.assertEqual(resp.status_code, 401)

    def test_avatars_unauthorized(self):
        resp = self.client.get("/v1/avatars")
        self.assertEqual(resp.status_code, 401)

    def test_wrong_key_unauthorized(self):
        resp = self.client.post(
            "/v1/sessions", json={}, headers={"X-API-Key": "wrong"}
        )
        self.assertEqual(resp.status_code, 401)

    def test_correct_key_allowed(self):
        resp = self.client.post(
            "/v1/sessions", json={"session_id": "s1"}, headers={"X-API-Key": "secret1"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["session_id"], "s1")

    def test_health_stays_open(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_ws_missing_key_rejected(self):
        self.client.post(
            "/v1/sessions", json={"session_id": "s1"}, headers={"X-API-Key": "secret1"}
        )
        with self.assertRaises(Exception):  # noqa: B017 - close-before-accept
            with self.client.websocket_connect("/v1/sessions/s1/audio"):
                pass

    def test_ws_wrong_key_rejected(self):
        self.client.post(
            "/v1/sessions", json={"session_id": "s1"}, headers={"X-API-Key": "secret1"}
        )
        with self.assertRaises(Exception):  # noqa: B017
            with self.client.websocket_connect("/v1/sessions/s1/audio?api_key=nope"):
                pass

    def test_ws_query_key_allowed(self):
        self.client.post(
            "/v1/sessions", json={"session_id": "s1"}, headers={"X-API-Key": "secret1"}
        )
        with self.client.websocket_connect(
            "/v1/sessions/s1/audio?api_key=secret1"
        ) as ws:
            ws.send_text(json.dumps({"type": "stop"}))
        resp = self.client.get(
            "/v1/sessions/s1/stats", headers={"X-API-Key": "secret1"}
        )
        self.assertEqual(resp.status_code, 200)

    def test_ws_header_key_allowed(self):
        self.client.post(
            "/v1/sessions", json={"session_id": "s1"}, headers={"X-API-Key": "secret1"}
        )
        with self.client.websocket_connect(
            "/v1/sessions/s1/audio", headers={"X-API-Key": "secret1"}
        ) as ws:
            ws.send_text(json.dumps({"type": "stop"}))


class TestSessionLimit(PublishTestCase):
    def test_session_limit_429(self):
        state.settings.max_sessions = 1
        try:
            r1 = self.client.post("/v1/sessions", json={"session_id": "s1"})
            self.assertEqual(r1.status_code, 200)
            r2 = self.client.post("/v1/sessions", json={"session_id": "s2"})
            self.assertEqual(r2.status_code, 429)
        finally:
            state.settings.max_sessions = 16

    def test_limit_released_after_close(self):
        state.settings.max_sessions = 1
        try:
            self.client.post("/v1/sessions", json={"session_id": "s1"})
            self.client.delete("/v1/sessions/s1")
            r = self.client.post("/v1/sessions", json={"session_id": "s2"})
            self.assertEqual(r.status_code, 200)
        finally:
            state.settings.max_sessions = 16


class TestBodyValidation(PublishTestCase):
    def test_invalid_session_id_422(self):
        resp = self.client.post("/v1/sessions", json={"session_id": "bad id!"})
        self.assertEqual(resp.status_code, 422)

    def test_oversized_session_id_422(self):
        resp = self.client.post(
            "/v1/sessions", json={"session_id": "x" * 100}
        )
        self.assertEqual(resp.status_code, 422)

    def test_valid_session_id_accepted(self):
        resp = self.client.post(
            "/v1/sessions", json={"session_id": "sess_A-b_1"}
        )
        self.assertEqual(resp.status_code, 200)


class TestAvatarIdValidation(PublishTestCase):
    """S4: avatar ids are path segments — reject escape payloads."""

    def test_traversal_rejected(self):
        resp = self.client.post(
            "/v1/sessions", json={"avatar_id": "../yongen"}
        )
        self.assertEqual(resp.status_code, 422)

    def test_path_separator_rejected(self):
        for payload in ("a/b", "a\\b"):
            resp = self.client.post("/v1/sessions", json={"avatar_id": payload})
            self.assertEqual(resp.status_code, 422, payload)

    def test_whitespace_and_unicode_rejected(self):
        for payload in ("a b", "刻晴", "a\tb"):
            resp = self.client.post("/v1/sessions", json={"avatar_id": payload})
            self.assertEqual(resp.status_code, 422, payload)

    def test_null_byte_rejected(self):
        resp = self.client.post("/v1/sessions", json={"avatar_id": "a\x00b"})
        self.assertEqual(resp.status_code, 422)

    def test_duplex_mode_rejects_too(self):
        resp = self.client.post(
            "/v1/sessions", json={"avatar_id": "../yongen", "mode": "duplex"}
        )
        self.assertEqual(resp.status_code, 422)

    def test_safe_charset_accepted(self):
        resp = self.client.post(
            "/v1/sessions", json={"avatar_id": "Yongen-2_1"}
        )
        self.assertEqual(resp.status_code, 200)

    def test_valid_avatar_id_predicate(self):
        from liveavatar.publish import _valid_avatar_id

        self.assertTrue(_valid_avatar_id("yongen"))
        self.assertTrue(_valid_avatar_id("A-b_9"))
        self.assertFalse(_valid_avatar_id("../etc"))
        self.assertFalse(_valid_avatar_id(""))

    def test_region_encoder_invalid_id_falls_back(self):
        from liveavatar.publish import _region_encoder_for

        self.assertIsNone(_region_encoder_for("../escape"))
        self.assertIsNone(_region_encoder_for("a/b"))


class TestWsFrameLimit(PublishTestCase):
    def test_oversized_frame_dropped(self):
        state.settings.max_ws_frame_bytes = 64
        try:
            self.client.post("/v1/sessions", json={"session_id": "s1"})
            with self.client.websocket_connect("/v1/sessions/s1/audio") as ws:
                ws.send_bytes(_pcm(200))  # 400 bytes > 64 → dropped
                time.sleep(0.2)
                ws.send_text(json.dumps({"type": "stop"}))
            time.sleep(0.1)
            body = self.client.get("/v1/sessions/s1/stats").json()
            self.assertEqual(body["adapter"]["pcm_chunks_pushed"], 0)
        finally:
            state.settings.max_ws_frame_bytes = 65536

    def test_normal_frame_accepted(self):
        state.settings.max_ws_frame_bytes = 64
        try:
            self.client.post("/v1/sessions", json={"session_id": "s1"})
            with self.client.websocket_connect("/v1/sessions/s1/audio") as ws:
                ws.send_bytes(_pcm(20))  # 40 bytes ≤ 64
                time.sleep(0.2)
                ws.send_text(json.dumps({"type": "stop"}))
            time.sleep(0.1)
            body = self.client.get("/v1/sessions/s1/stats").json()
            self.assertEqual(body["adapter"]["pcm_chunks_pushed"], 1)
        finally:
            state.settings.max_ws_frame_bytes = 65536


class TestSessionTokenScope(unittest.TestCase):
    """Session tokens (M-C task 17 base) carry an explicit scope claim."""

    def test_scope_claim(self):
        from liveavatar.publish import make_session_token

        token = make_session_token(
            api_key="k", api_secret="s", session_id="viewer", scope="session"
        )
        payload_b64 = token.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
        self.assertEqual(payload["scope"], "session")
        self.assertEqual(payload["sub"], "viewer")

    def test_default_scope_is_session(self):
        from liveavatar.publish import make_session_token

        token = make_session_token(api_key="k", api_secret="s", session_id="bot")
        payload = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "=="))
        self.assertEqual(payload["scope"], "session")


class TestStaticServing(PublishTestCase):
    """web/ exists in-repo → StaticFiles mount is active."""

    def test_app_js_served(self):
        resp = self.client.get("/app.js")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("javascript", resp.headers.get("content-type", ""))

    def test_unknown_path_404(self):
        resp = self.client.get("/definitely-not-here.txt")
        self.assertEqual(resp.status_code, 404)

    def test_traversal_blocked(self):
        resp = self.client.get("/..%2F..%2Fpyproject.toml")
        self.assertIn(resp.status_code, (403, 404))

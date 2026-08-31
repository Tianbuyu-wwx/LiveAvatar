"""M-D: HS256 session-token wiring into the publish service.

POST /v1/sessions mints a short-lived per-session token when
``api_secret`` is configured; WS handshakes may present it (``?token=`` /
``X-Session-Token`` / ``Authorization: Bearer``) instead of the static
``api_key``. Wrong session, tampered or expired tokens are rejected.
"""

from __future__ import annotations

import base64
import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from liveavatar.publish import PublishSettings, app, state
from liveavatar.publish.tokens import make_session_token, verify_session_token
from tests.test_publish import _pcm


def _b64_to_dict(seg: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))


class _TokenBase(unittest.TestCase):
    """WS-transport capture mode (WebSocketSink publishers) + key/secret."""

    KEY = "key1"
    SECRET = "sec1"

    def setUp(self) -> None:
        from tests.test_video_ws import _configure_ws_transport_mode

        _configure_ws_transport_mode()
        state.settings.api_key = self.KEY
        state.settings.api_secret = self.SECRET
        state.settings.token_ttl_s = 300
        self.client = TestClient(app)

    def tearDown(self) -> None:
        state.settings.api_key = ""
        state.settings.api_secret = ""
        state.pipeline = None

    def _create(self, session_id: str = "s1") -> dict:
        resp = self.client.post(
            "/v1/sessions",
            json={"session_id": session_id},
            headers={"X-API-Key": self.KEY},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()


class TestTokenIssuance(_TokenBase):
    def test_create_response_carries_session_token(self) -> None:
        body = self._create("s1")
        self.assertIn("session_token", body)
        claims = verify_session_token(body["session_token"], self.SECRET)
        self.assertIsNotNone(claims)
        self.assertEqual(claims["sub"], "s1")
        self.assertEqual(claims["scope"], "session")
        self.assertEqual(claims["iss"], self.KEY)
        self.assertLessEqual(claims["exp"] - claims["iat"], 300)

    def test_ttl_from_settings(self) -> None:
        state.settings.token_ttl_s = 42
        body = self._create("s2")
        claims = verify_session_token(body["session_token"], self.SECRET)
        self.assertEqual(claims["exp"] - claims["iat"], 42)

    def test_no_secret_no_token(self) -> None:
        state.settings.api_secret = ""
        body = self._create("s3")
        self.assertNotIn("session_token", body)


class TestWsTokenAuth(_TokenBase):
    def test_audio_ws_with_query_token_allowed(self) -> None:
        body = self._create("s1")
        token = body["session_token"]
        with self.client.websocket_connect(
            f"/v1/sessions/s1/audio?token={token}"
        ) as ws:
            ws.send_json({"type": "epoch", "epoch": 1})
            ws.send_bytes(_pcm(1600))
            ws.send_json({"type": "stop"})

    def _stub_ws(self, *, token=None, api_key=None, bearer=None):
        class _Stub:
            query_params = (
                {"token": token} if token else
                ({"api_key": api_key} if api_key else {})
            )
            headers = (
                {"authorization": bearer} if bearer else
                ({"x-session-token": token} if token else {})
            )

        return _Stub()

    def test_bearer_header_accepted(self) -> None:
        from liveavatar.publish.routes import _check_ws_auth

        body = self._create("s1")
        token = body["session_token"]
        self.assertTrue(
            _check_ws_auth(self._stub_ws(bearer=f"Bearer {token}"), "s1")
        )

    def test_session_token_header_accepted(self) -> None:
        from liveavatar.publish.routes import _check_ws_auth

        body = self._create("s1")
        token = body["session_token"]
        self.assertTrue(_check_ws_auth(self._stub_ws(token=token), "s1"))

    def test_video_ws_with_query_token_allowed(self) -> None:
        body = self._create("s1")
        token = body["session_token"]
        with self.client.websocket_connect(
            f"/v1/sessions/s1/video?token={token}"
        ) as ws:
            self.assertEqual(ws.receive_json()["type"], "ready")

    def test_token_for_other_session_rejected(self) -> None:
        self._create("s1")
        other = make_session_token(
            api_key=self.KEY, api_secret=self.SECRET, session_id="s2"
        )
        with self.assertRaises(Exception):  # noqa: B017 - close-before-accept
            with self.client.websocket_connect(
                f"/v1/sessions/s1/video?token={other}"
            ):
                pass

    def test_tampered_token_rejected(self) -> None:
        body = self._create("s1")
        head, payload, sig = body["session_token"].split(".")
        payload_dict = _b64_to_dict(payload)
        payload_dict["sub"] = "evil"
        evil_payload = base64.urlsafe_b64encode(
            json.dumps(payload_dict).encode()
        ).rstrip(b"=").decode()
        with self.assertRaises(Exception):  # noqa: B017
            with self.client.websocket_connect(
                f"/v1/sessions/s1/video?token={head}.{evil_payload}.{sig}"
            ):
                pass

    def test_expired_token_rejected(self) -> None:
        self._create("s1")
        expired = make_session_token(
            api_key=self.KEY, api_secret=self.SECRET,
            session_id="s1", ttl_s=-10,
        )
        with self.assertRaises(Exception):  # noqa: B017
            with self.client.websocket_connect(
                f"/v1/sessions/s1/video?token={expired}"
            ):
                pass

    def test_static_key_still_accepted(self) -> None:
        self._create("s1")
        with self.client.websocket_connect(
            "/v1/sessions/s1/video?api_key=key1"
        ) as ws:
            self.assertEqual(ws.receive_json()["type"], "ready")

    def test_wrong_scope_rejected(self) -> None:
        self._create("s1")
        rogue = make_session_token(
            api_key=self.KEY, api_secret=self.SECRET,
            session_id="s1", scope="admin",
        )
        with self.assertRaises(Exception):  # noqa: B017
            with self.client.websocket_connect(
                f"/v1/sessions/s1/video?token={rogue}"
            ):
                pass


class TestTokenEnv(unittest.TestCase):
    def test_from_env_reads_secret_and_ttl(self) -> None:
        env = {
            "LIVEAVATAR_API_KEY": "k",
            "LIVEAVATAR_API_SECRET": "s",
            "LIVEAVATAR_TOKEN_TTL_S": "60",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            settings = PublishSettings.from_env()
        self.assertEqual(settings.api_key, "k")
        self.assertEqual(settings.api_secret, "s")
        self.assertEqual(settings.token_ttl_s, 60)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

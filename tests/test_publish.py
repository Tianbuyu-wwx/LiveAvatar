# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for the publish service (FastAPI) using TestClient.

Runs in capture mode (no publisher factory) — sessions capture frames on
the adapter instead of publishing to a sink.
"""

from __future__ import annotations

import base64
import json
import time
import unittest

from fastapi.testclient import TestClient

from liveavatar.config import AvatarPoolConfig
from liveavatar.pipeline import AvatarPipeline
from liveavatar.publish import (
    PublishSettings,
    app,
    make_session_token,
    state,
    verify_session_token,
)
from liveavatar.worker import AvatarAssets, AvatarWorker
from tests.conftest import pcm as _pcm


class _ServiceWorker(AvatarWorker):
    """Minimal fake worker (4x4 BGR24 frames)."""

    def __init__(self, assets: AvatarAssets) -> None:
        super().__init__(assets, target_fps=25, width=4, height=4, batch_size=4)

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        return [(b"\x00" * 48, True) for _ in range(self.batch_size)]


class _FakeLease:
    def __init__(self, worker: AvatarWorker) -> None:
        self.worker = worker


class _FakeServicePool:
    """AvatarPool-compatible fake for service tests (no GPU, no files).

    Each lease gets a fresh worker: the real lease pool gives a session
    exclusive ownership of the avatar worker, so a shared instance here
    would break concurrent-session tests (asyncio.Lock bound to the first
    session's event loop).
    """

    def __init__(self) -> None:
        self._assets = AvatarAssets(
            avatar_id="yongen",
            data_dir="avatars/yongen",
            full_imgs_dir="avatars/yongen/full_imgs",
            coords_path="avatars/yongen/coords.pkl",
            latents_path="avatars/yongen/latents.pt",
            mask_dir="avatars/yongen/mask",
            mask_coords_path="avatars/yongen/mask_coords.pkl",
        )

    @property
    def available_avatars(self) -> list[str]:
        return ["yongen"]

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def acquire(self, session_id: str, avatar_id: str, **kwargs) -> _FakeLease:
        return _FakeLease(_ServiceWorker(self._assets))

    async def release_async(self, session_id: str) -> bool:
        return True

    def stats(self) -> dict:
        return {"fake": True}


def _configure_capture_mode() -> None:
    """Point the app at capture mode: no auth, fake pool, no files."""
    state.settings = PublishSettings()
    state.pipeline = AvatarPipeline(
        AvatarPoolConfig(avatar_data_root="/nonexistent"),
        pool=_FakeServicePool(),
    )
    state.pool_config = AvatarPoolConfig(avatar_data_root="/nonexistent")


class PublishTestCase(unittest.TestCase):
    """Base: fresh capture-mode state + TestClient per test."""

    def setUp(self) -> None:
        _configure_capture_mode()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        # Drop the pipeline so the next test starts clean.
        state.pipeline = None

    def _stats(self, session_id: str) -> dict:
        resp = self.client.get(f"/v1/sessions/{session_id}/stats")
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _poll_until(self, session_id: str, predicate, timeout_s: float = 5.0):
        """Poll session stats until ``predicate(body)`` holds.

        Deterministic replacement for fixed sleeps: the WS handler and the
        adapter consumer run on a background portal loop, so wall-clock
        sleeps are racy under CI load.
        """
        deadline = time.monotonic() + timeout_s
        body = self._stats(session_id)
        while not predicate(body):
            if time.monotonic() > deadline:
                self.fail(f"timeout waiting for stats condition: {body}")
            time.sleep(0.02)
            body = self._stats(session_id)
        return body


class TestHealthAndAvatars(PublishTestCase):
    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")

    def test_avatars_empty_root(self):
        resp = self.client.get("/v1/avatars")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["avatars"], [])
        self.assertEqual(body["data_root"], "/nonexistent")


class TestSessionToken(unittest.TestCase):
    """Session-token signing (M-C task 17 base): HS256 + short TTL."""

    def test_token_is_hs256_jwt(self):
        token = make_session_token(
            api_key="devkey", api_secret="secret", session_id="u1"
        )
        header_b64, payload_b64, sig = token.split(".")
        header = json.loads(base64.urlsafe_b64decode(header_b64 + "=="))
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
        self.assertEqual(header["alg"], "HS256")
        self.assertEqual(header["typ"], "JWT")
        self.assertEqual(payload["iss"], "devkey")
        self.assertEqual(payload["sub"], "u1")
        self.assertEqual(payload["scope"], "session")

    def test_token_deterministic_per_session(self):
        t1 = make_session_token(api_key="k", api_secret="s", session_id="a")
        t2 = make_session_token(api_key="k", api_secret="s", session_id="b")
        self.assertNotEqual(t1, t2)

    def test_verify_roundtrip_and_expiry(self):
        token = make_session_token(
            api_key="k", api_secret="s", session_id="u1", ttl_s=60
        )
        claims = verify_session_token(token, "s")
        self.assertIsNotNone(claims)
        self.assertEqual(claims["sub"], "u1")
        # Wrong secret → None.
        self.assertIsNone(verify_session_token(token, "wrong"))
        # Tampered payload → None.
        head, _, sig = token.split(".")
        self.assertIsNone(verify_session_token(f"{head}.eyJzdWIiOiJ4In0.{sig}", "s"))
        # Expired → None.
        expired = make_session_token(
            api_key="k", api_secret="s", session_id="u1", ttl_s=-10
        )
        self.assertIsNone(verify_session_token(expired, "s"))


class TestSessions(PublishTestCase):
    def test_create_session_capture_mode(self):
        resp = self.client.post("/v1/sessions", json={"session_id": "s1"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["session_id"], "s1")
        self.assertEqual(body["avatar_id"], "yongen")  # default placeholder
        self.assertEqual(body["transport"], "ws")
        self.assertEqual(body["sample_rate"], 16000)
        self.assertEqual(body["video_ws"], "/v1/sessions/s1/video")

    def test_create_session_generates_id(self):
        resp = self.client.post("/v1/sessions", json={})
        self.assertEqual(resp.status_code, 200)
        sid = resp.json()["session_id"]
        self.assertTrue(sid.startswith("sess_"))

    def test_delete_session(self):
        self.client.post("/v1/sessions", json={"session_id": "s1"})
        resp = self.client.delete("/v1/sessions/s1")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["closed"])
        # Second delete → 404.
        resp2 = self.client.delete("/v1/sessions/s1")
        self.assertEqual(resp2.status_code, 404)

    def test_stats_of_open_session(self):
        self.client.post("/v1/sessions", json={"session_id": "s1"})
        resp = self.client.get("/v1/sessions/s1/stats")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["session_id"], "s1")
        self.assertIn("adapter", body)

    def test_stats_unknown_session_404(self):
        resp = self.client.get("/v1/sessions/ghost/stats")
        self.assertEqual(resp.status_code, 404)


class TestWebSocketAudio(PublishTestCase):
    def test_ws_pcm_produces_frames(self):
        create = self.client.post("/v1/sessions", json={"session_id": "s1"})
        self.assertEqual(create.status_code, 200)

        with self.client.websocket_connect("/v1/sessions/s1/audio") as ws:
            # Send 3 chunks (60ms of audio → 12 frames at 25fps batch 4).
            for _ in range(3):
                ws.send_bytes(_pcm(320))
            # Poll inside the WS block: the consumer runs on the WS portal
            # loop, so keep the connection open until all frames land.
            body = self._poll_until(
                "s1", lambda b: b["adapter"]["frames_published"] >= 12
            )
            ws.send_text(json.dumps({"type": "stop"}))

        self.assertEqual(body["adapter"]["pcm_chunks_pushed"], 3)
        self.assertEqual(body["adapter"]["frames_published"], 12)

    def test_ws_epoch_control(self):
        """{"type":"epoch"} bumps the session epoch; pushes follow it.

        Binary frames always carry the session's current epoch, so they are
        never stale by construction — the stale-drop path is covered in
        test_pipeline (explicit epochs) and test_adapter (queue drain).
        """
        self.client.post("/v1/sessions", json={"session_id": "s1"})
        with self.client.websocket_connect("/v1/sessions/s1/audio") as ws:
            ws.send_text(json.dumps({"type": "epoch", "epoch": 5}))
            # Wait until the control message has actually been applied
            # server-side before pushing PCM.
            self._poll_until("s1", lambda b: b["epoch"] >= 5)
            ws.send_bytes(_pcm(320))
            body = self._poll_until(
                "s1", lambda b: b["adapter"]["pcm_chunks_pushed"] >= 1
            )
            self.assertEqual(body["epoch"], 5)
            self.assertEqual(body["adapter"]["frames_dropped_epoch"], 0)
            ws.send_text(json.dumps({"type": "cancel", "epoch": 6}))
            body = self._poll_until("s1", lambda b: b["epoch"] >= 6)
        self.assertEqual(body["epoch"], 6)

    def test_ws_unknown_session_rejected(self):
        # Starlette raises on close before accept; TestClient surfaces 403.
        with self.assertRaises(Exception):  # noqa: B017 - starlette error type varies by version
            with self.client.websocket_connect("/v1/sessions/ghost/audio"):
                pass


class TestStaticRoot(PublishTestCase):
    def test_root_responds(self):
        resp = self.client.get("/")
        # Either the demo HTML or the JSON fallback — both are 200.
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()

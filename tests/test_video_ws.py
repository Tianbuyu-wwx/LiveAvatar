"""End-to-end tests for the /v1/sessions/{sid}/video WS endpoint (R2 M1).

Runs on TestClient (CPU only) with the fake ServicePool: audio WS pushes
PCM → adapter → fake worker → WebSocketSink → video WS client.
"""

from __future__ import annotations

import time
import unittest

from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from liveavatar.config import AvatarPoolConfig
from liveavatar.pipeline import AvatarPipeline
from liveavatar.publish import (
    app,
    state,
)
from liveavatar.video_protocol import (
    FLAG_EOF,
    FLAG_EPOCH_BOUNDARY,
    FLAG_KEYFRAME,
    has_flag,
    unpack_region_payload,
    unpack_video_frame,
)
from tests.test_publish import _configure_capture_mode, _FakeServicePool

_PCM_CHUNK = b"\x01\x00" * 320  # 20 ms @16 kHz


def _configure_ws_transport_mode() -> None:
    """Capture mode, but with the real service publisher factory so the
    session publisher is a WebSocketSink (self-developed transport)."""
    _configure_capture_mode()
    from liveavatar.publish import _service_publisher_factory

    state.pipeline = AvatarPipeline(
        AvatarPoolConfig(avatar_data_root="/nonexistent"),
        pool=_FakeServicePool(),
        publisher_factory=_service_publisher_factory,
    )


class VideoWsTests(unittest.TestCase):
    def setUp(self) -> None:
        _configure_ws_transport_mode()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        state.pipeline = None

    def _create_session(self) -> str:
        resp = self.client.post("/v1/sessions", json={"avatar_id": "yongen"})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["session_id"]

    def _push_pcm(self, session_id: str, chunks: int = 8) -> None:
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/audio"
        ) as ws:
            for _ in range(chunks):
                ws.send_bytes(b"\x01\x00" * 320)  # 20 ms @16 kHz

    def test_ready_then_frames_then_close(self) -> None:
        session_id = self._create_session()
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video"
        ) as video:
            ready = video.receive_json()
            self.assertEqual(ready["type"], "ready")
            self.assertEqual(ready["codec"], "mjpeg_full")

            self._push_pcm(session_id, chunks=8)

            # Receive at least one binary wire frame (deterministic: the
            # audio WS only returns after all pushes are buffered, and the
            # adapter consumer runs on the portal loop).
            wire = None
            deadline = time.monotonic() + 10.0
            while wire is None:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, "no video frame received")
                msg = video.receive()
                if msg.get("bytes") is not None:
                    wire = msg["bytes"]
                # Ignore extra text frames (there are none today).

            header, payload = unpack_video_frame(wire)
            self.assertEqual(header.codec, 0)  # mjpeg_full
            self.assertTrue(has_flag(header.flags, FLAG_KEYFRAME))
            self.assertGreater(len(payload), 0)

    def test_keyframe_request_control(self) -> None:
        session_id = self._create_session()
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video"
        ) as video:
            self.assertEqual(video.receive_json()["type"], "ready")
            video.send_json({"type": "keyframe_request"})
            self._push_pcm(session_id, chunks=4)
            # First binary frame must be a keyframe (fresh client always
            # gets one; the request path is exercised by the sink tests).
            wire = None
            deadline = time.monotonic() + 10.0
            while wire is None:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0)
                msg = video.receive()
                if msg.get("bytes") is not None:
                    wire = msg["bytes"]
            header, _ = unpack_video_frame(wire)
            self.assertTrue(has_flag(header.flags, FLAG_KEYFRAME))

    def test_eof_on_session_close(self) -> None:
        session_id = self._create_session()
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video"
        ) as video:
            self.assertEqual(video.receive_json()["type"], "ready")
            resp = self.client.request(
                "DELETE", f"/v1/sessions/{session_id}"
            )
            self.assertEqual(resp.status_code, 200)
            # Closing the session stops the pipeline → sink.stop() → EOF
            # sentinel → server sends an EOF wire frame, then closes.
            saw_eof = False
            deadline = time.monotonic() + 10.0
            while not saw_eof:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, "no EOF received")
                msg = video.receive()
                if msg.get("bytes") is not None:
                    header, _ = unpack_video_frame(msg["bytes"])
                    saw_eof = has_flag(header.flags, FLAG_EOF)

    def test_no_sink_returns_4404(self) -> None:
        """Capture-mode pipeline (no publisher) → video WS must 4404."""
        _configure_capture_mode()
        session_id = self._create_session()
        with self.assertRaises(WebSocketDisconnect) as cm:
            with self.client.websocket_connect(
                f"/v1/sessions/{session_id}/video"
            ):
                pass
        self.assertEqual(cm.exception.code, 4404)

    def test_unknown_session_returns_4404(self) -> None:
        with self.assertRaises(WebSocketDisconnect) as cm:
            with self.client.websocket_connect(
                "/v1/sessions/nope/video"
            ):
                pass
        self.assertEqual(cm.exception.code, 4404)

    def test_missing_session_returns_404_rest(self) -> None:
        resp = self.client.get("/v1/sessions/nope/stats")
        self.assertEqual(resp.status_code, 404)


class VideoWsAuthTests(unittest.TestCase):
    """API-key protected video WS handshake."""

    def setUp(self) -> None:
        _configure_ws_transport_mode()
        state.settings.api_key = "secret-key"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        state.settings.api_key = ""
        state.pipeline = None

    def test_video_ws_requires_api_key(self) -> None:
        session_id = self._create_session()
        with self.assertRaises(WebSocketDisconnect) as cm:
            with self.client.websocket_connect(
                f"/v1/sessions/{session_id}/video"
            ):
                pass
        self.assertEqual(cm.exception.code, 4401)

    def test_video_ws_accepts_query_key(self) -> None:
        session_id = self._create_session()
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video?api_key=secret-key"
        ) as video:
            self.assertEqual(video.receive_json()["type"], "ready")

    def test_video_ws_accepts_header_key(self) -> None:
        session_id = self._create_session()
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video",
            headers={"X-API-Key": "secret-key"},
        ) as video:
            self.assertEqual(video.receive_json()["type"], "ready")

    def _create_session(self) -> str:
        resp = self.client.post(
            "/v1/sessions",
            json={"avatar_id": "yongen"},
            headers={"X-API-Key": "secret-key"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["session_id"]


class VideoWsInterruptTests(unittest.TestCase):
    """M3: interrupt / reconnect / multi-session end-to-end over the
    self-developed transport (all CPU, TestClient)."""

    def setUp(self) -> None:
        _configure_ws_transport_mode()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        state.pipeline = None

    def _create_session(self) -> str:
        resp = self.client.post("/v1/sessions", json={"avatar_id": "yongen"})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["session_id"]

    def _recv_frames(
        self, video, n_max: int, per_msg_timeout: float = 5.0
    ) -> list:
        """Collect up to n_max binary wire frames.

        ``receive()`` on the TestClient blocks forever, so each message is
        awaited through a helper thread with a timeout: a stream that goes
        idle before EOF (server paces with 0.5 s timeouts) simply stops the
        drain instead of hanging the suite on slow runners.
        """
        frames = []
        for _ in range(n_max):
            msg = self._receive_with_timeout(video, per_msg_timeout)
            if msg is None:
                break  # idle stream, no EOF yet — stop draining
            if msg.get("bytes") is None:
                continue
            header, _ = unpack_video_frame(msg["bytes"])
            if has_flag(header.flags, FLAG_EOF):
                break
            frames.append(header)
        return frames

    @staticmethod
    def _receive_with_timeout(video, timeout: float):
        import queue
        import threading

        q: queue.Queue = queue.Queue()

        def _worker() -> None:
            try:
                q.put(video.receive())
            except Exception as exc:  # surfaced in the caller thread
                q.put(exc)

        threading.Thread(target=_worker, daemon=True).start()
        try:
            result = q.get(timeout=timeout)
        except queue.Empty:
            return None
        if isinstance(result, Exception):
            raise result
        return result

    def test_interrupt_advances_epoch_with_boundary_keyframe(self) -> None:
        session_id = self._create_session()
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video"
        ) as video:
            self.assertEqual(video.receive_json()["type"], "ready")
            with self.client.websocket_connect(
                f"/v1/sessions/{session_id}/audio"
            ) as audio:
                audio.send_json({"type": "epoch", "epoch": 1})
                for _ in range(3):
                    audio.send_bytes(_PCM_CHUNK)
                # Drain epoch-1 frames.
                frames1 = self._recv_frames(video, 12)
                self.assertTrue(frames1)
                self.assertTrue(all(h.epoch == 1 for h in frames1))
                self.assertTrue(
                    has_flag(frames1[0].flags, FLAG_EPOCH_BOUNDARY)
                )

                # Interrupt: cancel + new epoch, then more audio.
                audio.send_json({"type": "cancel", "epoch": 2})
                audio.send_json({"type": "epoch", "epoch": 2})
                for _ in range(3):
                    audio.send_bytes(_PCM_CHUNK)
                frames2 = self._recv_frames(video, 12)
            self.assertTrue(frames2)
            # No stale frames: every post-interrupt frame is epoch 2, and
            # the first one is an epoch_boundary keyframe (client flushes).
            self.assertTrue(all(h.epoch == 2 for h in frames2))
            self.assertTrue(has_flag(frames2[0].flags, FLAG_EPOCH_BOUNDARY))
            self.assertTrue(has_flag(frames2[0].flags, FLAG_KEYFRAME))
            # Session stats: sink saw both epochs.
            stats = self.client.get(f"/v1/sessions/{session_id}/stats").json()
            self.assertGreaterEqual(stats["publisher"]["current_epoch"], 2)

    def test_reconnect_receives_fresh_keyframe(self) -> None:
        session_id = self._create_session()
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video"
        ) as first:
            self.assertEqual(first.receive_json()["type"], "ready")
            with self.client.websocket_connect(
                f"/v1/sessions/{session_id}/audio"
            ) as audio:
                for _ in range(2):
                    audio.send_bytes(_PCM_CHUNK)
            self._recv_frames(first, 8)
        # Client left; connect again — first frame must be a keyframe.
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video"
        ) as second:
            self.assertEqual(second.receive_json()["type"], "ready")
            with self.client.websocket_connect(
                f"/v1/sessions/{session_id}/audio"
            ) as audio:
                audio.send_json({"type": "epoch", "epoch": 1})
                audio.send_bytes(_PCM_CHUNK)
            headers = self._recv_frames(second, 4)
            self.assertTrue(headers)
            self.assertTrue(has_flag(headers[0].flags, FLAG_KEYFRAME))

    def test_three_sessions_isolated_streams(self) -> None:
        sids = [self._create_session() for _ in range(3)]
        videos = [
            self.client.websocket_connect(f"/v1/sessions/{sid}/video")
            for sid in sids
        ]
        try:
            for video in videos:
                ctx = video.__enter__()
                self.assertEqual(ctx.receive_json()["type"], "ready")
            # Push PCM to each session in turn; each video WS must see its
            # own sink's seq series starting from 0 (no cross-streaming).
            for sid in sids:
                with self.client.websocket_connect(
                    f"/v1/sessions/{sid}/audio"
                ) as audio:
                    for _ in range(2):
                        audio.send_bytes(_PCM_CHUNK)
            for sid, video in zip(sids, videos, strict=False):
                headers = self._recv_frames(video, 8)
                self.assertTrue(headers, f"no frames for {sid}")
                seqs = [h.seq for h in headers]
                self.assertEqual(seqs[0], 0, f"{sid} received foreign frame")
                self.assertEqual(
                    seqs, list(range(len(seqs))), f"{sid} seq not contiguous"
                )
                stats = self.client.get(f"/v1/sessions/{sid}/stats").json()
                self.assertGreater(stats["publisher"]["frames_published"], 0)
        finally:
            for video in videos:
                video.__exit__(None, None, None)


class VideoWsRegionCodecTests(unittest.TestCase):
    """M4: LIVEAVATAR_CODEC=region装配 —— ready 报 region_delta，帧为 patch。"""

    def setUp(self) -> None:
        import os
        import tempfile

        from liveavatar.region_codec import RegionSpec, write_region_json

        self._tmp = tempfile.TemporaryDirectory()
        state.settings = state.settings.__class__(
            livekit_url="", livekit_api_key="", livekit_api_secret=""
        )
        state.settings.codec = "region"
        state.pool_config = AvatarPoolConfig(avatar_data_root=self._tmp.name)
        # Fake avatar dir with a region.json (fake worker emits 4x4 frames).
        avatar_dir = os.path.join(self._tmp.name, "yongen")
        os.makedirs(avatar_dir, exist_ok=True)
        write_region_json(os.path.join(avatar_dir, "region.json"), RegionSpec(0, 0, 2, 2))
        from liveavatar.publish import _service_publisher_factory

        state.pipeline = AvatarPipeline(
            state.pool_config,
            pool=_FakeServicePool(),
            publisher_factory=_service_publisher_factory,
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        state.settings.codec = "mjpeg"
        state.pipeline = None
        self._tmp.cleanup()

    def test_ready_reports_region_delta(self) -> None:
        resp = self.client.post("/v1/sessions", json={"avatar_id": "yongen"})
        self.assertEqual(resp.status_code, 200, resp.text)
        session_id = resp.json()["session_id"]
        with self.client.websocket_connect(
            f"/v1/sessions/{session_id}/video"
        ) as video:
            ready = video.receive_json()
            self.assertEqual(ready["codec"], "region_delta")
            with self.client.websocket_connect(
                f"/v1/sessions/{session_id}/audio"
            ) as audio:
                audio.send_bytes(_PCM_CHUNK)
            msg = None
            for _ in range(64):  # bounded: never hang CI on a lost frame
                msg = video.receive()
                if msg.get("bytes") is not None:
                    break
            self.assertIsNotNone(msg and msg.get("bytes"))
            header, payload = unpack_video_frame(msg["bytes"])
            self.assertEqual(header.codec, 1)  # region_delta
            patches = unpack_region_payload(payload)
            self.assertTrue(patches)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

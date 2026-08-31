"""M-C: three-way concurrency correctness (fake worker, CPU only).

Three sessions run **simultaneously** (video + audio WS all open at the
same time, PCM interleaved) over the self-developed WS transport and a
fake pool — no GPU, no real models. Complements
``test_video_ws.test_three_sessions_isolated_streams`` (turn-by-turn):

1. Concurrent streaming: each session's video WS sees its own sink's
   strictly-increasing seq starting at 0, and only its own epoch.
2. Barge-in isolation: cancelling the epoch in one session never changes
   the epoch of frames flowing to the other two.
3. Close isolation: DELETEing one session EOFs its stream while the other
   two keep producing frames.

Note (CI lesson learned): TestClient ``receive()`` blocks forever, so every
receive is wrapped in a helper thread with a timeout — an idle stream stops
the drain instead of hanging the suite.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

from fastapi.testclient import TestClient

from liveavatar.publish import app, state
from liveavatar.video_protocol import (
    FLAG_EOF,
    has_flag,
    unpack_video_frame,
)
from tests.test_video_ws import _PCM_CHUNK, _configure_ws_transport_mode

_RECV_TIMEOUT = 5.0


def _receive_with_timeout(video, timeout: float = _RECV_TIMEOUT):
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


def _recv_frames(video, n_max: int) -> list:
    """Collect binary wire-frame headers until the stream goes idle/EOF."""
    frames = []
    for _ in range(n_max):
        msg = _receive_with_timeout(video)
        if msg is None:
            break
        if msg.get("bytes") is None:
            continue
        header, _ = unpack_video_frame(msg["bytes"])
        if has_flag(header.flags, FLAG_EOF):
            break
        frames.append(header)
    return frames


class _ThreeWayBase(unittest.TestCase):
    def setUp(self) -> None:
        _configure_ws_transport_mode()
        self.client = TestClient(app)
        self.sids = [self._create_session(i) for i in range(3)]

    def tearDown(self) -> None:
        state.pipeline = None

    def _create_session(self, index: int) -> str:
        resp = self.client.post(
            "/v1/sessions", json={"avatar_id": "yongen"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["session_id"]

    def _open_video(self, sid: str):
        ws = self.client.websocket_connect(f"/v1/sessions/{sid}/video")
        ctx = ws.__enter__()
        self.assertEqual(ctx.receive_json()["type"], "ready")
        return ws

    def _open_audio(self, sid: str):
        ws = self.client.websocket_connect(f"/v1/sessions/{sid}/audio")
        ws.__enter__()
        return ws


class TestConcurrentStreams(_ThreeWayBase):
    """Three fully concurrent sessions: interleaved pushes, isolated seq."""

    def test_interleaved_pushes_keep_streams_isolated(self) -> None:
        videos = [self._open_video(sid) for sid in self.sids]
        audios = [self._open_audio(sid) for sid in self.sids]
        # One distinct epoch per session.
        epochs = [10, 20, 30]
        for audio, epoch in zip(audios, epochs, strict=True):
            audio.send_json({"type": "epoch", "epoch": epoch})
        # Interleave pushes round-robin so frames are produced concurrently.
        for round_ in range(3):
            for audio in audios:
                audio.send_bytes(_PCM_CHUNK)
            del round_
        try:
            for sid, video, epoch in zip(
                self.sids, videos, epochs, strict=True
            ):
                headers = _recv_frames(video, 12)
                self.assertTrue(headers, f"no frames for {sid}")
                seqs = [h.seq for h in headers]
                # Own sink only: starts at 0, strictly increasing (drops are
                # legal on a slow client; foreign/duplicate seq is not).
                self.assertEqual(
                    seqs,
                    sorted(set(seqs)),
                    f"{sid} seq not strictly increasing: {seqs}",
                )
                self.assertEqual(seqs[0], 0, f"{sid} got a foreign frame")
                # Epoch isolation: only this session's epoch ever appears.
                self.assertEqual(
                    {h.epoch for h in headers},
                    {epoch},
                    f"{sid} saw foreign epoch: {sorted({h.epoch for h in headers})}",
                )
        finally:
            for audio in audios:
                audio.__exit__(None, None, None)
            for video in videos:
                video.__exit__(None, None, None)


class TestBargeinIsolation(_ThreeWayBase):
    """Cancelling one session must not touch the other two."""

    def test_cancel_in_one_session_others_unaffected(self) -> None:
        videos = [self._open_video(sid) for sid in self.sids]
        audios = [self._open_audio(sid) for sid in self.sids]
        for audio in audios:
            audio.send_json({"type": "epoch", "epoch": 1})
        for audio in audios:
            audio.send_bytes(_PCM_CHUNK)

        # Barge-in session 1 only: epoch 1 → 2.
        audios[1].send_json({"type": "cancel", "epoch": 2})
        audios[1].send_json({"type": "epoch", "epoch": 2})
        # All three keep producing after the interrupt.
        for audio in audios:
            audio.send_bytes(_PCM_CHUNK)
        try:
            for sid, video, idx in zip(
                self.sids, videos, range(3), strict=True
            ):
                headers = _recv_frames(video, 12)
                self.assertTrue(headers, f"no frames for {sid}")
                epochs = [h.epoch for h in headers]
                if idx == 1:
                    # Barge-in target: epoch-1 frames published *before* the
                    # interrupt are legitimately delivered; once the first
                    # epoch-2 frame arrives no stale epoch-1 frame may follow.
                    self.assertIn(2, epochs, f"{sid} never reached epoch 2")
                    first2 = epochs.index(2)
                    self.assertEqual(
                        set(epochs[first2:]),
                        {2},
                        f"{sid} stale epoch-1 frames after barge-in",
                    )
                else:
                    # Bystanders: the sibling's cancel must not change their
                    # epoch at all.
                    self.assertEqual(
                        set(epochs),
                        {1},
                        f"{sid} epochs {set(epochs)} != {{1}} "
                        "(barge-in leaked across sessions)",
                    )
        finally:
            for audio in audios:
                audio.__exit__(None, None, None)
            for video in videos:
                video.__exit__(None, None, None)


class TestCloseIsolation(_ThreeWayBase):
    """Deleting one session must not disturb the survivors."""

    def test_close_middle_session_others_keep_streaming(self) -> None:
        videos = [self._open_video(sid) for sid in self.sids]
        audios = [self._open_audio(sid) for sid in self.sids]
        for audio in audios:
            audio.send_json({"type": "epoch", "epoch": 1})
        for audio in audios:
            audio.send_bytes(_PCM_CHUNK)
        # Warm all streams so survivors have an active sink state.
        for video in videos:
            self.assertTrue(_recv_frames(video, 4))

        # Close session 1.
        resp = self.client.request("DELETE", f"/v1/sessions/{self.sids[1]}")
        self.assertEqual(resp.status_code, 200)
        # Its video WS must reach EOF.
        saw_eof = False
        deadline = time.monotonic() + _RECV_TIMEOUT
        while time.monotonic() < deadline and not saw_eof:
            msg = _receive_with_timeout(videos[1], 1.0)
            if msg is None:
                break
            if msg.get("bytes") is not None:
                header, _ = unpack_video_frame(msg["bytes"])
                saw_eof = has_flag(header.flags, FLAG_EOF)
        self.assertTrue(saw_eof, "closed session's video WS never EOFed")

        # Survivors keep producing frames after the close.
        for audio in (audios[0], audios[2]):
            audio.send_bytes(_PCM_CHUNK)
        try:
            for idx in (0, 2):
                headers = _recv_frames(videos[idx], 8)
                self.assertTrue(
                    headers,
                    f"survivor {self.sids[idx]} stopped streaming after "
                    "a sibling session was closed",
                )
        finally:
            for idx, audio in enumerate(audios):
                if idx != 1:
                    audio.__exit__(None, None, None)
            for idx, video in enumerate(videos):
                if idx != 1:
                    video.__exit__(None, None, None)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

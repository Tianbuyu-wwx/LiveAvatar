"""Tests for RealtimeAsrClient connect/read-loop/reconnect lifecycle.

Uses a fake WebSocket object (async-iterable, records sends) and patches
``websockets.connect`` to a scripted sequence of results, so reconnect
semantics are deterministic without real sockets.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from tests.conftest import wait_until as _wait_until

from liveavatar.audio_in.adapters import realtime_asr_client as rac
from liveavatar.audio_in.adapters.realtime_asr_client import RealtimeAsrClient

skip_no_ws = unittest.skipUnless(rac._HAS_WS, "websockets not installed")


class _Dropped(Exception):
    """Simulates the server closing the connection mid-stream."""


class _FakeWs:
    """Minimal async-iterable WebSocket double."""

    def __init__(self, incoming: list | None = None):
        self.sent: list = []
        self.incoming = list(incoming or [])
        self._closed = asyncio.Event()

    async def send(self, data) -> None:
        self.sent.append(data)

    def __aiter__(self) -> _FakeWs:
        return self

    async def __anext__(self) -> str:
        if self.incoming:
            item = self.incoming.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        # Stay open until close() is called, then end iteration (drop).
        await self._closed.wait()
        raise StopAsyncIteration

    async def close(self) -> None:
        self._closed.set()


def _patch_connect(results: list):
    """Patch websockets.connect to pop from ``results`` (Exception → raise)."""
    queue = list(results)

    async def fake_connect(url):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return mock.patch.object(rac.websockets, "connect", fake_connect)


@skip_no_ws
class TestConnect(unittest.IsolatedAsyncioTestCase):
    async def test_connect_sends_start_and_buffers_events(self):
        ws = _FakeWs(
            incoming=[
                json.dumps({"type": "vad", "kind": "speech_start", "ts_us": 1}),
                "not-json",  # malformed → skipped by the read loop
                json.dumps({"type": "eou", "confidence": 0.9, "silence_us": 5}),
                json.dumps({"type": "ack", "seq": 3}),
            ]
        )
        with _patch_connect([ws]):
            client = RealtimeAsrClient("ws://x/asr", "s1")
            await client.connect()

        self.assertTrue(client.is_connected)
        # Start message carries the session id.
        start = json.loads(ws.sent[0])
        self.assertEqual(start["action"], "start")
        self.assertEqual(start["session_id"], "s1")
        # Events buffered by type; malformed line skipped; envelope stripped.
        await _wait_until(lambda: len(client._eou_buf) == 1)
        self.assertEqual(client._vad_buf[0], {"kind": "speech_start"})
        self.assertEqual(client._eou_buf[0], {"confidence": 0.9, "silence_us": 5})
        self.assertEqual(client._ack_buf[0], {"type": "ack", "seq": 3})  # ack keeps envelope
        await client.close()
        self.assertFalse(client.is_connected)

    async def test_connect_failure_degrades_silently(self):
        with _patch_connect([ConnectionRefusedError("no server")]):
            client = RealtimeAsrClient("ws://x/asr", "s1")
            await client.connect()
        self.assertFalse(client.is_connected)
        self.assertIsNone(client._reader_task)


@skip_no_ws
class TestReconnect(unittest.IsolatedAsyncioTestCase):
    async def test_retries_until_first_success(self):
        with _patch_connect(
            [ConnectionRefusedError(), ConnectionRefusedError(), _FakeWs()]
        ):
            client = RealtimeAsrClient("ws://x/asr", "s1")
            await client.connect_with_reconnect(base_delay=0.01)
        self.assertTrue(client.is_connected)
        await client.close()

    async def test_reconnects_after_connection_drop(self):
        ws_a = _FakeWs(incoming=[json.dumps({"type": "asr", "text": "stale"})])
        ws_b = _FakeWs()
        with _patch_connect([ws_a, ws_b]):
            client = RealtimeAsrClient("ws://x/asr", "s1")
            await client.connect_with_reconnect(base_delay=0.01)
            self.assertIs(client._ws, ws_a)
            # Drop the connection: reader loop ends → supervisor reconnects.
            await ws_a.close()
            await _wait_until(lambda: client._ws is ws_b)
            self.assertTrue(client.is_connected)
            # Stale events from the dropped connection are discarded.
            self.assertEqual(client.drain_asr(), [])
            await client.close()

    async def test_close_stops_supervisor(self):
        with _patch_connect([_FakeWs()]):
            client = RealtimeAsrClient("ws://x/asr", "s1")
            await client.connect_with_reconnect(base_delay=0.01)
            supervisor = client._reconnect_task
            self.assertIsNotNone(supervisor)
            await client.close()
            self.assertTrue(supervisor.done())
            self.assertFalse(client._reconnect)

    async def test_close_during_retry_loop(self):
        """close() while still retrying the initial connect must not hang."""
        with _patch_connect([ConnectionRefusedError() for _ in range(10)]):
            client = RealtimeAsrClient("ws://x/asr", "s1")
            task = asyncio.create_task(
                client.connect_with_reconnect(base_delay=0.01)
            )
            await asyncio.sleep(0.03)  # let it fail a couple of times
            await asyncio.wait_for(client.close(), timeout=2.0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self.assertFalse(client.is_connected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

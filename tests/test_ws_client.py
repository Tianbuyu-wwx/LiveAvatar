# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""A4 (C6): self-written RFC 6455 WebSocket client tests.

Frame-encoding unit tests are dependency-free. Interop tests run the
client against a real ``websockets`` server (dev-only dependency used as
the test peer — the runtime code path has zero third-party deps).
"""

from __future__ import annotations

import asyncio
import unittest

try:
    import websockets.asyncio.server as _ws_server

    HAS_WEBSOCKETS = True
except ImportError:  # pragma: no cover — CI light env without dev extras
    _ws_server = None  # type: ignore[assignment]
    HAS_WEBSOCKETS = False

from liveavatar.ws_client import (
    OP_BINARY,
    OP_TEXT,
    WebSocketClient,
    WebSocketError,
    _encode_client_frame,
)

requires_ws = unittest.skipUnless(HAS_WEBSOCKETS, "websockets not installed")


# ------------------------------------------------------- frame unit tests


def test_frame_masked_and_short_length() -> None:
    frame = _encode_client_frame(OP_TEXT, b"hi", mask_key=b"\x01\x02\x03\x04")
    assert frame[0] == 0x81  # FIN + text
    assert frame[1] == 0x80 | 2  # MASK + len 2
    assert frame[2:6] == b"\x01\x02\x03\x04"
    payload = bytes(b ^ frame[2 + (i % 4)] for i, b in enumerate(frame[6:]))
    assert payload == b"hi"


def test_frame_length_boundaries() -> None:
    # 126 → 16-bit length header
    frame = _encode_client_frame(OP_BINARY, b"x" * 125, mask_key=b"\0\0\0\0")
    assert frame[1] == 0x80 | 125
    frame = _encode_client_frame(OP_BINARY, b"x" * 126, mask_key=b"\0\0\0\0")
    assert frame[1] == 0x80 | 126
    # 65536 → 64-bit length header
    frame = _encode_client_frame(OP_BINARY, b"x" * 65536, mask_key=b"\0\0\0\0")
    assert frame[1] == 0x80 | 127


def test_zero_mask_keeps_payload() -> None:
    frame = _encode_client_frame(OP_TEXT, b"plain", mask_key=b"\0\0\0\0")
    assert frame[6:] == b"plain"


# ------------------------------------------------------------- interop tests


class _EchoServer:
    """websockets echo server; records received messages and exposes pings."""

    def __init__(self) -> None:
        self.received: list = []
        self.server = None
        self.port = 0

    async def _handler(self, ws) -> None:
        async for message in ws:
            self.received.append(message)
            await ws.send(message)

    async def start(self) -> None:
        self.server = await _ws_server.serve(self._handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        self.server.close()
        await self.server.wait_closed()

    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/echo"


@requires_ws
class TestInterop(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = _EchoServer()
        await self.server.start()

    async def asyncTearDown(self) -> None:
        await self.server.stop()

    async def test_handshake_text_and_binary_roundtrip(self) -> None:
        async with await WebSocketClient.connect(self.server.url()) as client:
            await client.send("hello")
            assert await client.receive() == "hello"
            await client.send(b"\x00\x01\x02")
            assert await client.receive() == b"\x00\x01\x02"
        assert self.server.received == ["hello", b"\x00\x01\x02"]

    async def test_large_frame_over_16bit_boundary(self) -> None:
        payload = bytes(range(256)) * 300  # 76800 bytes > 65535
        async with await WebSocketClient.connect(self.server.url()) as client:
            await client.send(payload)
            echoed = await client.receive()
        assert echoed == payload

    async def test_async_iteration(self) -> None:
        client = await WebSocketClient.connect(self.server.url())
        await client.send("a")
        await client.send("b")
        messages = []
        async for message in client:
            messages.append(message)
            if len(messages) == 2:
                await client.close()
        assert messages == ["a", "b"]

    async def test_close_handshake_ends_iteration(self) -> None:
        client = await WebSocketClient.connect(self.server.url())
        await client.send("x")
        assert await client.receive() == "x"
        await client.close()
        assert client.is_closed
        assert await client.receive() is None

    async def test_answers_server_ping(self) -> None:
        """Server ping must be answered automatically or pong wait times out."""

        async def ping_handler(ws) -> None:
            await ws.send("before")
            pong = await ws.ping()  # resolves when the pong arrives
            await asyncio.wait_for(pong, timeout=5.0)
            await ws.send("after")

        server = await _ws_server.serve(ping_handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            client = await WebSocketClient.connect(f"ws://127.0.0.1:{port}/p")
            assert await client.receive() == "before"
            # server ping/pong exchange happens while we are waiting again
            assert await client.receive() == "after"
            await client.close()
        finally:
            server.close()
            await server.wait_closed()

    async def test_rejects_non_ws_url(self) -> None:
        try:
            await WebSocketClient.connect("http://127.0.0.1:1/x")
        except ValueError as exc:
            assert "ws://" in str(exc)
        else:
            raise AssertionError("expected ValueError for non-ws URL")

    async def test_connection_refused_degrades(self) -> None:
        # port 1 on localhost is closed; connect must raise, not hang
        try:
            await asyncio.wait_for(
                WebSocketClient.connect("ws://127.0.0.1:1/x"), timeout=5.0
            )
        except OSError:
            pass  # ConnectionRefusedError is the expected outcome

    async def test_malformed_handshake_rejected(self) -> None:
        """A plain HTTP endpoint (no 101) must raise WebSocketError."""

        async def plain_http(reader, writer) -> None:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(plain_http, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            try:
                await asyncio.wait_for(
                    WebSocketClient.connect(f"ws://127.0.0.1:{port}/x"), timeout=5.0
                )
            except WebSocketError as exc:
                assert "404" in str(exc)
            else:
                raise AssertionError("expected WebSocketError for non-101 response")
        finally:
            server.close()
            await server.wait_closed()

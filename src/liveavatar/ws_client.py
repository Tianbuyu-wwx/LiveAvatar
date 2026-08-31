"""Self-written RFC 6455 WebSocket client — A4 (C6).

A minimal, dependency-free client that replaces the ``websockets`` library
for outbound connections (RealtimeAsr microservice client). Implements the
subset of RFC 6455 the project actually uses:

* opening handshake (``ws://`` only — TLS termination happens at the
  reverse proxy, and the server never dials ``wss://``);
* masked client frames with 7/16/64-bit length encodings;
* text / binary messages, including fragmented (continuation) messages;
* automatic ``pong`` replies to ``ping`` and a cooperative close handshake;
* a ``websockets``-compatible call surface: ``connect(url)``, ``send``,
  ``receive()`` / async iteration and ``close()`` — so existing fakes and
  call sites keep working unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import struct
from urllib.parse import urlsplit

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


class WebSocketError(ConnectionError):
    """Raised when the server handshake violates RFC 6455."""


def _encode_client_frame(opcode: int, payload: bytes, mask_key: bytes | None = None) -> bytes:
    """Build one masked client→server frame (client frames MUST be masked)."""
    if mask_key is None:
        mask_key = os.urandom(4)
    n = len(payload)
    if n < 126:
        header = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
    elif n < 65536:
        header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
    else:
        header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
    masked = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
    return header + mask_key + masked


def _xor_mask(payload: bytes, mask_key: bytes) -> bytes:
    return bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))


class WebSocketClient:
    """A connected client-side WebSocket over asyncio streams."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer
        self._send_lock = asyncio.Lock()
        self._closed = False
        # fragmentation reassembly state
        self._frag_op: int | None = None
        self._frag_buf = bytearray()

    # ------------------------------------------------------------ handshake

    @classmethod
    async def connect(cls, url: str, *, open_timeout: float = 10.0) -> WebSocketClient:
        """Open a ``ws://`` connection and complete the RFC 6455 handshake."""
        parts = urlsplit(url)
        if parts.scheme != "ws":
            raise ValueError(f"only ws:// URLs are supported, got: {url}")
        if parts.hostname is None:
            raise ValueError(f"invalid WebSocket URL: {url}")
        host, port = parts.hostname, parts.port or 80
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=open_timeout
        )
        try:
            key = base64.b64encode(os.urandom(16)).decode()
            request = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            )
            writer.write(request.encode("ascii"))
            await asyncio.wait_for(writer.drain(), timeout=open_timeout)
            response = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=open_timeout)
        except BaseException:
            writer.close()
            raise

        headers: dict[str, str] = {}
        lines = response.decode("latin-1").split("\r\n")
        status = lines[0]
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        if len(status.split()) < 2 or status.split()[1] != "101":
            raise WebSocketError(f"handshake rejected: {status}")
        expected = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
        if headers.get("sec-websocket-accept") != expected:
            raise WebSocketError("handshake Sec-WebSocket-Accept mismatch")
        return cls(reader, writer)

    # ----------------------------------------------------------------- send

    async def send(self, data: bytes | str) -> None:
        """Send one unfragmented binary or text message."""
        if isinstance(data, str):
            opcode, payload = OP_TEXT, data.encode("utf-8")
        else:
            opcode, payload = OP_BINARY, bytes(data)
        frame = _encode_client_frame(opcode, payload)
        async with self._send_lock:
            self._writer.write(frame)
            await self._writer.drain()

    async def _send_pong(self, payload: bytes) -> None:
        frame = _encode_client_frame(OP_PONG, payload)
        async with self._send_lock:
            self._writer.write(frame)
            await self._writer.drain()

    async def _send_close_reply(self) -> None:
        frame = _encode_client_frame(OP_CLOSE, b"")
        try:
            async with self._send_lock:
                self._writer.write(frame)
                await self._writer.drain()
        except OSError:
            pass

    # -------------------------------------------------------------- receive

    async def _read_exact(self, n: int) -> bytes:
        return await self._reader.readexactly(n)

    async def receive(self) -> str | bytes | None:
        """Read the next message.

        Returns ``None`` once the connection is closed (either side). Ping
        frames are answered with pongs transparently.
        """
        while not self._closed:
            try:
                b1, b2 = await self._read_exact(2)
                opcode, fin = b1 & 0x0F, bool(b1 & 0x80)
                length = b2 & 0x7F
                if length == 126:
                    (length,) = struct.unpack("!H", await self._read_exact(2))
                elif length == 127:
                    (length,) = struct.unpack("!Q", await self._read_exact(8))
                mask = await self._read_exact(4) if b2 & 0x80 else None
                payload = await self._read_exact(length) if length else b""
            except (asyncio.IncompleteReadError, ConnectionError, OSError):
                self._closed = True
                return None
            if mask is not None:
                payload = _xor_mask(payload, mask)

            if opcode == OP_PING:
                await self._send_pong(payload)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                await self._send_close_reply()
                self._closed = True
                return None
            if opcode in (OP_TEXT, OP_BINARY):
                if fin:
                    if self._frag_op is not None:
                        raise WebSocketError("unfragmented message inside fragments")
                    return payload.decode("utf-8") if opcode == OP_TEXT else payload
                self._frag_op, self._frag_buf = opcode, bytearray(payload)
                continue
            if opcode == OP_CONT:
                if self._frag_op is None:
                    raise WebSocketError("continuation frame without a started message")
                self._frag_buf += payload
                if fin:
                    buf, op = bytes(self._frag_buf), self._frag_op
                    self._frag_op, self._frag_buf = None, bytearray()
                    return buf.decode("utf-8") if op == OP_TEXT else buf
                continue
            raise WebSocketError(f"unsupported opcode: {opcode:#x}")
        return None

    # ---------------------------------------------------------------- close

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        """Best-effort cooperative close (close frame + transport shutdown)."""
        if not self._closed:
            self._closed = True
            await self._send_close_reply()
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except OSError:
            pass

    # ------------------------------------------------------- async iteration

    async def __aenter__(self) -> WebSocketClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def __aiter__(self) -> WebSocketClient:
        return self

    async def __anext__(self) -> str | bytes:
        message = await self.receive()
        if message is None:
            raise StopAsyncIteration
        return message


# Module-level convenience used by call sites (and by tests that patch it).
connect = WebSocketClient.connect

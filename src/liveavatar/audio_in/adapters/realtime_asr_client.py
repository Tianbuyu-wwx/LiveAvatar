"""Shared WebSocket client for the RealtimeAsr microservice.

A single :class:`RealtimeAsrClient` manages one WebSocket connection per
session. It sends PCM frames as binary messages and buffers the JSON events
(``vad`` / ``asr`` / ``eou`` / ``ack``) returned by the service. The three
remote adapters (:class:`RemoteVadAdapter`, :class:`RemoteEouAdapter`,
:class:`RemoteAsrAdapter`) share this client and each drain only the events
of their own type.

Frame deduplication
-------------------
The worker calls ``vad.push_frame(frame)``, ``eou.push_frame(frame, ...)`` and
``asr.push_frame(frame)`` in sequence for the *same* frame. Each adapter
calls ``client.send_frame(frame)`` which deduplicates by ``frame.seq`` so
the PCM is sent to the service exactly once.

Event latency
-------------
Because the WebSocket is asynchronous, events from frame *N* typically arrive
while frame *N+1* is being processed. The adapters return whatever is
currently buffered, so events are delayed by ≈ one frame (20 ms). This is
acceptable for real-time audio and avoids blocking the worker's event loop.

Graceful degradation
--------------------
If the connection cannot be established, ``send_frame`` and ``drain_*``
become no-ops returning empty lists. The worker continues to run (without
VAD/EOU/ASR events) rather than crashing.

Transport
---------
Uses the self-written RFC 6455 client (:mod:`liveavatar.ws_client`) — the
``websockets`` library is not a runtime dependency.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
from typing import Any

from ... import ws_client
from ..frame import PCMFrame

logger = logging.getLogger("realtime_audio.adapters.client")


class RealtimeAsrClient:
    """Shared WebSocket client connecting to the RealtimeAsr service.

    Parameters
    ----------
    url : str
        WebSocket URL of the RealtimeAsr service (e.g.
        ``ws://127.0.0.1:8300/asr/stream``).
    session_id : str
        Session identifier forwarded to the service.
    """

    def __init__(self, url: str, session_id: str) -> None:
        self.url = url
        self.session_id = session_id
        self._ws: Any = None
        self._connected: bool = False
        self._reader_task: asyncio.Task | None = None
        self._last_sent_seq: int = -1
        self._vad_buf: collections.deque[dict[str, Any]] = collections.deque()
        self._asr_buf: collections.deque[dict[str, Any]] = collections.deque()
        self._eou_buf: collections.deque[dict[str, Any]] = collections.deque()
        self._ack_buf: collections.deque[dict[str, Any]] = collections.deque()
        self._send_lock = asyncio.Lock()
        # Reconnect state (only used by connect_with_reconnect).
        self._reconnect: bool = False
        self._closing: bool = False
        self._reconnect_task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def _connect_once(self) -> bool:
        """One connection attempt. True on success, False on failure."""
        try:
            self._ws = await ws_client.connect(self.url)
            start_msg = json.dumps(
                {"action": "start", "session_id": self.session_id, "sample_rate": 16000}
            )
            await self._ws.send(start_msg)
            self._connected = True
            self._reader_task = asyncio.create_task(self._read_loop())
            logger.info(
                "RealtimeAsrClient connected session_id=%s url=%s",
                self.session_id,
                self.url,
            )
            return True
        except Exception as e:
            logger.error("RealtimeAsrClient connect failed: %s", e)
            self._connected = False
            return False

    async def connect(self) -> None:
        """Open the WebSocket connection and start the reader task.

        If the connection fails, the client silently degrades —
        ``send_frame`` and ``drain_*`` become no-ops.
        """
        await self._connect_once()

    async def connect_with_reconnect(
        self,
        *,
        base_delay: float = 1.0,
        backoff: float = 2.0,
        max_delay: float = 30.0,
    ) -> None:
        """Connect with automatic reconnection after drops.

        Tries to connect with exponential backoff until the first success
        (returns once connected), then keeps a supervisor task running that
        re-establishes the connection whenever the read loop ends (drop).
        Stale events buffered from a dropped connection are discarded on
        reconnect so the worker never sees pre-drop ASR state. ``close()``
        stops the supervisor.
        """
        self._reconnect = True
        delay = base_delay
        while not self._closing:
            if await self._connect_once():
                self._reconnect_task = asyncio.create_task(
                    self._supervise(base_delay, backoff, max_delay)
                )
                return
            await asyncio.sleep(delay)
            delay = min(delay * backoff, max_delay)

    async def _supervise(self, base_delay: float, backoff: float, max_delay: float) -> None:
        """Await the reader task and reconnect with backoff when it ends."""
        delay = base_delay
        while self._reconnect and not self._closing:
            reader = self._reader_task
            if reader is not None:
                try:
                    await reader
                except asyncio.CancelledError:
                    pass
            if not self._reconnect or self._closing:
                return
            logger.warning(
                "RealtimeAsrClient connection lost session_id=%s; reconnecting in %.1fs",
                self.session_id,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * backoff, max_delay)
            # Discard events buffered by the dropped connection.
            self._vad_buf.clear()
            self._asr_buf.clear()
            self._eou_buf.clear()
            self._ack_buf.clear()
            while not self._closing and not await self._connect_once():
                await asyncio.sleep(delay)
                delay = min(delay * backoff, max_delay)

    async def _read_loop(self) -> None:
        """Background task: read JSON events and buffer by type."""
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                etype = event.get("type")
                if etype == "vad":
                    self._vad_buf.append(self._strip_event(event))
                elif etype == "asr":
                    self._asr_buf.append(self._strip_event(event))
                elif etype == "eou":
                    self._eou_buf.append(self._strip_event(event))
                elif etype == "ack":
                    self._ack_buf.append(event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("RealtimeAsrClient read loop ended: %s", e)
        finally:
            self._connected = False

    @staticmethod
    def _strip_event(event: dict[str, Any]) -> dict[str, Any]:
        """Remove transport-level fields (``type``, ``ts_us``) from a service event.

        The worker expects VAD events like ``{"kind": "speech_start", ...}``,
        ASR events like ``{"phase": "partial", "text": ..., ...}``, and EOU
        events like ``{"confidence": ..., "silence_us": ...}`` — without the
        ``type`` or ``ts_us`` envelope fields.
        """
        return {k: v for k, v in event.items() if k not in ("type", "ts_us")}

    # ------------------------------------------------------------------ send

    def send_frame(self, frame: PCMFrame) -> None:
        """Send a PCM frame as binary, deduplicating by ``frame.seq``.

        Non-blocking: schedules the async send via ``create_task``. If the
        connection is down, this is a no-op.
        """
        if not self._connected or self._ws is None:
            return
        if frame.seq == self._last_sent_seq:
            return  # already sent by another adapter sharing this client
        self._last_sent_seq = frame.seq
        asyncio.create_task(self._safe_send_bytes(frame.pcm_s16le))

    async def _safe_send_bytes(self, data: bytes) -> None:
        """Send binary data, logging errors without crashing."""
        try:
            async with self._send_lock:
                await self._ws.send(data)
        except Exception as e:
            logger.error("RealtimeAsrClient send failed: %s", e)
            self._connected = False

    def send_control(self, action: str, **kwargs: Any) -> None:
        """Send a JSON control message (``advance_epoch`` / ``flush`` / ``close``)."""
        if not self._connected or self._ws is None:
            return
        msg = json.dumps({"action": action, **kwargs})
        asyncio.create_task(self._safe_send_text(msg))

    async def _safe_send_text(self, data: str) -> None:
        try:
            async with self._send_lock:
                await self._ws.send(data)
        except Exception as e:
            logger.error("RealtimeAsrClient send failed: %s", e)
            self._connected = False

    # ----------------------------------------------------------------- drain

    def drain_vad(self) -> list[dict[str, Any]]:
        """Return and clear all buffered VAD events."""
        events = list(self._vad_buf)
        self._vad_buf.clear()
        return events

    def drain_asr(self) -> list[dict[str, Any]]:
        """Return and clear all buffered ASR events."""
        events = list(self._asr_buf)
        self._asr_buf.clear()
        return events

    def drain_eou(self) -> list[dict[str, Any]]:
        """Return and clear all buffered EOU events."""
        events = list(self._eou_buf)
        self._eou_buf.clear()
        return events

    def drain_acks(self) -> list[dict[str, Any]]:
        """Return and clear all buffered ACK events."""
        events = list(self._ack_buf)
        self._ack_buf.clear()
        return events

    # ---------------------------------------------------------------- close

    async def close(self) -> None:
        """Close the WebSocket connection and cancel the reader task."""
        self._closing = True
        self._reconnect = False
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None
        self._connected = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

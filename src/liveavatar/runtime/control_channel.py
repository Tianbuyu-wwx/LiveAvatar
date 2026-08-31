"""LiveKit control data-channel adapter.

Subscribes to ``data_received`` room events on the
``wisdomvii.realtime.control.v1`` topic (and topic-less messages for
backward compatibility with the current browser Thin Client, which does not
yet set a topic), parses JSON control envelopes, and dispatches them to the
RealtimeWorker's control queue.

Scope (Sprint 1, step 3): map Core intents (interrupt provisional/confirmed,
flush, close) and playback ACKs to worker actions. The browser currently
sends ``TransportEvent``-shaped messages (``{type, payload}``); the canonical
RealtimeContracts envelope (``{event_type, payload}``) is also accepted, so
the adapter is forward-compatible with the contract without a frontend
change.

Epoch authority stays in the worker: this adapter only forwards control
events via ``worker.push_control``. The worker's ``advance_epoch`` (triggered
by a confirmed interrupt) cancels the Tutor publisher through the
``on_epoch_advance`` callback, avoiding races between the data handler and
the worker loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("liveavatar.runtime.control_channel")

CONTROL_TOPIC = "wisdomvii.realtime.control.v1"


@dataclass
class ControlChannelStats:
    """Counters for the control data-channel adapter."""

    messages_received: int = 0
    topic_rejected: int = 0
    parse_errors: int = 0
    foreign_session: int = 0
    confirmed_interrupts: int = 0
    provisional_interrupts: int = 0
    flush_intents: int = 0
    close_intents: int = 0
    playback_acks: int = 0
    unknown_event_types: int = 0
    push_dropped: int = 0


class ControlChannelAdapter:
    """Bridge LiveKit data messages to the worker's control queue.

    Lifecycle::

        control = ControlChannelAdapter(worker, adapter.room, session_id)
        await control.start()      # register data_received handler
        ...
        await control.stop()       # clear references (room owns lifecycle)

    The adapter is passive: it only consumes data messages and forwards
    mapped control events to ``worker.push_control``.
    """

    def __init__(
        self,
        worker: Any,
        room: Any,
        session_id: str,
        *,
        control_topic: str = CONTROL_TOPIC,
    ) -> None:
        self.worker = worker
        self.room = room
        self.session_id = session_id
        self.control_topic = control_topic
        self.stats = ControlChannelStats()
        self._started = False

    # ------------------------------------------------------------------ start

    def start(self) -> None:
        """Register the ``data_received`` handler on the room."""
        if self._started or self.room is None:
            return
        self._started = True

        # Use the decorator form (consistent with LiveKitParticipantAdapter).
        @self.room.on("data_received")
        def _on_data(*args: Any, **kwargs: Any) -> None:
            self._on_data(*args, **kwargs)

    async def stop(self) -> None:
        """Clear references. The room's lifecycle is owned by the participant adapter."""
        self._started = False
        # LiveKit tears down handlers on room.disconnect(); explicit off() is
        # not consistently available across SDK versions, so we rely on that.

    # ------------------------------------------------------------ data entry

    def _on_data(self, *args: Any, **kwargs: Any) -> None:
        payload, topic = self._extract_payload_topic(args)
        if payload is None:
            return
        # Accept the control topic OR topic-less messages (current browser
        # publishes without a topic). Reject other named topics.
        if topic and topic != self.control_topic:
            self.stats.topic_rejected += 1
            return
        self.stats.messages_received += 1
        try:
            envelope = json.loads(payload.decode("utf-8"))
        except Exception:
            self.stats.parse_errors += 1
            return
        # Fire-and-forget the async handling; the data callback is sync.
        asyncio.create_task(self._handle_envelope(envelope))

    @staticmethod
    def _extract_payload_topic(args: tuple) -> tuple[bytes | None, str]:
        """Defensively extract (payload, topic) across SDK callback signatures.

        Supports the LiveKit ``DataPacket`` (single arg with ``.data`` /
        ``.topic``) and a positional ``(payload, participant, kind, topic)``
        shape for forward compatibility.
        """
        if not args:
            return None, ""
        first = args[0]
        # LiveKit DataPacket-style object (has .data, not .payload).
        if hasattr(first, "data") and not isinstance(first, (bytes, bytearray)):
            topic = getattr(first, "topic", "") or ""
            return first.data, topic
        # Positional (payload, participant, kind, topic).
        if isinstance(first, (bytes, bytearray)):
            topic = args[3] if len(args) >= 4 else ""
            return bytes(first), topic or ""
        return None, ""

    # ------------------------------------------------------------- dispatch

    async def _handle_envelope(self, envelope: dict[str, Any]) -> None:
        event = self._parse_and_map(envelope)
        if event is None:
            return
        pushed = await self.worker.push_control(event)
        if not pushed:
            self.stats.push_dropped += 1

    def _parse_and_map(self, envelope: dict[str, Any]) -> dict[str, Any] | None:
        """Pure mapping from a JSON envelope to a worker control event dict.

        Returns None when the message should be ignored (foreign session,
        unknown event type, unhandled intent). Dual-format: accepts both the
        canonical ``{event_type, payload}`` contract envelope and the
        browser's ``{type, payload}`` TransportEvent shape.
        """
        session_id = envelope.get("session_id")
        if session_id and session_id != self.session_id:
            self.stats.foreign_session += 1
            return None

        event_type = envelope.get("event_type") or envelope.get("type")
        payload = envelope.get("payload") or {}

        if event_type == "interrupt":
            # JSONL: payload.interrupt_event; TransportEvent: payload itself.
            interrupt_event = payload.get("interrupt_event", payload)
            kind = interrupt_event.get("kind")
            if kind == "confirmed":
                self.stats.confirmed_interrupts += 1
                return {"kind": "confirmed"}
            if kind == "provisional":
                self.stats.provisional_interrupts += 1
                return {"kind": "provisional"}
            self.stats.unknown_event_types += 1
            return None

        if event_type == "control":
            control_event = payload.get("control_event", payload)
            intent = control_event.get("intent")
            if intent == "flush":
                self.stats.flush_intents += 1
                return {"intent": "cancel_and_flush"}
            if intent == "close":
                self.stats.close_intents += 1
                return {"intent": "close"}
            self.stats.unknown_event_types += 1
            return None

        if event_type == "playback_ack":
            ack = payload.get("playback_ack", payload)
            self.stats.playback_acks += 1
            return {
                "intent": "playback_ack",
                "segment_seq": ack.get("segment_seq"),
                "consumed_pts_us": ack.get("consumed_pts_us"),
            }

        self.stats.unknown_event_types += 1
        return None

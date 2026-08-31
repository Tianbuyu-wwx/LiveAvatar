"""Wire envelope contracts for session worker events.

Adapted from WisdomVII ``RealtimeContracts`` — the event envelope spoken on
the worker output queue and the control channel. Every event carries
``epoch`` so consumers can drop stale-epoch messages after a barge-in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EVENT_TYPES = [
    "audio_frame",
    "asr",
    "vad",
    "eou",
    "interrupt",
    "control",
    "playback_ack",
    "residual_echo",
    "tts_audio",
    "avatar_frame",
    "task_event",
    "error",
]

EventType = Literal[
    "audio_frame",
    "asr",
    "vad",
    "eou",
    "interrupt",
    "control",
    "playback_ack",
    "residual_echo",
    "tts_audio",
    "avatar_frame",
    "task_event",
    "error",
]


@dataclass(slots=True)
class WordTiming:
    word: str
    start_us: int
    end_us: int


@dataclass(slots=True)
class AudioFrame:
    sample_rate: int = 16000
    channels: int = 1
    frame_duration_us: int = 20000
    pcm_s16le: bytes = b""
    discontinuity: bool = False
    sample_clock_pts_us: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pcm_s16le"] = self.pcm_s16le.decode("latin-1")
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioFrame:
        raw = data.copy()
        raw["pcm_s16le"] = raw["pcm_s16le"].encode("latin-1")
        return cls(**raw)


@dataclass(slots=True)
class AsrEvent:
    phase: Literal["partial", "final"] = "partial"
    text: str = ""
    stability: float = 0.0
    revision: int = 0
    words: list[WordTiming] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["words"] = [asdict(w) for w in self.words]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AsrEvent:
        raw = data.copy()
        raw["words"] = [WordTiming(**w) for w in raw.get("words", [])]
        return cls(**raw)


@dataclass(slots=True)
class VadEvent:
    kind: Literal["speech_start", "speech_end"] = "speech_start"
    energy_db: float = -96.0


@dataclass(slots=True)
class EouEvent:
    confidence: float = 0.0
    silence_us: int = 0


@dataclass(slots=True)
class InterruptEvent:
    kind: Literal["provisional", "confirmed", "rejected"] = "provisional"
    duck_gain: float = 1.0


@dataclass(slots=True)
class ControlEvent:
    intent: str = "noop"
    target_epoch: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "target_epoch": self.target_epoch,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlEvent:
        raw = data.copy()
        raw.setdefault("metadata", {})
        return cls(**raw)


@dataclass(slots=True)
class PlaybackAck:
    segment_seq: int = 0
    consumed_pts_us: int = 0


@dataclass(slots=True)
class ResidualEchoEvent:
    correlation: float = 0.0
    far_end_reference: bytes = b""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["far_end_reference"] = self.far_end_reference.decode("latin-1")
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualEchoEvent:
        raw = data.copy()
        raw["far_end_reference"] = raw["far_end_reference"].encode("latin-1")
        return cls(**raw)


@dataclass(slots=True)
class TtsAudioEvent:
    segment_seq: int
    epoch: int
    pts_us: int
    duration_us: int
    pcm_s16le: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_seq": self.segment_seq,
            "epoch": self.epoch,
            "pts_us": self.pts_us,
            "duration_us": self.duration_us,
            "pcm_s16le": self.pcm_s16le.hex(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TtsAudioEvent:
        raw = data.copy()
        raw["pcm_s16le"] = bytes.fromhex(raw["pcm_s16le"])
        return cls(**raw)


@dataclass(slots=True)
class AvatarFrameEvent:
    epoch: int
    pts_us: int
    frame_data: bytes
    format: Literal["i420", "rgb24", "bgr24", "h264"] = "i420"
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "pts_us": self.pts_us,
            "frame_data": self.frame_data.decode("latin-1"),
            "format": self.format,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvatarFrameEvent:
        raw = data.copy()
        raw["frame_data"] = raw["frame_data"].encode("latin-1")
        return cls(**raw)


@dataclass(slots=True)
class TaskEvent:
    task_id: str
    task_type: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskEvent:
        return cls(**data)


@dataclass(slots=True)
class ErrorEvent:
    source: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorEvent:
        return cls(**data)


_PAYLOAD_CLASSES: dict[str, Any] = {
    "audio_frame": AudioFrame,
    "asr": AsrEvent,
    "vad": VadEvent,
    "eou": EouEvent,
    "interrupt": InterruptEvent,
    "control": ControlEvent,
    "playback_ack": PlaybackAck,
    "residual_echo": ResidualEchoEvent,
    "tts_audio": TtsAudioEvent,
    "avatar_frame": AvatarFrameEvent,
    "task_event": TaskEvent,
    "error": ErrorEvent,
}


@dataclass(slots=True)
class Envelope:
    session_id: str
    event_type: EventType
    turn: int = 1
    epoch: int = 0
    seq: int = 0
    pts_us: int = 0
    deadline_us: int = 0
    trace_id: str = ""
    payload: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload_key = self._payload_key(self.event_type)
        if hasattr(self.payload, "to_dict"):
            payload_data = self.payload.to_dict()
        else:
            payload_data = asdict(self.payload) if self.payload else {}
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "epoch": self.epoch,
            "seq": self.seq,
            "pts_us": self.pts_us,
            "deadline_us": self.deadline_us,
            "trace_id": self.trace_id,
            "event_type": self.event_type,
            "payload": {payload_key: payload_data},
        }

    @staticmethod
    def _payload_key(event_type: str) -> str:
        if event_type == "audio_frame":
            return "audio_frame"
        if event_type == "playback_ack":
            return "playback_ack"
        return f"{event_type}_event"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Envelope:
        event_type = data.get("event_type", "")
        payload_map = data.get("payload", {})
        payload: Any = None
        if event_type == "audio_frame":
            payload = AudioFrame.from_dict(payload_map.get("audio_frame", {}))
        elif event_type == "playback_ack":
            payload = PlaybackAck(**payload_map.get("playback_ack", {}))
        elif event_type in _PAYLOAD_CLASSES:
            key = cls._payload_key(event_type)
            payload = _PAYLOAD_CLASSES[event_type].from_dict(payload_map.get(key, {}))
        return cls(
            session_id=data["session_id"],
            event_type=event_type,
            turn=data.get("turn", 1),
            epoch=data.get("epoch", 0),
            seq=data.get("seq", 0),
            pts_us=data.get("pts_us", 0),
            deadline_us=data.get("deadline_us", 0),
            trace_id=data.get("trace_id", ""),
            payload=payload,
        )


def validate_envelope(data: dict[str, Any]) -> list[str]:
    """Return a list of validation errors; empty means valid."""
    errors: list[str] = []
    required = ["session_id", "event_type", "seq", "pts_us"]
    for key in required:
        if key not in data:
            errors.append(f"missing {key}")
    event_type = data.get("event_type")
    if event_type and event_type not in _PAYLOAD_CLASSES:
        errors.append(f"unknown event_type: {event_type}")
    payload = data.get("payload")
    if payload is not None and not isinstance(payload, dict):
        errors.append("payload must be a dict")
    return errors

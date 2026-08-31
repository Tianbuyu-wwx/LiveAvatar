"""Service settings (env: LIVEKIT_*, LIVEAVATAR_*)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..duplex import DuplexSettings


@dataclass
class PublishSettings:
    """Service settings (env: LIVEKIT_*, LIVEAVATAR_*)."""

    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_room: str = "liveavatar"
    # Video transport: "ws" (self-developed, default) or "livekit".
    transport: str = "ws"
    # Self-developed ws codec: "mjpeg" (full frames) or "region"
    # (region-delta patches; requires the avatar's region.json).
    codec: str = "mjpeg"
    # Browser-reachable LiveKit URL (defaults to livekit_url).
    public_livekit_url: str = ""
    # Shared API key protecting REST + WS endpoints. Empty = auth disabled
    # (local dev / tests). When set, requests must present it via the
    # ``X-API-Key`` header (REST) or ``api_key`` query param / header (WS).
    api_key: str = ""
    # Maximum concurrent sessions (DoS guard for the GPU lease pool).
    max_sessions: int = 16
    # Maximum WebSocket binary frame size in bytes (PCM chunks are ~KB).
    max_ws_frame_bytes: int = 65536
    # Full-duplex spoke configuration (LIVEAVATAR_ASR_URL / LLM_* /
    # VOICE_CHAR / AEC / DUPLEX_AVATAR — see liveavatar.duplex).
    duplex: DuplexSettings = field(default_factory=DuplexSettings.from_env)

    @classmethod
    def from_env(cls) -> PublishSettings:
        transport = os.getenv("LIVEAVATAR_TRANSPORT", "ws").strip().lower()
        if transport not in ("ws", "livekit"):
            raise ValueError(
                f"LIVEAVATAR_TRANSPORT must be 'ws' or 'livekit', got {transport!r}"
            )
        codec = os.getenv("LIVEAVATAR_CODEC", "mjpeg").strip().lower()
        if codec not in ("mjpeg", "region"):
            raise ValueError(
                f"LIVEAVATAR_CODEC must be 'mjpeg' or 'region', got {codec!r}"
            )
        return cls(
            transport=transport,
            codec=codec,
            livekit_url=os.getenv("LIVEKIT_URL", ""),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            livekit_room=os.getenv("LIVEKIT_ROOM", "liveavatar"),
            public_livekit_url=os.getenv("PUBLIC_LIVEKIT_URL", ""),
            api_key=os.getenv("LIVEAVATAR_API_KEY", ""),
            max_sessions=int(os.getenv("LIVEAVATAR_MAX_SESSIONS", "16")),
            max_ws_frame_bytes=int(
                os.getenv("LIVEAVATAR_MAX_WS_FRAME_BYTES", "65536")
            ),
        )

    @property
    def livekit_enabled(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

"""Service settings (env: LIVEAVATAR_*)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..duplex import DuplexSettings


@dataclass
class PublishSettings:
    """Service settings (env: LIVEAVATAR_*)."""

    # Shared API key protecting REST + WS endpoints. Empty = auth disabled
    # (local dev / tests). When set, requests must present it via the
    # ``X-API-Key`` header (REST) or ``api_key`` query param / header (WS).
    api_key: str = ""
    # HMAC secret for short-lived session tokens (M-D task 17). When set
    # (together with api_key), POST /v1/sessions mints a per-session HS256
    # token that WS handshakes may present instead of the static key —
    # leaked browser credentials then expire after ``token_ttl_s``.
    api_secret: str = ""
    # Session-token TTL in seconds (only used when api_secret is set).
    token_ttl_s: int = 300
    # Maximum concurrent sessions (DoS guard for the GPU lease pool).
    max_sessions: int = 16
    # Maximum WebSocket binary frame size in bytes (PCM chunks are ~KB).
    max_ws_frame_bytes: int = 65536
    # Self-developed ws codec: "mjpeg" (full frames) or "region"
    # (region-delta patches; requires the avatar's region.json).
    codec: str = "mjpeg"
    # Full-duplex spoke configuration (LIVEAVATAR_ASR_URL / LLM_* /
    # VOICE_CHAR / AEC / DUPLEX_AVATAR — see liveavatar.duplex).
    duplex: DuplexSettings = field(default_factory=DuplexSettings.from_env)

    @classmethod
    def from_env(cls) -> PublishSettings:
        codec = os.getenv("LIVEAVATAR_CODEC", "mjpeg").strip().lower()
        if codec not in ("mjpeg", "region"):
            raise ValueError(
                f"LIVEAVATAR_CODEC must be 'mjpeg' or 'region', got {codec!r}"
            )
        return cls(
            codec=codec,
            api_key=os.getenv("LIVEAVATAR_API_KEY", ""),
            api_secret=os.getenv("LIVEAVATAR_API_SECRET", ""),
            token_ttl_s=int(os.getenv("LIVEAVATAR_TOKEN_TTL_S", "300")),
            max_sessions=int(os.getenv("LIVEAVATAR_MAX_SESSIONS", "16")),
            max_ws_frame_bytes=int(
                os.getenv("LIVEAVATAR_MAX_WS_FRAME_BYTES", "65536")
            ),
        )

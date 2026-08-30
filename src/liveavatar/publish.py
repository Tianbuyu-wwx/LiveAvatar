"""LiveAvatar publish service — FastAPI + WebSocket entry point.

Replaces the WisdomVII RealtimeCore session API with a minimal surface:

    POST   /v1/sessions              create session → {session_id, token, url}
    DELETE /v1/sessions/{sid}        close session (unpublish track)
    GET    /v1/sessions/{sid}/stats  adapter + publisher counters
    GET    /v1/avatars               list available avatars
    WS     /v1/sessions/{sid}/audio  stream PCM / send control messages
    GET    /health                   liveness probe
    GET    /                         web demo (served from web/)

WebSocket protocol
------------------
- Binary frames: raw PCM S16LE (16 kHz mono) — one chunk per frame.
- Text frames: JSON control messages::

    {"type": "epoch",  "epoch": 3}    start a new utterance epoch
    {"type": "cancel", "epoch": 4}    interrupt — drop stale audio+frames
    {"type": "stop"}                  flush and end

LiveKit integration
-------------------
When ``LIVEKIT_URL`` + ``LIVEKIT_API_KEY`` + ``LIVEKIT_API_SECRET`` are set,
each session joins the configured room as a publisher bot and streams the
avatar video track into it; the returned token lets the browser join the
same room and subscribe. Without LiveKit config the service still runs
(capture mode) — useful for tests and local previews.

Tokens are signed with the stdlib (HMAC-SHA256 JWT) — no extra dependency.

Run::

    uvicorn liveavatar.publish:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import AvatarPoolConfig
from .pipeline import AvatarPipeline, SessionState
from .pool import AvatarNotFound, AvatarPoolError

logger = logging.getLogger("liveavatar.publish")

# Optional LiveKit RTC SDK (only needed when publishing into a room).
try:
    from livekit import rtc  # type: ignore

    _HAS_LIVEKIT = True
except Exception:  # pragma: no cover
    _HAS_LIVEKIT = False
    rtc = None  # type: ignore


# ──────────────────────────────────────────────────── settings


@dataclass
class PublishSettings:
    """Service settings (env: LIVEKIT_*, LIVEAVATAR_*)."""

    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_room: str = "liveavatar"
    # Browser-reachable LiveKit URL (defaults to livekit_url).
    public_livekit_url: str = ""

    @classmethod
    def from_env(cls) -> PublishSettings:
        return cls(
            livekit_url=os.getenv("LIVEKIT_URL", ""),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            livekit_room=os.getenv("LIVEKIT_ROOM", "liveavatar"),
            public_livekit_url=os.getenv("PUBLIC_LIVEKIT_URL", ""),
        )

    @property
    def livekit_enabled(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)


# ───────────────────────────────────────────── token signing (stdlib)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwt_sign(payload: dict[str, Any], secret: str) -> str:
    """Sign a LiveKit-compatible JWT (HS256) using only the stdlib."""
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )
    sig = hmac.new(
        secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return signing_input + "." + _b64url(sig)


def make_access_token(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    ttl_s: int = 3600,
) -> str:
    """Mint a LiveKit room-join token for ``identity``."""
    now = int(time.time())
    payload = {
        "iss": api_key,
        "sub": identity,
        "iat": now,
        "nbf": now - 5,
        "exp": now + ttl_s,
        "name": identity,
        "video": {
            "roomJoin": True,
            "room": room,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
    }
    return _jwt_sign(payload, api_secret)


# ──────────────────────────────────────────────────────── app state


@dataclass
class AppState:
    settings: PublishSettings = field(default_factory=PublishSettings.from_env)
    pool_config: AvatarPoolConfig = field(default_factory=AvatarPoolConfig)
    pipeline: AvatarPipeline | None = None
    start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


state = AppState()


async def _ensure_pipeline() -> AvatarPipeline:
    """Lazily start the shared pipeline on first use."""
    async with state.start_lock:
        if state.pipeline is None:
            state.pipeline = AvatarPipeline(state.pool_config)
            await state.pipeline.start()
    return state.pipeline


async def _join_room(session_id: str) -> tuple[Any, Any]:
    """Join the LiveKit room as the avatar publisher bot.

    Returns (room, local_participant). Raises RuntimeError when the LiveKit
    SDK or configuration is missing.
    """
    if not _HAS_LIVEKIT:
        raise RuntimeError(
            "livekit package not installed; pip install 'liveavatar[livekit]'"
        )
    settings = state.settings
    token = make_access_token(
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
        identity=f"avatar-{session_id}",
        room=settings.livekit_room,
    )
    room = rtc.Room()
    await room.connect(settings.livekit_url, token)
    return room, room.local_participant


# ────────────────────────────────────────────────────────── FastAPI app

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Stop the shared pipeline on shutdown."""
    yield
    if state.pipeline is not None:
        await state.pipeline.stop()
        state.pipeline = None


app = FastAPI(title="LiveAvatar", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "livekit": state.settings.livekit_enabled})


@app.get("/v1/avatars")
async def list_avatars() -> JSONResponse:
    from .pool import discover_avatars

    avatars = discover_avatars(state.pool_config.avatar_data_root)
    return JSONResponse(
        {
            "data_root": state.pool_config.avatar_data_root,
            "avatars": sorted(avatars.keys()),
        }
    )


@app.post("/v1/sessions")
async def create_session(body: dict[str, Any] | None = None) -> JSONResponse:
    """Create a session: acquire the avatar lease and join the room.

    Body (all optional): ``{"session_id": "...", "avatar_id": "yongen"}``.
    """
    body = body or {}
    session_id: str = body.get("session_id") or f"sess_{int(time.time() * 1000):x}"
    avatar_id: str = body.get("avatar_id") or _default_avatar_id()

    pipeline = await _ensure_pipeline()

    room = participant = None
    if state.settings.livekit_enabled:
        room, participant = await _join_room(session_id)

    try:
        await pipeline.open_session(
            session_id,
            avatar_id,
            local_participant=participant,
            room=room,
        )
    except AvatarNotFound as exc:
        if room is not None:
            await room.disconnect()
        return JSONResponse({"error": str(exc)}, status_code=404)
    except AvatarPoolError as exc:
        if room is not None:
            await room.disconnect()
        return JSONResponse({"error": str(exc)}, status_code=503)

    resp: dict[str, Any] = {
        "session_id": session_id,
        "avatar_id": avatar_id,
        "livekit": state.settings.livekit_enabled,
        "sample_rate": 16000,
        "sample_format": "s16le",
    }
    if state.settings.livekit_enabled:
        resp["url"] = (
            state.settings.public_livekit_url or state.settings.livekit_url
        )
        resp["room"] = state.settings.livekit_room
        resp["token"] = make_access_token(
            api_key=state.settings.livekit_api_key,
            api_secret=state.settings.livekit_api_secret,
            identity=session_id,
            room=state.settings.livekit_room,
        )
    return JSONResponse(resp)


def _default_avatar_id() -> str:
    """First alphabetically-available avatar, or a placeholder."""
    from .pool import discover_avatars

    avatars = discover_avatars(state.pool_config.avatar_data_root)
    return sorted(avatars.keys())[0] if avatars else "yongen"


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str) -> JSONResponse:
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse({"error": "no session"}, status_code=404)
    ok = await pipeline.close_session(session_id)
    return JSONResponse({"closed": ok}, status_code=200 if ok else 404)


@app.get("/v1/sessions/{session_id}/stats")
async def session_stats(session_id: str) -> JSONResponse:
    pipeline = state.pipeline
    if pipeline is None or pipeline.get_session(session_id) is None:
        return JSONResponse({"error": "no session"}, status_code=404)
    return JSONResponse(pipeline.session_stats(session_id))


@app.websocket("/v1/sessions/{session_id}/audio")
async def audio_ws(websocket: WebSocket, session_id: str) -> None:
    """Stream PCM into the session; JSON text frames control the epoch."""
    pipeline = await _ensure_pipeline()
    session: SessionState | None = pipeline.get_session(session_id)
    if session is None:
        await websocket.close(code=4404, reason="session not found")
        return

    await websocket.accept()
    sample_rate = 16000
    logger.info("ws_connected", extra={"session_id": session_id})
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if (text := msg.get("text")) is not None:
                try:
                    ctrl = json.loads(text)
                except json.JSONDecodeError:
                    continue
                ctype = ctrl.get("type")
                if ctype == "cancel":
                    pipeline.cancel_epoch(session_id, int(ctrl.get("epoch", 0)))
                elif ctype == "epoch":
                    assert session.adapter is not None
                    session.epoch = max(session.epoch, int(ctrl.get("epoch", 0)))
                    session.adapter.cancel_epoch(session.epoch)
                elif ctype == "stop":
                    break

            elif (pcm := msg.get("bytes")) is not None:
                await pipeline.push_pcm(
                    session_id,
                    pcm,
                    sample_rate=sample_rate,
                )
    except WebSocketDisconnect:
        pass
    finally:
        logger.info(
            "ws_disconnected",
            extra={"session_id": session_id, "samples": session.samples_pushed},
        )


# ────────────────────────────────────────── static web demo (optional)


@app.get("/")
async def index() -> Any:
    index_html = os.path.join(_web_dir(), "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return JSONResponse(
        {"service": "LiveAvatar", "docs": "/docs", "health": "/health"}
    )


def _web_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "web")


@app.get("/{path:path}")
async def static_files(path: str) -> Any:
    full = os.path.normpath(os.path.join(_web_dir(), path))
    if not full.startswith(os.path.normpath(_web_dir())):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if os.path.isfile(full):
        return FileResponse(full)
    return JSONResponse({"error": "not found"}, status_code=404)

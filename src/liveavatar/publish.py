"""LiveAvatar publish service — FastAPI + WebSocket entry point.

Replaces the WisdomVII RealtimeCore session API with a minimal surface:

    POST   /v1/sessions              create session → {session_id, token, url}
    DELETE /v1/sessions/{sid}        close session (unpublish track)
    GET    /v1/sessions/{sid}/stats  adapter + publisher counters
    GET    /v1/avatars               list available avatars
    WS     /v1/sessions/{sid}/audio  stream PCM / send control messages
    GET    /health                   liveness probe
    GET    /                         web demo (served from web/)

Session modes (POST /v1/sessions body ``mode``)
-----------------------------------------------
- ``push`` (default): the client streams TTS-ready PCM and receives
  avatar video — the pipeline path.
- ``duplex``: a full-duplex star session (:mod:`liveavatar.duplex`) —
  the server runs VAD/EOU/ASR → LLM → TTS → Avatar on the microphone
  audio and sends synthesized speech + events back over the same audio
  WS. Spokes are configured via ``LIVEAVATAR_ASR_URL``,
  ``LIVEAVATAR_LLM_*``, ``LIVEAVATAR_VOICE_CHAR``, ``LIVEAVATAR_AEC``
  and ``LIVEAVATAR_DUPLEX_AVATAR`` (all optional; defaults: reference
  ASR, echo (no LLM), FakeTts, no AEC, audio-only).

WebSocket protocol
------------------
- Binary frames: raw PCM S16LE (16 kHz mono) — one chunk per frame.
- Text frames: JSON control messages::

    {"type": "epoch",  "epoch": 3}    start a new utterance epoch
    {"type": "cancel", "epoch": 4}    interrupt — drop stale audio+frames
    {"type": "stop"}                  flush and end

Video transport
---------------
``LIVEAVATAR_TRANSPORT`` selects the video delivery path:

- ``ws`` (default): the self-developed transport — frames are encoded
  (MJPEG) and fanned out by :class:`~liveavatar.ws_sink.WebSocketSink`;
  browsers subscribe at ``WS /v1/sessions/{sid}/video`` and render with
  ``web/player.js``. No extra infrastructure needed.
- ``livekit``: each session joins the configured room as a publisher bot
  (requires ``LIVEKIT_URL`` + ``LIVEKIT_API_KEY`` + ``LIVEKIT_API_SECRET``);
  the returned token lets the browser join the same room and subscribe.

Without any configuration the service still runs (capture mode) — useful
for tests and local previews.

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
import re
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import AvatarPoolConfig
from .duplex import DuplexSession, DuplexSettings
from .pipeline import AvatarPipeline, SessionState
from .pool import AvatarNotFound, AvatarPoolError

logger = logging.getLogger("liveavatar.publish")

# S4: avatar ids become filesystem path segments (avatar_data_root/<id>) —
# restrict to a safe charset so traversal/escape payloads are rejected at
# session creation and before any path join.
_AVATAR_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _valid_avatar_id(avatar_id: str) -> bool:
    return bool(_AVATAR_ID_RE.fullmatch(avatar_id))


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
            max_ws_frame_bytes=int(os.getenv("LIVEAVATAR_MAX_WS_FRAME_BYTES", "65536")),
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
    can_publish: bool = True,
) -> str:
    """Mint a LiveKit room-join token for ``identity``.

    ``can_publish=False`` for browser tokens — viewers must only subscribe;
    the avatar publisher bot keeps its own can_publish=True token.
    """
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
            "canPublish": can_publish,
            "canSubscribe": True,
            "canPublishData": can_publish,
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
    # Full-duplex sessions (mode="duplex") — keyed by session id.
    duplex_sessions: dict[str, DuplexSession] = field(default_factory=dict)
    # Shared TTS VoicePool (GPT-SoVITS) — created lazily on the first
    # duplex session that configures LIVEAVATAR_VOICE_CHAR.
    voice_pool: Any = None
    voice_pool_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


state = AppState()


async def _ensure_pipeline() -> AvatarPipeline:
    """Lazily start the shared pipeline on first use."""
    async with state.start_lock:
        if state.pipeline is None:
            state.pipeline = AvatarPipeline(
                state.pool_config,
                publisher_factory=_service_publisher_factory,
            )
            await state.pipeline.start()
    return state.pipeline


def _service_publisher_factory(session: SessionState) -> Any:
    """Create the per-session video publisher for the service.

    - LiveKit session (local_participant present) → LiveKit track publisher.
    - Otherwise → self-developed WebSocketSink (R2 transport): frames are
      encoded and fanned out to ``/v1/sessions/{sid}/video`` consumers.
    """
    if getattr(session, "local_participant", None) is not None:
        from .video_publisher import AvatarVideoPublisher

        cfg = state.pool_config
        return AvatarVideoPublisher(
            session.local_participant,
            session.session_id,
            width=cfg.width,
            height=cfg.height,
            target_fps=cfg.target_fps,
        )
    from .ws_sink import MjpegFrameEncoder, WebSocketSink

    cfg = state.pool_config
    encoder: Any = None
    if state.settings.codec == "region":
        encoder = _region_encoder_for(session.avatar_id)
    return WebSocketSink(
        encoder=encoder if encoder is not None else MjpegFrameEncoder(),
        target_fps=cfg.target_fps,
        width=cfg.width,
        height=cfg.height,
        quality=80,
    )


def _region_encoder_for(avatar_id: str) -> Any:
    """RegionFrameEncoder for the avatar, or None when region.json is
    missing / degenerate (caller falls back to full-frame MJPEG)."""
    from .region_codec import RegionFrameEncoder, load_region_json, region_spec_from_masks

    if not _valid_avatar_id(avatar_id):
        return None
    region_path = os.path.join(
        state.pool_config.avatar_data_root, avatar_id, "region.json"
    )
    try:
        spec = load_region_json(region_path)
        return RegionFrameEncoder(spec)
    except (OSError, ValueError, KeyError):
        # No region.json — try to derive it from the masks on the fly.
        mask_dir = os.path.join(
            state.pool_config.avatar_data_root, avatar_id, "mask"
        )
        cfg = state.pool_config
        mask_spec = region_spec_from_masks(mask_dir, cfg.width, cfg.height)
        if mask_spec is None:
            return None
        return RegionFrameEncoder(mask_spec)


async def _ensure_voice_pool() -> Any:
    """Lazily create + start the shared TTS VoicePool (duplex mode)."""
    async with state.voice_pool_lock:
        if state.voice_pool is None:
            from .voice.config import VoicePoolConfig
            from .voice.pool import VoicePool

            state.voice_pool = VoicePool(VoicePoolConfig())
            await state.voice_pool.start()
        return state.voice_pool


def _duplex_video_sink(avatar_id: str) -> Any:
    """WebSocketSink publisher for a duplex session's avatar spoke."""
    from .ws_sink import MjpegFrameEncoder, WebSocketSink

    cfg = state.pool_config
    encoder: Any = None
    if state.settings.codec == "region":
        encoder = _region_encoder_for(avatar_id)
    return WebSocketSink(
        encoder=encoder if encoder is not None else MjpegFrameEncoder(),
        target_fps=cfg.target_fps,
        width=cfg.width,
        height=cfg.height,
        quality=80,
    )


async def _open_duplex_session(session_id: str, avatar_id: str) -> JSONResponse:
    """Create + start a full-duplex session (star topology over WS)."""
    if session_id in state.duplex_sessions:
        return JSONResponse({"error": "session already open"}, status_code=409)
    if len(state.duplex_sessions) >= state.settings.max_sessions:
        return JSONResponse({"error": "session limit reached"}, status_code=429)

    ds = state.settings.duplex
    voice_pool: Any = None
    if ds.char_id:
        try:
            voice_pool = await _ensure_voice_pool()
        except Exception:
            logger.exception("voice_pool_start_failed")
            return JSONResponse(
                {"error": "voice pool failed to start"}, status_code=503
            )

    avatar_pool: Any = None
    sink: Any = None
    if ds.with_avatar:
        try:
            pipeline = await _ensure_pipeline()
            avatar_pool = pipeline.pool
            sink = _duplex_video_sink(avatar_id)
        except Exception:
            logger.exception("duplex_avatar_unavailable")
            # Audio-only fallback — never fail the session for video.

    session = DuplexSession(
        session_id,
        avatar_id,
        settings=ds,
        voice_pool=voice_pool,
        avatar_pool=avatar_pool,
        sink=sink,
    )
    await session.start()
    state.duplex_sessions[session_id] = session

    resp: dict[str, Any] = {
        "session_id": session_id,
        "avatar_id": avatar_id,
        "mode": "duplex",
        "transport": "ws",
        "spokes": ds.describe(),
        "sample_rate": 16000,
        "sample_format": "s16le",
    }
    if sink is not None:
        resp["video_ws"] = f"/v1/sessions/{session_id}/video"
    return JSONResponse(resp)


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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402


class CreateSessionBody(BaseModel):
    """POST /v1/sessions request body (all fields optional)."""

    session_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Client-chosen session id; autogenerated when omitted.",
    )
    avatar_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Avatar to lease; defaults to the first available.",
    )
    mode: str = Field(
        default="push",
        pattern=r"^(push|duplex)$",
        description=(
            "'push' (default): client streams TTS-ready PCM through the "
            "avatar adapter. 'duplex': full-duplex star session — the "
            "server runs VAD/EOU/ASR → LLM → TTS → Avatar on mic audio."
        ),
    )


def _check_auth(request: Request) -> JSONResponse | None:
    """Return a 401 response when the request is not authorized, else None."""
    key = state.settings.api_key
    if not key:
        return None
    if request.headers.get("X-API-Key") == key:
        return None
    return JSONResponse({"error": "unauthorized"}, status_code=401)


def _check_ws_auth(websocket: WebSocket) -> bool:
    """True when the WS handshake is authorized (or auth is disabled)."""
    key = state.settings.api_key
    if not key:
        return True
    provided = websocket.query_params.get("api_key") or websocket.headers.get(
        "x-api-key"
    )
    return provided == key


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Stop all sessions, the voice pool and the pipeline on shutdown."""
    from .observability import configure_logging

    configure_logging()
    yield
    for sid in list(state.duplex_sessions):
        session = state.duplex_sessions.pop(sid)
        try:
            await session.stop()
        except Exception:
            logger.exception("duplex_session_stop_failed", extra={"session_id": sid})
    if state.voice_pool is not None:
        try:
            await state.voice_pool.stop()
        except Exception:
            logger.exception("voice_pool_stop_failed")
        state.voice_pool = None
    if state.pipeline is not None:
        await state.pipeline.stop()
        state.pipeline = None


app = FastAPI(title="LiveAvatar", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "livekit": state.settings.livekit_enabled})


@app.get("/metrics")
async def metrics_endpoint(request: Request) -> Any:
    """Prometheus exposition (opt-in via LIVEAVATAR_METRICS=on)."""
    unauthorized = _check_auth(request)
    if unauthorized is not None:
        return unauthorized
    from .observability import (
        _METRICS_CONTENT_TYPE,
        METRICS_ENV,
        metrics_enabled,
        render_metrics,
    )

    if not metrics_enabled():
        return JSONResponse(
            {"error": f"metrics disabled (set {METRICS_ENV}=on)"},
            status_code=404,
        )
    from fastapi.responses import Response

    return Response(render_metrics(state), media_type=_METRICS_CONTENT_TYPE)


@app.get("/v1/avatars")
async def list_avatars(request: Request) -> JSONResponse:
    unauthorized = _check_auth(request)
    if unauthorized is not None:
        return unauthorized
    from .pool import discover_avatars

    avatars = discover_avatars(state.pool_config.avatar_data_root)
    return JSONResponse(
        {
            "data_root": state.pool_config.avatar_data_root,
            "avatars": sorted(avatars.keys()),
        }
    )


@app.post("/v1/sessions")
async def create_session(
    request: Request, body: CreateSessionBody | None = None
) -> JSONResponse:
    """Create a session: acquire the avatar lease and join the room.

    Body (all optional): ``{"session_id": "...", "avatar_id": "yongen",
    "mode": "push" | "duplex"}``.
    """
    unauthorized = _check_auth(request)
    if unauthorized is not None:
        return unauthorized
    body = body or CreateSessionBody()
    session_id = body.session_id or f"sess_{secrets.token_hex(8)}"
    avatar_id = body.avatar_id or _default_avatar_id()

    # S4: reject unsafe avatar ids before any filesystem use (both modes).
    if not _valid_avatar_id(avatar_id):
        return JSONResponse({"error": "invalid avatar_id"}, status_code=400)

    if body.mode == "duplex":
        return await _open_duplex_session(session_id, avatar_id)

    pipeline = await _ensure_pipeline()

    if len(pipeline.sessions) >= state.settings.max_sessions:
        return JSONResponse(
            {"error": "session limit reached"}, status_code=429
        )

    # Transport selection: "livekit" joins the room as a publisher bot;
    # "ws" (default) leaves publisher creation to the service factory,
    # which installs a WebSocketSink served at /v1/sessions/{sid}/video.
    use_livekit = False
    room = participant = None
    if state.settings.transport == "livekit":
        if not state.settings.livekit_enabled:
            return JSONResponse(
                {
                    "error": "transport=livekit requires LIVEKIT_URL, "
                    "LIVEKIT_API_KEY and LIVEKIT_API_SECRET"
                },
                status_code=503,
            )
        try:
            room, participant = await _join_room(session_id)
        except RuntimeError as exc:  # livekit SDK missing
            return JSONResponse({"error": str(exc)}, status_code=503)
        use_livekit = True

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
        "mode": "push",
        "transport": state.settings.transport,
        "livekit": use_livekit,
        "sample_rate": 16000,
        "sample_format": "s16le",
        "video_ws": f"/v1/sessions/{session_id}/video",
    }
    if use_livekit:
        resp["url"] = (
            state.settings.public_livekit_url or state.settings.livekit_url
        )
        resp["room"] = state.settings.livekit_room
        # Browser token: subscribe-only (no publish / data channels).
        resp["token"] = make_access_token(
            api_key=state.settings.livekit_api_key,
            api_secret=state.settings.livekit_api_secret,
            identity=session_id,
            room=state.settings.livekit_room,
            can_publish=False,
        )
    return JSONResponse(resp)


def _default_avatar_id() -> str:
    """First alphabetically-available avatar, or a placeholder."""
    from .pool import discover_avatars

    avatars = discover_avatars(state.pool_config.avatar_data_root)
    return sorted(avatars.keys())[0] if avatars else "yongen"


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> JSONResponse:
    unauthorized = _check_auth(request)
    if unauthorized is not None:
        return unauthorized
    duplex = state.duplex_sessions.pop(session_id, None)
    if duplex is not None:
        await duplex.stop()
        return JSONResponse({"closed": True})
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse({"error": "no session"}, status_code=404)
    ok = await pipeline.close_session(session_id)
    return JSONResponse({"closed": ok}, status_code=200 if ok else 404)


@app.get("/v1/sessions/{session_id}/stats")
async def session_stats(session_id: str, request: Request) -> JSONResponse:
    unauthorized = _check_auth(request)
    if unauthorized is not None:
        return unauthorized
    duplex = state.duplex_sessions.get(session_id)
    if duplex is not None:
        return JSONResponse(duplex.stats())
    pipeline = state.pipeline
    if pipeline is None or pipeline.get_session(session_id) is None:
        return JSONResponse({"error": "no session"}, status_code=404)
    return JSONResponse(pipeline.session_stats(session_id))


@app.websocket("/v1/sessions/{session_id}/audio")
async def audio_ws(websocket: WebSocket, session_id: str) -> None:
    """Stream PCM into the session; JSON text frames control the epoch.

    Push mode: binary frames are TTS-ready PCM fed to the avatar adapter.
    Duplex mode: binary frames are microphone audio processed through the
    star topology (VAD/EOU/ASR → LLM → TTS → Avatar); the server sends
    synthesized PCM back as binary frames and pipeline events (asr/vad/
    eou/control/error) as JSON text frames.
    """
    if not _check_ws_auth(websocket):
        await websocket.close(code=4401, reason="unauthorized")
        return
    duplex = state.duplex_sessions.get(session_id)
    if duplex is not None:
        await _duplex_audio_ws(websocket, duplex)
        return
    pipeline = await _ensure_pipeline()
    session: SessionState | None = pipeline.get_session(session_id)
    if session is None:
        await websocket.close(code=4404, reason="session not found")
        return

    await websocket.accept()
    sample_rate = 16000
    max_frame = state.settings.max_ws_frame_bytes
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
                if len(pcm) > max_frame:
                    # Oversized frame — drop instead of buffering (DoS guard).
                    logger.warning(
                        "ws_frame_too_large",
                        extra={"session_id": session_id, "bytes": len(pcm)},
                    )
                    continue
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


async def _duplex_audio_ws(websocket: WebSocket, duplex: DuplexSession) -> None:
    """Full-duplex audio loop: mic uplink + TTS/event downlink.

    The duplex session is closed when the audio socket drops (or on
    ``DELETE /v1/sessions/{sid}``), mirroring the short-lived session
    lifecycle of the push mode.
    """
    await websocket.accept()
    max_frame = state.settings.max_ws_frame_bytes
    logger.info(
        "ws_connected",
        extra={"session_id": duplex.session_id, "mode": "duplex"},
    )

    async def _sender() -> None:
        """Pump worker output (PCM + events) to the browser."""
        try:
            while True:
                item = await duplex.out_queue.dequeue()
                if item["kind"] == "pcm":
                    await websocket.send_bytes(item["data"])
                else:
                    await websocket.send_json(
                        {"type": item["event_type"], **item["payload"]}
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "duplex_sender_failed",
                extra={"session_id": duplex.session_id},
            )

    sender_task = asyncio.create_task(_sender())
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
                if ctype in ("cancel", "epoch"):
                    # Barge-in: the worker is the epoch authority.
                    new_epoch = duplex.cancel_epoch()
                    logger.info(
                        "duplex_barge_in",
                        extra={"session_id": duplex.session_id, "epoch": new_epoch},
                    )
                elif ctype == "stop":
                    break
            elif (pcm := msg.get("bytes")) is not None:
                if len(pcm) > max_frame:
                    logger.warning(
                        "ws_frame_too_large",
                        extra={"session_id": duplex.session_id, "bytes": len(pcm)},
                    )
                    continue
                await duplex.push_pcm(pcm)
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        if state.duplex_sessions.get(duplex.session_id) is duplex:
            state.duplex_sessions.pop(duplex.session_id, None)
        await duplex.stop()
        logger.info(
            "ws_disconnected",
            extra={"session_id": duplex.session_id, "mode": "duplex"},
        )


# ────────────────────────────────────── video WS (self-developed transport)


@app.websocket("/v1/sessions/{session_id}/video")
async def video_ws(websocket: WebSocket, session_id: str) -> None:
    """Subscribe to the session's avatar video stream (R2 transport).

    Server → client: binary wire frames (docs/PROTOCOL.md), one JSON
    ``ready`` message first. Client → server: JSON control messages
    (``hello`` / ``feedback`` / ``keyframe_request``).
    """
    if not _check_ws_auth(websocket):
        await websocket.close(code=4401, reason="unauthorized")
        return
    # Duplex sessions carry their own sink; push sessions live on the
    # pipeline (started lazily only when a push session is actually needed).
    duplex = state.duplex_sessions.get(session_id)
    sink: Any = duplex.sink if duplex is not None else None
    pipeline: AvatarPipeline | None = None
    if duplex is None:
        pipeline = await _ensure_pipeline()
        session: SessionState | None = pipeline.get_session(session_id)
        sink = getattr(session, "publisher", None) if session else None
    from .video_protocol import (
        VideoFrameHeader,
        make_flags,
        pack_video_frame,
    )
    from .ws_sink import EOF_SENTINEL, VideoClient, WebSocketSink

    if not isinstance(sink, WebSocketSink):
        await websocket.close(code=4404, reason="no video sink for session")
        return

    await websocket.accept()
    client: VideoClient = sink.add_client()
    logger.info(
        "video_ws_connected",
        extra={"session_id": session_id, "clients": sink.client_count},
    )

    async def _recv_loop() -> None:
        """Consume client control messages until disconnect."""
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") in ("websocket.disconnect",):
                    return
                if (text := msg.get("text")) is not None:
                    try:
                        ctrl = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    ctype = ctrl.get("type")
                    if ctype == "keyframe_request":
                        sink.request_keyframe(client)
                    elif ctype == "feedback":
                        # M5 adaptive quality: client congestion report.
                        sink.apply_feedback(client, ctrl)
                    # hello: consumed, no-op.
                elif msg.get("bytes") is not None:
                    # Binary uplink is not part of the protocol.
                    return
        except Exception:  # pragma: no cover - disconnect noise
            return

    recv_task = asyncio.create_task(_recv_loop())
    try:
        cfg = state.pool_config
        codec_names = {0: "mjpeg_full", 1: "region_delta"}
        await websocket.send_json(
            {
                "type": "ready",
                "codec": codec_names.get(sink._encoder.codec, "unknown"),
                "target_fps": sink.target_fps or cfg.target_fps,
                "width": cfg.width,
                "height": cfg.height,
            }
        )
        while True:
            try:
                wire = await asyncio.wait_for(client.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # Session gone without stop() (e.g. pipeline teardown)?
                if duplex is not None:
                    if not duplex._running:
                        break
                elif pipeline is not None and pipeline.get_session(session_id) is None:
                    break
                continue
            if wire is EOF_SENTINEL:
                eof = pack_video_frame(
                    VideoFrameHeader(
                        flags=make_flags(eof=True),
                        codec=0,
                        quality=1,
                        seq=0,
                        epoch=sink.current_epoch,
                        pts_us=0,
                        width=cfg.width,
                        height=cfg.height,
                    ),
                    b"",
                )
                await websocket.send_bytes(eof)
                break
            await websocket.send_bytes(wire)
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError: the ASGI transport rejects sends after the client
        # already disconnected (close/completion race) — normal teardown.
        pass
    finally:
        recv_task.cancel()
        sink.remove_client(client)
        logger.info(
            "video_ws_disconnected",
            extra={"session_id": session_id, "clients": sink.client_count},
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


if os.path.isdir(_web_dir()):
    # Starlette StaticFiles handles traversal/404 safely; mounted last so all
    # API routes and /health match first.
    from fastapi.staticfiles import StaticFiles  # noqa: E402

    app.mount("/", StaticFiles(directory=_web_dir(), html=True), name="web")

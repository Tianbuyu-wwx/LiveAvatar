# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Session lifecycle: pipeline/voice-pool start, duplex sessions."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from ..duplex import DuplexSession
from ..pipeline import AvatarPipeline
from ..pool import discover_avatars
from .encoders import _service_publisher_factory
from .encoders import _ws_sink_for as _duplex_video_sink
from .state import state

logger = logging.getLogger("liveavatar.publish")


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


async def _ensure_voice_pool() -> Any:
    """Lazily create + start the shared TTS VoicePool (duplex mode)."""
    async with state.voice_pool_lock:
        if state.voice_pool is None:
            from ..voice.config import VoicePoolConfig
            from ..voice.pool import VoicePool

            state.voice_pool = VoicePool(VoicePoolConfig())
            await state.voice_pool.start()
        return state.voice_pool


async def _open_duplex_session(session_id: str, avatar_id: str) -> Any:
    """Create + start a full-duplex session (star topology over WS)."""
    from fastapi.responses import JSONResponse

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


def _default_avatar_id() -> str:
    """First alphabetically-available avatar, or a placeholder."""
    avatars = discover_avatars(state.pool_config.avatar_data_root)
    return sorted(avatars.keys())[0] if avatars else "yongen"


@asynccontextmanager
async def _lifespan(app: Any):
    """Stop all sessions, the voice pool and the pipeline on shutdown."""
    from ..observability import configure_logging

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

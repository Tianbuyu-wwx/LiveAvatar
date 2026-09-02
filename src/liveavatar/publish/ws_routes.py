# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""WebSocket routes: audio uplink (push/duplex) + video downlink (R2)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from ..duplex import DuplexSession
from ..pipeline import AvatarPipeline, SessionState
from .routes import _check_ws_auth, app
from .session_manager import _ensure_pipeline
from .state import state

logger = logging.getLogger("liveavatar.publish")


@app.websocket("/v1/sessions/{session_id}/audio")
async def audio_ws(websocket: WebSocket, session_id: str) -> None:
    """Stream PCM into the session; JSON text frames control the epoch.

    Push mode: binary frames are TTS-ready PCM fed to the avatar adapter.
    Duplex mode: binary frames are microphone audio processed through the
    star topology (VAD/EOU/ASR → LLM → TTS → Avatar); the server sends
    synthesized PCM back as binary frames and pipeline events (asr/vad/
    eou/control/error) as JSON text frames.
    """
    if not _check_ws_auth(websocket, session_id):
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
    if not _check_ws_auth(websocket, session_id):
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
    from ..video_protocol import (
        VideoFrameHeader,
        make_flags,
        pack_video_frame,
    )
    from ..ws_sink import EOF_SENTINEL, VideoClient, WebSocketSink

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

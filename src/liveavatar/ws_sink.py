# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""WebSocketSink — PublishSink backend that fans frames out to WS clients.

The self-developed transport backend (R2): the sink encodes each
:class:`~liveavatar.worker.AvatarFrame` into a wire frame
(:mod:`liveavatar.video_protocol`) and offers it to every connected
video-WS client via per-client :class:`~liveavatar._common.loopqueue.LoopFreeQueue`.

Design invariants:

- **Never blocks the pipeline**: client queues use drop-oldest; a slow
  client only loses its own frames (TTS audio stays the master clock).
- **Loop-free**: publish happens on the adapter's loop, clients consume on
  their own WS-portal loops — all queues are LoopFreeQueue instances.
- **Keyframe discipline**: a client that (re)connects or explicitly
  requests one receives a keyframe as its first frame; ``cancel_epoch``
  forces the next frame to be a keyframe flagged ``epoch_boundary``.
- **Self-contained frames** (MJPEG / intra-only region patches): any frame
  can be dropped without corrupting the stream — matches the epoch
  drop-stale-frames semantics exactly.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ._common.loopqueue import LoopFreeQueue
from .adaptive import FeedbackAggregator, FeedbackSignals, QualityController
from .video_protocol import (
    CODEC_MJPEG_FULL,
    VideoFrameHeader,
    make_flags,
    pack_video_frame,
)
from .worker import AvatarFrame

logger = logging.getLogger("liveavatar.ws_sink")

#: Sentinel enqueued into every client queue by ``stop()``: the video WS
#: endpoint sends an ``eof_stream`` frame upon receiving it.
EOF_SENTINEL = None


@runtime_checkable
class FrameEncoder(Protocol):
    """Encodes one avatar frame into a wire payload."""

    codec: int

    def encode(
        self, frame: AvatarFrame, *, keyframe: bool, quality: int
    ) -> bytes: ...


class MjpegFrameEncoder:
    """Full-frame JPEG encoder (BGR24 in, JPEG out) — the v1 baseline codec."""

    codec = CODEC_MJPEG_FULL

    def __init__(self) -> None:
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np

    def encode(
        self, frame: AvatarFrame, *, keyframe: bool, quality: int
    ) -> bytes:
        img = self._np.frombuffer(frame.frame_data, dtype=self._np.uint8).reshape(
            frame.height, frame.width, 3
        )
        ok, buf = self._cv2.imencode(
            ".jpg", img, [int(self._cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:  # pragma: no cover - cv2 failure is effectively fatal
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()


@dataclass
class VideoClient:
    """One connected video-WS consumer (queue + keyframe bookkeeping)."""

    client_id: int
    queue: LoopFreeQueue[bytes | None] = field(
        default_factory=lambda: LoopFreeQueue(maxsize=4)
    )
    wants_keyframe: bool = True
    frames_sent: int = 0
    frames_dropped: int = 0


@dataclass
class WebSocketSinkStats:
    frames_seen: int = 0
    frames_published: int = 0
    frames_dropped_epoch: int = 0
    frames_dropped_closed: int = 0
    encode_errors: int = 0
    client_frames_dropped: int = 0


class WebSocketSink:
    """Fan-out video sink for the self-developed WS transport.

    Satisfies the :class:`~liveavatar.sinks.PublishSink` protocol, so it
    plugs directly into the streaming pipeline wherever a publisher is
    expected.
    """

    def __init__(
        self,
        *,
        encoder: FrameEncoder | None = None,
        target_fps: int = 25,
        width: int = 512,
        height: int = 512,
        quality: int = 80,
        client_queue_size: int = 4,
        keyframe_interval_us: int = 1_000_000,
    ) -> None:
        self._encoder: FrameEncoder = encoder or MjpegFrameEncoder()
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.quality = quality
        self._client_queue_size = client_queue_size
        self._keyframe_interval_us = keyframe_interval_us
        self._current_epoch = 0
        self._boundary_pending = False
        self._seq = 0
        self._last_keyframe_pts_us: int | None = None
        self._closed = False
        self._stats = WebSocketSinkStats()
        self._clients: dict[int, VideoClient] = {}
        self._next_client_id = 0
        # publish_frame runs on the adapter's loop while add/remove/request
        # can come from WS-portal loops — guard the client table.
        self._lock = threading.Lock()
        # M5 adaptive quality (D6): client feedback → EWMA → tier machine.
        self.aggregator = FeedbackAggregator()
        self.controller = QualityController()

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:  # PublishSink protocol
        return None

    async def stop(self) -> None:  # PublishSink protocol
        self._closed = True
        with self._lock:
            clients = list(self._clients.values())
        for client in clients:
            client.queue.close()

    # ------------------------------------------------------------ client mgmt

    def add_client(self) -> VideoClient:
        """Register a consumer; it receives a keyframe as its first frame."""
        with self._lock:
            client = VideoClient(
                client_id=self._next_client_id,
                queue=LoopFreeQueue(maxsize=self._client_queue_size),
                wants_keyframe=True,
            )
            self._next_client_id += 1
            self._clients[client.client_id] = client
            return client

    def remove_client(self, client: VideoClient) -> None:
        with self._lock:
            self._clients.pop(client.client_id, None)
        client.queue.close()

    def request_keyframe(self, client: VideoClient) -> None:
        client.wants_keyframe = True

    def apply_feedback(self, client: VideoClient, msg: dict[str, Any]) -> None:
        """Consume one client feedback report and adapt the tier (M5).

        Expected client message (all fields optional, defaults 0)::

            {"type": "feedback", "seq_gaps": 3, "frames": 120,
             "kbps": 640.0, "fps": 24.0}
        """
        frames = max(0, int(msg.get("frames") or 0))
        gaps = max(0, int(msg.get("seq_gaps") or 0))
        signals = FeedbackSignals(
            seq_gap_rate=gaps / frames if frames > 0 else 0.0,
            kbps=float(msg.get("kbps") or 0.0),
            fps=float(msg.get("fps") or 0.0),
        )
        self.aggregator.update(signals)
        if self.controller.update(self.aggregator):
            tier = self.controller.tier
            self.quality = tier.quality
            self._keyframe_interval_us = tier.keyframe_interval_us
            # Re-sync every client at the new parameters immediately.
            with self._lock:
                for c in self._clients.values():
                    c.wants_keyframe = True
            logger.info(
                "ws_sink_tier_changed",
                extra={"tier": tier.name, "quality": tier.quality},
            )

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    # ------------------------------------------------------------- publishing

    async def publish_frame(self, frame: AvatarFrame, epoch: int) -> bool:
        self._stats.frames_seen += 1
        if self._closed:
            self._stats.frames_dropped_closed += 1
            return False
        if epoch < self._current_epoch:
            self._stats.frames_dropped_epoch += 1
            return False

        boundary = self._boundary_pending or epoch > self._current_epoch
        self._current_epoch = epoch
        self._boundary_pending = False

        with self._lock:
            clients = list(self._clients.values())
        keyframe = bool(clients) and (
            boundary
            or any(c.wants_keyframe for c in clients)
            or self._last_keyframe_pts_us is None
            or frame.pts_us - self._last_keyframe_pts_us
            >= self._keyframe_interval_us
        )

        try:
            payload = self._encoder.encode(
                frame, keyframe=keyframe, quality=self.quality
            )
        except Exception:
            self._stats.encode_errors += 1
            logger.exception(
                "ws_sink_encode_error",
                extra={"epoch": epoch, "pts_us": frame.pts_us},
            )
            return False

        header = VideoFrameHeader(
            flags=make_flags(keyframe=keyframe, epoch_boundary=boundary),
            codec=self._encoder.codec,
            quality=self.quality,
            seq=self._seq,
            epoch=epoch,
            pts_us=frame.pts_us,
            width=frame.width,
            height=frame.height,
        )
        self._seq = (self._seq + 1) % 2**16
        if keyframe:
            self._last_keyframe_pts_us = frame.pts_us
        wire = pack_video_frame(header, payload)
        if keyframe:
            # Clear per-client keyframe requests — they all receive it.
            for c in clients:
                c.wants_keyframe = False

        for client in clients:
            # offer() returns the number of DROPPED items (0 = delivered).
            if client.queue.offer(wire):
                client.frames_dropped += 1
                self._stats.client_frames_dropped += 1
            else:
                client.frames_sent += 1
        self._stats.frames_published += 1
        return True

    def cancel_epoch(self, new_epoch: int) -> None:
        """Advance the epoch; the next published frame is a keyframe flagged
        ``epoch_boundary`` so browsers can flush their jitter buffers."""
        if new_epoch > self._current_epoch:
            self._current_epoch = new_epoch
            self._boundary_pending = True

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def stats(self) -> dict[str, Any]:
        return {
            "frames_seen": self._stats.frames_seen,
            "frames_published": self._stats.frames_published,
            "frames_dropped_epoch": self._stats.frames_dropped_epoch,
            "frames_dropped_closed": self._stats.frames_dropped_closed,
            "encode_errors": self._stats.encode_errors,
            "client_frames_dropped": self._stats.client_frames_dropped,
            "clients": self.client_count,
            "codec": self._encoder.codec,
            "current_epoch": self._current_epoch,
            "quality": self.quality,
            "tier": self.controller.tier.name,
            "smoothed_gap_rate": round(self.aggregator.smoothed_gap_rate, 5),
        }

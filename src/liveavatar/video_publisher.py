"""Avatar video publisher: stream AvatarFrame objects to a LiveKit video track.

Mirrors the audio publisher but for video:
- Input: ``AvatarFrame`` (BGR24 bytes + pts_us + epoch + is_speaking).
- Output: a published ``LocalVideoTrack`` backed by a ``VideoSource``.
- Cancellation: epoch-based — a confirmed interrupt bumps the epoch and
  stale-epoch frames are dropped before capture, so video stops within
  one frame (≤ 40ms at 25 fps) of an interrupt.

The publisher is single-session and single-track. It does NOT own the
AvatarWorker — the caller feeds it ``AvatarFrame`` objects produced by the
worker's ``synthesize_video_stream``.

Frame format
------------
``AvatarFrame.frame_data`` is raw BGR24 (OpenCV convention). LiveKit's
``VideoSource.capture_frame`` expects I420 (YUV 4:2:0). The conversion
happens in ``_capture_frame`` which:
1. Tries ``cv2.cvtColor(..., COLOR_BGR2YUV_I420)`` when cv2 is available.
2. Falls back to a pure-numpy BT.601 limited-range conversion with
   identical output (±1 rounding).

Tests override ``_capture_frame`` to avoid the livekit/cv2 dependency.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .worker import AvatarFrame

# Optional LiveKit RTC SDK (heavy native dependency).
try:
    from livekit import rtc  # type: ignore

    _HAS_LIVEKIT = True
except Exception:  # pragma: no cover - exercised only when livekit absent
    _HAS_LIVEKIT = False
    rtc = None  # type: ignore

logger = logging.getLogger("liveavatar.video_publisher")


@dataclass
class VideoPublisherStats:
    """Counters for the Avatar video publisher."""

    frames_seen: int = 0
    frames_published: int = 0
    frames_dropped_epoch: int = 0
    frames_dropped_timeout: int = 0
    frames_silence: int = 0
    bytes_published: int = 0
    track_published: bool = False


def _numpy_bgr24_to_i420(bgr_data: bytes, width: int, height: int) -> bytes:
    """Pure-numpy BT.601 limited-range conversion (matches cv2 output ±1)."""
    import numpy as np

    bgr = np.frombuffer(bgr_data, dtype=np.uint8).reshape(height, width, 3)
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    y = (16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0).clip(0, 255).astype(np.uint8)
    u = (128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0).clip(0, 255).astype(np.uint8)
    v = (128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0).clip(0, 255).astype(np.uint8)

    # 2×2 downsample for U and V (I420 = 4:2:0).
    return y.tobytes() + u[::2, ::2].tobytes() + v[::2, ::2].tobytes()


def _bgr24_to_i420(bgr_data: bytes, width: int, height: int) -> bytes:
    """Convert BGR24 pixel data to I420 (YUV 4:2:0) planar format.

    Fast path: ``cv2.cvtColor(..., COLOR_BGR2YUV_I420)`` when OpenCV is
    available. Fallback: pure-numpy BT.601 limited-range conversion with
    identical output (±1 rounding) so behaviour is deterministic across
    environments. Output layout: Y plane (H×W) + U plane (H/2×W/2) +
    V plane (H/2×W/2).
    """
    try:
        import cv2
        import numpy as np

        bgr = np.frombuffer(bgr_data, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).tobytes()
    except ImportError:
        return _numpy_bgr24_to_i420(bgr_data, width, height)


class AvatarVideoPublisher:
    """Publish Avatar video frames to a LiveKit room as a cancellable track.

    Lifecycle::

        publisher = AvatarVideoPublisher(
            room.local_participant, session_id, width=720, height=1280
        )
        await publisher.start()                              # publish track
        await publisher.publish_frame(avatar_frame, epoch)   # feed a frame
        publisher.cancel_epoch(new_epoch)                    # interrupt
        await publisher.stop()                               # unpublish

    The publisher mirrors the audio publisher's epoch-cancellation model:
    ``publish_frame`` checks ``frame_epoch < current_epoch`` before capturing
    and aborts immediately on stale epoch.
    """

    def __init__(
        self,
        local_participant: Any,
        session_id: str,
        *,
        width: int = 720,
        height: int = 1280,
        target_fps: int = 25,
        capture_timeout_s: float = 0.2,
    ) -> None:
        self.local_participant = local_participant
        self.session_id = session_id
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.capture_timeout_s = capture_timeout_s
        self._current_epoch = 0
        self._source: Any = None  # rtc.VideoSource
        self._track: Any = None  # rtc.LocalVideoTrack
        self._publication: Any = None  # rtc.LocalTrackPublication
        self.stats = VideoPublisherStats()

    # ------------------------------------------------------------------ start

    async def start(self) -> None:
        """Create the VideoSource, LocalVideoTrack and publish it."""
        if not _HAS_LIVEKIT:
            raise RuntimeError(
                "livekit package is not installed; "
                "install the 'livekit' extra to use AvatarVideoPublisher"
            )
        if self._track is not None:
            return
        self._source = rtc.VideoSource(self.width, self.height)
        self._track = rtc.LocalVideoTrack.create_video_track(
            f"avatar-{self.session_id}", self._source
        )
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_CAMERA
        self._publication = await self.local_participant.publish_track(
            self._track, options
        )
        self.stats.track_published = True
        logger.info(
            "avatar_video_publisher_started",
            extra={
                "session_id": self.session_id,
                "width": self.width,
                "height": self.height,
                "fps": self.target_fps,
                "track_sid": str(getattr(self._publication, "sid", None)),
            },
        )

    # -------------------------------------------------------------- publish

    async def publish_frame(self, frame: AvatarFrame, epoch: int) -> bool:
        """Feed one AvatarFrame to the track, respecting the current epoch.

        Returns True if the frame was published, False if it was dropped
        (stale epoch or capture backpressure timeout).
        """
        self.stats.frames_seen += 1
        if self._source is None:
            raise RuntimeError("publisher not started; call start() first")

        # Epoch check: drop stale-epoch frames immediately.
        if epoch < self._current_epoch:
            self.stats.frames_dropped_epoch += 1
            return False

        try:
            await self._capture_frame(frame)
        except asyncio.TimeoutError:
            self.stats.frames_dropped_timeout += 1
            return False

        self.stats.frames_published += 1
        self.stats.bytes_published += len(frame.frame_data)
        if not frame.is_speaking:
            self.stats.frames_silence += 1
        return True

    async def _capture_frame(self, avatar_frame: AvatarFrame) -> None:
        """Convert BGR24 → I420 and capture into the LiveKit VideoSource.

        Override in tests to avoid the livekit/cv2 dependency.
        """
        i420_data = _bgr24_to_i420(
            avatar_frame.frame_data,
            avatar_frame.width,
            avatar_frame.height,
        )
        video_frame = rtc.VideoFrame(
            width=avatar_frame.width,
            height=avatar_frame.height,
            type=rtc.VideoBufferType.I420,
            data=i420_data,
        )
        # LiveKit 1.1+ VideoSource.capture_frame is synchronous; wrap it in
        # a thread with a timeout so backpressure does not block the event
        # loop indefinitely.
        await asyncio.wait_for(
            asyncio.to_thread(
                self._source.capture_frame,
                video_frame,
                timestamp_us=avatar_frame.pts_us,
            ),
            timeout=self.capture_timeout_s,
        )

    # -------------------------------------------------------------- cancel

    def cancel_epoch(self, new_epoch: int) -> None:
        """Advance the cancellation epoch; stale-epoch frames are dropped.

        Monotonic: a lower or equal value is ignored (same as the audio
        publisher).
        """
        if new_epoch > self._current_epoch:
            self._current_epoch = new_epoch

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    # ------------------------------------------------------------------ stop

    async def stop(self) -> None:
        """Unpublish the track and release resources."""
        if self._publication is not None and self.local_participant is not None:
            track_sid = getattr(self._publication, "sid", None) or getattr(
                self._track, "sid", None
            )
            if track_sid is not None:
                try:
                    await self.local_participant.unpublish_track(track_sid)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("avatar_video_unpublish_error: %s", exc)
        self._publication = None
        self._track = None
        self._source = None
        self.stats.track_published = False

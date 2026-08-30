"""Pluggable publish sinks: LiveKit (existing), RTMP (ffmpeg), preview.

A ``PublishSink`` is the narrow interface the streaming pipeline needs from
a video output backend. ``AvatarVideoPublisher`` (LiveKit) already matches
it structurally; :class:`RtmpSink` adds an ffmpeg-backed RTMP/FLV output
without requiring livekit.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .worker import AvatarFrame

logger = logging.getLogger("liveavatar.sinks")


@runtime_checkable
class PublishSink(Protocol):
    """Structural protocol for video output backends."""

    async def start(self) -> None:
        """Open/prepare the output."""

    async def publish_frame(self, frame: AvatarFrame, epoch: int) -> bool:
        """Feed one frame; return False when dropped (stale epoch, etc.)."""

    def cancel_epoch(self, new_epoch: int) -> None:
        """Advance the cancellation epoch (monotonic)."""

    async def stop(self) -> None:
        """Flush and close the output."""

    @property
    def current_epoch(self) -> int: ...

    def stats(self) -> dict[str, Any]: ...


@dataclass
class RtmpSinkStats:
    frames_seen: int = 0
    frames_published: int = 0
    frames_dropped_epoch: int = 0


class RtmpSink:
    """Publish avatar frames to an RTMP endpoint via ffmpeg stdin.

    Requires an ``ffmpeg`` binary on PATH. BGR24 raw frames are piped in and
    encoded to H.264/FLV by ffmpeg — no extra Python dependencies.

    Epoch semantics mirror :class:`AvatarVideoPublisher`: stale frames are
    dropped before write; ``cancel_epoch`` is monotonic.
    """

    def __init__(
        self,
        rtmp_url: str,
        *,
        width: int = 512,
        height: int = 512,
        target_fps: int = 25,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        if width % 2 or height % 2:
            raise ValueError("RtmpSink requires even width/height (yuv420p)")
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.ffmpeg_bin = ffmpeg_bin
        self._current_epoch = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._stats = RtmpSinkStats()

    # ------------------------------------------------------------------ cmd

    def _ffmpeg_cmd(self) -> list[str]:
        return [
            self.ffmpeg_bin,
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.target_fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-f", "flv",
            self.rtmp_url,
        ]

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(  # noqa: S603 - fixed arg list
            self._ffmpeg_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(
            "rtmp_sink_started",
            extra={"url": self.rtmp_url, "wh": f"{self.width}x{self.height}"},
        )

    async def publish_frame(self, frame: AvatarFrame, epoch: int) -> bool:
        self._stats.frames_seen += 1
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("rtmp sink not started; call start() first")
        if epoch < self._current_epoch:
            self._stats.frames_dropped_epoch += 1
            return False
        expected = self.width * self.height * 3
        if len(frame.frame_data) != expected:
            raise ValueError(
                f"frame size {len(frame.frame_data)} != {expected} "
                f"(BGR24 {self.width}x{self.height})"
            )
        self._proc.stdin.write(frame.frame_data)
        self._stats.frames_published += 1
        return True

    def cancel_epoch(self, new_epoch: int) -> None:
        if new_epoch > self._current_epoch:
            self._current_epoch = new_epoch

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def stats(self) -> dict[str, Any]:
        return {
            "frames_seen": self._stats.frames_seen,
            "frames_published": self._stats.frames_published,
            "frames_dropped_epoch": self._stats.frames_dropped_epoch,
        }

    async def stop(self) -> None:
        if self._proc is not None:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except Exception:  # pragma: no cover - broken pipe on kill
                    pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
                self._proc.wait(timeout=2)
            self._proc = None
        logger.info("rtmp_sink_stopped", extra={"url": self.rtmp_url})

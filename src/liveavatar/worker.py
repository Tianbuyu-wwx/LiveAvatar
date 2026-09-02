# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""AvatarWorker abstract base class and AvatarFrame data structure.

Defines the streaming video inference interface that:
- ``MuseTalkAvatarWorker`` implements (real GPU inference)
- ``StaticAvatarWorker`` implements (degradation fallback)
- Fake workers implement (testing, no torch dependency)

The interface mirrors a streaming TTS worker, but:
- Input: PCM S16LE bytes (audio chunk from TTS) instead of text.
- Output: ``AvatarFrame`` (BGR24 video frame) instead of PCM bytes.
- Each PCM chunk produces ``batch_size`` video frames (MuseTalk infers
  multiple frames per audio feature batch for throughput).
"""

from __future__ import annotations

import abc
import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from .lease import CancelToken

logger = logging.getLogger("liveavatar.worker")


@dataclass(slots=True)
class AvatarFrame:
    """A single video frame produced by an AvatarWorker.

    Attributes
    ----------
    frame_data : bytes
        Raw BGR24 pixel data (OpenCV convention), ``width * height * 3`` bytes.
    pts_us : int
        Presentation timestamp in microseconds — aligned with the TTS PCM
        chunk's pts_us. This is the canonical clock for audio-video sync.
    epoch : int
        The epoch when this frame was produced. Frames with ``epoch <
        current_epoch`` must be discarded by the publisher.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    is_speaking : bool
        Whether this frame was generated from speech audio (True) or
        silence/idle (False). Idle frames use the avatar's default pose.
    """

    frame_data: bytes
    pts_us: int
    epoch: int
    width: int
    height: int
    is_speaking: bool = True


@dataclass(slots=True)
class AvatarWorkerStats:
    """Lifetime statistics for an AvatarWorker."""

    pcm_chunks_consumed: int = 0
    frames_produced: int = 0
    frames_silence: int = 0
    inference_batches: int = 0
    cancel_count: int = 0
    errors: int = 0
    total_inference_ms: float = 0.0


@dataclass
class AvatarAssets:
    """Preprocessed avatar data paths (one per avatar_id).

    A directory under ``avatar_data_root`` containing the output of
    ``prepare_avatar.py``:
    - full_imgs/ : original frames (sorted 0.jpg, 1.jpg, ...)
    - coords.pkl : face detection coordinates per frame
    - latents.pt : VAE-encoded reference latents per frame
    - mask/ : blending masks per frame
    - mask_coords.pkl : mask crop coordinates per frame
    """

    avatar_id: str
    data_dir: str
    full_imgs_dir: str
    coords_path: str
    latents_path: str
    mask_dir: str
    mask_coords_path: str


class AvatarWorker(abc.ABC):
    """Abstract base for streaming video avatar inference.

    Subclasses implement ``_infer_batch`` (GPU-specific) and ``_paste_back``
    (compositing). The base class handles:
    - Async streaming via ``asyncio.to_thread`` (keeps the event loop free).
    - Cooperative cancellation via ``CancelToken``.
    - ``_infer_lock`` to serialize GPU access (one inference at a time).
    - Statistics tracking.
    """

    def __init__(
        self,
        assets: AvatarAssets,
        *,
        target_fps: int = 25,
        width: int = 720,
        height: int = 1280,
        batch_size: int = 4,
    ) -> None:
        self.assets = assets
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self._infer_lock = asyncio.Lock()
        self._busy = False
        self.stats = AvatarWorkerStats()

    @property
    def avatar_id(self) -> str:
        return self.assets.avatar_id

    @property
    def busy(self) -> bool:
        return self._busy

    async def synthesize_video_stream(
        self,
        pcm_s16le: bytes,
        *,
        pts_us: int,
        epoch: int,
        cancel_token: CancelToken | None = None,
    ) -> AsyncGenerator[AvatarFrame, None]:
        """Convert one PCM chunk into ``batch_size`` video frames.

        Parameters
        ----------
        pcm_s16le : bytes
            PCM S16LE audio chunk (16kHz mono), typically 320 samples = 20ms.
        pts_us : int
            Presentation timestamp of the audio chunk. The first output frame
            inherits this PTS; subsequent frames advance by ``1/fps`` seconds.
        epoch : int
            Current epoch for cancellation tracking.
        cancel_token : CancelToken | None
            Cooperative cancellation flag. Checked between frames.

        Yields
        ------
        AvatarFrame
            One frame at a time, in PTS order.
        """
        token = cancel_token or CancelToken()
        async with self._infer_lock:
            self._busy = True
            self.stats.pcm_chunks_consumed += 1
            try:
                # Pre-inference cancellation check: if the token was already
                # cancelled (e.g. interrupt arrived before this chunk started
                # inference), skip the expensive GPU forward pass entirely.
                if token.cancelled:
                    self.stats.cancel_count += 1
                    return

                frame_pts_increment = 1_000_000 // self.target_fps
                current_pts = pts_us

                t0 = time.monotonic()
                frames = await asyncio.to_thread(
                    self._infer_batch, pcm_s16le
                )
                self.stats.total_inference_ms += (time.monotonic() - t0) * 1000
                self.stats.inference_batches += 1

                for frame_data, is_speaking in frames:
                    if token.cancelled:
                        self.stats.cancel_count += 1
                        break
                    yield AvatarFrame(
                        frame_data=frame_data,
                        pts_us=current_pts,
                        epoch=epoch,
                        width=self.width,
                        height=self.height,
                        is_speaking=is_speaking,
                    )
                    self.stats.frames_produced += 1
                    if not is_speaking:
                        self.stats.frames_silence += 1
                    current_pts += frame_pts_increment
            except Exception:
                self.stats.errors += 1
                logger.exception(
                    "avatar_inference_error",
                    extra={"avatar_id": self.avatar_id, "epoch": epoch},
                )
                raise
            finally:
                self._busy = False

    @abc.abstractmethod
    def _infer_batch(
        self, pcm_s16le: bytes
    ) -> list[tuple[bytes, bool]]:
        """Run GPU inference on one PCM chunk.

        Returns a list of ``(frame_data_bgr24, is_speaking)`` tuples,
        length == ``self.batch_size``. Called in a worker thread via
        ``asyncio.to_thread`` — must be thread-safe and non-async.

        Subclasses implement the actual model inference (MuseTalk UNet+VAE,
        static frame, etc.).
        """
        ...

    def cancel(self) -> None:
        """Mark the worker as cancelled (for external use)."""
        self._busy = False

"""StaticAvatarWorker — degradation fallback that emits a fixed image.

Used as the ``fallback_worker`` on :class:`AvatarStreamingAdapter` when the
primary MuseTalk worker fails repeatedly. The static worker:

- Loads the avatar's first reference frame from ``full_imgs/`` (or a
  caller-supplied default BGR24 buffer when no on-disk avatar exists).
- Returns that frame for every PCM chunk with ``is_speaking=False``.
- Never raises (barring disk I/O errors at construction time), so it keeps
  the video track alive while audio playback continues unaffected.

This is the second link in the degradation chain:
``MuseTalk → Static → Audio-only``. When even the static worker fails
(e.g. no avatar data at all), the adapter logs the error and the
``AvatarVideoPublisher`` simply stops receiving frames — audio keeps
flowing on its own track.
"""

from __future__ import annotations

import logging
import os

from .worker import AvatarAssets, AvatarWorker

logger = logging.getLogger("liveavatar.static_worker")


class StaticAvatarWorker(AvatarWorker):
    """Fallback worker that emits a fixed frame for every PCM chunk.

    Parameters
    ----------
    assets : AvatarAssets
        Preprocessed avatar data paths. ``full_imgs/`` is scanned for the
        first image to use as the static frame.
    default_frame_bgr : bytes | None
        Pre-encoded BGR24 buffer used when ``assets.full_imgs_dir`` is empty
        or unreadable. Must be ``width * height * 3`` bytes. When ``None``
        and no on-disk image exists, a black frame is generated.
    """

    def __init__(
        self,
        assets: AvatarAssets,
        *,
        target_fps: int = 25,
        width: int = 720,
        height: int = 1280,
        batch_size: int = 4,
        default_frame_bgr: bytes | None = None,
    ) -> None:
        super().__init__(
            assets,
            target_fps=target_fps,
            width=width,
            height=height,
            batch_size=batch_size,
        )
        self._frame_index = 0
        self._static_frame = self._load_static_frame(default_frame_bgr)
        logger.info(
            "static_worker_init",
            extra={
                "avatar_id": self.avatar_id,
                "frame_bytes": len(self._static_frame),
                "batch_size": batch_size,
            },
        )

    def _load_static_frame(
        self, default_frame_bgr: bytes | None
    ) -> bytes:
        """Load the first frame from ``full_imgs/`` or use the fallback."""
        # Prefer the caller-supplied default (no disk I/O).
        if default_frame_bgr is not None:
            return self._resize_bytes(default_frame_bgr)

        # Try to read the first frame from full_imgs/.
        frame = self._read_first_full_img()
        if frame is not None:
            return self._resize_bytes(frame)

        # Last resort: solid black BGR24 frame.
        return b"\x00" * (self.width * self.height * 3)

    def _read_first_full_img(self) -> bytes | None:
        """Read the first image from ``assets.full_imgs_dir`` as BGR24."""
        try:
            import glob

            import cv2

            if not os.path.isdir(self.assets.full_imgs_dir):
                return None
            candidates = sorted(
                glob.glob(
                    os.path.join(self.assets.full_imgs_dir, "*.[jpJP][pnPN]*[gG]")
                )
            )
            if not candidates:
                return None
            img = cv2.imread(candidates[0])
            if img is None:
                return None
            return img.tobytes()
        except Exception:
            logger.exception(
                "static_worker_read_failed",
                extra={"full_imgs_dir": self.assets.full_imgs_dir},
            )
            return None

    def _resize_bytes(self, bgr_bytes: bytes) -> bytes:
        """Resize a BGR24 byte buffer to the worker's target dimensions."""
        if len(bgr_bytes) == self.width * self.height * 3:
            return bgr_bytes
        try:
            import cv2
            import numpy as np

            # Infer source height assuming width matches; fall back to square.
            src_len = len(bgr_bytes)
            src_pixels = src_len // 3
            import math

            src_h = int(math.isqrt(src_pixels))
            if src_h * src_h != src_pixels:
                # Not square — assume width=self.width.
                src_w = self.width
                src_h = src_pixels // src_w
                if src_w * src_h != src_pixels:
                    # Cannot infer shape — return black frame of target size.
                    return b"\x00" * (self.width * self.height * 3)
            else:
                src_w = src_h
            arr = np.frombuffer(bgr_bytes, dtype=np.uint8).reshape(
                src_h, src_w, 3
            )
            resized = cv2.resize(arr, (self.width, self.height))
            return resized.tobytes()
        except Exception:
            logger.exception(
                "static_worker_resize_failed",
                extra={"frame_bytes": len(bgr_bytes)},
            )
            return b"\x00" * (self.width * self.height * 3)

    # ----------------------------------------------------------- inference

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        """Return ``batch_size`` copies of the static frame, ``is_speaking=False``.

        The frame index advances so each frame gets a unique PTS via the
        base class's ``synthesize_video_stream`` (mirrors the MuseTalk
        worker's behavior for sync continuity).
        """
        self._frame_index += self.batch_size
        return [
            (self._static_frame, False) for _ in range(self.batch_size)
        ]

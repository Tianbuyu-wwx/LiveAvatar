"""Offline batch renderer: whole wav → mp4 (Roadmap: batch_renderer).

Usage::

    python -m liveavatar.batch_renderer \
        --audio speech.wav --avatar yongen --save out.mp4

The renderer feeds the entire PCM to the worker in one call; the worker's
offline path (long-audio branch of ``_infer_batch``) renders every frame
with correct per-frame temporal audio features. No LiveKit required.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from .config import AvatarPoolConfig
from .pool import AvatarPool
from .worker import AvatarFrame, AvatarWorker

logger = logging.getLogger("liveavatar.batch_renderer")


async def render_to_file(
    worker: AvatarWorker,
    pcm_s16le: bytes,
    sample_rate: int,
    out_path: str,
) -> int:
    """Render ``pcm_s16le`` via ``worker`` and write BGR frames to ``out_path``.

    Returns the number of frames written. Uses cv2.VideoWriter (mp4v).
    """
    import cv2

    written = 0
    writer = None
    async for frame in worker.synthesize_video_stream(
        pcm_s16le, pts_us=0, epoch=0
    ):
        img = _frame_to_bgr(frame)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(
                out_path, fourcc, worker.target_fps,
                (frame.width, frame.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer: {out_path}")
        writer.write(img)
        written += 1
    if writer is not None:
        writer.release()
    logger.info(
        "batch_render_done",
        extra={"out": out_path, "frames": written},
    )
    return written


def _frame_to_bgr(frame: AvatarFrame):
    import numpy as np

    return np.frombuffer(frame.frame_data, dtype=np.uint8).reshape(
        frame.height, frame.width, 3
    )


async def _run(args: argparse.Namespace) -> int:
    from .preview import load_wav_mono_16k

    device = args.device or os.getenv("LIVEAVATAR_DEVICE", "cuda")
    config = AvatarPoolConfig(
        avatar_data_root=args.avatar_data_root or os.getenv(
            "LIVEAVATAR_AVATAR_DATA_ROOT", "avatars"
        ),
        device=device,
        batch_size=args.batch_size,
    )
    pool = AvatarPool(config)
    await pool.start()
    try:
        lease = await pool.acquire("batch", args.avatar, timeout=args.acquire_timeout)
        try:
            pcm, duration_s = load_wav_mono_16k(args.audio)
            print(f"[batch] audio: {args.audio} ({duration_s:.2f}s)")
            return await render_to_file(lease.worker, pcm, 16000, args.save)
        finally:
            await pool.release_async("batch")
    finally:
        await pool.stop()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True, help="16-bit PCM wav file")
    ap.add_argument("--avatar", required=True, help="avatar_id under --avatar-data-root")
    ap.add_argument("--avatar-data-root", default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: LIVEAVATAR_DEVICE)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--acquire-timeout", type=float, default=120.0)
    ap.add_argument("--save", required=True, help="output mp4 path")
    args = ap.parse_args(argv)
    written = asyncio.run(_run(args))
    print(f"[batch] wrote {written} frames → {args.save}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main", "render_to_file"]

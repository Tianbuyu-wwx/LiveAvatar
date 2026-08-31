"""Local preview (mode B): wav file → MuseTalk avatar → window / mp4.

No LiveKit required. Joins the avatar pool, streams the wav through the
pipeline in real time and renders the produced frames with OpenCV.

Examples::

    # window preview (press q to quit)
    python -m liveavatar.preview --audio data/audio/yongen.wav --avatar yongen

    # render to mp4 instead
    python -m liveavatar.preview --audio data/audio/yongen.wav \\
        --avatar yongen --save out.mp4

Configuration comes from ``LIVEAVATAR_*`` env vars / ``.env`` (device,
avatar_data_root, model paths) just like the service.
"""

from __future__ import annotations

import argparse
import asyncio
import wave
from collections import deque
from typing import Any

import numpy as np

from .adapter import AvatarStreamingAdapter
from .config import AvatarPoolConfig
from .pool import AvatarPool
from .worker import AvatarFrame


class PreviewSink:
    """Publisher-compatible sink that buffers frames for the display loop.

    Implements the AvatarVideoPublisher surface used by the adapter
    (``publish_frame`` / ``cancel_epoch`` / ``current_epoch``) without any
    LiveKit dependency.
    """

    def __init__(self, maxlen: int = 256) -> None:
        self.frames: deque[AvatarFrame] = deque(maxlen=maxlen)
        self._current_epoch = 0

    async def publish_frame(self, frame: AvatarFrame, epoch: int) -> bool:
        if epoch < self._current_epoch:
            return False
        self.frames.append(frame)
        return True

    def cancel_epoch(self, new_epoch: int) -> None:
        if new_epoch > self._current_epoch:
            self._current_epoch = new_epoch

    @property
    def current_epoch(self) -> int:
        return self._current_epoch


def load_wav_mono_16k(path: str) -> tuple[bytes, float]:
    """Read a wav file and return (pcm_s16le_bytes, duration_seconds).

    Requires mono 16-bit PCM WAV. Other sample rates are resampled with
    linear interpolation; stereo is downmixed.
    """
    with wave.open(path, "rb") as w:
        nchannels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    if sampwidth != 2:
        raise SystemExit(
            f"unsupported sample width {sampwidth * 8}-bit; only 16-bit PCM wav is accepted"
        )

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if nchannels > 1:
        audio = audio.reshape(-1, nchannels).mean(axis=1).astype(np.float32)

    if framerate != 16000:
        duration = len(audio) / framerate
        target_n = int(duration * 16000)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_n),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio.astype(np.int16).tobytes(), nframes / framerate


async def run_preview(args: argparse.Namespace) -> None:
    import cv2

    config_kwargs: dict[str, Any] = {}
    if args.avatar_data_root:
        config_kwargs["avatar_data_root"] = args.avatar_data_root
    config = AvatarPoolConfig(**config_kwargs)
    pool = AvatarPool(config)
    await pool.start()
    try:
        lease = await pool.acquire("preview", args.avatar, timeout=args.acquire_timeout)
        worker: Any = lease.worker
        print(f"[preview] avatar loaded: {worker.avatar_id} (device={config.device})")

        pcm, duration_s = load_wav_mono_16k(args.audio)
        print(f"[preview] audio: {args.audio} ({duration_s:.2f}s)")

        sink: Any = PreviewSink()
        adapter = AvatarStreamingAdapter(worker=worker, publisher=sink, session_id="preview")
        await adapter.start()

        writer = None
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(
                args.save, fourcc, config.target_fps, (config.width, config.height)
            )
            print(f"[preview] saving → {args.save}")

        # Slice the wav in batch-sized chunks so each push maps 1:1 to one
        # MuseTalk inference batch (batch_size frames = batch audio duration).
        # This keeps video duration == audio duration.
        sample_rate = 16000
        batch_samples = worker.batch_size * sample_rate // worker.target_fps
        chunk_bytes = batch_samples * 2
        chunk_interval = batch_samples / sample_rate
        pts_us = 0
        stop = False

        async def feed() -> None:
            nonlocal pts_us, stop
            for offset in range(0, len(pcm), chunk_bytes):
                chunk = pcm[offset : offset + chunk_bytes]
                if len(chunk) < chunk_bytes:
                    chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))
                await adapter.push_pcm(chunk, pts_us, 0)
                pts_us += batch_samples * 1_000_000 // sample_rate
                await asyncio.sleep(chunk_interval)
            stop = True

        feed_task = asyncio.create_task(feed())
        shown = 0
        while True:
            await asyncio.sleep(0.005)
            while sink.frames:
                frame = sink.frames.popleft()
                img = np.frombuffer(frame.frame_data, dtype=np.uint8).reshape(
                    frame.height, frame.width, 3
                )
                if writer is not None:
                    writer.write(img)
                if not args.save:
                    cv2.imshow("LiveAvatar preview", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop = True
                shown += 1
            if stop and not sink.frames:
                break

        await feed_task
        # Give the consumer a moment to flush the tail.
        for _ in range(20):
            while sink.frames:
                frame = sink.frames.popleft()
                img = np.frombuffer(frame.frame_data, dtype=np.uint8).reshape(
                    frame.height, frame.width, 3
                )
                if writer is not None:
                    writer.write(img)
                shown += 1
            await asyncio.sleep(0.05)

        await adapter.stop()
        if writer is not None:
            writer.release()
        try:
            # headless OpenCV builds (CI / opencv-python-headless) have no
            # highgui; destroyAllWindows raises there even when never shown.
            cv2.destroyAllWindows()
        except cv2.error:  # pragma: no cover - depends on the cv2 build
            pass
        print(f"[preview] done: {shown} frames, adapter stats: {vars(adapter.stats)}")
    finally:
        await pool.release_async("preview")
        await pool.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True, help="16-bit PCM wav file")
    ap.add_argument("--avatar", required=True, help="avatar_id under --avatar-data-root")
    ap.add_argument("--avatar-data-root", default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: LIVEAVATAR_DEVICE)")
    ap.add_argument("--acquire-timeout", type=float, default=120.0)
    ap.add_argument("--save", default=None, help="save rendered frames to an mp4 file")
    args = ap.parse_args()

    if args.device is not None:
        # Override env-derived device before config construction.
        import os

        os.environ["LIVEAVATAR_DEVICE"] = args.device
    asyncio.run(run_preview(args))


if __name__ == "__main__":
    main()

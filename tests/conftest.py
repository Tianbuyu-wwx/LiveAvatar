# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Shared test helpers (T2): fake avatar assets, PCM, WAV, async polling,
and the TestClient WS race helpers (T3).

Plain (non-fixture) helpers so unittest-style classes and pytest tests can
both use them: ``from tests.conftest import make_assets, pcm``. Deduplicated
from test_adapter / test_pipeline / test_pool / test_static_worker /
test_worker / test_publish / test_preview / audio_in/test_reconnect /
test_concurrency3 / test_video_ws.
"""

from __future__ import annotations

import asyncio
import queue
import struct
import threading
import wave

from liveavatar.video_protocol import FLAG_EOF, has_flag, unpack_video_frame
from liveavatar.worker import AvatarAssets


def make_assets(
    avatar_id: str = "nahida", *, full_imgs_dir: str | None = None
) -> AvatarAssets:
    """Minimal AvatarAssets with fake paths (no files accessed)."""
    base = f"avatars/{avatar_id}/"
    return AvatarAssets(
        avatar_id=avatar_id,
        data_dir=base,
        full_imgs_dir=(
            full_imgs_dir if full_imgs_dir is not None else base + "full_imgs"
        ),
        coords_path=base + "coords.pkl",
        latents_path=base + "latents.pt",
        mask_dir=base + "mask",
        mask_coords_path=base + "mask_coords.pkl",
    )


def pcm(samples: int = 320, value: int = 1) -> bytes:
    """PCM S16LE chunk: ``samples`` repeats of the signed 16-bit ``value``."""
    return value.to_bytes(2, "little", signed=True) * samples


def write_wav(
    path: str,
    samples: list[int],
    framerate: int,
    nchannels: int = 1,
    sampwidth: int = 2,
) -> None:
    """Write a WAV file (stereo frames are interleaved identical channels)."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        frames = b"".join(struct.pack("<h", s) for s in samples)
        if nchannels > 1:
            frames = frames * nchannels
        wf.writeframes(frames)


async def wait_until(cond, timeout: float = 2.0) -> None:
    """Poll ``cond`` on the running loop until true or timeout (AssertionError)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


# ── TestClient WS race helpers (T3, single copy) ────────────────────────
# ``receive()`` on the TestClient blocks forever, so each message is
# awaited through a helper thread with a timeout: a stream that goes idle
# before EOF (server paces with 0.5 s timeouts) simply stops the drain
# instead of hanging the suite on slow runners.

RECV_TIMEOUT = 5.0


def receive_with_timeout(video, timeout: float = RECV_TIMEOUT):
    """One WS message via a helper thread; None on timeout, exceptions
    surfaced in the caller thread."""
    q: queue.Queue = queue.Queue()

    def _worker() -> None:
        try:
            q.put(video.receive())
        except Exception as exc:  # surfaced in the caller thread
            q.put(exc)

    threading.Thread(target=_worker, daemon=True).start()
    try:
        result = q.get(timeout=timeout)
    except queue.Empty:
        return None
    if isinstance(result, Exception):
        raise result
    return result


def recv_frames(
    video, n_max: int, per_msg_timeout: float = RECV_TIMEOUT
) -> list:
    """Collect up to ``n_max`` binary wire-frame headers until the stream
    goes idle / hits EOF."""
    frames = []
    for _ in range(n_max):
        msg = receive_with_timeout(video, per_msg_timeout)
        if msg is None:
            break  # idle stream, no EOF yet — stop draining
        if msg.get("bytes") is None:
            continue
        header, _ = unpack_video_frame(msg["bytes"])
        if has_flag(header.flags, FLAG_EOF):
            break
        frames.append(header)
    return frames

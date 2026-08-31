"""Shared test helpers (T2): fake avatar assets, PCM, WAV, async polling.

Plain (non-fixture) helpers so unittest-style classes and pytest tests can
both use them: ``from tests.conftest import make_assets, pcm``. Deduplicated
from test_adapter / test_pipeline / test_pool / test_static_worker /
test_worker / test_publish / test_preview / audio_in/test_reconnect.
"""

from __future__ import annotations

import asyncio
import struct
import wave

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

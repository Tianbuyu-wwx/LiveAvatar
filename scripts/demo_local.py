"""Run the publish service on CPU with a synthetic worker (no GPU).

Frames are an animated 128x128 pattern instead of real MuseTalk lip-sync;
everything else (pipeline, adapter, WebSocketSink, /video WS, web demo) is
the real service code path. Used to demo and smoke-test the self-developed
video transport:

    python scripts/demo_local.py --port 8000
    → open http://127.0.0.1:8000/
"""

from __future__ import annotations

import argparse
import time

import uvicorn

from liveavatar.config import AvatarPoolConfig
from liveavatar.pipeline import AvatarPipeline
from liveavatar.publish import PublishSettings, app, state
from liveavatar.worker import AvatarAssets, AvatarWorker

_WIDTH = 128
_HEIGHT = 128


class _PatternWorker(AvatarWorker):
    """Animated diagonal-gradient worker (CPU, no model files)."""

    def __init__(self) -> None:
        super().__init__(
            AvatarAssets(
                avatar_id="demo",
                data_dir="nonexistent",
                full_imgs_dir="nonexistent",
                coords_path="nonexistent",
                latents_path="nonexistent",
                mask_dir="nonexistent",
                mask_coords_path="nonexistent",
            ),
            target_fps=25,
            width=_WIDTH,
            height=_HEIGHT,
            batch_size=4,
        )
        self._t0 = time.perf_counter()

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        import numpy as np

        h, w = _HEIGHT, _WIDTH
        # Static background gradient (so the region encoder's background
        # hash stays stable) + an animated band inside the mouth rect.
        yy, xx = np.mgrid[0:h, 0:w]
        r = (xx * 255 // w) % 256
        g = (yy * 255 // h) % 256
        b = np.full((h, w), 90, np.uint8)
        t = (time.perf_counter() - self._t0) * 2.0
        band = (np.sin(yy / 6 + t) + 1) * 110
        mask = (xx >= _WIDTH // 4) & (xx < 3 * _WIDTH // 4) & (
            yy >= _HEIGHT // 4
        ) & (yy < _HEIGHT // 4 + _HEIGHT // 3)
        b = np.where(mask, band.astype(np.uint8), b)
        img = np.stack([r, g, b], axis=-1).astype(np.uint8)  # BGR-ish
        raw = img.tobytes()
        return [(raw, True) for _ in range(self.batch_size)]


class _FakeLease:
    def __init__(self, worker: AvatarWorker) -> None:
        self.worker = worker


class _DemoPool:
    """AvatarPool-compatible fake: one shared pattern worker."""

    def __init__(self) -> None:
        self._worker = _PatternWorker()

    @property
    def available_avatars(self) -> list[str]:
        return ["demo"]

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def acquire(self, session_id: str, avatar_id: str, **kwargs) -> _FakeLease:
        return _FakeLease(self._worker)

    async def release_async(self, session_id: str) -> bool:
        return True

    def stats(self) -> dict:
        return {"demo": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--codec",
        choices=("mjpeg", "region"),
        default="mjpeg",
        help="ws transport codec (region needs region.json; this demo "
        "writes one automatically for the pattern avatar)",
    )
    args = parser.parse_args()

    import os

    from liveavatar.region_codec import RegionSpec, write_region_json

    state.settings = PublishSettings(
        livekit_url="", livekit_api_key="", livekit_api_secret=""
    )
    state.settings.codec = args.codec
    state.pool_config = AvatarPoolConfig(avatar_data_root="nonexistent")
    if args.codec == "region":
        # The default avatar id falls back to "yongen" when nothing is
        # discoverable, so park the demo region spec where the factory
        # looks for it.
        avatar_dir = os.path.join("data", "avatars", "yongen")
        os.makedirs(avatar_dir, exist_ok=True)
        write_region_json(
            os.path.join(avatar_dir, "region.json"),
            RegionSpec(_WIDTH // 4, _HEIGHT // 4, _WIDTH // 2, _HEIGHT // 3),
        )
        state.pool_config = AvatarPoolConfig(avatar_data_root="data/avatars")
    from liveavatar.publish import _service_publisher_factory

    state.pipeline = AvatarPipeline(
        state.pool_config,
        pool=_DemoPool(),  # type: ignore[arg-type]
        publisher_factory=_service_publisher_factory,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

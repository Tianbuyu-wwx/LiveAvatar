# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Encoding hot-path micro-benchmarks (PERF-2) — CPU only.

Times the three encode-side hot paths of the self-developed video
transport, ``--frames N`` each (default 1000), after a 20-frame warmup:

- ``protocol_pack`` : VideoFrameHeader + pack_region_frame — the 26-byte
  header, patch table and payload validation on the wire path;
- ``region_encode`` : RegionFrameEncoder.encode steady state — full-frame
  background hash + mouth-region JPEG + payload pack (the ~10x-smaller
  common case);
- ``mjpeg_encode``  : MjpegFrameEncoder.encode — full-canvas JPEG (the
  v1 baseline codec and the region fallback).

Record-only in the nightly job (no pass/fail threshold): shared runners
are too noisy for micro-timing gates. The JSON report is archived as a
workflow artifact and compared against the committed baseline report
(docs/性能门禁基线_*.md) when hunting regressions.
"""

from __future__ import annotations

import argparse
import json
import time

from wsperf import _percentile, _synthetic_image

from liveavatar.region_codec import RegionFrameEncoder, RegionSpec
from liveavatar.video_protocol import (
    FLAG_KEYFRAME,
    Patch,
    VideoFrameHeader,
    pack_region_frame,
)
from liveavatar.worker import AvatarFrame
from liveavatar.ws_sink import MjpegFrameEncoder

_WARMUP_FRAMES = 20
_REGION_RECT = (156, 156, 200, 200)  # mouth-ish patch on the 512² canvas


def _bench(name: str, frames: int, durations_ms: list[float], out_sizes: list[int]) -> dict:
    ordered = sorted(durations_ms)
    return {
        "bench": name,
        "frames": frames,
        "total_ms": round(sum(durations_ms), 1),
        "mean_ms": round(sum(durations_ms) / frames, 3),
        "p50_ms": round(_percentile(ordered, 50), 3),
        "p95_ms": round(_percentile(ordered, 95), 3),
        "max_ms": round(ordered[-1], 3),
        "avg_out_bytes": round(sum(out_sizes) / frames, 1),
    }


def bench_protocol_pack(frame_img, frames: int, quality: int) -> dict:
    """pack_region_frame with a realistic region JPEG (re-encoded once)."""
    import cv2

    rx, ry, rw, rh = _REGION_RECT
    ok, buf = cv2.imencode(".jpg", frame_img[ry : ry + rh, rx : rx + rw],
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    jpeg = buf.tobytes()

    durations: list[float] = []
    sizes: list[int] = []
    for i in range(frames):
        header = VideoFrameHeader(
            flags=FLAG_KEYFRAME if i == 0 else 0,
            codec=1,
            quality=quality,
            seq=i % 2**16,
            epoch=0,
            pts_us=i * 40_000,
            width=frame_img.shape[1],
            height=frame_img.shape[0],
        )
        t0 = time.perf_counter()
        wire = pack_region_frame(header, [Patch(x=rx, y=ry, w=rw, h=rh, jpeg=jpeg)])
        durations.append((time.perf_counter() - t0) * 1000)
        sizes.append(len(wire))
    return _bench("protocol_pack", frames, durations, sizes)


def bench_region_encode(frame_bytes: bytes, width: int, height: int,
                        frames: int, quality: int) -> dict:
    """Steady-state RegionFrameEncoder: static background, moving mouth."""
    rx, ry, rw, rh = _REGION_RECT
    encoder = RegionFrameEncoder(RegionSpec(x=rx, y=ry, w=rw, h=rh))

    durations: list[float] = []
    sizes: list[int] = []
    for i in range(frames):
        # Mutate only the mouth rect so every frame after the first takes
        # the region-patch path (background hash stays stable).
        img = bytearray(frame_bytes)
        img[(ry * width + rx) * 3] = (i * 7) % 256
        frame = AvatarFrame(
            frame_data=bytes(img), pts_us=i * 40_000, epoch=0,
            width=width, height=height,
        )
        t0 = time.perf_counter()
        wire = encoder.encode(frame, keyframe=False, quality=quality)
        durations.append((time.perf_counter() - t0) * 1000)
        sizes.append(len(wire))
    return _bench("region_encode", frames, durations, sizes)


def bench_mjpeg_encode(frame_bytes: bytes, width: int, height: int,
                       frames: int, quality: int) -> dict:
    encoder = MjpegFrameEncoder()
    durations: list[float] = []
    sizes: list[int] = []
    for i in range(frames):
        frame = AvatarFrame(
            frame_data=frame_bytes, pts_us=i * 40_000, epoch=0,
            width=width, height=height,
        )
        t0 = time.perf_counter()
        wire = encoder.encode(frame, keyframe=True, quality=quality)
        durations.append((time.perf_counter() - t0) * 1000)
        sizes.append(len(wire))
    return _bench("mjpeg_encode", frames, durations, sizes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args(argv)

    import cv2

    img = _synthetic_image()
    height, width = img.shape[:2]
    frame_bytes = img.tobytes()

    benches = [
        bench_protocol_pack(img, args.frames, args.quality),
        bench_region_encode(frame_bytes, width, height, args.frames, args.quality),
        bench_mjpeg_encode(frame_bytes, width, height, args.frames, args.quality),
    ]
    report = {
        "mode": "micro_bench",
        "width": width,
        "height": height,
        "quality": args.quality,
        "cv2_version": cv2.__version__,
        "benches": benches,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

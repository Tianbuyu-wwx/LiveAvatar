# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Measurement client for the R2 self-developed video transport (M0 tool).

Pure CPU — no torch, no GPU, no server required for the synthetic mode.

Modes
-----
``--synthetic N --fps F [--codec 0|1] [--quality Q]``
    Generate a synthetic 512x512 stream in-process, encode it with cv2
    when available (realistic payload sizes; deterministic pseudo payload
    otherwise), and measure throughput / inter-arrival statistics. This
    is the M0 baseline mode and also the tool self-test path.

``--url ws://host/... [--api-key K] [--duration S]``
    Connect to a live ``/v1/sessions/{sid}/video`` endpoint (M1+) and
    measure the real stream. Requires the optional ``websockets``
    package (``uv pip install websockets``) — tool-only dependency.

The JSON report is printed to stdout; the statistics logic lives in
:class:`StreamStats` so tests can drive it without any I/O.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

from liveavatar.video_protocol import (
    CODEC_REGION_DELTA,
    FLAG_EPOCH_BOUNDARY,
    FLAG_KEYFRAME,
    Patch,
    VideoFrameHeader,
    pack_region_frame,
    pack_video_frame,
    unpack_video_frame,
)

# Typical synthetic region: a 200x200 patch near the center (lip area).
_REGION_RECT = (156, 156, 200, 200)
_FALLBACK_FULL_BYTES = 35_000  # typical 512^2 JPEG q80 size
_FALLBACK_PATCH_BYTES = 5_000  # typical 200^2 JPEG q80 size


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Percentile of an ascending-sorted list (nearest-rank)."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, round(pct / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


@dataclass
class StreamStats:
    """Frame-level stream statistics collector (no I/O, unit-testable)."""

    start_ts: float | None = None
    first_frame_ts: float | None = None
    last_frame_ts: float | None = None
    frames: int = 0
    wire_bytes: int = 0
    payload_bytes: int = 0
    keyframes: int = 0
    epoch_boundaries: int = 0
    max_epoch_seen: int = -1
    stale_dropped: int = 0
    gaps_ms: list[float] = field(default_factory=list)
    boundary_gaps_ms: list[float] = field(default_factory=list)

    def feed(self, header: VideoFrameHeader, wire_size: int, ts: float) -> bool:
        """Account for one received frame; returns False when stale-dropped."""
        if header.epoch < self.max_epoch_seen:
            self.stale_dropped += 1
            return False
        if self.start_ts is None:
            self.start_ts = ts
        if self.first_frame_ts is None:
            self.first_frame_ts = ts
        else:
            assert self.last_frame_ts is not None  # set alongside first_frame_ts
            gap = (ts - self.last_frame_ts) * 1000.0
            self.gaps_ms.append(gap)
            if header.flags & FLAG_EPOCH_BOUNDARY:
                self.boundary_gaps_ms.append(gap)
        self.last_frame_ts = ts
        self.frames += 1
        self.wire_bytes += wire_size
        self.keyframes += int(bool(header.flags & FLAG_KEYFRAME))
        self.epoch_boundaries += int(bool(header.flags & FLAG_EPOCH_BOUNDARY))
        self.max_epoch_seen = max(self.max_epoch_seen, header.epoch)
        return True

    def report(self, *, codec: int, fps_nominal: float | None = None) -> dict:
        """Snapshot the statistics as a JSON-ready dict."""
        duration = 0.0
        if self.last_frame_ts is not None and self.first_frame_ts is not None:
            duration = self.last_frame_ts - self.first_frame_ts
        gaps = sorted(self.gaps_ms)
        boundary_gaps = sorted(self.boundary_gaps_ms)
        return {
            "codec": codec,
            "frames": self.frames,
            "stale_dropped": self.stale_dropped,
            "duration_s": round(duration, 3),
            "fps": round(self.frames / duration, 2) if duration > 0 else 0.0,
            "fps_nominal": fps_nominal,
            "wire_kbps": round(self.wire_bytes * 8 / 1000 / duration, 1)
            if duration > 0
            else 0.0,
            "wire_mb_total": round(self.wire_bytes / 1e6, 2),
            "avg_frame_kb": round(self.wire_bytes / self.frames / 1000, 1)
            if self.frames
            else 0.0,
            "gap_ms_p50": round(_percentile(gaps, 50), 2),
            "gap_ms_p95": round(_percentile(gaps, 95), 2),
            "gap_ms_max": round(gaps[-1], 2) if gaps else 0.0,
            "first_frame_ms": round(
                ((self.first_frame_ts or 0) - (self.start_ts or 0)) * 1000, 2
            ),
            "keyframes": self.keyframes,
            "epoch_boundaries": self.epoch_boundaries,
            "boundary_gap_ms_p50": round(_percentile(boundary_gaps, 50), 2),
            "max_epoch_seen": self.max_epoch_seen,
        }


# ─────────────────────────────────────────────────────── synthetic mode


def _synthetic_image(width: int = 512, height: int = 512):
    """Deterministic BGR test image (gradient + face-ish shapes)."""
    import cv2
    import numpy as np

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(30, 90, width, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(60, 120, height, dtype=np.uint8)[:, None]
    cv2.circle(img, (width // 2, height // 2 - 40), 60, (80, 120, 200), -1)
    cv2.rectangle(
        img,
        (width // 2 - 50, height // 2 + 40),
        (width // 2 + 50, height // 2 + 90),
        (40, 60, 160),
        -1,
    )
    return img


def _jpeg_or_fallback(img, quality: int) -> bytes:
    """Encode via cv2 when available; else a deterministic byte blob."""
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return buf.tobytes()
    except ImportError:
        pass
    return bytes(range(256)) * (_FALLBACK_FULL_BYTES // 256 + 1)


def run_synthetic(
    n_frames: int = 500,
    fps: float = 25.0,
    codec: int = CODEC_REGION_DELTA,
    quality: int = 80,
    width: int = 512,
    height: int = 512,
    pace: bool = True,
) -> dict:
    """Generate, encode and measure a synthetic stream. Returns the report."""
    img = _synthetic_image(width, height)
    rx, ry, rw, rh = _REGION_RECT
    patch_img = img[ry : ry + rh, rx : rx + rw]

    stats = StreamStats()
    encode_ms: list[float] = []
    seq = 0
    interval = 1.0 / fps
    t0 = time.monotonic()

    for i in range(n_frames):
        if pace:
            target = t0 + i * interval
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        te = time.perf_counter()

        is_key = (i % max(1, int(fps))) == 0  # ≥1 keyframe per second
        pts_us = int(i * 1_000_000 / fps)
        header = VideoFrameHeader(
            flags=FLAG_KEYFRAME if is_key else 0,
            codec=codec,
            quality=quality,
            seq=seq,
            epoch=0,
            pts_us=pts_us,
            width=width,
            height=height,
        )
        seq = (seq + 1) % 2**16

        if codec == CODEC_REGION_DELTA:
            patches: list[Patch] = []
            if is_key:
                patches.append(
                    Patch(x=0, y=0, w=width, h=height, jpeg=_jpeg_or_fallback(img, quality))
                )
            else:
                patches.append(
                    Patch(x=rx, y=ry, w=rw, h=rh, jpeg=_jpeg_or_fallback(patch_img, quality))
                )
            wire = pack_region_frame(header, patches)
        else:
            wire = pack_video_frame(header, _jpeg_or_fallback(img, quality))

        encode_ms.append((time.perf_counter() - te) * 1000)

        # Receive-side accounting: unpack back and feed stats (true roundtrip).
        rh2, _payload = unpack_video_frame(wire)
        stats.feed(rh2, len(wire), time.monotonic())

    enc = sorted(encode_ms)
    report = stats.report(codec=codec, fps_nominal=fps)
    report["encode_ms_p50"] = round(_percentile(enc, 50), 2)
    report["encode_ms_p95"] = round(_percentile(enc, 95), 2)
    report["mode"] = "synthetic"
    return report


# ─────────────────────────────────────────────────────────────── url mode


async def run_url(url: str, api_key: str = "", duration_s: float = 10.0) -> dict:
    """Connect to a live /video endpoint and measure the real stream (M1+)."""
    try:
        import websockets  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "url mode requires the optional 'websockets' package: "
            "uv pip install websockets"
        ) from exc

    headers = {"X-API-Key": api_key} if api_key else None
    stats = StreamStats()
    ready: dict | None = None
    async with websockets.connect(url, additional_headers=headers, open_timeout=10) as ws:  # type: ignore[attr-defined]
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s:
            remaining = max(0.1, duration_s - (time.monotonic() - t0))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                break
            ts = time.monotonic()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "ready":
                    ready = data
                continue
            header, _payload = unpack_video_frame(msg)
            stats.feed(header, len(msg), ts)

    report = stats.report(
        codec=(ready or {}).get("codec", -1), fps_nominal=(ready or {}).get("target_fps")
    )
    report["mode"] = "url"
    report["url"] = url
    return report


# ───────────────────────────────────────────────────────────────────── CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", type=int, metavar="N", help="synthetic mode: frame count")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--codec", type=int, default=CODEC_REGION_DELTA, choices=(0, 1))
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument(
        "--url", help="url mode: ws://host/v1/sessions/<sid>/video (M1+)"
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--duration", type=float, default=10.0, help="url mode seconds")
    args = parser.parse_args(argv)

    if args.url:
        import asyncio

        report = asyncio.run(run_url(args.url, args.api_key, args.duration))
    elif args.synthetic:
        report = run_synthetic(
            n_frames=args.synthetic,
            fps=args.fps,
            codec=args.codec,
            quality=args.quality,
        )
    else:
        parser.error("choose one of --synthetic N or --url ws://...")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

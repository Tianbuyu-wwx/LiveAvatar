# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""End-to-end benchmark for the self-developed WS transport (R2 M6).

Pure CPU: starts the real publish service in-process (uvicorn on
localhost, synthetic pattern worker from ``demo_local``), then drives it
over **real sockets** — PCM in via ``WS /audio``, wire frames out via
``WS /video``. Exercises the full production code path: adapter →
worker → WebSocketSink → /video endpoint, across three event loops.

Measured per codec (JSON to stdout):

- ``startup_latency_ms``   first PCM push → first video frame arrival
- ``fps`` / ``wire_kbps``  delivered rate and bandwidth (wsperf stats)
- ``gap_ms_p50/p95``       inter-arrival jitter
- ``drift_ms_p50/p95``     arrival deviation vs the ideal 40 ms cadence
- ``interrupt_latency_ms`` epoch bump → first ``epoch_boundary`` frame
- ``reconnect_keyframe_ms`` video WS reconnect → first keyframe

Usage::

    python scripts/e2e_bench.py --codec mjpeg --seconds 10
    python scripts/e2e_bench.py --codec region --seconds 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time

import httpx2 as httpx  # httpx2 is the dev-extra HTTP client (no httpx)
import uvicorn
from websockets.asyncio.client import connect as ws_connect
from wsperf import StreamStats, _percentile

from liveavatar.video_protocol import (
    FLAG_EPOCH_BOUNDARY,
    FLAG_KEYFRAME,
    unpack_video_frame,
)

_HOST = "127.0.0.1"
_CHUNK_BYTES = 3200  # 100 ms of silence at 16 kHz S16LE mono
_CHUNK_S = 0.1


# ─────────────────────────────────────────────────────── service bootstrap


def _start_server(codec: str) -> tuple[uvicorn.Server, threading.Thread, int]:
    """Boot the real service (CPU synthetic worker) on an ephemeral port.

    Mirrors ``demo_local.main`` so the bench drives the exact demo path.
    """
    from demo_local import _DemoPool

    from liveavatar.config import AvatarPoolConfig
    from liveavatar.pipeline import AvatarPipeline
    from liveavatar.publish import (
        PublishSettings,
        _service_publisher_factory,
        app,
        state,
    )
    from liveavatar.region_codec import RegionSpec, write_region_json

    state.settings = PublishSettings()
    state.settings.codec = codec
    if codec == "region":
        avatar_dir = os.path.join("data", "avatars", "yongen")
        os.makedirs(avatar_dir, exist_ok=True)
        write_region_json(
            os.path.join(avatar_dir, "region.json"),
            RegionSpec(32, 32, 64, 42),  # matches demo_local's 128x128 pattern
        )
        state.pool_config = AvatarPoolConfig(avatar_data_root="data/avatars")
    else:
        state.pool_config = AvatarPoolConfig(avatar_data_root="nonexistent")
    state.pipeline = AvatarPipeline(
        state.pool_config,
        pool=_DemoPool(),  # type: ignore[arg-type]
        publisher_factory=_service_publisher_factory,
    )

    config = uvicorn.Config(app, host=_HOST, port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:  # pragma: no cover - boot wait
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, port


# ────────────────────────────────────────────────────────── measurement


async def _read_video(
    url: str, stats: StreamStats, obs: dict, stop: asyncio.Event
) -> None:
    """Consume /video frames into ``stats`` until ``stop`` is set."""
    async with ws_connect(url) as video:
        ready = json.loads(await video.recv())
        if obs.get("ready") is None:
            obs["ready"] = ready
        while not stop.is_set():
            try:
                msg = await asyncio.wait_for(video.recv(), timeout=0.25)
            except (asyncio.TimeoutError, TimeoutError):
                continue
            if isinstance(msg, str):
                continue
            ts = time.perf_counter()
            header, _ = unpack_video_frame(msg)
            stats.feed(header, len(msg), ts)
            obs["arrivals"].append(ts)
            if header.flags & FLAG_EPOCH_BOUNDARY:
                obs["boundary_arrivals"].append(ts)
            if header.flags & FLAG_KEYFRAME:
                obs["keyframe_arrivals"].append(ts)


async def _stream_audio(url: str, seconds: int, obs: dict) -> None:
    """Push silence PCM in real time; bump the epoch at 60% through."""
    async with ws_connect(url) as audio:
        await audio.send(json.dumps({"type": "epoch", "epoch": 1}))
        silence = b"\x00" * _CHUNK_BYTES
        t0 = time.perf_counter()
        obs["first_pcm_at"] = t0
        t_interrupt = t0 + seconds * 0.6
        n = 0
        while True:
            now = time.perf_counter()
            if now >= t0 + seconds:
                break
            if obs.get("interrupt_sent") is None and now >= t_interrupt:
                await audio.send(json.dumps({"type": "epoch", "epoch": 2}))
                obs["interrupt_sent"] = time.perf_counter()
            await audio.send(silence)
            n += 1
            target = t0 + n * _CHUNK_S
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
        obs["audio_done"] = time.perf_counter()
        await audio.send(json.dumps({"type": "stop"}))


async def _reconnect_probe(url: str, obs: dict) -> None:
    """Close-and-reopen /video; measure time to the first keyframe."""
    t_open = time.perf_counter()
    async with ws_connect(url) as video:
        await video.recv()  # ready
        deadline = t_open + 5.0
        while time.perf_counter() < deadline:
            try:
                msg = await asyncio.wait_for(video.recv(), timeout=0.25)
            except (asyncio.TimeoutError, TimeoutError):
                continue
            if isinstance(msg, str):
                continue
            header, _ = unpack_video_frame(msg)
            if header.flags & FLAG_KEYFRAME:
                obs["reconnect_keyframe_ms"] = (time.perf_counter() - t_open) * 1000
                return


def _p50(vals: list[float]) -> float:
    return _percentile(sorted(vals), 50) if vals else 0.0


def _p95(vals: list[float]) -> float:
    return _percentile(sorted(vals), 95) if vals else 0.0


async def _run_codec(codec: str, seconds: int) -> dict:
    server, _thread, port = _start_server(codec)
    base = f"ws://{_HOST}:{port}"
    stats = StreamStats()
    obs: dict = {"arrivals": [], "boundary_arrivals": [], "keyframe_arrivals": []}
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"http://{_HOST}:{port}/v1/sessions", json={"avatar_id": "demo"}
            )
            resp.raise_for_status()
            sid = resp.json()["session_id"]

            video_url = f"{base}/v1/sessions/{sid}/video"
            audio_url = f"{base}/v1/sessions/{sid}/audio"

            audio_task = asyncio.create_task(_stream_audio(audio_url, seconds, obs))
            stop1 = asyncio.Event()
            reader1 = asyncio.create_task(_read_video(video_url, stats, obs, stop1))

            # Wait for the interrupt (epoch bump) plus ~1 s of post-interrupt
            # frames — enough for the latency sample and steady-state stats.
            while obs.get("interrupt_sent") is None and not audio_task.done():
                await asyncio.sleep(0.05)
            await asyncio.sleep(1.0)

            # Reconnect probe: kill the reader socket, pause (simulated
            # outage), reopen and time the first keyframe. Audio keeps
            # streaming so the new socket receives frames immediately.
            stop1.set()
            reader1.cancel()
            try:
                await reader1
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            await asyncio.sleep(0.5)  # simulated outage
            await _reconnect_probe(video_url, obs)

            await audio_task
            await http.delete(f"http://{_HOST}:{port}/v1/sessions/{sid}")

        await asyncio.sleep(0.3)

        report = stats.report(codec=codec, fps_nominal=25.0)
        report["mode"] = "e2e_real_sockets"
        report["codec_setting"] = codec
        report["ready"] = obs.get("ready")

        arrivals = obs["arrivals"]
        if arrivals and obs.get("first_pcm_at") is not None:
            report["startup_latency_ms"] = round(
                (arrivals[0] - obs["first_pcm_at"]) * 1000, 1
            )
        if len(arrivals) > 8:
            first = arrivals[0]
            drift = [
                (a - (first + i * 0.04)) * 1000 for i, a in enumerate(arrivals)
            ]
            report["drift_ms_p50"] = round(_p50(drift), 1)
            report["drift_ms_p95"] = round(_p95(drift), 1)
        boundary_after = [
            ts
            for ts in obs["boundary_arrivals"]
            if obs.get("interrupt_sent") is not None and ts >= obs["interrupt_sent"]
        ]
        if boundary_after:
            report["interrupt_latency_ms"] = round(
                (boundary_after[0] - obs["interrupt_sent"]) * 1000, 1
            )
        if obs.get("reconnect_keyframe_ms") is not None:
            report["reconnect_keyframe_ms"] = round(obs["reconnect_keyframe_ms"], 1)
        report["epoch_boundaries_total"] = len(obs["boundary_arrivals"])
        return report
    finally:
        server.should_exit = True
        await asyncio.sleep(0.2)


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", choices=("mjpeg", "region"), default="mjpeg")
    parser.add_argument("--seconds", type=int, default=10)
    args = parser.parse_args(argv)
    report = await _run_codec(args.codec, args.seconds)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main_async()))

"""Capacity report scaffold for concurrent sessions (M-C, CPU only).

Boots the real publish service (uvicorn + synthetic pattern worker), then
drives ``--sessions N`` **simultaneous** sessions over real sockets: each
session pushes real-time PCM and consumes video frames exactly like the
browser demo, with one mid-stream barge-in per session.

Emitted per session (JSON): delivered fps, wire kbps, inter-arrival gap
p50/p95, drift p50/p95, startup and interrupt latency, dropped frames.
Aggregate: total egress Mbps and the M-C gate check (interrupt p95 ≤ 90 ms,
per-session fps ≥ 20, no session starvation).

Usage::

    python scripts/capacity_report.py --sessions 3 --seconds 10
    python scripts/capacity_report.py --sessions 3 --seconds 10 \
        --markdown docs/容量报告_2026-08-31.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time

import httpx2 as httpx
import uvicorn
from websockets.asyncio.client import connect as ws_connect
from wsperf import StreamStats, _percentile

from demo_local import _PatternWorker

from liveavatar.config import AvatarPoolConfig
from liveavatar.pipeline import AvatarPipeline
from liveavatar.publish import (
    PublishSettings,
    _service_publisher_factory,
    app,
    state,
)
from liveavatar.video_protocol import (
    FLAG_EPOCH_BOUNDARY,
    unpack_video_frame,
)

_HOST = "127.0.0.1"
_CHUNK_BYTES = 3200  # 100 ms of silence at 16 kHz S16LE mono
_CHUNK_S = 0.1

# M-C gates: per-session delivered fps floor and the transport interruption
# budget (see project constraint: end-to-end interruption delay ≤ 90 ms).
_FPS_FLOOR = 20.0
_INTERRUPT_BUDGET_MS = 90.0


# ─────────────────────────────────────────────────────── service bootstrap


class _PerLeasePool:
    """Demo pool that hands each lease its own pattern worker.

    Mirrors the real lease pool's exclusive worker ownership so N
    concurrent sessions never share an event-loop-bound lock.
    """

    @property
    def available_avatars(self) -> list[str]:
        return ["demo"]

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def acquire(self, session_id: str, avatar_id: str, **kwargs):
        class _Lease:
            worker = _PatternWorker()

        return _Lease()

    async def release_async(self, session_id: str) -> bool:
        return True

    def stats(self) -> dict:
        return {"per_lease": True}


def _start_server() -> tuple[uvicorn.Server, threading.Thread, int]:
    state.settings = PublishSettings()
    state.settings.codec = "mjpeg"
    state.pool_config = AvatarPoolConfig(avatar_data_root="nonexistent")
    state.pipeline = AvatarPipeline(
        state.pool_config,
        pool=_PerLeasePool(),
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


# ────────────────────────────────────────────────────────── per-session run


async def _read_video(url: str, stats: StreamStats, obs: dict, stop: asyncio.Event) -> None:
    async with ws_connect(url) as video:
        obs["ready"] = json.loads(await video.recv())
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


async def _stream_audio(url: str, seconds: int, obs: dict) -> None:
    async with ws_connect(url) as audio:
        await audio.send(json.dumps({"type": "epoch", "epoch": 1}))
        silence = b"\x00" * _CHUNK_BYTES
        t0 = time.perf_counter()
        obs["first_pcm_at"] = t0
        t_interrupt = t0 + seconds * 0.6
        n = 0
        while time.perf_counter() < t0 + seconds:
            now = time.perf_counter()
            if obs.get("interrupt_sent") is None and now >= t_interrupt:
                await audio.send(json.dumps({"type": "epoch", "epoch": 2}))
                obs["interrupt_sent"] = time.perf_counter()
            await audio.send(silence)
            n += 1
            target = t0 + n * _CHUNK_S
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
        await audio.send(json.dumps({"type": "stop"}))


async def _run_session(http, base: str, ws_base: str, seconds: int) -> dict:
    resp = await http.post(f"{base}/v1/sessions", json={"avatar_id": "demo"})
    resp.raise_for_status()
    sid = resp.json()["session_id"]

    stats = StreamStats()
    obs: dict = {"arrivals": [], "boundary_arrivals": []}
    audio_task = asyncio.create_task(
        _stream_audio(f"{ws_base}/v1/sessions/{sid}/audio", seconds, obs)
    )
    stop = asyncio.Event()
    reader = asyncio.create_task(
        _read_video(f"{ws_base}/v1/sessions/{sid}/video", stats, obs, stop)
    )
    await audio_task
    await asyncio.sleep(0.3)  # let the tail frames drain
    stop.set()
    reader.cancel()
    try:
        await reader
    except asyncio.CancelledError:
        pass
    await http.delete(f"{base}/v1/sessions/{sid}")

    report = stats.report(codec="mjpeg", fps_nominal=25.0)
    report["session_id"] = sid
    arrivals = obs["arrivals"]
    if arrivals and obs.get("first_pcm_at") is not None:
        report["startup_latency_ms"] = round(
            (arrivals[0] - obs["first_pcm_at"]) * 1000, 1
        )
    if len(arrivals) > 8:
        first = arrivals[0]
        drift = [(a - (first + i * 0.04)) * 1000 for i, a in enumerate(arrivals)]
        report["drift_ms_p50"] = round(_percentile(sorted(drift), 50), 1)
        report["drift_ms_p95"] = round(_percentile(sorted(drift), 95), 1)
    boundary_after = [
        ts
        for ts in obs["boundary_arrivals"]
        if obs.get("interrupt_sent") is not None and ts >= obs["interrupt_sent"]
    ]
    if boundary_after:
        report["interrupt_latency_ms"] = round(
            (boundary_after[0] - obs["interrupt_sent"]) * 1000, 1
        )
    return report


def _gates(sessions: list[dict]) -> dict:
    interrupt = [
        s["interrupt_latency_ms"]
        for s in sessions
        if s.get("interrupt_latency_ms") is not None
    ]
    fps_vals = [s["fps"] for s in sessions]
    interrupt_p95 = _percentile(sorted(interrupt), 95) if interrupt else None
    return {
        "fps_floor": _FPS_FLOOR,
        "fps_min": round(min(fps_vals), 1) if fps_vals else 0.0,
        "fps_ok": bool(fps_vals) and min(fps_vals) >= _FPS_FLOOR,
        "interrupt_budget_ms": _INTERRUPT_BUDGET_MS,
        "interrupt_p95_ms": interrupt_p95,
        "interrupt_ok": (
            interrupt_p95 is not None and interrupt_p95 <= _INTERRUPT_BUDGET_MS
        ),
        "starved_sessions": sum(
            1 for s in sessions if s.get("frames", 0) == 0
        ),
    }


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--markdown", type=str, default=None,
                        help="also write a markdown report to this path")
    args = parser.parse_args(argv)

    server, _thread, port = _start_server()
    base = f"http://{_HOST}:{port}"
    ws_base = f"ws://{_HOST}:{port}"
    try:
        async with httpx.AsyncClient() as http:
            # Create all sessions first so they truly overlap in time.
            sessions = await asyncio.gather(
                *(
                    _run_session(http, base, ws_base, args.seconds)
                    for _ in range(args.sessions)
                )
            )
        total_kbps = sum(s.get("wire_kbps", 0.0) for s in sessions)
        report = {
            "mode": "capacity_concurrent",
            "codec_setting": "mjpeg",
            "sessions": args.sessions,
            "seconds": args.seconds,
            "total_egress_mbps": round(total_kbps / 1000, 2),
            "per_session": sessions,
            "gates": _gates(list(sessions)),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.markdown:
            _write_markdown(args.markdown, report)
        gates = report["gates"]
        return 0 if (gates["fps_ok"] and gates["interrupt_ok"]) else 1
    finally:
        server.should_exit = True
        await asyncio.sleep(0.2)


def _write_markdown(path: str, report: dict) -> None:
    lines = [
        "# 容量报告（自研 WS 传输，CPU 合成 worker）",
        "",
        f"- 并发会话数：{report['sessions']}",
        f"- 每会话时长：{report['seconds']} s",
        f"- 总出口带宽：{report['total_egress_mbps']} Mbps",
        "",
        "| 会话 | fps | 码率 kbps | gap p95 ms | drift p95 ms | 启动 ms | 打断 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in report["per_session"]:
        lines.append(
            f"| {s['session_id']} | {s.get('fps', 0):.1f} | "
            f"{s.get('wire_kbps', 0):.0f} | {s.get('gap_ms_p95', '-')} | "
            f"{s.get('drift_ms_p95', '-')} | "
            f"{s.get('startup_latency_ms', '-')} | "
            f"{s.get('interrupt_latency_ms', '-')} |"
        )
    g = report["gates"]
    lines += [
        "",
        "## 门禁",
        "",
        f"- 单会话 fps ≥ {g['fps_floor']}：实测最低 {g['fps_min']} → "
        f"{'通过' if g['fps_ok'] else '未通过'}",
        f"- 打断延迟 ≤ {g['interrupt_budget_ms']} ms（p95）：实测 "
        f"{g['interrupt_p95_ms']} ms → {'通过' if g['interrupt_ok'] else '未通过'}",
        f"- 零饥饿会话：{g['starved_sessions']} 个空会话 → "
        f"{'通过' if g['starved_sessions'] == 0 else '未通过'}",
        "",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main_async()))

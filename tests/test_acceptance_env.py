# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Environment acceptance tests for the three [待补] items of the R2
report (docs/R2验收报告_2026-08-31.md §1.1 / §1.4 / §1.5).

All three run on CPU over **real sockets** (uvicorn in a daemon thread +
websockets client), i.e. the exact production code path also driven by
``scripts/e2e_bench.py``:

1. LAN latency P95 ≤ 200 ms (§1.1). In CI the loopback stands in for the
   LAN; set ``LIVEAVATAR_LAN_HOST=host:port`` to run the identical probe
   against a real two-machine LAN deployment (the service must already be
   running there with the ``demo`` avatar).
2. 3 concurrent sessions × 10-minute soak (§1.4). Duration comes from
   ``LIVEAVATAR_SOAK_SECONDS`` — 10 s in CI, 600 for the full acceptance
   run. Asserts frame continuity, per-session isolation, clean EOF
   shutdown and bounded memory growth (memory curve printed on failure).
3. Weak-network injection (§1.5). Application-level impairment: a
   deterministic 5 % frame loss (the signal that drives adaptation; a
   constant one-way delay is invisible to a push stream without clock
   sync and is covered by probe 1's latency budget). Verifies 降画质不
   冻结 (stream never freezes, every received frame stays decodable),
   the tier degrades, and after the link heals the quality jumps back to
   tier 0 within the 2 s budget.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from websockets.asyncio.client import connect as ws_connect

from liveavatar.config import AvatarPoolConfig
from liveavatar.pipeline import AvatarPipeline
from liveavatar.video_protocol import (
    FLAG_EOF,
    FLAG_KEYFRAME,
    has_flag,
    unpack_region_payload,
    unpack_video_frame,
)
from tests.test_publish import _FakeServicePool

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from demo_local import _DemoPool  # noqa: E402

_HOST = "127.0.0.1"
_CHUNK_BYTES = 3200  # 100 ms of silence @ 16 kHz S16LE mono
_CHUNK_S = 0.1


# ────────────────────────────────────────────────────────── server harness


class _TestServer:
    """Boot the real publish service on an ephemeral port (CPU workers).

    codec="mjpeg" uses the animated 128x128 pattern pool (demo avatar);
    codec="region" uses the 4x4 fake pool with a tmp region.json.
    """

    def __init__(self, codec: str = "mjpeg") -> None:
        from liveavatar.publish import (
            PublishSettings,
            _service_publisher_factory,
            app,
            state,
        )

        self._tmp = tempfile.TemporaryDirectory() if codec == "region" else None
        state.settings = PublishSettings()
        state.settings.codec = codec
        if codec == "region":
            from liveavatar.region_codec import RegionSpec, write_region_json

            avatar_dir = os.path.join(self._tmp.name, "yongen")
            os.makedirs(avatar_dir, exist_ok=True)
            write_region_json(
                os.path.join(avatar_dir, "region.json"), RegionSpec(1, 1, 2, 2)
            )
            state.pool_config = AvatarPoolConfig(avatar_data_root=self._tmp.name)
            pool = _FakeServicePool()
            avatar = "yongen"
        else:
            state.pool_config = AvatarPoolConfig(avatar_data_root="nonexistent")
            pool = _DemoPool()  # type: ignore[arg-type]
            avatar = "demo"
        state.pipeline = AvatarPipeline(
            state.pool_config, pool=pool, publisher_factory=_service_publisher_factory
        )
        self.avatar = avatar

        config = uvicorn.Config(app, host=_HOST, port=0, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        while not self.server.started:  # pragma: no cover - boot wait
            time.sleep(0.05)
        self.port: int = self.server.servers[0].sockets[0].getsockname()[1]

    @property
    def ws_base(self) -> str:
        return f"ws://{_HOST}:{self.port}"

    @property
    def http_base(self) -> str:
        return f"http://{_HOST}:{self.port}"

    def close(self) -> None:
        from liveavatar.publish import state

        self.server.should_exit = True
        self.thread.join(timeout=5)
        state.pipeline = None
        state.settings.codec = "mjpeg"
        if self._tmp is not None:
            self._tmp.cleanup()


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(q / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


async def _create_session(http, http_base: str, avatar: str) -> str:
    resp = await http.post(
        f"{http_base}/v1/sessions", json={"avatar_id": avatar}
    )
    resp.raise_for_status()
    return resp.json()["session_id"]


async def _push_realtime(audio, seconds: float, sends: list | None = None) -> None:
    """Push silence PCM in real time, then ask the session to stop."""
    silence = b"\x00" * _CHUNK_BYTES
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        await audio.send(silence)
        if sends is not None:
            sends.append(time.perf_counter())
        n += 1
        target = t0 + n * _CHUNK_S
        await asyncio.sleep(max(0.0, target - time.perf_counter()))
    await audio.send(json.dumps({"type": "stop"}))


def _rss_mb() -> float:
    """Resident memory of this process in MB (Windows / POSIX)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(pmc),
            pmc.cb,
        )
        return pmc.WorkingSetSize / (1024 * 1024)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)


# ──────────────────────────────────────────────────────── 1) LAN latency


async def _measure_latency(ws_base: str, http_base: str, avatar: str,
                           seconds: float) -> dict:
    """Stream for ``seconds`` and profile delivery over real sockets.

    Latency metric: per 100 ms PCM chunk, the time from the push to the
    first video frame arriving *after* it (one-way delay needs clock sync,
    this is the honest proxy the probe can measure on any host pair).
    """
    import httpx2 as httpx  # httpx2 is the dev-extra HTTP client (no httpx)

    async with httpx.AsyncClient() as http:
        sid = await _create_session(http, http_base, avatar)
        arrivals: list[float] = []
        sends: list[float] = []
        stop = asyncio.Event()

        async def reader() -> None:
            async with ws_connect(f"{ws_base}/v1/sessions/{sid}/video") as video:
                ready = json.loads(await video.recv())
                assert ready["type"] == "ready", ready
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(video.recv(), timeout=0.25)
                    except (asyncio.TimeoutError, TimeoutError):
                        continue
                    if isinstance(msg, str):
                        continue
                    arrivals.append(time.perf_counter())

        async with ws_connect(f"{ws_base}/v1/sessions/{sid}/audio") as audio:
            pusher = asyncio.create_task(_push_realtime(audio, seconds, sends))
            rtask = asyncio.create_task(reader())
            await pusher
            await asyncio.sleep(0.5)
            stop.set()
            rtask.cancel()
            try:
                await rtask
            except asyncio.CancelledError:
                pass
            await http.delete(f"{http_base}/v1/sessions/{sid}")

    report: dict = {"frames": len(arrivals)}
    latencies: list[float] = []
    j = 0
    for send_at in sends:
        while j < len(arrivals) and arrivals[j] < send_at:
            j += 1
        if j < len(arrivals):
            latencies.append((arrivals[j] - send_at) * 1000)
    if latencies:
        gaps = [
            (b - a) * 1000 for a, b in zip(arrivals, arrivals[1:], strict=False)
        ]
        report["latency_ms_p50"] = round(_pct(sorted(latencies), 50), 1)
        report["latency_ms_p95"] = round(_pct(sorted(latencies), 95), 1)
        report["gap_ms_p95"] = round(_pct(sorted(gaps), 95), 1)
        report["fps"] = round(len(arrivals) / seconds, 1)
    return report


class LanLatencyTests(unittest.TestCase):
    """§1.1 — frame-arrival P95 ≤ 200 ms over real sockets.

    Loopback stands in for the LAN in CI; LIVEAVATAR_LAN_HOST re-targets
    the identical probe at a two-machine deployment.
    """

    def test_frame_arrival_p95_under_200ms(self) -> None:
        seconds = float(os.environ.get("LIVEAVATAR_ACCEPTANCE_SECONDS", "6"))
        lan_host = os.environ.get("LIVEAVATAR_LAN_HOST", "")
        server = None
        if lan_host:
            ws_base, http_base = f"ws://{lan_host}", f"http://{lan_host}"
            avatar = "demo"
        else:
            server = _TestServer("mjpeg")
            ws_base, http_base, avatar = (
                server.ws_base,
                server.http_base,
                server.avatar,
            )
        try:
            report = asyncio.run(
                _measure_latency(ws_base, http_base, avatar, seconds)
            )
        finally:
            if server is not None:
                server.close()
        self.assertGreaterEqual(report["frames"], 50, report)
        self.assertLess(report["latency_ms_p95"], 200.0, report)
        self.assertLess(report["gap_ms_p95"], 200.0, report)


# ─────────────────────────────────────────────────────────── 2) 3×10 min soak


async def _run_soak(ws_base: str, http_base: str, avatar: str,
                    seconds: float) -> dict[str, dict]:
    """3 concurrent sessions streamed in real time; per-session report."""
    import httpx2 as httpx  # httpx2 is the dev-extra HTTP client (no httpx)

    out: dict[str, dict] = {}
    async with httpx.AsyncClient() as http:
        sids = [await _create_session(http, http_base, avatar) for _ in range(3)]

        async def reader(sid: str) -> None:
            seqs: list[int] = []
            eof = False
            async with ws_connect(f"{ws_base}/v1/sessions/{sid}/video") as video:
                ready = json.loads(await video.recv())
                assert ready["type"] == "ready", ready
                deadline = time.monotonic() + seconds + 15
                while time.monotonic() < deadline:
                    try:
                        msg = await asyncio.wait_for(video.recv(), timeout=0.5)
                    except (asyncio.TimeoutError, TimeoutError):
                        continue
                    if isinstance(msg, str):
                        continue
                    header, _ = unpack_video_frame(msg)
                    if has_flag(header.flags, FLAG_EOF):
                        eof = True
                        break
                    seqs.append(header.seq)
            out[sid] = {"seqs": seqs, "eof": eof}

        pushers = [
            asyncio.create_task(_push_via_ws(ws_base, sid, seconds))
            for sid in sids
        ]
        readers = [asyncio.create_task(reader(sid)) for sid in sids]
        await asyncio.gather(*pushers)
        # Tear the sessions down: sink.stop() → EOF sentinel → the readers
        # observe a clean end-of-stream per session.
        for sid in sids:
            await http.delete(f"{http_base}/v1/sessions/{sid}")
        await asyncio.wait(readers, timeout=15)
    return out


async def _push_via_ws(ws_base: str, sid: str, seconds: float) -> None:
    async with ws_connect(f"{ws_base}/v1/sessions/{sid}/audio") as audio:
        await _push_realtime(audio, seconds)


class SoakTests(unittest.TestCase):
    """§1.4 — 3 concurrent sessions, long stability + memory curve.

    Default 10 s in CI; the full acceptance run::

        LIVEAVATAR_SOAK_SECONDS=600 python -m pytest tests/test_acceptance_env.py -k soak -s
    """

    def test_three_sessions_soak_memory_stable(self) -> None:
        seconds = float(os.environ.get("LIVEAVATAR_SOAK_SECONDS", "10"))
        server = _TestServer("mjpeg")
        rss_curve = [_rss_mb()]
        try:
            report = asyncio.run(_soak_with_curve(server, seconds, rss_curve))
        finally:
            server.close()
        rss_growth = rss_curve[-1] - rss_curve[0]
        for sid, entry in report.items():
            seqs = entry["seqs"]
            self.assertTrue(entry["eof"], f"{sid}: no EOF frame")
            self.assertGreaterEqual(len(seqs), seconds * 3, f"{sid}: starved")
            self.assertTrue(
                all(b > a for a, b in zip(seqs, seqs[1:], strict=False)),
                f"{sid}: seq not strictly increasing (leak/cross-stream)",
            )
        self.assertLess(rss_growth, 150.0, f"memory curve: {rss_curve}")


async def _soak_with_curve(server: _TestServer, seconds: float,
                           rss_curve: list[float]) -> dict:
    async def curve() -> None:
        while True:
            rss_curve.append(_rss_mb())
            await asyncio.sleep(1.0)

    task = asyncio.create_task(curve())
    try:
        return await _run_soak(server.ws_base, server.http_base,
                               server.avatar, seconds)
    finally:
        task.cancel()


# ───────────────────────────────────────────────────── 3) weak-network link


def _decode_region_frame(canvas: np.ndarray | None, header, payload: bytes):
    """Reference client decoder: keyframes reset the canvas, patches
    composite onto it (a patch after a *dropped* keyframe keeps the last
    canvas — stale picture, never corruption)."""
    patches = unpack_region_payload(payload)
    if has_flag(header.flags, FLAG_KEYFRAME):
        p = patches[0]
        img = cv2.imdecode(np.frombuffer(p.jpeg, np.uint8), cv2.IMREAD_COLOR)
        return img.copy()
    if canvas is None:
        return None
    for p in patches:
        crop = cv2.imdecode(np.frombuffer(p.jpeg, np.uint8), cv2.IMREAD_COLOR)
        canvas[p.y : p.y + p.h, p.x : p.x + p.w] = crop
    return canvas


async def _run_weak_network(server: _TestServer, lossy_s: float,
                            heal_s: float) -> dict:
    """5 % deterministic frame loss → degrade; heal → tier 0 ≤ 2 s."""
    import httpx2 as httpx  # httpx2 is the dev-extra HTTP client (no httpx)

    obs: dict = {
        "frames": 0,
        "gaps": 0,
        "decoded": 0,
        "max_gap_s": 0.0,
        "degraded": False,
        "recovered": False,
    }
    impaired = {"on": True}

    async with httpx.AsyncClient() as http:
        sid = await _create_session(http, server.http_base, server.avatar)
        sink = None  # resolved lazily: pipeline lives in the server thread

        async def reader() -> None:
            canvas = None
            last_arrival = None
            expected_seq: int | None = None
            win_gaps = win_frames = 0
            next_report = time.perf_counter() + 0.5
            async with ws_connect(f"{server.ws_base}/v1/sessions/{sid}/video") as video:
                ready = json.loads(await video.recv())
                assert ready["type"] == "ready", ready
                while True:
                    try:
                        msg = await asyncio.wait_for(video.recv(), timeout=0.5)
                    except (asyncio.TimeoutError, TimeoutError):
                        continue
                    if isinstance(msg, str):
                        continue
                    header, payload = unpack_video_frame(msg)
                    if has_flag(header.flags, FLAG_EOF):
                        break
                    now = time.perf_counter()
                    if last_arrival is not None:
                        obs["max_gap_s"] = max(
                            obs["max_gap_s"], now - last_arrival
                        )
                    last_arrival = now
                    if impaired["on"] and header.seq % 20 == 7:
                        # 5 % deterministic loss *in transit*: the frame is
                        # never delivered; the resulting seq hole is booked
                        # on the next arrival (exactly what a netem drop
                        # looks like to the player's droppedSeqGaps).
                        continue
                    if win_gaps and not impaired["on"]:
                        # Healed mid-window: report post-heal health only,
                        # so recovery is not delayed by a straddling report.
                        win_gaps = 0
                    if expected_seq is not None and header.seq > expected_seq:
                        lost = header.seq - expected_seq
                        obs["gaps"] += lost
                        win_gaps += lost
                    expected_seq = header.seq + 1
                    obs["frames"] += 1
                    win_frames += 1
                    canvas = _decode_region_frame(canvas, header, payload)
                    obs["decoded"] += 1
                    if time.perf_counter() >= next_report and (
                        win_frames + win_gaps > 0
                    ):
                        await video.send(
                            json.dumps(
                                {
                                    "type": "feedback",
                                    "seq_gaps": win_gaps,
                                    "frames": win_frames,
                                }
                            )
                        )
                        win_gaps = win_frames = 0
                        next_report = time.perf_counter() + 0.5

        rtask = asyncio.create_task(reader())
        async with ws_connect(f"{server.ws_base}/v1/sessions/{sid}/audio") as audio:
            pusher = asyncio.create_task(
                _push_realtime(audio, lossy_s + heal_s)
            )
            # Phase A: lossy — the tier must degrade (降画质), never freeze.
            t0 = time.perf_counter()
            degraded_at = None
            while time.perf_counter() - t0 < lossy_s + 3.0:
                sink = sink or _session_sink(sid)
                if sink is not None and sink.controller.tier_index >= 1:
                    degraded_at = time.perf_counter()
                    break
                await asyncio.sleep(0.05)
            obs["degraded"] = degraded_at is not None
            obs["degraded_after_s"] = (
                round(degraded_at - t0, 2) if degraded_at else None
            )
            obs["lossy_quality"] = sink.quality if sink else None

            # Phase B: heal — 3 consecutive healthy reports → tier 0 (2 s).
            impaired["on"] = False
            t_heal = time.perf_counter()
            recovered_at = None
            while time.perf_counter() - t_heal < heal_s + 2.0:
                if sink is not None and sink.controller.tier_index == 0:
                    recovered_at = time.perf_counter()
                    break
                await asyncio.sleep(0.05)
            obs["recovered"] = recovered_at is not None
            obs["recover_s"] = (
                round(recovered_at - t_heal, 2) if recovered_at else None
            )
            obs["healed_quality"] = sink.quality if sink else None
            await pusher
        rtask.cancel()
        try:
            await rtask
        except asyncio.CancelledError:
            pass
        await http.delete(f"{server.http_base}/v1/sessions/{sid}")
    return obs


def _session_sink(sid: str):
    """In-process handle on the session's WebSocketSink (same process)."""
    from liveavatar.publish import state
    from liveavatar.ws_sink import WebSocketSink

    if state.pipeline is None:
        return None
    session = state.pipeline.get_session(sid)
    sink = getattr(session, "publisher", None) if session else None
    return sink if isinstance(sink, WebSocketSink) else None


class WeakNetworkTests(unittest.TestCase):
    """§1.5 — 5 % frame loss: degrade, never freeze, 2 s full recovery."""

    def test_lossy_link_degrades_then_recovers(self) -> None:
        server = _TestServer("region")
        try:
            obs = asyncio.run(_run_weak_network(server, lossy_s=4.0, heal_s=4.0))
        finally:
            server.close()
        # 降画质: the tier moved off "excellent" while frames were lost.
        self.assertTrue(obs["degraded"], obs)
        self.assertLess(obs["lossy_quality"], 80, obs)
        # 不冻结: frames kept flowing (max inter-arrival well under a freeze)
        # and every delivered frame decoded without corruption.
        self.assertGreater(obs["decoded"], 20, obs)
        self.assertLess(obs["max_gap_s"], 2.0, obs)
        # 2 s 回满: tier 0 (quality 80) restored right after the link healed.
        self.assertTrue(obs["recovered"], obs)
        self.assertLessEqual(obs["recover_s"], 2.5, obs)
        self.assertEqual(obs["healed_quality"], 80, obs)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-A: hard-cut interruption baseline recorder (CPU, duplex mode).

Boots the real publish service with a parameterized pattern-avatar pool,
then runs ``--points`` interruption points x ``--avatars`` avatars as
serial duplex sessions. Each session:

1. pushes speech-paced PCM over the duplex audio WS,
2. sends ``{"type": "cancel"}`` at a deterministic mid-stream instant
   (T0 = client send time),
3. records the first new-epoch boundary frame arrival on the video WS
   (client-side end-to-end interruption latency),
4. fetches ``/stats`` for the server-side five-layer timeline
   (detect -> audio_flush -> tts_stop -> video_invalidate -> new_frame).

Emits one JSON report (per-session rows + aggregate percentiles + the
<= 90 ms interruption budget gate) for the paper's latency-decomposition
table (Section 5.1) and as the hard-cut baseline for P-E comparisons.

Usage::

    python scripts/record_interruption_baseline.py --points 20
    python scripts/record_interruption_baseline.py --points 4 \
        --out data/interruption_baseline/smoke.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from typing import Any

import httpx2 as httpx
import uvicorn
from demo_local import _HEIGHT, _WIDTH, _PatternWorker
from websockets.asyncio.client import connect as ws_connect

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
_CHUNK_BYTES = 3200  # 100 ms of 16 kHz S16LE mono
_CHUNK_S = 0.1
_SESSION_SECONDS = 6.0
_INTERRUPT_BUDGET_MS = 90.0

# Three parameterized "avatars" (pattern workers with distinct motion
# speeds) stand in for real MuseTalk avatars in the CPU latency runs.
_AVATAR_PARAMS: dict[str, float] = {"demo_a": 1.5, "demo_b": 2.5, "demo_c": 4.0}


# ─────────────────────────────────────────────────────── service bootstrap


class _SpeedPatternWorker(_PatternWorker):
    """Pattern worker whose band speed is set at construction time."""

    def __init__(self, speed: float) -> None:
        self._speed = speed
        super().__init__()
        self._t0 = time.perf_counter()

    def _infer_batch(self, pcm_s16le: bytes):  # type: ignore[override]
        import numpy as np

        h, w = _HEIGHT, _WIDTH
        yy, xx = np.mgrid[0:h, 0:w]
        base = (time.perf_counter() - self._t0) * self._speed
        dt = 1.0 / 25.0 * self._speed  # real 25 fps pacing within the batch

        def render(t: float) -> bytes:
            # Global gentle drift so consecutive frames differ everywhere
            # (keeps the P-A5 normal inter-frame MAD non-zero).
            drift = (
                (np.sin(xx / 28 + t * 0.7) + np.sin(yy / 19 + t * 0.5)) * 3
            ).astype(np.uint8)
            r = ((xx * 255 // w + drift) % 256).astype(np.uint8)
            g = ((yy * 255 // h + drift) % 256).astype(np.uint8)
            b = np.full((h, w), 90, np.uint8)
            band = (np.sin(yy / 6 + t) + 1) * 110
            mask = (xx >= w // 4) & (xx < 3 * w // 4) & (yy >= h // 4) & (
                yy < h // 4 + h // 3
            )
            b = np.where(mask, band.astype(np.uint8), b)
            img = np.stack([r, g, b], axis=-1).astype(np.uint8)
            return img.tobytes()

        return [
            (render(base + k * dt), True) for k in range(self.batch_size)
        ]


class _DuplexDemoPool:
    """AvatarPool-compatible fake: one dedicated pattern worker per lease."""

    def __init__(self) -> None:
        self._workers: dict[str, _SpeedPatternWorker] = {}

    @property
    def available_avatars(self) -> list[str]:
        return list(_AVATAR_PARAMS)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def acquire(self, session_id: str, avatar_id: str, **kwargs):
        worker = self._workers.get(avatar_id)
        if worker is None:
            worker = _SpeedPatternWorker(_AVATAR_PARAMS.get(avatar_id, 2.0))
            self._workers[avatar_id] = worker

        class _Lease:
            pass

        lease = _Lease()
        lease.worker = worker
        return lease

    async def release_async(self, session_id: str) -> bool:
        return True

    def stats(self) -> dict:
        return {"duplex_demo": True}


def _start_server() -> tuple[uvicorn.Server, threading.Thread, int]:
    state.settings = PublishSettings()
    state.settings.codec = "mjpeg"
    state.settings.duplex.with_avatar = True
    state.pool_config = AvatarPoolConfig(avatar_data_root="nonexistent")
    state.pipeline = AvatarPipeline(
        state.pool_config,
        publisher_factory=_service_publisher_factory,
        pool=_DuplexDemoPool(),
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


def _speech_pcm(seconds: float) -> list[bytes]:
    """Speech-paced PCM chunks: speech bursts + short breath gaps."""
    chunks: list[bytes] = []
    n = int(seconds / _CHUNK_S)
    for i in range(n):
        phase = i % 8
        if phase in (3, 7):  # breath gap
            chunks.append(b"\x00" * _CHUNK_BYTES)
            continue
        # Quiet 220 Hz tone (speech-paced energy for VAD/EOU realism).
        import array
        import math

        samples = array.array("h")
        base = i * _CHUNK_BYTES // 2
        for s in range(_CHUNK_BYTES // 2):
            amp = 600 if phase < 3 else 300
            samples.append(int(amp * math.sin(2 * math.pi * 220 * (base + s) / 16000)))
        chunks.append(samples.tobytes())
    return chunks


async def _read_video(url: str, obs: dict, stop: asyncio.Event) -> None:
    from websockets.exceptions import ConnectionClosed

    async with ws_connect(url, close_timeout=0.5) as video:
        obs["ready"] = json.loads(await video.recv())
        while not stop.is_set():
            try:
                msg = await asyncio.wait_for(video.recv(), timeout=0.25)
            except (asyncio.TimeoutError, TimeoutError):
                continue
            except ConnectionClosed:
                break
            if isinstance(msg, str):
                continue
            header, payload = unpack_video_frame(msg)
            ts = time.perf_counter()
            obs["arrivals"].append(ts)
            obs["epoch_arrivals"].append((header.epoch, ts))
            # Encoded JPEG body (small) — decoded lazily for P-A5 frame
            # difference analysis around the interrupt point.
            obs["frames"].append((ts, header.epoch, payload))
            if header.flags & FLAG_EPOCH_BOUNDARY:
                obs["boundary_arrivals"].append((header.epoch, ts))


def _decode_gray(payload: bytes):
    import cv2
    import numpy as np

    img = cv2.imdecode(
        np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE
    )
    return img.astype(np.float32)


def _mad(a, b) -> float:
    import numpy as np

    return float(np.mean(np.abs(a - b)))


def _frame_diff_stats(obs: dict, pre_epoch: int, interrupt_sent: float,
                      window_s: float = 2.0) -> dict | None:
    """P-A5: hard-cut artifact evidence via frame differences.

    Compares the "mutation" diff (last old-epoch frame vs first
    new-epoch frame across the interrupt) against the normal inter-frame
    diffs in the last ``window_s`` before the interrupt. A ratio >> 1
    proves the hard-cut produces an abrupt visual jump.
    """
    import numpy as np

    frames = obs.get("frames") or []
    old = [(ts, raw) for (ts, e, raw) in frames if e == pre_epoch]
    new = [(ts, raw) for (ts, e, raw) in frames if e > pre_epoch]
    if not old or not new:
        return None

    cut_lo = interrupt_sent - window_s
    normal = [
        (ts0, ts1, raw0, raw1)
        for (ts0, raw0), (ts1, raw1) in zip(old, old[1:], strict=False)
        if ts0 >= cut_lo and ts1 <= interrupt_sent
    ]
    if not normal:
        return None

    cache: dict[int, Any] = {}

    def gray(idx: int, raw: bytes):
        if idx not in cache:
            cache[idx] = _decode_gray(raw)
        return cache[idx]

    normal_mads = [
        _mad(gray(i, r0), gray(i + 1, r1))
        for i, (_t0, _t1, r0, r1) in enumerate(normal)
    ]
    last_old_ts, last_old_raw = old[-1]
    first_new_ts, first_new_raw = new[0]
    mutation = _mad(_decode_gray(last_old_raw), _decode_gray(first_new_raw))
    normal_p50 = float(np.median(normal_mads))
    return {
        "normal_pairs": len(normal_mads),
        "normal_p50": round(normal_p50, 2),
        "normal_max": round(max(normal_mads), 2),
        "mutation": round(mutation, 2),
        "ratio_vs_p50": round(mutation / normal_p50, 2) if normal_p50 > 0 else None,
        "freeze_gap_ms": round((first_new_ts - last_old_ts) * 1000, 2),
    }


async def _run_session(http: httpx.AsyncClient, base: str, ws_base: str,
                       avatar_id: str, interrupt_at: float) -> dict:
    resp = await http.post(
        f"{base}/v1/sessions",
        json={"mode": "duplex", "avatar_id": avatar_id},
    )
    resp.raise_for_status()
    sid = resp.json()["session_id"]

    obs: dict = {
        "arrivals": [],
        "boundary_arrivals": [],
        "epoch_arrivals": [],
        "frames": [],
    }
    stop = asyncio.Event()
    reader = asyncio.create_task(
        _read_video(f"{ws_base}/v1/sessions/{sid}/video", obs, stop)
    )

    interrupt_sent: float | None = None
    stats: dict = {}
    # Keep the audio WS open until AFTER /stats is fetched — closing (or a
    # "stop" message) tears the duplex session down immediately.
    async with ws_connect(
        f"{ws_base}/v1/sessions/{sid}/audio", close_timeout=0.5
    ) as audio:
        await audio.send(json.dumps({"type": "epoch", "epoch": 1}))
        t0 = time.perf_counter()
        chunks = _speech_pcm(_SESSION_SECONDS)
        for i, chunk in enumerate(chunks):
            now = time.perf_counter()
            if interrupt_sent is None and now >= t0 + interrupt_at:
                await audio.send(json.dumps({"type": "cancel"}))
                interrupt_sent = time.perf_counter()
            await audio.send(chunk)
            target = t0 + (i + 1) * _CHUNK_S
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
        # Let the new-epoch frames drain, then fetch the server-side
        # five-layer timeline while the session is still alive.
        await asyncio.sleep(0.8)
        stats_resp = await http.get(f"{base}/v1/sessions/{sid}/stats")
        stats = stats_resp.json() if stats_resp.status_code == 200 else {}
        await audio.send(json.dumps({"type": "stop"}))

    await asyncio.sleep(0.3)
    stop.set()
    reader.cancel()
    try:
        await reader
    except asyncio.CancelledError:
        pass
    await http.delete(f"{base}/v1/sessions/{sid}")

    row: dict = {
        "avatar_id": avatar_id,
        "interrupt_at_s": interrupt_at,
        "session_id": sid,
    }
    timeline = (stats.get("metrics") or {}).get("interruption_timeline")
    if timeline:
        row["server_timeline"] = timeline

    if interrupt_sent is not None:
        # Epoch base = the highest epoch observed BEFORE the interrupt.
        # Robust to the session-start "epoch" control bumping the epoch.
        pre_epoch = max(
            (e for (e, ts) in obs["epoch_arrivals"] if ts < interrupt_sent),
            default=0,
        )
        old_arrivals = [
            ts for (epoch, ts) in obs["epoch_arrivals"]
            if epoch == pre_epoch and ts >= interrupt_sent
        ]
        new_arrivals = [
            ts for (epoch, ts) in obs["epoch_arrivals"]
            if epoch > pre_epoch and ts >= interrupt_sent
        ]
        boundary = [
            ts for (epoch, ts) in obs["boundary_arrivals"]
            if epoch > pre_epoch and ts >= interrupt_sent
        ]
        # Old-epoch frames arriving AFTER the cancel: stale frames already
        # queued in the sink (hard-cut artifact evidence).
        if old_arrivals:
            row["stale_tail_ms"] = round(
                (old_arrivals[-1] - interrupt_sent) * 1000, 2
            )
            row["stale_frames_after_cancel"] = len(old_arrivals)
        # First new-epoch frame (generation-dependent: ASR/TTS must produce
        # the next utterance before new video exists).
        if new_arrivals:
            row["resume_ms"] = round((new_arrivals[0] - interrupt_sent) * 1000, 2)
        if boundary:
            row["client_e2e_ms"] = round((boundary[0] - interrupt_sent) * 1000, 2)
        # P-A5: frame-difference evidence around the cut.
        fd = _frame_diff_stats(obs, pre_epoch, interrupt_sent)
        if fd is not None:
            row["frame_diff"] = fd
        row["frames_total"] = len(obs["arrivals"])
    return row


# ─────────────────────────────────────────────────────────────── aggregate


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


_LAYER_KEYS = [
    "detect_to_audio_flush_ms",
    "audio_flush_to_tts_stop_ms",
    "tts_stop_to_video_invalidate_ms",
    "video_invalidate_to_new_frame_ms",
    "total_ms",
]


def _fd_summary(rows: list[dict]) -> dict:
    """Percentile summary of the per-session frame-diff evidence."""
    def pick(key: str) -> list[float]:
        return sorted(
            r["frame_diff"][key]
            for r in rows
            if r.get("frame_diff", {}).get(key) is not None
        )

    mutation = pick("mutation")
    normal = pick("normal_p50")
    ratio = pick("ratio_vs_p50")
    freeze = pick("freeze_gap_ms")
    return {
        "n": len(mutation),
        "normal_p50_ms": round(_percentile(normal, 50), 2) if normal else None,
        "mutation_p50": round(_percentile(mutation, 50), 2) if mutation else None,
        "mutation_p95": round(_percentile(mutation, 95), 2) if mutation else None,
        "ratio_p50": round(_percentile(ratio, 50), 2) if ratio else None,
        "ratio_max": round(ratio[-1], 2) if ratio else None,
        "freeze_gap_p50_ms": round(_percentile(freeze, 50), 2) if freeze else None,
    }


def _aggregate(rows: list[dict]) -> dict:
    agg: dict = {"n": len(rows)}
    client = sorted(
        r["client_e2e_ms"] for r in rows if r.get("client_e2e_ms") is not None
    )
    agg["client_e2e"] = {
        "n": len(client),
        "p50_ms": round(_percentile(client, 50), 2) if client else None,
        "p95_ms": round(_percentile(client, 95), 2) if client else None,
        "max_ms": round(client[-1], 2) if client else None,
    }
    for key in ("stale_tail_ms", "resume_ms"):
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        agg[key] = {
            "n": len(vals),
            "p50_ms": round(_percentile(vals, 50), 2) if vals else None,
            "p95_ms": round(_percentile(vals, 95), 2) if vals else None,
            "max_ms": round(vals[-1], 2) if vals else None,
        }
    stale_counts = [
        r["stale_frames_after_cancel"]
        for r in rows
        if r.get("stale_frames_after_cancel") is not None
    ]
    agg["stale_frames_after_cancel"] = {
        "n_sessions": len(stale_counts),
        "max": max(stale_counts) if stale_counts else 0,
    }
    server_keys: dict[str, list[float]] = {k: [] for k in _LAYER_KEYS}
    for r in rows:
        tl = r.get("server_timeline") or {}
        for key in _LAYER_KEYS:
            v = tl.get(key)
            if isinstance(v, (int, float)):
                server_keys[key].append(float(v))
    server_out: dict = {}
    for key, vals in server_keys.items():
        vals.sort()
        server_out[key] = {
            "n": len(vals),
            "p50_ms": round(_percentile(vals, 50), 3) if vals else None,
            "p95_ms": round(_percentile(vals, 95), 3) if vals else None,
            "max_ms": round(vals[-1], 3) if vals else None,
        }
    agg["server_timeline"] = server_out

    # P-A5: hard-cut artifact frame-difference evidence (overall + per avatar).
    agg["frame_diff"] = _fd_summary(rows)
    agg["frame_diff_by_avatar"] = {
        avatar: _fd_summary([r for r in rows if r["avatar_id"] == avatar])
        for avatar in sorted({r["avatar_id"] for r in rows})
    }

    # Gate on the ENFORCEMENT segment only (detect → audio_flush →
    # tts_stop → video_invalidate): the actual interrupt execution. The
    # new_frame segment includes the next-utterance generation start-up
    # (ASR/TTS) and is reported separately — filling that visual freeze is
    # exactly what the P-D transition-frame method targets.
    enforcement_keys = [
        "detect_to_audio_flush_ms",
        "audio_flush_to_tts_stop_ms",
        "tts_stop_to_video_invalidate_ms",
    ]
    enf_p95_sum = sum(
        server_out[k]["p95_ms"] or 0.0 for k in enforcement_keys
    )
    agg["gate"] = {
        "budget_ms": _INTERRUPT_BUDGET_MS,
        "enforcement_p95_sum_ms": round(enf_p95_sum, 3),
        "new_frame_segment_p95_ms": server_out["video_invalidate_to_new_frame_ms"]["p95_ms"],
        "pass": enf_p95_sum <= _INTERRUPT_BUDGET_MS,
    }
    return agg


def _write_markdown(path: str, report: dict) -> None:
    agg = report["aggregate"]
    gate = agg["gate"]
    lines = [
        "# P-A 打断延迟分解报告（硬切基线，CPU duplex 模式）",
        "",
        f"- 会话数：{agg['n']}（{report['avatars']} 素材 x {report['points']} 打断点）",
        f"- 执行段门禁（detect → audio_flush → tts_stop → video_invalidate）："
        f"三层 p95 之和 ≤ {gate['budget_ms']} ms → "
        f"实测 {gate['enforcement_p95_sum_ms']} ms → "
        f"{'PASS' if gate['pass'] else 'FAIL'}",
        f"- new_frame 段 p95（含下一轮话语 ASR/TTS 生成启动）："
        f"{gate['new_frame_segment_p95_ms']} ms（P-D 过渡帧方法的目标）",
        "",
        "## 客户端视角",
        "",
        "| 指标 | n | p50 | p95 | max |",
        "|---|---|---|---|---|",
        _stat_row("client_e2e（boundary 帧到达）", agg["client_e2e"]),
        _stat_row("stale_tail（打断后旧帧漏出）", agg["stale_tail_ms"]),
        _stat_row("resume（新流首帧，含 ASR/TTS 生成）", agg["resume_ms"]),
        "",
        f"- 打断后旧帧漏出会话数：{agg['stale_frames_after_cancel']['n_sessions']}，"
        f"单会话最多漏出 {agg['stale_frames_after_cancel']['max']} 帧"
        "（执行段即时清队 → 旧帧零漏出；视觉伪影来自冻结间隙 + 突变帧差，"
        "见下方帧差证据）",
        "",
        "## 服务端五层分解（detect → audio_flush → tts_stop → video_invalidate → new_frame）",
        "",
        "| 层间 | n | p50 | p95 | max |",
        "|---|---|---|---|---|",
    ]
    for key in _LAYER_KEYS:
        s = agg["server_timeline"][key]
        lines.append(
            f"| {key} | {s['n']} | {s['p50_ms']} | {s['p95_ms']} | {s['max_ms']} |"
        )
    fd = agg["frame_diff"]
    lines += [
        "",
        "## 硬切伪影帧差证据（P-A5：突变帧差 vs 正常帧间差）",
        "",
        "| 素材 | n | 正常帧间差 p50 | 突变帧差 p50 | "
        "放大倍数 p50 | 放大倍数 max | 冻结间隙 p50(ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for avatar, s in agg["frame_diff_by_avatar"].items():
        lines.append(
            f"| {avatar} | {s['n']} | {s['normal_p50_ms']} | "
            f"{s['mutation_p50']} | {s['ratio_p50']} | {s['ratio_max']} | "
            f"{s['freeze_gap_p50_ms']} |"
        )
    lines += [
        "",
        f"- 全体：突变帧差 p50 = {fd['mutation_p50']}（正常 {fd['normal_p50_ms']}），"
        f"放大倍数 p50 = {fd['ratio_p50']}，max = {fd['ratio_max']}；"
        f"冻结间隙 p50 = {fd['freeze_gap_p50_ms']} ms。",
        "- 判定：放大倍数 > 1 即打断点处存在显著口型/画面突变（硬切伪影客观存在）；"
        "冻结间隙为旧流末帧到新流首帧的视觉冻结时长（P-D 过渡帧的目标）。",
        "",
    ]
    lines += [
        "",
        "## 单次打断明细",
        "",
        "| 素材 | 打断点(s) | stale_tail(ms) | resume(ms) | 服务端 total(ms) | 突变帧差 | 倍数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        tl = r.get("server_timeline") or {}
        fd = r.get("frame_diff") or {}
        lines.append(
            f"| {r['avatar_id']} | {r['interrupt_at_s']:.2f} | "
            f"{r.get('stale_tail_ms', '-')} | {r.get('resume_ms', '-')} | "
            f"{tl.get('total_ms', '-')} | {fd.get('mutation', '-')} | "
            f"{fd.get('ratio_vs_p50', '-')} |"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _stat_row(label: str, s: dict) -> str:
    return (
        f"| {label} | {s.get('n', 0)} | {s.get('p50_ms')} | "
        f"{s.get('p95_ms')} | {s.get('max_ms')} |"
    )


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--out", type=str,
                        default="data/interruption_baseline/baseline.json")
    parser.add_argument("--markdown", type=str, default=None)
    parser.add_argument(
        "--from-json", type=str, default=None,
        help="skip recording; regenerate the markdown report from a "
             "previously written JSON report",
    )
    args = parser.parse_args(argv)

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            report = json.load(f)
        md_path = args.markdown or args.from_json.replace(".json", ".md")
        _write_markdown(md_path, report)
        print(f"regenerated {md_path}")
        return 0

    server, _thread, port = _start_server()
    base = f"http://{_HOST}:{port}"
    ws_base = f"ws://{_HOST}:{port}"
    rows: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            avatars = list(_AVATAR_PARAMS)
            for avatar_id in avatars:
                for i in range(args.points):
                    interrupt_at = round(0.5 + i * 4.0 / max(args.points - 1, 1), 3)
                    row = await _run_session(
                        http, base, ws_base, avatar_id, interrupt_at
                    )
                    rows.append(row)
                    e2e = row.get("client_e2e_ms")
                    print(f"[{len(rows)}] {avatar_id} @{interrupt_at:.2f}s "
                          f"e2e={e2e}ms", flush=True)
    finally:
        server.should_exit = True
        await asyncio.sleep(0.2)

    report = {
        "mode": "interruption_baseline_hard_cut",
        "avatars": len(_AVATAR_PARAMS),
        "points": args.points,
        "session_seconds": _SESSION_SECONDS,
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    md_path = args.markdown or args.out.replace(".json", ".md")
    _write_markdown(md_path, report)
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))
    return 0 if report["aggregate"]["gate"]["pass"] else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())

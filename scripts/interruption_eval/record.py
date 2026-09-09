# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-B: interruption-eval media recorder (CPU, duplex mode).

Boots the real publish service with audio-driven mouth-proxy workers and
a seeded streaming TTS (``SynthTts``), then records per-session media for
the offline analyzer:

- ``frames/f00001.jpg …`` — the MJPEG video track as received on the
  video WS (lossless: the recorder raises the sink's per-client queue
  depth via ``LIVEAVATAR_CLIENT_QUEUE_SIZE``), with a frame timeline
  (arrival ts, epoch, boundary flag);
- ``audio.wav`` — the agent's TTS audio as received on the audio WS
  (what the pipeline actually emitted);
- ``meta.json`` — seed(s), phoneme schedule(s), interrupt point
  (utterance-relative), epoch-2 anchor, client latency row, server
  five-layer timeline.

Session protocol (design B — "audio in → barge-in → resume"):

1. a speech burst arms VAD; EOU fires the scripted ASR final; the agent
   answers with seeded utterance 1 (epoch 1) streamed in 160 ms chunks —
   exactly one avatar inference batch (4 frames @ 25 fps) per chunk, so
   audio and video share the 40 ms/frame utterance clock;
2. at the cut instant the recorder sends ``{"type": "cancel"}``;
3. after a short pause the user "speaks again" (second burst): a second
   ASR final spawns the epoch-2 answer (seeded utterance 2) — the same
   resume behavior as the P-A baseline, giving the transition something
   to bridge INTO.

For each seed a no-interrupt ``reference`` session is also recorded:
the V-APT reference trajectory (same seed → same audio → same closed-
loop mouth trajectory without the cut).

The recorder swaps the worker default TTS (``FakeTts`` → ``SynthTts``)
in-process: the synthetic utterance flows through the REAL pipeline
(VAD/EOU → scripted ASR → streaming TTS → avatar fan-out), so the hard-
cut behavior under study is genuine pipeline behavior, not a simulation.

Session directory layout under ``--out``::

    <out>/reference_<avatar>_seed<seed>/{meta.json,audio.wav,frames/}
    <out>/interrupt_<avatar>_seed<seed>_t<int>s/…

Usage::

    python scripts/interruption_eval/record.py --smoke
    python scripts/interruption_eval/record.py --points 20
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import math
import os
import threading
import time
import wave
from collections.abc import AsyncGenerator
from typing import Any

import httpx2 as httpx
import uvicorn
from audio_synth import Utterance, synth_utterance
from mouth_proxy import MouthProxyWorker
from websockets.asyncio.client import connect as ws_connect

from liveavatar.config import AvatarPoolConfig
from liveavatar.pipeline import AvatarPipeline
from liveavatar.publish import (
    PublishSettings,
    _service_publisher_factory,
    app,
    state,
)
from liveavatar.runtime.fake_tts import FakeTtsSegment
from liveavatar.video_protocol import (
    FLAG_EOF,
    FLAG_EPOCH_BOUNDARY,
    unpack_video_frame,
)

_HOST = "127.0.0.1"
_UPLINK_S = 0.1  # mic uplink chunk (and trigger pacing step)
_UPLINK_BYTES = 3200  # 100 ms of 16 kHz S16LE mono
# TTS chunk = one avatar batch (4 frames @ 25 fps = 160 ms): audio and
# video then share the same 40 ms/frame utterance clock with no drift.
_TTS_CHUNK_S = 0.16
_UTT_S = 6.0
_UTT2_S = 2.5  # post-interrupt response duration
_TRIGGER_TONE_S = 0.6  # speech burst that arms VAD
_TRIGGER_SILENCE_S = 0.5  # > EOU 400 ms silence → scripted ASR final
_TRIGGER2_TONE_S = 0.3  # shorter second burst (resume must fit the window)
_RESUME_PAUSE_S = 0.15  # user pause between barge-in and second burst
_DRAIN_S = 1.6  # post-response wait before /stats (let final frames land)
_RESUME_DEADLINE_S = 8.0  # give up waiting for the epoch-2 anchor
_AVATARS = ("mouth_a", "mouth_b", "mouth_c")
_FRAMES_PER_BATCH = 4
_FRAME_PERIOD_S = _TTS_CHUNK_S / _FRAMES_PER_BATCH  # 0.04
_SCHEMA = 2

# Holder read by SynthTts at construction (sessions run serially).
_EVAL: dict[str, Any] = {"utterances": []}


# ─────────────────────────────────────────────────────── eval TTS swap


class SynthTts:
    """Streaming TTS emitting the seeded eval utterances at 1× realtime.

    Mirrors the ``FakeTts`` interface the worker consumes
    (``synthesize_stream`` / ``cancel_epoch`` / ``pop_played_segments``).
    Each synthesis call (one per scripted ASR final) pops the NEXT
    utterance from ``_EVAL["utterances"]`` — utterance 1 for the first
    response, utterance 2 for the post-interrupt response — so the
    ground-truth phoneme schedules in meta.json match the audio the
    mouth proxy actually tracks. Chunks are 160 ms = exactly one avatar
    batch, paced at 1× realtime (speech-paced cut). Mid-stream
    cancellation is handled by the worker's TTS-task cancellation (epoch
    advance), as with any streaming TTS.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.segment_counter = 0
        self.active_segments: list[FakeTtsSegment] = []
        self._queue: list[Utterance] = list(_EVAL["utterances"])
        self._idx = 0

    def _next_utterance(self) -> Utterance:
        if self._idx >= len(self._queue):
            raise RuntimeError(
                "no eval utterance left — recorder must register one per "
                "ASR final (utterance 1 + optional post-cut response)"
            )
        utt = self._queue[self._idx]
        self._idx += 1
        return utt

    def synthesize(self, text: str, epoch: int, pts_us: int) -> list[FakeTtsSegment]:
        """Sync fallback: the whole utterance as one segment."""
        utt = self._next_utterance()
        self.segment_counter += 1
        seg = FakeTtsSegment(
            segment_seq=self.segment_counter,
            text=text,
            epoch=epoch,
            pcm_s16le=utt.pcm_s16le,
            pts_us=pts_us,
            duration_us=int(len(utt.pcm_s16le) / 2 / utt.sr * 1_000_000),
        )
        self.active_segments.append(seg)
        return [seg]

    async def synthesize_stream(
        self, text: str, epoch: int, pts_us: int
    ) -> AsyncGenerator[FakeTtsSegment, None]:
        """Yield 160 ms chunks (= 1 avatar batch) paced at 1× realtime."""
        utt = self._next_utterance()
        chunk_bytes = int(utt.sr * _TTS_CHUNK_S) * 2
        pts = pts_us
        for i in range(0, len(utt.pcm_s16le), chunk_bytes):
            await asyncio.sleep(_TTS_CHUNK_S)
            pcm = utt.pcm_s16le[i : i + chunk_bytes]
            self.segment_counter += 1
            seg = FakeTtsSegment(
                segment_seq=self.segment_counter,
                text=text,
                epoch=epoch,
                pcm_s16le=pcm,
                pts_us=pts,
                duration_us=int(len(pcm) / 2 / utt.sr * 1_000_000),
            )
            pts += seg.duration_us
            self.active_segments.append(seg)
            yield seg

    def cancel_epoch(self, epoch: int) -> int:
        before = len(self.active_segments)
        self.active_segments = [s for s in self.active_segments if s.epoch >= epoch]
        return before - len(self.active_segments)

    def pop_played_segments(self, consumed_pts_us: int) -> list[FakeTtsSegment]:
        played = [
            s for s in self.active_segments if s.pts_us + s.duration_us <= consumed_pts_us
        ]
        self.active_segments = [
            s for s in self.active_segments if s.pts_us + s.duration_us > consumed_pts_us
        ]
        return played


def _install_synth_tts() -> None:
    """Swap the worker default TTS to SynthTts (in-process, eval only)."""
    import liveavatar.runtime.worker as worker_mod

    worker_mod.FakeTts = SynthTts  # type: ignore[assignment]


# ─────────────────────────────────────────────────────── service bootstrap


class _MouthProxyPool:
    """AvatarPool-compatible fake: one dedicated mouth proxy per avatar."""

    def __init__(self, avatar_ids: list[str]) -> None:
        self._ids = list(avatar_ids)
        self._workers: dict[str, MouthProxyWorker] = {}
        self._seen_sessions: set[str] = set()

    @property
    def available_avatars(self) -> list[str]:
        return list(self._ids)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def acquire(self, session_id: str, avatar_id: str, **kwargs: Any):
        worker = self._workers.get(avatar_id)
        if worker is None:
            worker = MouthProxyWorker(avatar_id)
            self._workers[avatar_id] = worker
        if session_id not in self._seen_sessions:
            worker.reset()  # fresh idle mouth per eval session
            self._seen_sessions.add(session_id)

        class _Lease:
            pass

        lease = _Lease()
        lease.worker = worker
        return lease

    async def release_async(self, session_id: str) -> bool:
        return True

    def stats(self) -> dict:
        return {"mouth_proxy_pool": True}


def _start_server(
    avatars: list[str], *, transition_frames: int = 3
) -> tuple[uvicorn.Server, threading.Thread, int]:
    state.settings = PublishSettings()
    state.settings.codec = "mjpeg"
    # Lossless capture: the eval recorder must see every rendered frame
    # (a dropped frame would bias the offline trajectory metrics).
    state.settings.client_queue_size = 512
    state.settings.duplex.with_avatar = True
    # P-D: barge-in transition length (0 = hard-cut baseline replay).
    state.settings.duplex.avatar_transition_frames = max(0, transition_frames)
    state.pool_config = AvatarPoolConfig(avatar_data_root="nonexistent")
    state.pipeline = AvatarPipeline(
        state.pool_config,
        publisher_factory=_service_publisher_factory,
        pool=_MouthProxyPool(avatars),
    )
    config = uvicorn.Config(app, host=_HOST, port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:  # pragma: no cover - boot wait
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, port


# ────────────────────────────────────────────────────────── audio helpers


def _tone_pcm(seconds: float, freq: int = 220, amp: int = 600) -> list[bytes]:
    """Speech-paced trigger chunks (100 ms) that arm VAD/EOU."""
    chunks: list[bytes] = []
    n_chunks = int(seconds / _UPLINK_S)
    for i in range(n_chunks):
        samples = bytearray()
        base = i * _UPLINK_BYTES // 2
        for s in range(_UPLINK_BYTES // 2):
            samples += int(
                amp * math.sin(2 * math.pi * freq * (base + s) / 16000)
            ).to_bytes(2, "little", signed=True)
        chunks.append(bytes(samples))
    return chunks


_SILENCE_CHUNK = b"\x00" * _UPLINK_BYTES


async def _pace(upto: float) -> None:
    await asyncio.sleep(max(0.0, upto - time.perf_counter()))


# ────────────────────────────────────────────────────────── WS readers


async def _read_video(url: str, obs: dict, stop: asyncio.Event, frames_dir: str) -> None:
    """Record the video WS: frame files + arrival/epoch/boundary timeline."""
    from websockets.exceptions import ConnectionClosed

    idx = 0
    async with ws_connect(url, close_timeout=0.5) as video:
        obs["video_ready"] = json.loads(await video.recv())
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
            if (header.flags & FLAG_EOF) or not payload:
                continue  # EOF sentinel — not a renderable frame
            ts = time.perf_counter()
            idx += 1
            fname = f"f{idx:05d}.jpg"
            with open(os.path.join(frames_dir, fname), "wb") as f:
                f.write(payload)
            obs["frames"].append(
                {
                    "ts": ts,
                    "epoch": header.epoch,
                    "boundary": bool(header.flags & FLAG_EPOCH_BOUNDARY),
                    "file": fname,
                }
            )
            obs["epoch_arrivals"].append((header.epoch, ts))
            if header.flags & FLAG_EPOCH_BOUNDARY:
                obs["boundary_arrivals"].append((header.epoch, ts))


def _make_audio_reader(ws: Any, obs: dict, stop: asyncio.Event) -> Any:
    """Task body: record one audio WS downlink (raw TTS PCM + events)."""
    from websockets.exceptions import ConnectionClosed

    async def _run() -> None:
        while not stop.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.25)
            except (asyncio.TimeoutError, TimeoutError):
                continue
            except ConnectionClosed:
                break
            ts = time.perf_counter()
            if isinstance(msg, bytes):
                obs["tts_chunks"].append((ts, len(msg)))
                obs["tts_pcm"].append(msg)
            else:
                obs["events"].append((ts, msg[:300]))

    return _run()


# ────────────────────────────────────────────────────────── per-session run


def _first_epoch2_frame_ts(obs: dict) -> float | None:
    for f in obs["frames"]:
        if f["epoch"] > 1:
            return f["ts"]
    return None


def _first_driven_epoch2_frame_ts(obs: dict, skip: int) -> float | None:
    """First epoch-2 frame NOT part of the P-D transition.

    With a transition configured, the first ``skip`` epoch-2 frames are
    the mouth-close bridge (published at cancel time, before utterance
    2) — the session anchor for pacing and the phoneme clock is the
    first DRIVEN frame after them. ``skip=0`` (hard cut) degenerates to
    :func:`_first_epoch2_frame_ts`.
    """
    seen = 0
    for f in obs["frames"]:
        if f["epoch"] > 1:
            if seen < skip:
                seen += 1
                continue
            return f["ts"]
    return None


async def _run_session(
    http: httpx.AsyncClient,
    base: str,
    ws_base: str,
    *,
    avatar_id: str,
    seed: int,
    interrupt_at: float | None,
    out_dir: str,
    utt_s: float = _UTT_S,
    utt2_s: float = _UTT2_S,
    transition_frames: int = 0,
) -> dict:
    """Record one eval session (interrupted or reference) to ``out_dir``."""
    utterance = synth_utterance(utt_s, seed)
    utt2 = synth_utterance(utt2_s, seed + 1000)
    _EVAL["utterances"] = (
        [utterance] if interrupt_at is None else [utterance, utt2]
    )
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

    resp = await http.post(
        f"{base}/v1/sessions",
        json={"mode": "duplex", "avatar_id": avatar_id},
    )
    resp.raise_for_status()
    sid = resp.json()["session_id"]
    t0 = time.perf_counter()

    obs: dict = {
        "frames": [],
        "epoch_arrivals": [],
        "boundary_arrivals": [],
        "tts_chunks": [],
        "tts_pcm": [],
        "events": [],
    }
    stop = asyncio.Event()
    video_task = asyncio.create_task(
        _read_video(
            f"{ws_base}/v1/sessions/{sid}/video", obs, stop,
            os.path.join(out_dir, "frames"),
        )
    )
    audio_task: asyncio.Task | None = None
    interrupt_sent: float | None = None
    stats: dict = {}
    error: str | None = None
    async with ws_connect(
        f"{ws_base}/v1/sessions/{sid}/audio", close_timeout=0.5
    ) as audio:
        # One audio WS: concurrent downlink reader + this send/control
        # coroutine (a second connection would steal out_queue items).
        audio_task = asyncio.create_task(_make_audio_reader(audio, obs, stop))
        await audio.send(json.dumps({"type": "epoch", "epoch": 1}))
        # Trigger 1: speech burst + EOU silence → scripted ASR final → TTS.
        for i, chunk in enumerate(_tone_pcm(_TRIGGER_TONE_S)):
            await audio.send(chunk)
            await _pace(t0 + (i + 1) * _UPLINK_S)
        for i in range(int(_TRIGGER_SILENCE_S / _UPLINK_S)):
            await audio.send(_SILENCE_CHUNK)
            await _pace(t0 + _TRIGGER_TONE_S + (i + 1) * _UPLINK_S)

        # Wait for the first TTS chunk (utterance-position anchor).
        deadline = time.perf_counter() + 5.0
        while not obs["tts_chunks"] and time.perf_counter() < deadline:
            await asyncio.sleep(0.02)
        if not obs["tts_chunks"]:
            error = "no tts audio within 5 s"
        elif interrupt_at is not None:
            tts_start = obs["tts_chunks"][0][0]
            await _pace(tts_start + interrupt_at)
            await audio.send(json.dumps({"type": "cancel"}))
            interrupt_sent = time.perf_counter()
            # Design B: the user speaks again after the barge-in — a
            # second ASR final spawns the epoch-2 answer (resume).
            await _pace(interrupt_sent + _RESUME_PAUSE_S)
            t2 = time.perf_counter()
            for i, chunk in enumerate(_tone_pcm(_TRIGGER2_TONE_S)):
                await audio.send(chunk)
                await _pace(t2 + (i + 1) * _UPLINK_S)
            for i in range(int(_TRIGGER_SILENCE_S / _UPLINK_S)):
                await audio.send(_SILENCE_CHUNK)
                await _pace(t2 + _TRIGGER2_TONE_S + (i + 1) * _UPLINK_S)
            # Wait for the epoch-2 response to play out (anchor = first
            # DRIVEN epoch-2 video frame, generated from its first TTS
            # chunk — transition bridge frames arrive earlier and don't
            # count).
            deadline = time.perf_counter() + _RESUME_DEADLINE_S
            while (
                _first_driven_epoch2_frame_ts(obs, transition_frames) is None
                and time.perf_counter() < deadline
            ):
                await asyncio.sleep(0.02)
            utt2_ts = _first_driven_epoch2_frame_ts(obs, transition_frames)
            if utt2_ts is None:
                error = "no epoch-2 response within deadline"
                await asyncio.sleep(1.0)
            else:
                # Pace past the end of utterance 2 (client-anchored).
                await _pace(utt2_ts + utt2_s + 0.3)
        else:
            tts_start = obs["tts_chunks"][0][0]
            await _pace(tts_start + utt_s + 0.8)
        await asyncio.sleep(_DRAIN_S)
        stats_resp = await http.get(f"{base}/v1/sessions/{sid}/stats")
        stats = stats_resp.json() if stats_resp.status_code == 200 else {}
        await audio.send(json.dumps({"type": "stop"}))

    await asyncio.sleep(0.3)
    stop.set()
    for task in (video_task, audio_task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await http.delete(f"{base}/v1/sessions/{sid}")

    row: dict = {
        "kind": "interrupt" if interrupt_at is not None else "reference",
        "avatar_id": avatar_id,
        "seed": seed,
        "schema": _SCHEMA,
        "session_id": sid,
        "frame_period_s": _FRAME_PERIOD_S,
        "tts_chunk_s": _TTS_CHUNK_S,
        "utterance_duration_s": utt_s,
        "interrupt_at_s": interrupt_at,
    }
    if error is not None:
        row["error"] = error
        return row

    # audio.wav — the TTS audio actually received from the pipeline.
    pcm_all = b"".join(obs["tts_pcm"])
    with wave.open(os.path.join(out_dir, "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm_all)
    row["tts_samples_received"] = len(pcm_all) // 2

    tts_start = obs["tts_chunks"][0][0]
    row["tts_start_s"] = round(tts_start - t0, 4)
    if interrupt_sent is not None:
        row["interrupt_sent_s"] = round(interrupt_sent - t0, 4)
        row["cut_in_utterance_s"] = round(interrupt_sent - tts_start, 4)
        row["tts_samples_at_cancel"] = (
            sum(n for ts, n in obs["tts_chunks"] if ts <= interrupt_sent) // 2
        )
        # Epoch-2 anchors (client clock): ``utt2_start_s`` = first
        # epoch-2 frame (the P-A L5 "first new frame"; with P-D this is
        # the first transition bridge frame), ``utt2_driven_start_s`` =
        # first utterance-2-driven frame (the phoneme-clock anchor —
        # equals ``utt2_start_s`` when the transition is off). The
        # audio.wav sample offset points at the utterance-2 audio start
        # (transition frames arrive before any utterance-2 audio).
        row["transition_frames"] = max(0, transition_frames)
        utt2_ts = _first_epoch2_frame_ts(obs)
        driven_ts = _first_driven_epoch2_frame_ts(obs, transition_frames)
        if utt2_ts is not None and driven_ts is not None:
            row["utt2_start_s"] = round(utt2_ts - t0, 4)
            row["utt2_driven_start_s"] = round(driven_ts - t0, 4)
            row["utt2_audio_sample_offset"] = (
                sum(n for ts, n in obs["tts_chunks"] if ts <= utt2_ts) // 2
            )
            row["utterance_2"] = {"seed": seed + 1000, **utt2.to_meta()}

    # Client-side latency row (same semantics as the P-A baseline).
    if interrupt_sent is not None:
        pre_epoch = max(
            (e for (e, ts) in obs["epoch_arrivals"] if ts < interrupt_sent),
            default=0,
        )
        old_arr = [
            ts for (e, ts) in obs["epoch_arrivals"]
            if e == pre_epoch and ts >= interrupt_sent
        ]
        new_arr = [
            ts for (e, ts) in obs["epoch_arrivals"]
            if e > pre_epoch and ts >= interrupt_sent
        ]
        boundary = [
            ts for (e, ts) in obs["boundary_arrivals"]
            if e > pre_epoch and ts >= interrupt_sent
        ]
        if old_arr:
            row["stale_tail_ms"] = round((old_arr[-1] - interrupt_sent) * 1000, 2)
        if new_arr:
            row["resume_ms"] = round((new_arr[0] - interrupt_sent) * 1000, 2)
        if boundary:
            row["client_e2e_ms"] = round((boundary[0] - interrupt_sent) * 1000, 2)

    timeline = (stats.get("metrics") or {}).get("interruption_timeline")
    if timeline:
        row["server_timeline"] = timeline

    row["frame_timeline"] = [
        {
            "ts_rel": round(f["ts"] - t0, 4),
            "epoch": f["epoch"],
            "boundary": f["boundary"],
            "file": f["file"],
        }
        for f in obs["frames"]
    ]
    row["tts_chunk_timeline"] = [
        {"ts_rel": round(ts - t0, 4), "bytes": n} for (ts, n) in obs["tts_chunks"]
    ]
    row["control_events"] = [
        {"ts_rel": round(ts - t0, 4), "event": ev} for (ts, ev) in obs["events"]
    ]
    row["utterance"] = utterance.to_meta()
    row["recorded_at"] = _dt.datetime.now().isoformat(timespec="seconds")

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    return row


# ─────────────────────────────────────────────────────────────────── main


def _session_dir(root: str, avatar_id: str, seed: int,
                 interrupt_at: float | None) -> str:
    if interrupt_at is None:
        name = f"reference_{avatar_id}_seed{seed}"
    else:
        name = f"interrupt_{avatar_id}_seed{seed}_t{interrupt_at:.2f}s"
    return os.path.join(root, name)


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--avatars", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--duration", type=float, default=_UTT_S)
    parser.add_argument("--duration2", type=float, default=_UTT2_S,
                        help="post-interrupt response duration")
    parser.add_argument("--out", type=str,
                        default="data/interruption_eval/dataset")
    parser.add_argument("--transition", type=int, default=3,
                        help="P-D barge-in transition frames (0 = hard cut)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny grid: 2 avatars x 2 points, 2 seeds")
    args = parser.parse_args(argv)

    if args.smoke:
        avatars = list(_AVATARS)[:2]
        points, seeds = 2, 2
    else:
        avatars = list(_AVATARS)[: max(1, args.avatars)]
        points, seeds = max(1, args.points), max(1, args.seeds)

    _install_synth_tts()
    server, _thread, port = _start_server(avatars,
                                          transition_frames=args.transition)
    base = f"http://{_HOST}:{port}"
    ws_base = f"ws://{_HOST}:{port}"

    interrupt_ats = [
        round(0.5 + i * 4.0 / max(points - 1, 1), 3) for i in range(points)
    ]
    rows: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            # Reference sessions: one per seed (V-APT reference curve).
            for seed in range(1, seeds + 1):
                out_dir = _session_dir(args.out, avatars[0], seed, None)
                row = await _run_session(
                    http, base, ws_base, avatar_id=avatars[0], seed=seed,
                    interrupt_at=None, out_dir=out_dir, utt_s=args.duration,
                )
                rows.append(row)
                print(f"[ref {seed}/{seeds}] tts@{row.get('tts_start_s')}s",
                      flush=True)
            # Interrupted sessions: avatars × points, seed = point index.
            for avatar_id in avatars:
                for i, interrupt_at in enumerate(interrupt_ats):
                    seed = i + 1
                    out_dir = _session_dir(args.out, avatar_id, seed, interrupt_at)
                    row = await _run_session(
                        http, base, ws_base, avatar_id=avatar_id, seed=seed,
                        interrupt_at=interrupt_at, out_dir=out_dir,
                        utt_s=args.duration, utt2_s=args.duration2,
                        transition_frames=args.transition,
                    )
                    rows.append(row)
                    print(
                        f"[{len(rows)}] {avatar_id} @{interrupt_at:.2f}s "
                        f"e2e={row.get('client_e2e_ms')}ms "
                        f"resume={row.get('resume_ms')}ms "
                        f"cut@{row.get('cut_in_utterance_s')}s"
                        f"{' ERR:' + row['error'] if row.get('error') else ''}",
                        flush=True,
                    )
    finally:
        server.should_exit = True
        await asyncio.sleep(0.2)

    report = {
        "mode": "interruption_eval_media",
        "schema": _SCHEMA,
        "avatars": avatars,
        "points": points,
        "seeds": seeds,
        "transition_frames": max(0, args.transition),
        "rows": [
            {k: v for k, v in r.items() if k != "frame_timeline"} for r in rows
        ],
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "record_index.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    errors = [r for r in rows if r.get("error")]
    print(f"recorded {len(rows)} sessions ({len(errors)} errors) → {args.out}")
    return 1 if errors else 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())

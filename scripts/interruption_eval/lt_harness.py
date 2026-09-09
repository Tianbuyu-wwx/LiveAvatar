# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-E: LiveTalking baseline harness — unified interruption protocol + recording.

Runs inside the LiveTalking venv (needs aiortc + aiohttp; numpy only for
``prepare``). Drives Design-B sessions against a running LiveTalking server:

    prepare  — synthesize the deterministic utterance wavs (same seeds/audio
               as the LiveAvatar arm: utt1 = synth(seed), utt2 = synth(seed+1000))
    session  — one Design-B session: WebRTC connect → /humanaudio utt1 →
               /interrupt_talk at t_int (from SSE playback anchor) →
               /humanaudio utt2 → server-side /record mp4 + meta.json
    run      — the full matrix loop (avatars × 20 seeds × repeats)

Latency/metric anchors (documented for the paper):
- ``utt1_start`` / ``utt2_start``: SSE ``{"status": "start"}`` events fired at
  the WebRTC OUTPUT side (playback), client wall clock.
- ``interrupt_sent``: client wall clock just before POST /interrupt_talk.
- Content-level timing (audio stop, utt2 audio start, visual resume) is
  derived offline from the server-side recording by cross-correlating the
  recorded PCM with the known utterance wavs (sample-accurate) — see
  lt_extract.py.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import aiohttp

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from audio_synth import Utterance, synth_utterance  # noqa: E402

_UTT_S = 6.0
_UTT2_S = 2.5  # post-interrupt response duration (mirrors record.py)
_POLL_HZ = 40.0


def write_wav(path: Path, utt: Utterance) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(utt.sr)
        w.writeframes(utt.pcm_s16le)


def cmd_prepare(args: argparse.Namespace) -> int:
    out = Path(args.wav_dir)
    for seed in range(1, args.seeds + 1):
        utt1 = synth_utterance(_UTT_S, seed)
        utt2 = synth_utterance(_UTT2_S, seed + 1000)
        write_wav(out / f"utt1_seed{seed}.wav", utt1)
        write_wav(out / f"utt2_seed{seed}.wav", utt2)
        (out / f"events_seed{seed}.json").write_text(
            json.dumps(
                {
                    "utt1": utt1.to_meta(),
                    "utt2": utt2.to_meta(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    print(f"prepared {args.seeds} utterance pairs -> {out}")
    return 0


def _interrupt_grid(points: int) -> list[float]:
    return [round(0.5 + i * 4.0 / max(points - 1, 1), 3) for i in range(points)]


class LTClient:
    """Async client for one LiveTalking session (WebRTC + HTTP + SSE)."""

    def __init__(self, base: str, avatar: str) -> None:
        self.base = base.rstrip("/")
        self.avatar = avatar
        self.sessionid: str | None = None
        self.sse_events: list[dict[str, Any]] = []
        self.speak_log: list[tuple[float, bool]] = []
        self._sse_task: asyncio.Task | None = None
        self._poll_task: asyncio.Task | None = None
        self._http: aiohttp.ClientSession | None = None

    async def connect(self, http: aiohttp.ClientSession, pc: Any) -> None:
        self._http = http
        # recv-only audio+video transceivers
        pc.addTransceiver("audio", direction="recvonly")
        pc.addTransceiver("video", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        async with http.post(
            f"{self.base}/offer",
            json={"sdp": pc.localDescription.sdp, "type": "offer", "avatar": self.avatar},
        ) as resp:
            body = await resp.json()
        if "sessionid" not in body or body.get("code") == -1:
            raise RuntimeError(f"offer failed: {body}")
        self.sessionid = str(body["sessionid"])
        answer = body["sdp"]
        from aiortc import RTCSessionDescription

        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
        self._sse_task = asyncio.create_task(self._sse_loop())

    async def _sse_loop(self) -> None:
        assert self._http and self.sessionid
        try:
            async with self._http.get(
                f"{self.base}/sse", params={"sessionid": self.sessionid}
            ) as resp:
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        payload = json.loads(line[len("data: "):])
                    except json.JSONDecodeError:
                        continue
                    payload["_ts"] = time.perf_counter()
                    self.sse_events.append(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.sse_events.append({"_ts": time.perf_counter(), "_sse_error": str(exc)})

    async def start_speak_poll(self) -> None:
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop_speak_poll(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def _poll_loop(self) -> None:
        assert self._http and self.sessionid
        while True:
            try:
                async with self._http.post(
                    f"{self.base}/is_speaking", json={"sessionid": self.sessionid}
                ) as resp:
                    body = await resp.json()
                speaking = bool(body.get("data"))
            except Exception:  # noqa: BLE001
                speaking = self.speak_log[-1][1] if self.speak_log else False
            self.speak_log.append((time.perf_counter(), speaking))
            await asyncio.sleep(1.0 / _POLL_HZ)

    async def record(self, action: str) -> None:
        assert self._http and self.sessionid
        async with self._http.post(
            f"{self.base}/record",
            json={"sessionid": self.sessionid, "type": action},
        ) as resp:
            body = await resp.json()
        if body.get("code") not in (0, None):
            raise RuntimeError(f"record {action} failed: {body}")

    async def send_audio(self, wav_path: Path) -> None:
        assert self._http and self.sessionid
        data = aiohttp.FormData()
        data.add_field("sessionid", self.sessionid)
        data.add_field(
            "file", wav_path.read_bytes(), filename=wav_path.name,
            content_type="audio/wav",
        )
        async with self._http.post(f"{self.base}/humanaudio", data=data) as resp:
            body = await resp.json()
        if body.get("code") not in (0, None):
            raise RuntimeError(f"humanaudio failed: {body}")

    async def interrupt(self) -> float:
        assert self._http and self.sessionid
        t0 = time.perf_counter()
        async with self._http.post(
            f"{self.base}/interrupt_talk", json={"sessionid": self.sessionid}
        ) as resp:
            body = await resp.json()
        if body.get("code") not in (0, None):
            raise RuntimeError(f"interrupt_talk failed: {body}")
        return t0

    async def fetch_recording(self, out_path: Path) -> None:
        assert self._http and self.sessionid
        async with self._http.get(
            f"{self.base}/record/{self.sessionid}"
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"record download failed: {resp.status}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(await resp.read())

    def wait_event(self, status: str, after_ts: float, timeout: float) -> float | None:
        deadline = time.perf_counter() + timeout
        seen = 0
        while time.perf_counter() < deadline:
            for ev in self.sse_events[seen:]:
                seen += 1
                if ev.get("status") == status and ev.get("_ts", 0) >= after_ts:
                    return ev["_ts"]
            time.sleep(0.002)
        return None


async def run_session(
    base: str, avatar: str, seed: int, interrupt_at: float, repeat: int,
    wav_dir: Path, out_dir: Path, *, poll_speak: bool = True,
) -> dict:
    from aiortc import RTCPeerConnection

    out_dir.mkdir(parents=True, exist_ok=True)
    utt1_path = wav_dir / f"utt1_seed{seed}.wav"
    utt2_path = wav_dir / f"utt2_seed{seed}.wav"
    client = LTClient(base, avatar)
    pc = RTCPeerConnection()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http:
        await client.connect(http, pc)
        await asyncio.sleep(0.3)  # let the peer + render loop settle

        await client.record("start_record")
        t_before_utt1 = time.perf_counter()
        await client.send_audio(utt1_path)
        utt1_start = client.wait_event("start", t_before_utt1, timeout=10.0)
        if utt1_start is None:
            raise RuntimeError("no SSE start event for utterance 1")

        if poll_speak:
            await client.start_speak_poll()

        # schedule the interrupt relative to the playback-side anchor
        delay = interrupt_at - (time.perf_counter() - utt1_start)
        if delay > 0:
            await asyncio.sleep(delay)
        t0 = await client.interrupt()
        await client.send_audio(utt2_path)
        t_utt2_post = time.perf_counter()

        utt2_start = client.wait_event("start", t_utt2_post, timeout=15.0)
        utt2_end = client.wait_event("end", t_utt2_post, timeout=30.0)
        await asyncio.sleep(0.8)  # tail margin (last frames drain)

        await client.stop_speak_poll()
        await client.record("end_record")
        await asyncio.sleep(0.5)  # server muxes the mp4 synchronously

        mp4_path = out_dir / "recording.mp4"
        await client.fetch_recording(mp4_path)

    meta = {
        "kind": "interrupt",
        "system": "livetalking",
        "protocol": "lt_design_b",
        "sessionid": client.sessionid,
        "avatar_id": avatar,
        "seed": seed,
        "repeat": repeat,
        "interrupt_at_s": interrupt_at,
        "utt1_wav": utt1_path.name,
        "utt2_wav": utt2_path.name,
        "utt1": {"seed": seed, "duration_s": _UTT_S, "sr": 16000},
        "utterance_2": {"seed": seed + 1000, "duration_s": _UTT2_S, "sr": 16000},
        "perf_clock": {
            "utt1_start": utt1_start,
            "interrupt_sent": t0,
            "utt2_post": t_utt2_post,
            "utt2_start": utt2_start,
            "utt2_end": utt2_end,
        },
        "interrupt_to_utt2_post_ms": (t_utt2_post - t0) * 1000.0,
        "sse_events": client.sse_events,
        "speak_log": [[t, s] for t, s in client.speak_log],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        await pc.close()
    except Exception:  # noqa: BLE001
        pass
    return meta


async def run_reference(
    base: str, avatar: str, seed: int, repeat: int, wav_dir: Path, out_dir: Path
) -> dict:
    """Uninterrupted recording of utt1 (V-APT reference curve)."""
    from aiortc import RTCPeerConnection

    out_dir.mkdir(parents=True, exist_ok=True)
    utt1_path = wav_dir / f"utt1_seed{seed}.wav"
    client = LTClient(base, avatar)
    pc = RTCPeerConnection()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http:
        await client.connect(http, pc)
        await asyncio.sleep(0.3)
        await client.record("start_record")
        t_before = time.perf_counter()
        await client.send_audio(utt1_path)
        utt1_start = client.wait_event("start", t_before, timeout=10.0)
        if utt1_start is None:
            raise RuntimeError("no SSE start event (reference)")
        await asyncio.sleep(_UTT_S + 1.5)  # full utterance + tail margin
        await client.record("end_record")
        await asyncio.sleep(0.5)
        mp4_path = out_dir / "recording.mp4"
        await client.fetch_recording(mp4_path)
    meta = {
        "kind": "reference",
        "system": "livetalking",
        "protocol": "lt_design_b",
        "sessionid": client.sessionid,
        "avatar_id": avatar,
        "seed": seed,
        "repeat": repeat,
        "utt1_wav": utt1_path.name,
        "utt1": {"seed": seed, "duration_s": _UTT_S, "sr": 16000},
        "perf_clock": {"utt1_start": utt1_start},
        "sse_events": client.sse_events,
        "speak_log": [[t, s] for t, s in client.speak_log],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        await pc.close()
    except Exception:  # noqa: BLE001
        pass
    return meta


async def cmd_run(args: argparse.Namespace) -> int:
    wav_dir = Path(args.wav_dir)
    out_root = Path(args.out_root)
    avatars = [a for a in args.avatars.split(",") if a]
    grid = _interrupt_grid(args.points)
    rows: list[dict] = []
    if args.references:
        # Uninterrupted reference sessions (V-APT baseline curves), one per
        # seed on the first avatar — mirrors record.py's convention.
        for seed in range(1, args.points + 1):
            out_dir = out_root / f"reference_{avatars[0]}_seed{seed}"
            try:
                meta = await run_reference(
                    args.base, avatars[0], seed, 0, wav_dir, out_dir
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] ref seed{seed}: {exc}")
                rows.append({"avatar_id": avatars[0], "seed": seed,
                             "repeat": 0, "kind": "reference",
                             "error": str(exc)})
                continue
            rows.append(meta)
            print(f"[ref {seed}/{args.points}]", flush=True)
            await asyncio.sleep(args.session_gap)
    for avatar in avatars:
        for repeat in range(1, args.repeats + 1):
            for i, t_int in enumerate(grid):
                seed = i + 1
                out_dir = out_root / f"interrupt_{avatar}_seed{seed}_t{t_int:.2f}s_r{repeat}"
                try:
                    meta = await run_session(
                        args.base, avatar, seed, t_int, repeat, wav_dir, out_dir
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[FAIL] {avatar} seed{seed} r{repeat}: {exc}")
                    rows.append(
                        {"avatar_id": avatar, "seed": seed, "repeat": repeat,
                         "interrupt_at_s": t_int, "error": str(exc)}
                    )
                    continue
                rows.append(meta)
                print(
                    f"[{len(rows)}] {avatar} seed{seed} r{repeat} @{t_int:.2f}s "
                    f"utt2@{(meta['perf_clock']['utt2_start'] or 0):.3f}",
                    flush=True,
                )
                await asyncio.sleep(args.session_gap)
    (out_root / "record_index.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fails = sum(1 for r in rows if r.get("error"))
    print(f"done: {len(rows)} sessions, {fails} failures")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="synthesize utterance wavs")
    p.add_argument("--wav-dir", default="data/interruption_eval/lt_wavs")
    p.add_argument("--seeds", type=int, default=20)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("session", help="run one Design-B session")
    p.add_argument("--base", default="http://127.0.0.1:8010")
    p.add_argument("--avatar", default="yongen")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--interrupt-at", type=float, required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--wav-dir", default="data/interruption_eval/lt_wavs")
    p.add_argument("--out", required=True)
    p.add_argument("--no-poll", action="store_true")
    p.set_defaults(fn=None)

    p = sub.add_parser("run", help="full matrix loop")
    p.add_argument("--base", default="http://127.0.0.1:8010")
    p.add_argument("--avatars", default="yongen")
    p.add_argument("--points", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--references", action="store_true",
                   help="also record uninterrupted reference sessions")
    p.add_argument("--wav-dir", default="data/interruption_eval/lt_wavs")
    p.add_argument("--out-root", required=True)
    p.add_argument("--session-gap", type=float, default=1.0)
    p.set_defaults(fn=cmd_run)

    args = ap.parse_args()
    if args.cmd == "session":
        out = Path(args.out)
        meta = asyncio.run(
            run_session(
                args.base, args.avatar, args.seed, args.interrupt_at,
                args.repeat, Path(args.wav_dir), out, poll_speak=not args.no_poll,
            )
        )
        print(json.dumps(meta["perf_clock"], indent=2))
        return 0
    if inspect.iscoroutinefunction(args.fn):
        return asyncio.run(args.fn(args))
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

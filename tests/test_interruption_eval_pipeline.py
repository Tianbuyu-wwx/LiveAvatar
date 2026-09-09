# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-B pipeline unit tests: extractor, analyze, report, record round-trip.

The record round-trip boots the real publish service (CPU, duplex) with
the SynthTts swap and records one tiny interrupted + one reference
session — the same machinery as the full dataset run, miniaturized.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pytest

_SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "interruption_eval"
)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import metrics as m  # noqa: E402

try:
    import cv2  # noqa: F401

    _HAS_CV2 = True
except Exception:  # pragma: no cover - light CI envs
    _HAS_CV2 = False

_NEED_CV2 = pytest.mark.skipif(not _HAS_CV2, reason="cv2 required")

try:
    from liveavatar.audio_in.frame import PCMFrame  # noqa: F401

    _HAS_AUDIO = True
except Exception:  # pragma: no cover - light CI envs
    _HAS_AUDIO = False

_NEED_AUDIO = pytest.mark.skipif(not _HAS_AUDIO, reason="audio_in required")


# ─────────────────────────────────────────────── openness extractor


@_NEED_CV2
class TestExtractor:
    def test_monotonic_and_calibrated(self) -> None:
        import analyze as az
        from mouth_proxy import render_face

        ops = np.linspace(0.05, 0.95, 10)
        vals = [az.extract_openness(render_face(float(op), t_s=0.3)) for op in ops]
        assert all(b >= a - 0.02 for a, b in zip(vals, vals[1:], strict=False))
        assert vals[-1] > vals[0] + 0.5  # strongly increasing overall
        # Mid-range linearity (spike result: pixel area ≈ linear in openness).
        mid = az.extract_openness(render_face(0.5, t_s=0.3))
        assert abs(mid - 0.5) < 0.15

    def test_idle_frame_reads_closed(self) -> None:
        import analyze as az
        from mouth_proxy import render_face

        assert az.extract_openness(render_face(0.02, t_s=0.0)) < 0.12


# ──────────────────────────────────────────────────────── Spearman ρ


class TestSpearman:
    def test_perfect_monotonic(self) -> None:
        from report import spearman_rho

        assert spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
        assert spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_ties_use_average_ranks(self) -> None:
        from report import _average_ranks, spearman_rho

        assert _average_ranks(np.array([1.0, 2.0, 2.0, 3.0])).tolist() == [
            1.0, 2.5, 2.5, 4.0,
        ]
        rho = spearman_rho([1, 2, 3, 4], [1, 2, 2, 4])
        assert 0.85 < rho < 1.0

    def test_constant_series_is_zero(self) -> None:
        from report import spearman_rho

        assert spearman_rho([2, 2, 2, 2], [1, 2, 3, 4]) == 0.0


# ──────────────────────────────── analyze over fabricated mini-dataset


def _write_session(
    out_dir: str,
    *,
    kind: str,
    seed: int = 7,
    utt_s: float = 3.0,
    interrupt_sent_s: float | None = None,
    tts_start_s: float = 0.5,
) -> None:
    """Fabricate a recorded session (recorder design B): frames + meta.

    Epoch-1 frames start at ``tts_start_s`` (frame 0 = utterance frame 0
    on the 40 ms clock). Interrupted sessions add two stale epoch-1
    frames past the cut, a ~0.7 s screen-freeze gap (NO frames — the
    browser holds the last one), then epoch-2 frames driven by a second
    seeded utterance (lead-in pause → closed mouth).
    """
    import cv2
    from audio_synth import synth_utterance
    from mouth_proxy import render_face

    period = 0.04
    utt = synth_utterance(utt_s, seed)
    utt2 = synth_utterance(2.5, seed + 1000)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cut_u = (
        round(interrupt_sent_s - tts_start_s, 4)
        if interrupt_sent_s is not None and kind == "interrupt"
        else None
    )
    gap_s = 0.7
    stale = 2
    if cut_u is not None:
        n1 = int(cut_u / period) + stale
        n2 = 30  # 1.2 s of epoch-2 content
        u2_0 = cut_u + gap_s
    else:
        n1 = int(utt_s / period)
        n2 = 0
        u2_0 = None

    specs: list[tuple[float, int, float]] = []  # (ts_rel, epoch, openness)
    for k1 in range(n1):
        u = k1 * period
        specs.append((tts_start_s + u, 1, max(0.05, utt.openness_at(u))))
    for k2 in range(n2):
        u = u2_0 + k2 * period
        specs.append((tts_start_s + u, 2, max(0.05, utt2.openness_at(k2 * period))))

    timeline = []
    for i, (t, epoch, op) in enumerate(specs):
        fname = f"f{i:05d}.jpg"
        img = render_face(float(op), t_s=t)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        buf.tofile(os.path.join(frames_dir, fname))
        timeline.append(
            {"ts_rel": round(t, 4), "epoch": epoch,
             "boundary": epoch == 2, "file": fname}
        )

    meta = {
        "kind": kind,
        "avatar_id": "mouth_a",
        "seed": seed,
        "schema": 2,
        "utterance_duration_s": utt_s,
        "interrupt_at_s": interrupt_sent_s,
        "interrupt_sent_s": interrupt_sent_s,
        "tts_start_s": tts_start_s,
        "frame_period_s": period,
        "tts_chunk_s": 0.16,
        "cut_in_utterance_s": (
            round(interrupt_sent_s - tts_start_s, 4)
            if interrupt_sent_s is not None
            else None
        ),
        "frame_timeline": timeline,
        "utterance": utt.to_meta(),
    }
    if u2_0 is not None:
        first_e2 = next(fr for fr in timeline if fr["epoch"] == 2)
        meta["utt2_start_s"] = first_e2["ts_rel"]
        meta["utterance_2"] = {"seed": seed + 1000, **utt2.to_meta()}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


@_NEED_CV2
class TestAnalyzeFabricated:
    @pytest.fixture()
    def dataset(self, tmp_path: pytest.TempPathFactory) -> dict:
        from audio_synth import synth_utterance

        root = str(tmp_path)
        _write_session(os.path.join(root, "reference_mouth_a_seed7"), kind="reference")
        # Place the cut mid-vowel (high openness) so the freeze-then-idle
        # jump is large; fall back to 1.0 s if the seed has no such vowel.
        utt = synth_utterance(3.0, 7)
        u_cut = next(
            (u / 1000.0 for u in range(500, 2600)
             if utt.openness_at(u / 1000.0) >= 0.6),
            1.0,
        )
        assert utt.openness_at(u_cut - 0.02) >= 0.5  # pre-cut mouth is open
        cut = 0.5 + u_cut  # tts_start_s = 0.5
        _write_session(
            os.path.join(root, "interrupt_mouth_a_seed7_t1.50s"),
            kind="interrupt",
            interrupt_sent_s=cut,
        )
        return {"root": root, "cut": cut}

    def test_end_to_end_metrics(self, dataset: dict) -> None:
        import analyze as az

        d = os.path.join(dataset["root"], "interrupt_mouth_a_seed7_t1.50s")
        ref = os.path.join(dataset["root"], "reference_mouth_a_seed7")
        out = az.analyze_session(d, ref)
        with open(os.path.join(d, "metrics.json"), encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == out  # metrics.json mirrors the return value

        # The frozen mouth deviates from the still-speaking reference
        # inside the transition window → positive V-APT.
        assert out["v_apt"]["area"] > 0.2
        assert out["v_apt"]["apt_ms"] > 8.0
        # The jump from the frozen open mouth to the epoch-2 lead-in
        # pause (closed) is detected.
        assert out["smoothness"]["mutation"] > 0.2
        assert out["smoothness"]["freeze_frames"] >= 0
        # Post-cut audio is silence then utterance 2 (starts with a
        # pause) while the mouth freezes open → most frames mismatch.
        assert out["mismatch"]["rate"] > 0.5
        assert out["mismatch"]["audio"] == "pause+utt2"

        # Half-syllable flag matches the direct metric call on the meta.
        meta_path = os.path.join(d, "meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        expected = m.half_syllable_flag(
            meta["cut_in_utterance_s"], meta["utterance"]["events"]
        )
        assert out["half_syllable"]["flag"] == expected

    def test_reference_session_gets_no_cut_metrics(self, dataset: dict) -> None:
        import analyze as az

        d = os.path.join(dataset["root"], "reference_mouth_a_seed7")
        out = az.analyze_session(d)
        assert "v_apt" not in out
        assert out["note"]

    def test_analyze_all_and_report(self, dataset: dict) -> None:
        import analyze as az
        import report as rp

        rows = az.analyze_all(dataset["root"])
        assert len(rows) == 1
        assert "error" not in rows[0]

        rep = rp.aggregate(dataset["root"])
        assert rep["n_interrupt_sessions"] == 1
        assert rep["overall"]["v_apt_ms"]["p50"] > 0
        assert 0.0 <= rep["overall"]["half_syllable_rate"] <= 1.0
        md = rp._md(rep)
        assert "P-B" in md and "V-APT" in md


# ───────────────────────────────────── blind key / Spearman round-trip


class TestSpearmanGate:
    def _fabricate(self, root: str, reverse: bool) -> tuple[str, str]:
        sessions = []
        for i in range(6):
            d = os.path.join(root, f"interrupt_mouth_a_seed{i + 1}_t1.00s")
            os.makedirs(d, exist_ok=True)
            vapt = i * 100.0 + 50.0
            with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "kind": "interrupt",
                        "avatar_id": "mouth_a",
                        "seed": i + 1,
                        "v_apt": {"apt_ms": vapt, "area": 0.0, "recover_idx": None,
                                  "span_frames": 0, "window_frames": 0},
                        "mismatch": {"rate": vapt / 600.0, "delta_s": 0.6},
                        "smoothness": {"mutation": vapt / 600.0, "baseline_p50": 0.01,
                                       "ratio": 2.0, "freeze_frames": 1},
                    },
                    f,
                )
            sessions.append(f"interrupt_mouth_a_seed{i + 1}_t1.00s")
        key = {f"clip_{k:02d}": d for k, d in enumerate(sessions, start=1)}
        key_path = os.path.join(root, "key.json")
        with open(key_path, "w", encoding="utf-8") as f:
            json.dump(key, f)
        ratings = os.path.join(root, "ratings.csv")
        with open(ratings, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "naturalness_1to5", "comfort_1to5", "comment"])
            for k in range(1, 7):
                # Ratings rank by clip order; "reverse" flips the rating
                # direction against the (fixed, ascending) metric values.
                w.writerow([f"clip_{k:02d}", (7 - k) if reverse else k, "", ""])
        return key_path, ratings

    def test_aligned_ratings_pass_gate(self, tmp_path) -> None:
        import report as rp

        key, ratings = self._fabricate(str(tmp_path), reverse=False)
        rc = rp.cmd_spearman(
            argparse_ns(ratings=ratings, key=key, out=str(tmp_path / "s.json"))
        )
        assert rc == 0
        with open(tmp_path / "s.json", encoding="utf-8") as f:
            out = json.load(f)
        assert out["gate"]["pass"] is True
        assert out["spearman"]["v_apt_ms"] == pytest.approx(1.0)

    def test_anti_correlated_ratings_fail_gate(self, tmp_path) -> None:
        import report as rp

        key, ratings = self._fabricate(str(tmp_path), reverse=True)
        rc = rp.cmd_spearman(
            argparse_ns(ratings=ratings, key=key, out=None)
        )
        assert rc == 2  # gate exit code


def argparse_ns(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)


# ─────────────────────────────────── blind mux: grid alignment (P-B7)


@_NEED_CV2
class TestBlindMux:
    def _fabricate_session(self, root: str) -> str:
        from mouth_proxy import render_face

        d = os.path.join(root, "interrupt_mouth_a_seed1_t1.00s")
        os.makedirs(os.path.join(d, "frames"), exist_ok=True)
        # Frames: 6 on the 40 ms grid, a 0.28 s freeze gap (NO frames —
        # the screen holds), then 2 epoch-2 frames.
        ts = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.48, 0.52]
        ops = [0.2, 0.4, 0.55, 0.7, 0.8, 0.85, 0.05, 0.08]
        timeline = []
        for i, (t, op) in enumerate(zip(ts, ops, strict=True)):
            fn = f"f{i:03d}.png"
            ok, buf = cv2.imencode(".png", render_face(float(op), t_s=t))
            assert ok
            buf.tofile(os.path.join(d, "frames", fn))
            timeline.append(
                {"ts_rel": t, "epoch": 1 if i < 6 else 2,
                 "boundary": False, "file": fn}
            )
        # TTS chunks: two before the cut, one after the resume — tones
        # at distinct frequencies so placement is verifiable.
        def _pcm(freq: float, dur: float) -> bytes:
            t = np.arange(int(16000 * dur)) / 16000
            return (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16).tobytes()

        segs = [(0.02, _pcm(300.0, 0.2)), (0.24, _pcm(500.0, 0.2)),
                (0.62, _pcm(700.0, 0.2))]
        import wave

        with wave.open(os.path.join(d, "audio.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(s for _, s in segs))
        meta = {
            "kind": "interrupt",
            "avatar_id": "mouth_a",
            "seed": 1,
            "frame_period_s": 0.04,
            "frame_timeline": timeline,
            "tts_chunk_timeline": [
                {"ts_rel": tc, "bytes": len(s)} for tc, s in segs
            ],
        }
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return d

    def test_grid_expand_holds_freeze_gap(self) -> None:
        import report as rp

        timeline = [
            {"ts_rel": 0.0}, {"ts_rel": 0.04}, {"ts_rel": 0.20},
            {"ts_rel": 0.48},
        ]
        grid = rp._grid_expand(timeline, 0.04)
        assert len(grid) == 13  # 1 + 1 + 4 (Δ=0.16) + 7 (Δ=0.28)
        assert grid[2] is grid[5]  # hold repeats the last pre-gap frame
        assert grid[6] is timeline[3]

    def test_aligned_audio_places_chunks_with_silence_gap(self, tmp_path) -> None:
        import wave

        import report as rp

        d = self._fabricate_session(str(tmp_path))
        out = os.path.join(str(tmp_path), "aligned.wav")
        dur = rp._aligned_audio(d, out, t0_s=0.0)
        assert abs(dur - 0.82) < 1e-6
        with wave.open(out, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        assert len(pcm) == 13120  # 0.82 s at 16 kHz
        assert np.abs(pcm[320:340]).max() > 1000  # chunk 1 present
        assert np.abs(pcm[7100:9500]).max() == 0  # post-cut gap = silence
        assert np.abs(pcm[9920:9960]).max() > 1000  # resume chunk placed

    def test_write_clip_grid_frame_count(self, tmp_path) -> None:
        import cv2
        import report as rp

        d = self._fabricate_session(str(tmp_path))
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        mp4 = os.path.join(str(tmp_path), "clip.mp4")
        assert rp._write_clip(d, mp4, meta["frame_timeline"], 25.0, 0.04)
        cap = cv2.VideoCapture(mp4)
        cnt = 0
        while cap.read()[0]:
            cnt += 1
        cap.release()
        # 6 real frames + 6 holds across the 0.28 s gap + 2 = 14.
        assert cnt == 14

    def test_cmd_blind_end_to_end(self, tmp_path) -> None:
        import report as rp

        root = str(tmp_path)
        d = self._fabricate_session(root)
        with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"kind": "interrupt", "avatar_id": "mouth_a", "seed": 1,
                 "v_apt": {"apt_ms": 120.0}, "error": None},
                f,
            )
        rc = rp.cmd_blind(argparse_ns(data=root, n=1, seed=0, out=None))
        assert rc == 0
        blind = os.path.join(root, "blind")
        assert os.path.isfile(os.path.join(blind, "clip_01.mp4"))
        assert os.path.isfile(os.path.join(blind, "clip_01.wav"))
        assert os.path.isfile(os.path.join(root, "key.json"))


# ───────────────────────────────────────── record round-trip (service)


@_NEED_AUDIO
@_NEED_CV2
class TestRecordRoundTrip:
    def test_tiny_interrupt_and_reference(self, tmp_path, monkeypatch) -> None:
        import asyncio

        import httpx2 as httpx
        import record as rec

        import liveavatar.runtime.worker as worker_mod

        # Swap the default TTS for the duration of this test only
        # (monkeypatch restores FakeTts for the other suites).
        monkeypatch.setattr(worker_mod, "FakeTts", rec.SynthTts)
        server, _thread, port = rec._start_server(["mouth_a"])
        base = f"http://127.0.0.1:{port}"
        ws_base = f"ws://127.0.0.1:{port}"
        try:

            async def run() -> tuple[dict, dict]:
                async with httpx.AsyncClient(timeout=30.0) as http:
                    ref_dir = str(tmp_path / "reference_mouth_a_seed1")
                    ref = await rec._run_session(
                        http, base, ws_base, avatar_id="mouth_a", seed=1,
                        interrupt_at=None, out_dir=ref_dir, utt_s=1.5,
                    )
                    int_dir = str(tmp_path / "interrupt_mouth_a_seed1_t0.50s")
                    row = await rec._run_session(
                        http, base, ws_base, avatar_id="mouth_a", seed=1,
                        interrupt_at=0.5, out_dir=int_dir, utt_s=1.5,
                    )
                return ref, row

            ref, row = asyncio.run(run())
        finally:
            server.should_exit = True

        # Reference: full utterance received, media on disk. 1.5 s at
        # 160 ms chunks = 10 chunks → 40 frames, lossless capture.
        assert ref.get("error") is None
        assert os.path.isfile(os.path.join(str(tmp_path / "reference_mouth_a_seed1"), "audio.wav"))
        assert ref["tts_samples_received"] > 8000  # ≥ 0.5 s of audio
        assert len(ref["frame_timeline"]) >= 36

        # Interrupted: cut measured on the client timeline; the second
        # trigger spawns the epoch-2 (resume) response.
        assert row.get("error") is None
        cut = row["cut_in_utterance_s"]
        assert 0.2 < cut < 0.9  # target 0.5 with realtime-pacing jitter
        d = str(tmp_path / "interrupt_mouth_a_seed1_t0.50s")
        assert os.path.isfile(os.path.join(d, "audio.wav"))
        e2 = [fr for fr in row["frame_timeline"] if fr["epoch"] > 1]
        assert len(e2) >= 8, "epoch-2 (resume) response must render frames"
        assert row.get("utt2_start_s") is not None
        assert row.get("resume_ms") is not None
        assert row.get("utterance_2", {}).get("events")
        assert row["tts_samples_received"] > 12000  # cut tail + utt2
        # Frame files must exist where the timeline says they do.
        first_frame = row["frame_timeline"][0]["file"]
        assert os.path.isfile(os.path.join(d, "frames", first_frame))
        last_frame = row["frame_timeline"][-1]["file"]
        assert os.path.isfile(os.path.join(d, "frames", last_frame))
        # Phoneme schedule matches the seed (1.5 s utterance).
        assert row["utterance"]["events"], "ground-truth events must be present"

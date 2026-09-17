"""P-E4: LiveTalking full matrix chain (yongen then sun), batched.

Runs the LT server as subprocess.Popen with stdout/stderr redirected to
FILE HANDLES (never pipes). PowerShell's Start-Process -RedirectStandardOutput
creates an unread pipe; LiveTalking's log volume fills it and the server
dies after ~2 sessions. File redirection (this script, or `*> file` in the
shell job host) has no such cap and servers run stably.

Batching: the per-session native-thread leak degrades the pipeline after
4-7 sessions, so 2 sessions run per server process, restart between
batches.

GPU-contention gate: when the GPU is loaded by another process (game,
video call, a second MuseTalk server left alive), MuseTalk's UNET batch-16
forward stretches from ~0.1s to 1-10s (py-spy 2026-09-16: inference thread
pinned inside UNET, render thread blocked at whisper.feat_queue.put), the
audio pipeline consumes at ~0.1x realtime, SSE eventpoints fire late, and
every session dies with "no SSE start event for utterance 2" — silently
poisoning the dataset. A raw matmul probe does NOT catch this (1.7ms even
with a game at 71% GPU util: the working set is too small to trigger the
VRAM spill). The gate is therefore END-TO-END: after the server listens,
one micro reference session is recorded and gated on its utt1 push→
playback latency (healthy 2-4s, crawling 10-20s).
"""
from __future__ import annotations

import argparse
import http.client
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = r"E:\lt\.venv\Scripts\python.exe"
HARNESS = r"E:\lva\scripts\interruption_eval\lt_harness.py"
WORKDIR = r"E:\lt"
WAV_DIR = r"E:\lva\data\interruption_eval\lt_wavs"
DATA_ROOT = r"E:\lva\data\interruption_eval"
POINTS = 20
REPEATS = 3
# After the SSE pipeline fix the per-session thread leak no longer
# compounds across active sessions, so a full 2xPOINTS batch fits inside
# one healthy server window (measured: utt2 latency grows ~20 ms/session,
# last session still ~6.6 s vs the 6.5 s probe threshold margin).
# Sweep re-runs preskip complete batches, so BATCH only bounds how much
# a GPU-CONTENDED abort throws away.
BATCH = 10
# utt1 push→playback latency above which the server is declared contended
# (healthy 2-4s; crawling 10-20s). The probe costs one extra active session
# per fresh server (3 total incl. the batch) — still inside the healthy
# window given clean per-session teardown.
PROBE_THRESHOLD_MS = 6500.0


def kill_all_servers() -> None:
    """taskkill every app.py python tree, then wait until port 8010 is gone."""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
         "Where-Object { $_.CommandLine -match 'app\\.py' } | "
         "ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>$null }"],
        capture_output=True)
    for _ in range(30):
        if not _port_listening():
            return
        time.sleep(2)
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
             "Where-Object { $_.CommandLine -match 'app\\.py' } | "
             "ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>$null }"],
            capture_output=True)


def _port_listening() -> bool:
    # 5s: the MuseTalk server holds the GIL during startup/first-session
    # warmup; a 2s probe times out and misclassifies a healthy server.
    # ANY HTTP status counts: GET / is a 302 redirect (http.client does not
    # follow it), so requiring 200 here misclassified every healthy server
    # as SERVER-TIMEOUT (observed 6x before diagnosis).
    conn = http.client.HTTPConnection("127.0.0.1", 8010, timeout=5)
    try:
        conn.request("GET", "/")
        return conn.getresponse().status > 0
    except Exception:  # noqa: BLE001
        return False
    finally:
        conn.close()


def probe_server(avatar: str) -> tuple[bool, str]:
    """End-to-end GPU/health gate: one micro reference session, judged on
    its utt1 push-to-playback latency. Returns (healthy, detail)."""
    scratch = Path(WORKDIR) / f"lt_probe_{avatar}_{int(time.time())}"
    cmd = [PY, HARNESS, "run", "--base", "http://127.0.0.1:8010",
           "--avatars", avatar, "--points", str(POINTS),
           "--repeats", str(REPEATS), "--wav-dir", WAV_DIR,
           "--out-root", str(scratch), "--retries", "0",
           "--ref-range", "0:1", "--references", "--sess-range", "0:0",
           "--index-name", "index_probe.json"]
    try:
        rc = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=120).returncode
        meta_path = scratch / f"reference_{avatar}_seed1" / "meta.json"
        if rc == 0 and meta_path.is_file():
            ms = float(json.loads(
                meta_path.read_text(encoding="utf-8"))["utt1_start_latency_ms"])
            return ms <= PROBE_THRESHOLD_MS, f"utt1_latency={ms:.0f}ms"
        return False, "probe-session-failed"
    except Exception as exc:  # noqa: BLE001
        return False, f"probe-error:{exc}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def start_server(avatar: str, log_base: Path) -> subprocess.Popen | None:
    kill_all_servers()
    out_log = open(log_base.with_suffix(".out.log"), "ab")  # noqa: SIM115
    err_log = open(log_base.with_suffix(".err.log"), "ab")  # noqa: SIM115
    proc = subprocess.Popen(
        [PY, "app.py", "--transport", "webrtc", "--model", "musetalk",
         "--avatar_id", avatar, "--tts", "edgetts",
         "--listenport", "8010", "--stun="],
        cwd=WORKDIR, stdout=out_log, stderr=err_log)
    # 180 x ~2s = 6min window: MuseTalk cold start (weights + CUDA ctx +
    # avatar latents + warmup) takes ~3.5-4min; a 3min window misclassifies
    # a healthy loading server as SERVER-TIMEOUT (observed 3x in a row).
    for _ in range(180):
        time.sleep(2)
        if proc.poll() is not None:
            print(f"SERVER-DIED-STARTUP-{avatar} rc={proc.returncode}", flush=True)
            out_log.close()
            err_log.close()
            return None
        if _port_listening():
            # Listening != healthy: gate on the micro-session probe. Two
            # attempts (a transient probe failure retried once); persistent
            # failure = GPU contended — skip the batch instead of writing
            # crawled recordings into the dataset.
            for attempt in (1, 2):
                ok, detail = probe_server(avatar)
                if ok:
                    break
                print(f"PROBE-UNHEALTHY-{avatar} attempt={attempt} {detail}",
                      flush=True)
                if attempt == 1:
                    time.sleep(30)
            if not ok:
                print(f"GPU-CONTENDED-{avatar} {detail} — "
                      f"close other GPU apps, then re-run the chain",
                      flush=True)
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True)
                out_log.close()
                err_log.close()
                return None
            print(f"SERVER-READY-{avatar} pid={proc.pid} {detail}", flush=True)
            return proc
    print(f"SERVER-TIMEOUT-{avatar}", flush=True)
    # taskkill the whole tree: proc.kill() only kills the venv shim; the uv
    # python real body survives and keeps holding port 8010.
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                   capture_output=True)
    return None


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    for _ in range(15):
        time.sleep(2)
        if not _port_listening():
            return
        kill_all_servers()


def run_batch(avatar: str, out_root: Path, ref_spec: str | None,
              sess_spec: str | None, tag: str) -> int:
    cmd = [PY, HARNESS, "run", "--base", "http://127.0.0.1:8010",
           "--avatars", avatar, "--points", str(POINTS),
           "--repeats", str(REPEATS), "--wav-dir", WAV_DIR,
           "--out-root", str(out_root), "--session-gap", "1.0",
           "--retries", "2", "--index-name", f"index_{tag}.json"]
    if ref_spec:
        # "0:0" = empty sess range: a references-only batch must NOT fall
        # through into the full interrupt grid (that ran 60 sessions on one
        # server, hit the ~5-session thread-leak degradation, and wrote
        # meta.json for degraded utt2_start=None sessions).
        cmd += ["--ref-range", ref_spec, "--references", "--sess-range", "0:0"]
    if sess_spec:
        cmd += ["--sess-range", sess_spec]
    rc = subprocess.run(cmd).returncode
    print(f"BATCH-{tag}-EXIT={rc}", flush=True)
    return rc


def _batch_complete(avatar: str, out_root: Path, lo: int, hi: int,
                    kind: str) -> bool:
    """True when every session of batch [lo:hi) already has meta.json.

    Lets sweep re-runs skip server starts (~4.5min each) for batches whose
    sessions were all recorded by an earlier pass. Session dir names mirror
    lt_harness.cmd_run; the interrupt grid formula is reused from the harness
    module to keep a single source of truth.
    """
    if kind == "ref":
        names = [f"reference_{avatar}_seed{n}" for n in range(lo + 1, hi + 1)]
    else:
        import importlib.util

        spec = importlib.util.spec_from_file_location("lt_harness", HARNESS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        grid = mod._interrupt_grid(POINTS)
        names = []
        for gi in range(lo, hi):
            i = gi % POINTS
            repeat = gi // POINTS + 1
            names.append(
                f"interrupt_{avatar}_seed{i + 1}_t{grid[i]:.2f}s_r{repeat}")
    return all((out_root / n / "meta.json").is_file() for n in names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--avatars", default="yongen,sun")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()
    batch = args.batch

    overall_rc = 0
    for avatar in [a for a in args.avatars.split(",") if a]:
        print(f"LT-{avatar}-PHASE-START", flush=True)
        out_root = Path(DATA_ROOT) / f"lt_{avatar}"
        out_root.mkdir(parents=True, exist_ok=True)
        log_base = Path(WORKDIR) / f"lt_server_{avatar}"

        # references: POINTS total
        n_ref_batches = (POINTS + batch - 1) // batch
        for b in range(n_ref_batches):
            lo, hi = b * batch, min((b + 1) * batch, POINTS)
            if _batch_complete(avatar, out_root, lo, hi, "ref"):
                print(f"BATCH-preSkip-ref_{b}", flush=True)
                continue
            srv = start_server(avatar, log_base)
            if srv is None:
                overall_rc = 1
                print(f"BATCH-skip-ref_{b}", flush=True)
                continue
            run_batch(avatar, out_root, f"{lo}:{hi}", None, f"ref_{avatar}_{b}")
            stop_server(srv)

        # interrupt sessions: POINTS * REPEATS total
        total_sess = POINTS * REPEATS
        n_sess_batches = (total_sess + batch - 1) // batch
        for b in range(n_sess_batches):
            lo, hi = b * batch, min((b + 1) * batch, total_sess)
            if _batch_complete(avatar, out_root, lo, hi, "sess"):
                print(f"BATCH-preSkip-sess_{b}", flush=True)
                continue
            srv = start_server(avatar, log_base)
            if srv is None:
                overall_rc = 1
                print(f"BATCH-skip-sess_{b}", flush=True)
                continue
            run_batch(avatar, out_root, None, f"{lo}:{hi}", f"sess_{avatar}_{b}")
            stop_server(srv)
        print(f"LT-{avatar}-PHASE-DONE", flush=True)
    kill_all_servers()
    print("LT-CHAIN-DONE", flush=True)
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())

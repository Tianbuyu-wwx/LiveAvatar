# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Nightly performance gate (PERF-1) — CPU only, no server needed.

Runs the wsperf synthetic stream in-process and enforces the regression
gates committed in ``perf_baseline.json``:

- delivered fps ≥ 20 (25 fps nominal, pacing-based);
- wire bitrate within ±20 % of the committed baseline — catches silent
  protocol/encoder payload regressions that fps cannot see.

Exit code 0 = all gates pass, 1 = any gate failed (the nightly job goes
red). The gate logic lives in :func:`check_perf` so tests can drive it
without any I/O.

Maintenance: after a *deliberate* protocol/encoder change, re-record the
baseline and commit the diff::

    python scripts/perf_gate.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wsperf import run_synthetic

_BASELINE_PATH = Path(__file__).resolve().parent / "perf_baseline.json"
_FPS_FLOOR = 20.0
_BITRATE_TOLERANCE = 0.20
_FRAMES = 750  # 30 s worth at the 25 fps nominal pace
_FPS_NOMINAL = 25.0


def check_perf(report: dict, baseline: dict) -> dict:
    """Pure gate logic (no I/O) — unit-testable.

    ``report`` is a wsperf synthetic-mode report; ``baseline`` is the
    parsed ``perf_baseline.json``.
    """
    ref = float(baseline["wire_kbps"])
    tol = float(baseline.get("bitrate_tolerance", _BITRATE_TOLERANCE))
    lo, hi = ref * (1.0 - tol), ref * (1.0 + tol)
    bitrate = float(report["wire_kbps"])
    fps = float(report["fps"])
    fps_ok = fps >= _FPS_FLOOR
    bitrate_ok = lo <= bitrate <= hi
    return {
        "fps_floor": _FPS_FLOOR,
        "fps": fps,
        "fps_ok": fps_ok,
        "bitrate_ref_kbps": ref,
        "bitrate_band_kbps": [round(lo, 1), round(hi, 1)],
        "bitrate_kbps": bitrate,
        "bitrate_ok": bitrate_ok,
        "encode_ms_p95": float(report.get("encode_ms_p95", 0.0)),
        "ok": bool(fps_ok and bitrate_ok),
    }


def _load_baseline(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"baseline missing: {path} — record one with "
            "'python scripts/perf_gate.py --update-baseline' and commit it"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frames", type=int, default=_FRAMES, help="synthetic frame count"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="re-record perf_baseline.json from this run (then commit it)",
    )
    args = parser.parse_args(argv)

    report = run_synthetic(n_frames=args.frames, fps=_FPS_NOMINAL)

    if args.update_baseline:
        baseline = {
            "wire_kbps": report["wire_kbps"],
            "bitrate_tolerance": _BITRATE_TOLERANCE,
            "fps_floor": _FPS_FLOOR,
            "frames": args.frames,
            "fps_nominal": _FPS_NOMINAL,
        }
        _BASELINE_PATH.write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
        )
        print(f"baseline written: {_BASELINE_PATH}", file=sys.stderr)
        print(json.dumps(report, indent=2))
        return 0

    baseline = _load_baseline(_BASELINE_PATH)
    verdict = check_perf(report, baseline)
    print(json.dumps({"report": report, "gates": verdict}, indent=2))
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

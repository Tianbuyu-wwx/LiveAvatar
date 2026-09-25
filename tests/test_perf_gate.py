# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""O3 PERF-1: perf-gate logic (pure functions, CPU only).

Covers the fps-floor and bitrate-band checks of
``scripts/perf_gate.py`` without running the synthetic stream.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

perf_gate = importlib.import_module("perf_gate")


def _report(**overrides) -> dict:
    base = {"fps": 25.0, "wire_kbps": 700.0, "encode_ms_p95": 0.5}
    base.update(overrides)
    return base


def _baseline(wire_kbps: float = 700.0) -> dict:
    return {"wire_kbps": wire_kbps, "bitrate_tolerance": 0.2}


class TestPerfGate(unittest.TestCase):
    def test_all_pass(self) -> None:
        gates = perf_gate.check_perf(_report(), _baseline())
        self.assertTrue(gates["ok"])
        self.assertTrue(gates["fps_ok"])
        self.assertTrue(gates["bitrate_ok"])

    def test_bitrate_band_boundaries_inclusive(self) -> None:
        # 700 ±20 % → [560, 840]; band edges must pass (inclusive).
        for kbps in (560.0, 840.0):
            gates = perf_gate.check_perf(_report(wire_kbps=kbps), _baseline())
            self.assertTrue(gates["bitrate_ok"], kbps)

    def test_bitrate_above_band_fails(self) -> None:
        gates = perf_gate.check_perf(_report(wire_kbps=841.0), _baseline())
        self.assertFalse(gates["bitrate_ok"])
        self.assertFalse(gates["ok"])

    def test_bitrate_below_band_fails(self) -> None:
        gates = perf_gate.check_perf(_report(wire_kbps=559.0), _baseline())
        self.assertFalse(gates["bitrate_ok"])
        self.assertFalse(gates["ok"])

    def test_fps_below_floor_fails(self) -> None:
        gates = perf_gate.check_perf(_report(fps=19.9), _baseline())
        self.assertFalse(gates["fps_ok"])
        self.assertFalse(gates["ok"])

    def test_fps_floor_boundary_inclusive(self) -> None:
        gates = perf_gate.check_perf(_report(fps=20.0), _baseline())
        self.assertTrue(gates["fps_ok"])

    def test_baseline_without_tolerance_uses_default(self) -> None:
        gates = perf_gate.check_perf(
            _report(wire_kbps=700.0 * 1.2), {"wire_kbps": 700.0}
        )
        self.assertTrue(gates["bitrate_ok"])
        gates = perf_gate.check_perf(
            _report(wire_kbps=700.0 * 1.21), {"wire_kbps": 700.0}
        )
        self.assertFalse(gates["bitrate_ok"])

    def test_band_values_in_verdict(self) -> None:
        gates = perf_gate.check_perf(_report(), _baseline())
        self.assertEqual(gates["bitrate_band_kbps"], [560.0, 840.0])
        self.assertEqual(gates["fps_floor"], 20.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

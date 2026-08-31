"""M-D: capacity-report scaffold gates (pure functions, CPU only).

Covers the gate logic and markdown rendering of
``scripts/capacity_report.py`` without booting the service: fps floor,
interruption budget (≤ 90 ms), zero-starvation, and report output.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

capacity_report = importlib.import_module("capacity_report")


def _session(**overrides) -> dict:
    base = {
        "session_id": "s1",
        "frames": 120,
        "fps": 25.0,
        "wire_kbps": 760.0,
        "interrupt_latency_ms": 5.0,
        "gap_ms_p95": 45.0,
    }
    base.update(overrides)
    return base


class TestGates(unittest.TestCase):
    def test_all_pass(self) -> None:
        gates = capacity_report._gates([_session() for _ in range(3)])
        self.assertTrue(gates["fps_ok"])
        self.assertTrue(gates["interrupt_ok"])
        self.assertEqual(gates["starved_sessions"], 0)
        self.assertAlmostEqual(gates["fps_min"], 25.0)
        self.assertAlmostEqual(gates["interrupt_p95_ms"], 5.0)

    def test_low_fps_fails(self) -> None:
        gates = capacity_report._gates(
            [_session(fps=19.0), _session(fps=25.0), _session(fps=25.0)]
        )
        self.assertFalse(gates["fps_ok"])
        self.assertAlmostEqual(gates["fps_min"], 19.0)

    def test_interrupt_over_budget_fails(self) -> None:
        gates = capacity_report._gates(
            [_session(interrupt_latency_ms=91.0)] * 3
        )
        self.assertFalse(gates["interrupt_ok"])
        self.assertAlmostEqual(gates["interrupt_p95_ms"], 91.0)

    def test_missing_interrupt_is_reported_not_crash(self) -> None:
        gates = capacity_report._gates(
            [_session(interrupt_latency_ms=None), _session(), _session()]
        )
        # p95 is computed over the sessions that produced a measurement.
        self.assertIn("interrupt_p95_ms", gates)
        self.assertAlmostEqual(gates["interrupt_p95_ms"], 5.0)

    def test_starved_sessions_counted(self) -> None:
        gates = capacity_report._gates([_session(frames=0), _session()])
        self.assertEqual(gates["starved_sessions"], 1)


class TestMarkdown(unittest.TestCase):
    def test_report_written_with_gates(self) -> None:
        report = {
            "sessions": 3,
            "seconds": 6,
            "total_egress_mbps": 1.71,
            "per_session": [_session(session_id=f"s{i}") for i in range(3)],
            "gates": capacity_report._gates(
                [_session() for _ in range(3)]
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "容量报告.md"
            capacity_report._write_markdown(str(path), report)
            text = path.read_text(encoding="utf-8")
        self.assertIn("并发会话数：3", text)
        self.assertIn("| s0 |", text)
        self.assertIn("打断延迟 ≤", text)
        self.assertIn("通过", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

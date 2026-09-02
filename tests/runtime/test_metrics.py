# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for runtime SessionMetrics (idempotent recorders, latency windows)."""

from __future__ import annotations

import unittest

from liveavatar.runtime.metrics import SessionMetrics


class TestSessionMetrics(unittest.TestCase):
    def test_first_packet_idempotent(self):
        m = SessionMetrics(session_id="s1")
        self.assertTrue(m.record_first_packet())
        first = m.first_packet_ns
        self.assertIsNotNone(first)
        self.assertFalse(m.record_first_packet())
        self.assertEqual(m.first_packet_ns, first)

    def test_first_playback_idempotent(self):
        m = SessionMetrics(session_id="s1")
        self.assertTrue(m.record_first_playback())
        self.assertFalse(m.record_first_playback())

    def test_flush_without_interrupt_returns_none(self):
        m = SessionMetrics(session_id="s1")
        self.assertIsNone(m.record_flush())
        self.assertEqual(m.flush_count, 0)

    def test_interrupt_to_flush_window(self):
        m = SessionMetrics(session_id="s1")
        m.record_interrupt()
        duration = m.record_flush()
        self.assertIsNotNone(duration)
        self.assertGreaterEqual(duration, 0.0)
        self.assertEqual(m.interrupt_count, 1)
        self.assertEqual(m.flush_count, 1)
        self.assertAlmostEqual(m.interrupt_to_flush_ms, duration, places=6)

    def test_flush_without_new_interrupt_uses_last_window(self):
        m = SessionMetrics(session_id="s1")
        m.record_interrupt()
        d1 = m.record_flush()
        d2 = m.record_flush()
        # Both flushes measure from the SAME last interrupt (the window is
        # not reset), so the second can only be >= the first on the
        # monotonic clock. No new interrupt was recorded.
        self.assertIsNotNone(d1)
        self.assertIsNotNone(d2)
        self.assertGreaterEqual(d2, d1)
        self.assertEqual(m.interrupt_count, 1)
        self.assertEqual(m.flush_count, 2)

    def test_first_to_playback_none_until_both_recorded(self):
        m = SessionMetrics(session_id="s1")
        self.assertIsNone(m.first_to_playback_ms)
        m.record_first_packet()
        self.assertIsNone(m.first_to_playback_ms)
        m.record_first_playback()
        self.assertIsNotNone(m.first_to_playback_ms)
        self.assertGreaterEqual(m.first_to_playback_ms, 0.0)

    def test_summary_shape(self):
        m = SessionMetrics(session_id="s1")
        summary = m.summary()
        self.assertEqual(summary["session_id"], "s1")
        self.assertIsNone(summary["first_packet_ns"])
        self.assertIsNone(summary["first_packet_to_first_playback_ms"])
        self.assertEqual(summary["interrupt_count"], 0)
        self.assertEqual(summary["flush_count"], 0)
        m.record_first_packet()
        m.record_first_playback()
        m.record_interrupt()
        m.record_flush()
        full = m.summary()
        self.assertIsNotNone(full["first_packet_ns"])
        self.assertIsNotNone(full["interrupt_to_flush_ms"])
        self.assertEqual(full["interrupt_count"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

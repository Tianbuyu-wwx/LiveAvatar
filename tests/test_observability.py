# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for observability: /metrics export + trace-id logging (CPU only)."""

from __future__ import annotations

import logging
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from liveavatar.observability import (
    METRICS_ENV,
    TraceIdFilter,
    configure_logging,
    metrics_enabled,
    render_metrics,
)


def _fake_sink(clients: int = 1) -> SimpleNamespace:
    """WebSocketSink-duck-typed publisher (stats() + client_count)."""
    return SimpleNamespace(
        client_count=clients,
        stats=lambda: {
            "frames_seen": 10,
            "frames_published": 8,
            "frames_dropped_epoch": 1,
            "frames_dropped_closed": 1,
            "encode_errors": 0,
            "client_frames_dropped": 2,
        },
    )


class TestMetricsEnabled(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(metrics_enabled())

    def test_on_variants(self):
        for raw in ("on", "1", "true", "yes", "ON"):
            with mock.patch.dict(os.environ, {METRICS_ENV: raw}):
                self.assertTrue(metrics_enabled(), raw)

    def test_explicit_off(self):
        with mock.patch.dict(os.environ, {METRICS_ENV: "off"}):
            self.assertFalse(metrics_enabled())


class TestTraceIdFilter(unittest.TestCase):
    def _record(self) -> logging.LogRecord:
        return logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="x", args=(), exc_info=None,
        )

    def test_trace_id_from_session_id(self):
        record = self._record()
        record.session_id = "sess_abc"  # what extra= produces at call sites
        self.assertTrue(TraceIdFilter().filter(record))
        self.assertEqual(record.trace_id, "sess_abc")

    def test_trace_id_defaults_to_dash(self):
        record = self._record()
        self.assertTrue(TraceIdFilter().filter(record))
        self.assertEqual(record.trace_id, "-")

    def test_configure_logging_idempotent(self):
        root = logging.getLogger()
        before = len(root.handlers)
        configure_logging()
        after_first = len(root.handlers)
        configure_logging()
        self.assertEqual(len(root.handlers), after_first)
        self.assertLessEqual(after_first - before, 1)


class TestRenderMetrics(unittest.TestCase):
    def test_empty_state_renders_zeroes(self):
        state = SimpleNamespace(pipeline=None, duplex_sessions={})
        text = render_metrics(state)
        self.assertIn('liveavatar_sessions_active 0', text)
        self.assertIn('liveavatar_sessions_push 0', text)
        self.assertIn('liveavatar_sessions_duplex 0', text)
        self.assertIn('liveavatar_video_clients 0', text)
        self.assertIn('liveavatar_uptime_seconds ', text)
        self.assertTrue(text.endswith("\n"))

    def test_aggregates_push_and_duplex_sinks(self):
        sink_a = _fake_sink(clients=2)
        sink_b = _fake_sink(clients=1)
        state = SimpleNamespace(
            pipeline=SimpleNamespace(
                sessions={"s1": SimpleNamespace(publisher=sink_a)}
            ),
            duplex_sessions={"d1": SimpleNamespace(sink=sink_b)},
        )
        text = render_metrics(state)
        self.assertIn('liveavatar_sessions_active 2', text)
        self.assertIn('liveavatar_sessions_push 1', text)
        self.assertIn('liveavatar_sessions_duplex 1', text)
        self.assertIn('liveavatar_video_clients 3', text)
        # Counters summed across both sinks.
        self.assertIn('liveavatar_frames_seen_total 20', text)
        self.assertIn('liveavatar_frames_published_total 16', text)
        self.assertIn('liveavatar_frames_dropped_total 4', text)
        self.assertIn('liveavatar_client_frames_dropped_total 4', text)
        # Prom exposition markers present.
        self.assertIn('# TYPE liveavatar_frames_seen_total counter', text)

    def test_non_sink_publisher_ignored(self):
        # Publisher without stats() must not break the export.
        state = SimpleNamespace(
            pipeline=SimpleNamespace(
                sessions={"s1": SimpleNamespace(publisher=object())}
            ),
            duplex_sessions={},
        )
        text = render_metrics(state)
        self.assertIn('liveavatar_sessions_push 1', text)
        self.assertIn('liveavatar_frames_seen_total 0', text)

    def test_session_without_publisher(self):
        state = SimpleNamespace(
            pipeline=SimpleNamespace(
                sessions={"s1": SimpleNamespace(publisher=None)}
            ),
            duplex_sessions={"d1": SimpleNamespace(sink=None)},
        )
        text = render_metrics(state)
        self.assertIn('liveavatar_sessions_active 2', text)
        self.assertIn('liveavatar_video_clients 0', text)


class TestMetricsEndpoint(unittest.TestCase):
    def setUp(self):
        from liveavatar.publish import app, state

        self.state = state
        self._saved = (state.pipeline, dict(state.duplex_sessions))
        state.pipeline = None
        state.duplex_sessions = {}
        self.client = TestClient(app)

    def tearDown(self):
        self.state.pipeline, d = self._saved
        self.state.duplex_sessions = d

    def test_disabled_returns_404(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 404)
        self.assertIn(METRICS_ENV, resp.json()["error"])

    def test_enabled_returns_prometheus_text(self):
        with mock.patch.dict(os.environ, {METRICS_ENV: "on"}):
            resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/plain", resp.headers["content-type"])
        self.assertIn("liveavatar_sessions_active 0", resp.text)

    def test_enabled_aggregates_live_state(self):
        self.state.pipeline = SimpleNamespace(
            sessions={"s1": SimpleNamespace(publisher=_fake_sink(clients=1))}
        )
        with mock.patch.dict(os.environ, {METRICS_ENV: "on"}):
            resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIn('liveavatar_sessions_push 1', resp.text)
        self.assertIn('liveavatar_frames_seen_total 10', resp.text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

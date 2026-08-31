"""Observability: Prometheus ``/metrics`` export + trace-id logging.

Self-written, zero-dependency implementations (project convention):

- **Prometheus text exposition** (version 0.0.4) — the format is a handful
  of plain-text lines, so ``prometheus_client`` is not warranted. Metrics
  are gathered at *scrape* time from the live session objects, so no
  instrumentation plumbing is needed at the call sites.
- **Trace-id logging** — every service log call site already passes
  ``session_id`` via ``extra=``; that value is the correlation id for the
  session's whole chain. :class:`TraceIdFilter` surfaces it as
  ``%(trace_id)s`` for formatters.

The endpoint is gated behind ``LIVEAVATAR_METRICS=on`` (off by default so
local dev / tests don't expose internals).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

METRICS_ENV = "LIVEAVATAR_METRICS"

# Process start (monotonic enough for uptime display purposes).
_PROCESS_START = time.time()

_METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def metrics_enabled(env: dict[str, str] | None = None) -> bool:
    """True when ``LIVEAVATAR_METRICS`` opts the /metrics endpoint on."""
    source = env if env is not None else os.environ
    return source.get(METRICS_ENV, "").strip().lower() in ("on", "1", "true", "yes")


class TraceIdFilter(logging.Filter):
    """Inject ``record.trace_id`` from the record's ``session_id`` extra.

    Records without a ``session_id`` (pipeline lifecycle, startup) get
    ``"-"``. Safe to attach to any handler; idempotent per record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = str(getattr(record, "session_id", "") or "-")
        return True


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(trace_id)s] %(message)s"
_log_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a trace-id-aware handler to the root logger (idempotent)."""
    global _log_configured
    if _log_configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(TraceIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    _log_configured = True


# ──────────────────────────────────────────────── Prometheus export


def _fmt_gauge(name: str, help_text: str, value: float) -> list[str]:
    return [
        f"# HELP {name} {help_text}.",
        f"# TYPE {name} gauge",
        f"{name} {value}",
    ]


def _fmt_counter(name: str, help_text: str, value: float) -> list[str]:
    return [
        f"# HELP {name} {help_text}.",
        f"# TYPE {name} counter",
        f"{name} {value}",
    ]


def _sink_stats(sink: Any) -> dict[str, Any] | None:
    """Stats dict from a WebSocketSink-like publisher, else None."""
    stats_fn = getattr(sink, "stats", None)
    if not callable(stats_fn):
        return None
    try:
        stats = stats_fn()
    except Exception:  # pragma: no cover - defensive, stats() is a dataclass
        return None
    return stats if isinstance(stats, dict) else None


def render_metrics(state: Any) -> str:
    """Render the Prometheus exposition from live service state.

    ``state`` is the publish module's ``AppState`` (duck-typed: needs
    ``pipeline`` and ``duplex_sessions``). Counters aggregate every
    WebSocketSink attached to a push-session publisher or a duplex sink.
    """
    push_sessions: list[Any] = []
    duplex_sessions: list[Any] = []
    pipeline = getattr(state, "pipeline", None)
    if pipeline is not None:
        push_sessions = list(getattr(pipeline, "sessions", {}).values())
    duplex_sessions = list(getattr(state, "duplex_sessions", {}).values())

    sinks: list[Any] = []
    for session in push_sessions:
        publisher = getattr(session, "publisher", None)
        if publisher is not None:
            sinks.append(publisher)
    for session in duplex_sessions:
        sink = getattr(session, "sink", None)
        if sink is not None:
            sinks.append(sink)

    lines: list[str] = []
    lines += _fmt_gauge(
        "liveavatar_sessions_active",
        "Currently active sessions",
        len(push_sessions) + len(duplex_sessions),
    )
    lines += _fmt_gauge(
        "liveavatar_sessions_push", "Active push-mode sessions", len(push_sessions)
    )
    lines += _fmt_gauge(
        "liveavatar_sessions_duplex", "Active duplex-mode sessions", len(duplex_sessions)
    )

    client_count = 0
    totals = {
        "frames_seen": 0,
        "frames_published": 0,
        "frames_dropped_epoch": 0,
        "frames_dropped_closed": 0,
        "encode_errors": 0,
        "client_frames_dropped": 0,
    }
    for sink in sinks:
        clients = getattr(sink, "client_count", 0)
        if isinstance(clients, int):
            client_count += clients
        stats = _sink_stats(sink)
        if stats is None:
            continue
        for key in totals:
            value = stats.get(key, 0)
            if isinstance(value, int):
                totals[key] += value
    lines += _fmt_gauge(
        "liveavatar_video_clients", "Connected video-WS clients", client_count
    )

    lines += _fmt_counter(
        "liveavatar_frames_seen_total",
        "Video frames received by sinks",
        totals["frames_seen"],
    )
    lines += _fmt_counter(
        "liveavatar_frames_published_total",
        "Video frames encoded and fanned out",
        totals["frames_published"],
    )
    dropped = totals["frames_dropped_epoch"] + totals["frames_dropped_closed"]
    lines += _fmt_counter(
        "liveavatar_frames_dropped_total", "Video frames dropped by sinks", dropped
    )
    lines += _fmt_counter(
        "liveavatar_encode_errors_total",
        "Video frame encoding errors",
        totals["encode_errors"],
    )
    lines += _fmt_counter(
        "liveavatar_client_frames_dropped_total",
        "Frames dropped per-client (slow consumer queue full)",
        totals["client_frames_dropped"],
    )

    lines += _fmt_gauge(
        "liveavatar_uptime_seconds", "Process uptime in seconds", time.time() - _PROCESS_START
    )
    return "\n".join(lines) + "\n"

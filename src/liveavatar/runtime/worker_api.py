# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Structural interface the worker mixins rely on (REF-2).

``ingest.IngestMixin`` and ``llm_turn.LlmTurnMixin`` annotate their ``self``
as :class:`RealtimeWorkerCore` instead of importing ``RealtimeWorker`` —
a Protocol keeps the dependency one-way (worker.py → mixins, never back)
so mypy sees no import cycle and the mixins stay decoupled from the hub's
concrete state layout. ``RealtimeWorker`` satisfies the Protocol
structurally; nothing to register.
"""

from __future__ import annotations

from typing import Any, Protocol


class RealtimeWorkerCore(Protocol):
    """Members of :class:`~liveavatar.runtime.worker.RealtimeWorker` used by
    the ingest / llm_turn mixins (attribute access surface only)."""

    session_id: str
    epoch: int
    _running: bool
    metrics: Any
    stats: Any
    tts: Any
    vad: Any
    eou: Any
    asr: Any
    avatar_adapter: Any
    text_source: Any
    output_queue: Any
    _tts_tasks: Any
    _history: list[dict[str, str]]
    history_limit: int
    _vad_active: bool
    _ptt_mode: bool
    _tutor_speaking: bool
    _headphones: bool
    _aec: Any
    _consumed_pts_us: int
    _last_valley_cut: dict[str, Any] | None

    def advance_epoch(self) -> int: ...

    def _make_envelope(
        self, event_type: Any, payload: Any, *, pts_us: int = 0
    ) -> dict[str, Any]: ...

    def set_headphones(self, value: bool) -> None: ...

    def set_ptt_mode(self, value: bool) -> None: ...

    async def _run_llm_turn(self, text: str, epoch: int, pts_us: int) -> None: ...

    async def _run_tts_stream(self, text: str, epoch: int, pts_us: int) -> None: ...

    def _dispatch_tts(self, text: str, epoch: int, pts_us: int) -> None: ...

    def _emit_tts_segment(self, seg: Any) -> None: ...

    def _cancel_tts_tasks(self) -> None: ...

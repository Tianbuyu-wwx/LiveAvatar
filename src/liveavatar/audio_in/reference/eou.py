# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Silence-based end-of-utterance detector reference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..frame import PCMFrame
from ..interfaces import EndOfUtteranceDetector


@dataclass
class SilenceEouDetector(EndOfUtteranceDetector):
    """Emit EOU after a configurable silence duration following speech."""

    silence_needed_us: int = 500000  # 500 ms
    _silence_start_us: int | None = field(default=None, init=False)
    _pending: bool = field(default=False, init=False)

    def push_frame(self, frame: PCMFrame, vad_active: bool) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if vad_active:
            self._silence_start_us = None
            self._pending = True
        elif self._pending:
            if self._silence_start_us is None:
                self._silence_start_us = frame.pts_us
            silence = frame.pts_us - self._silence_start_us
            if silence >= self.silence_needed_us:
                events.append(
                    {
                        "confidence": min(1.0, silence / self.silence_needed_us),
                        "silence_us": silence,
                    }
                )
                self.reset()
        return events

    def reset(self) -> None:
        self._silence_start_us = None
        self._pending = False

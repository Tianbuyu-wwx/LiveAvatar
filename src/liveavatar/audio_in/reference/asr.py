# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Scripted ASR adapter for deterministic control-flow tests.

It does not perform real speech recognition. Instead it matches simple
keyword sequences to demonstrate partial/final/revision semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..frame import PCMFrame
from ..interfaces import StreamingAsrAdapter


@dataclass
class ScriptedAsrAdapter(StreamingAsrAdapter):
    """Deterministic ASR that emits partial/final events based on frame energy."""

    revision: int = 0
    pending_text: str = ""
    emitted_text: str = ""
    energy_threshold_db: float = -50.0
    silence_frames_for_eou: int = 5
    silence_count: int = 0
    frame_count: int = 0
    script: list[str] = field(default_factory=lambda: ["你好", "请讲题", "谢谢"] * 100)

    def push_frame(self, frame: PCMFrame) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        energy = frame.energy_db()
        self.frame_count += 1

        if energy > self.energy_threshold_db:
            self.silence_count = 0
            idx = min(self.frame_count // 10, len(self.script) - 1)
            text = self.script[idx]
            if text != self.pending_text:
                self.revision += 1
                self.pending_text = text
                events.append(
                    {
                        "phase": "partial",
                        "text": text,
                        "stability": 0.6,
                        "revision": self.revision,
                        "words": [],
                    }
                )
        else:
            self.silence_count += 1
            if self.pending_text and self.silence_count >= self.silence_frames_for_eou:
                self.revision += 1
                events.append(
                    {
                        "phase": "final",
                        "text": self.pending_text,
                        "stability": 1.0,
                        "revision": self.revision,
                        "words": [
                            {"word": self.pending_text, "start_us": 0, "end_us": frame.pts_us}
                        ],
                    }
                )
                self.emitted_text = self.pending_text
                self.pending_text = ""
        return events

    def flush(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.pending_text:
            self.revision += 1
            events.append(
                {
                    "phase": "final",
                    "text": self.pending_text,
                    "stability": 1.0,
                    "revision": self.revision,
                    "words": [{"word": self.pending_text, "start_us": 0, "end_us": 0}],
                }
            )
            self.emitted_text = self.pending_text
            self.pending_text = ""
        return events

    def advance_epoch(self, epoch: int) -> None:
        self.revision = 0
        self.pending_text = ""
        self.emitted_text = ""
        self.silence_count = 0
        self.frame_count = 0

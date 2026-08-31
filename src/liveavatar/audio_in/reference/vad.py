"""Energy-based VAD reference implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..frame import PCMFrame
from ..interfaces import VoiceActivityDetector


@dataclass
class EnergyVad(VoiceActivityDetector):
    """Simple energy VAD with hysteresis."""

    threshold_db: float = -45.0
    release_db: float = -50.0
    hangover_frames: int = 5
    _is_speech: bool = field(default=False, init=False)
    _hangover: int = field(default=0, init=False)

    def push_frame(self, frame: PCMFrame) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        energy = frame.energy_db()

        if not self._is_speech and energy > self.threshold_db:
            self._is_speech = True
            self._hangover = 0
            events.append({"kind": "speech_start", "energy_db": energy})
        elif self._is_speech:
            if energy < self.release_db:
                self._hangover += 1
                if self._hangover >= self.hangover_frames:
                    self._is_speech = False
                    events.append({"kind": "speech_end", "energy_db": energy})
            else:
                self._hangover = 0
        return events

    def reset(self) -> None:
        self._is_speech = False
        self._hangover = 0

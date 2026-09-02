# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Audio frontend adapter interfaces.

All adapters are intentionally stateful and single-session oriented.
Implementations must handle epoch advance and deadline/discard semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .frame import PCMFrame


class AudioFrontend(ABC):
    """Transform raw captured audio into canonical PCM frames."""

    @abstractmethod
    def push(self, raw_audio: bytes, capture_ts_us: int) -> list[PCMFrame]:
        """Return zero or more canonical frames."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class StreamingAsrAdapter(ABC):
    """Produce partial/final transcript events from a stream of PCM frames."""

    @abstractmethod
    def push_frame(self, frame: PCMFrame) -> list[dict[str, Any]]:
        """Return zero or more ASR events (partial/final)."""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> list[dict[str, Any]]:
        """Force final recognition and reset internal state."""
        raise NotImplementedError

    @abstractmethod
    def advance_epoch(self, epoch: int) -> None:
        """Discard state tied to older epochs."""
        raise NotImplementedError


class VoiceActivityDetector(ABC):
    """Detect speech start/end from PCM frames."""

    @abstractmethod
    def push_frame(self, frame: PCMFrame) -> list[dict[str, Any]]:
        """Return zero or more VAD events."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class EndOfUtteranceDetector(ABC):
    """Detect end of user utterance."""

    @abstractmethod
    def push_frame(self, frame: PCMFrame, vad_active: bool) -> list[dict[str, Any]]:
        """Return zero or more EOU events."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class ResidualEchoDetector(ABC):
    """Estimate residual echo from far-end reference."""

    @abstractmethod
    def push_frame(self, frame: PCMFrame, far_end_pcm: bytes | None) -> dict[str, Any] | None:
        """Return residual echo event or None."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

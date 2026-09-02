# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""RealtimeAudio: canonical PCM frame processing and audio frontend interfaces."""

from .buffer import BoundedFrameBuffer, DropPolicy
from .frame import FRAME_DURATION_US, SAMPLE_RATE, PCMFrame, SampleClock
from .interfaces import (
    AudioFrontend,
    EndOfUtteranceDetector,
    ResidualEchoDetector,
    StreamingAsrAdapter,
    VoiceActivityDetector,
)
from .reference.asr import ScriptedAsrAdapter
from .reference.echo import ZeroLagEchoDetector
from .reference.eou import SilenceEouDetector
from .reference.vad import EnergyVad

__all__ = [
    "PCMFrame",
    "SampleClock",
    "FRAME_DURATION_US",
    "SAMPLE_RATE",
    "BoundedFrameBuffer",
    "DropPolicy",
    "AudioFrontend",
    "StreamingAsrAdapter",
    "VoiceActivityDetector",
    "EndOfUtteranceDetector",
    "ResidualEchoDetector",
    "ScriptedAsrAdapter",
    "EnergyVad",
    "SilenceEouDetector",
    "ZeroLagEchoDetector",
]

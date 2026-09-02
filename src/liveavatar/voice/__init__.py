# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""NVCFramework Voice Pool: fixed-voice worker pool for real-time TTS.

Solves the P0 cross-talk risk of the single-global ``tts_pipeline`` in
``api_v2.py`` by giving each character a dedicated, immutable worker.

Public API::

    from voice_pool import VoicePool, VoicePoolConfig, VoiceLease

    pool = VoicePool(VoicePoolConfig.from_env())
    await pool.start()
    lease = await pool.acquire("sess_123", "nahida")
    async for pcm in lease.worker.synthesize_stream("你好"):
        ...
    pool.release("sess_123")
"""

from __future__ import annotations

from .config import VoicePoolConfig
from .lease import CancelToken, VoiceLease
from .pool import (
    CharacterNotFound,
    GpuMemoryExhausted,
    VoicePool,
    VoicePoolError,
    VoicePoolExhausted,
    discover_characters,
)
from .worker import CharacterAssets, NvcWorker, NvcWorkerStats

__all__ = [
    # Config
    "VoicePoolConfig",
    # Lease & cancellation
    "CancelToken",
    "VoiceLease",
    # Worker
    "CharacterAssets",
    "NvcWorker",
    "NvcWorkerStats",
    # Pool
    "VoicePool",
    "VoicePoolError",
    "VoicePoolExhausted",
    "GpuMemoryExhausted",
    "CharacterNotFound",
    "discover_characters",
]

__version__ = "0.1.0"

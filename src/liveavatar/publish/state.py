# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Shared mutable service state (single instance per process)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..config import AvatarPoolConfig
from ..duplex import DuplexSession
from ..pipeline import AvatarPipeline
from .settings import PublishSettings


@dataclass
class AppState:
    settings: PublishSettings = field(default_factory=PublishSettings.from_env)
    pool_config: AvatarPoolConfig = field(default_factory=AvatarPoolConfig)
    pipeline: AvatarPipeline | None = None
    start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Full-duplex sessions (mode="duplex") — keyed by session id.
    duplex_sessions: dict[str, DuplexSession] = field(default_factory=dict)
    # Shared TTS VoicePool (GPT-SoVITS) — created lazily on the first
    # duplex session that configures LIVEAVATAR_VOICE_CHAR.
    voice_pool: Any = None
    voice_pool_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Process-wide singleton — every submodule imports this same instance.
state = AppState()

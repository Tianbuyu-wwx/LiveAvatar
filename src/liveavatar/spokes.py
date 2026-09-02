# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Shared optional-spoke assembly (A2).

Single source of truth for the guarded imports and "configured-or-degrade"
resolution of the :class:`~liveavatar.runtime.worker.RealtimeWorker` spokes:
the remote ASR trio (VAD/EOU/ASR), NLMS AEC, voice pool + streaming TTS,
avatar pool + streaming adapter, and the LLM text source.

``DuplexSession`` and the former worker runtime used to carry near-identical
import-guard + warning + construction blocks; both now delegate here so the
resolution rules can never drift apart. Every resolver degrades to ``None``
(with a warning) instead of raising when an optional extra is missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

# ── Guarded imports — the single source for optional-dependency flags ──
try:
    from liveavatar.audio_in.adapters.realtime_asr_client import RealtimeAsrClient
    from liveavatar.audio_in.adapters.remote_asr import RemoteAsrAdapter
    from liveavatar.audio_in.adapters.remote_eou import RemoteEouAdapter
    from liveavatar.audio_in.adapters.remote_vad import RemoteVadAdapter

    HAS_REMOTE = True
except Exception:
    HAS_REMOTE = False

try:
    from liveavatar.audio_in.adapters.nlms_echo import NlmsAec

    HAS_AEC = True
except Exception:
    HAS_AEC = False

try:
    from liveavatar.tts import NvcStreamingTtsAdapter
    from liveavatar.voice.config import VoicePoolConfig
    from liveavatar.voice.pool import VoicePool

    HAS_VOICE_POOL = True
except Exception:
    HAS_VOICE_POOL = False

try:
    from liveavatar.adapter import AvatarStreamingAdapter
    from liveavatar.pool import AvatarPool
    from liveavatar.static_worker import StaticAvatarWorker
    from liveavatar.worker import AvatarAssets

    HAS_AVATAR = True
except Exception:
    HAS_AVATAR = False

try:
    from liveavatar.text_source import OpenAIChatTextSource

    HAS_TEXT_SOURCE = True
except Exception:
    HAS_TEXT_SOURCE = False


@dataclass(slots=True)
class RemoteAsrSpokes:
    """Remote VAD/EOU/ASR trio backed by one shared RealtimeAsrClient."""

    client: Any
    vad: Any
    eou: Any
    asr: Any


def resolve_remote_asr(
    asr_url: str | None, session_id: str, *, logger: logging.Logger
) -> RemoteAsrSpokes | None:
    """Build the remote ASR trio, or None (reference adapters fallback)."""
    if not asr_url:
        return None
    if HAS_REMOTE:
        client = RealtimeAsrClient(asr_url, session_id)
        return RemoteAsrSpokes(
            client=client,
            vad=RemoteVadAdapter(client),
            eou=RemoteEouAdapter(client),
            asr=RemoteAsrAdapter(client),
        )
    logger.warning(
        "asr_url set but remote adapters unavailable; using reference adapters"
    )
    return None


def resolve_aec(enable_aec: bool, *, logger: logging.Logger) -> Any:
    """NLMS AEC when enabled and importable, else None (with a warning)."""
    if not enable_aec:
        return None
    if HAS_AEC:
        return NlmsAec()
    logger.warning("enable_aec set but NlmsAec unavailable; AEC disabled")
    return None


def resolve_text_source(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    logger: logging.Logger,
) -> Any:
    """OpenAI-compatible chat LLM spoke, or None (echo fallback)."""
    if not (base_url and model):
        return None
    if HAS_TEXT_SOURCE:
        return OpenAIChatTextSource(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
        )
    logger.warning("llm configured but text_source unavailable")
    return None


def resolve_voice_pool(
    voice_pool: Any, voice_pool_config: Any, worker_factory: Any
) -> tuple[Any, bool]:
    """Resolve the TTS pool: external pool wins, else build an owned one.

    Returns ``(pool, owns_pool)``. An owned pool is started/stopped by the
    caller; a shared pool is owned by whoever passed it in.
    """
    if voice_pool is not None:
        return voice_pool, False
    if voice_pool_config is not None and HAS_VOICE_POOL:
        return VoicePool(voice_pool_config, worker_factory=worker_factory), True
    return None, False


def default_voice_pool_config() -> Any:
    """Default VoicePoolConfig when the extra is available, else None."""
    return VoicePoolConfig() if HAS_VOICE_POOL else None


def resolve_avatar_pool(
    avatar_pool: Any, avatar_pool_config: Any, worker_factory: Any
) -> tuple[Any, bool]:
    """Resolve the avatar pool: external pool wins, else build an owned one.

    Returns ``(pool, owns_pool)`` — mirrors :func:`resolve_voice_pool`.
    """
    if avatar_pool is not None:
        return avatar_pool, False
    if avatar_pool_config is not None and HAS_AVATAR:
        return AvatarPool(avatar_pool_config, worker_factory=worker_factory), True
    return None, False


def build_tts_adapter(pool: Any, session_id: str, char_id: str) -> Any:
    """Streaming TTS adapter on ``pool`` for ``char_id``, or None."""
    if pool is None or not char_id or not HAS_VOICE_POOL:
        return None
    return NvcStreamingTtsAdapter(
        pool=pool,
        session_id=session_id,
        char_id=char_id,
        sample_rate=16000,
    )


def resolve_avatar_adapter(
    pool: Any,
    session_id: str,
    avatar_id: str,
    publisher: Any,
    *,
    fallback_worker: Any = None,
    degrade_after_errors: int | None = None,
) -> Any:
    """AvatarStreamingAdapter feeding ``publisher`` from ``pool``.

    None when the pool/publisher is missing or the avatar extra is not
    installed (caller degrades to audio-only).
    """
    if pool is None or publisher is None or not HAS_AVATAR:
        return None
    kwargs: dict[str, Any] = {}
    if fallback_worker is not None:
        kwargs["fallback_worker"] = fallback_worker
    if degrade_after_errors is not None:
        kwargs["degrade_after_errors"] = degrade_after_errors
    return AvatarStreamingAdapter(
        pool=pool,
        publisher=publisher,
        session_id=session_id,
        avatar_id=avatar_id,
        **kwargs,
    )


def static_fallback_worker(avatar_id: str) -> Any:
    """Degradation fallback: StaticAvatarWorker with empty assets.

    With no on-disk assets the static worker emits a solid black frame
    (see ``StaticAvatarWorker._load_static_frame``). None when the avatar
    extra is not installed.
    """
    if not HAS_AVATAR:
        return None
    return StaticAvatarWorker(
        AvatarAssets(
            avatar_id=avatar_id,
            data_dir="",
            full_imgs_dir="",
            coords_path="",
            latents_path="",
            mask_dir="",
            mask_coords_path="",
        )
    )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Fixed-voice worker pool with lease management and fair queuing.

Built on :class:`wisdomvii_common.pool.WorkerPool`. The ``VoicePool`` manages a
set of character-specific ``NvcWorker`` instances. Each worker's weights are
loaded once and never switched, eliminating the P0 cross-talk risk of the
single-global ``tts_pipeline`` in ``api_v2.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

from liveavatar._common.pool import (
    GpuMemoryExhausted as CommonGpuMemoryExhausted,
)
from liveavatar._common.pool import (
    PoolError as CommonPoolError,
)
from liveavatar._common.pool import (
    PoolExhausted as CommonPoolExhausted,
)
from liveavatar._common.pool import (
    ResourceNotFound as CommonResourceNotFound,
)
from liveavatar._common.pool import (
    WorkerPool,
)

from .config import VoicePoolConfig
from .lease import VoiceLease
from .worker import CharacterAssets, NvcWorker

logger = logging.getLogger("liveavatar.voice.pool")


# ────────────────────────────────────────────────────────────── exceptions


class VoicePoolError(CommonPoolError):
    """Base exception for voice pool errors."""


class VoicePoolExhausted(CommonPoolExhausted):
    """All workers for a character are busy and the acquire timed out."""


class GpuMemoryExhausted(CommonGpuMemoryExhausted):
    """Cannot load another character worker (max_workers reached)."""


class CharacterNotFound(CommonResourceNotFound):
    """The requested character_id has no assets in weights_root."""


# ─────────────────────────────────────────────── character discovery


def discover_characters(weights_root: str) -> dict[str, CharacterAssets]:
    """Scan ``weights_root`` for character folders and resolve assets.

    Each subfolder must contain at least one ``.ckpt`` (GPT), one ``.pth``
    (SoVITS) and one ``.wav`` (reference audio). The reference text is the
    audio filename without extension (project convention).

    Returns a dict keyed by ``char_id`` (folder name), sorted alphabetically.
    """
    root = Path(weights_root)
    if not root.exists():
        return {}

    chars: dict[str, CharacterAssets] = {}
    for char_dir in sorted(root.iterdir()):
        if not char_dir.is_dir():
            continue
        ckpts = sorted(char_dir.glob("*.ckpt"))
        pths = sorted(char_dir.glob("*.pth"))
        wavs = sorted(char_dir.glob("*.wav"))
        if not ckpts or not pths or not wavs:
            continue
        ref_audio = wavs[0]
        chars[char_dir.name] = CharacterAssets(
            char_id=char_dir.name,
            gpt_path=str(ckpts[0]).replace("\\", "/"),
            sovits_path=str(pths[0]).replace("\\", "/"),
            ref_audio_path=str(ref_audio).replace("\\", "/"),
            ref_text=ref_audio.stem,
        )
    return chars


# ────────────────────────────────────────────────────────── VoicePool


class VoicePool(WorkerPool[NvcWorker, CharacterAssets]):
    """Singleton manager for character-pinned TTS workers.

    Parameters
    ----------
    config : VoicePoolConfig
        Pool configuration.
    worker_factory : Callable[[CharacterAssets], NvcWorker] | None
        Optional factory to create workers (for testing). When ``None``,
        workers are created via :meth:`_default_worker_factory` which loads
        real GPT-SoVITS TTS instances.
    """

    _resource_not_found_error = CharacterNotFound
    _pool_exhausted_error = VoicePoolExhausted
    _gpu_memory_exhausted_error = GpuMemoryExhausted

    def __init__(
        self,
        config: VoicePoolConfig,
        *,
        worker_factory=None,
    ) -> None:
        super().__init__(
            config,
            worker_factory=worker_factory,
            resources=discover_characters(config.weights_root),
            logger=logger,
        )

    @property
    def _resource_kind(self) -> str:
        return "character"

    @property
    def _preloaded_resource_ids(self) -> list[str]:
        return list(self._config.preloaded_chars)

    @property
    def _characters(self) -> dict[str, CharacterAssets]:
        """Backward-compatible alias for the shared ``_resources`` map."""
        return self._resources

    @_characters.setter
    def _characters(self, value: dict[str, CharacterAssets]) -> None:
        self._resources = value

    def _create_lease(
        self,
        session_id: str,
        resource_id: str,
        worker: NvcWorker,
        ttl: float,
    ) -> VoiceLease:
        return cast(VoiceLease, VoiceLease.create(
            session_id=session_id,
            resource_id=resource_id,
            worker=worker,
            ttl=ttl,
        ))

    @property
    def characters(self) -> dict[str, CharacterAssets]:
        """Map of discovered character IDs to their assets."""
        return dict(self._resources)

    def _default_worker_factory(self, assets: CharacterAssets) -> NvcWorker:
        """Create a real NvcWorker with a GPT-SoVITS TTS instance.

        This imports torch and GPT-SoVITS internals, so it's only called
        when a real worker is needed (not in tests).
        """
        # GPT-SoVITS engine code is vendored at <repo>/third_party/GPT_SoVITS.
        engine_root = Path.cwd() / "third_party"
        if not (engine_root / "GPT_SoVITS").exists():
            # Fallback for source checkouts run from another working dir.
            engine_root = Path(__file__).resolve().parents[3] / "third_party"
        sys.path.insert(0, str(engine_root))
        sys.path.insert(0, str(engine_root / "GPT_SoVITS"))

        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # type: ignore

        config_dict = {
            "custom": {
                "bert_base_path": self._config.bert_path,
                "cnhuhbert_base_path": self._config.cnhuhbert_path,
                "device": self._config.device,
                "is_half": self._config.is_half,
                "t2s_weights_path": assets.gpt_path,
                "vits_weights_path": assets.sovits_path,
                "version": "v2Pro",
            }
        }
        tts_config = TTS_Config(config_dict)
        tts = TTS(tts_config)
        tts.set_ref_audio(assets.ref_audio_path)

        return NvcWorker(
            assets=assets,
            tts=tts,
            target_sample_rate=self._config.target_sample_rate,
        )

    # Backward-compatible aliases.
    loaded_workers = WorkerPool.loaded_workers
    active_leases = WorkerPool.active_leases

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of pool state for observability."""
        base = super().stats()
        base["available_characters"] = list(base["available_resources"])
        now = __import__("time").monotonic()
        base["workers"] = {
            cid: w.to_dict() for cid, w in self._workers.items()
        }
        base["leases"] = {
            sid: lease.to_dict()
            for sid, lease in self._leases.items()
            if not lease.is_expired(now)
        }
        return base

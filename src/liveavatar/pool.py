# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Fixed-avatar worker pool with lease management and fair queuing.

Built on :class:`liveavatar._common.pool.WorkerPool`. Each worker's avatar
data (face coords, latents, masks) is loaded once at construction and never
switched, providing the same cross-talk isolation guarantee as the voice pool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from ._common.pool import (
    GpuMemoryExhausted as CommonGpuMemoryExhausted,
)
from ._common.pool import (
    PoolError as CommonPoolError,
)
from ._common.pool import (
    PoolExhausted as CommonPoolExhausted,
)
from ._common.pool import (
    ResourceNotFound as CommonResourceNotFound,
)
from ._common.pool import (
    WorkerPool,
)
from .config import AvatarPoolConfig
from .lease import AvatarLease
from .worker import AvatarAssets, AvatarWorker

logger = logging.getLogger("liveavatar.pool")


# ────────────────────────────────────────────────────────────── exceptions


class AvatarPoolError(CommonPoolError):
    """Base exception for avatar pool errors."""


class AvatarPoolExhausted(CommonPoolExhausted):
    """All workers for an avatar are busy and the acquire timed out."""


class GpuMemoryExhausted(CommonGpuMemoryExhausted):
    """Cannot load another avatar worker (max_workers reached)."""


class AvatarNotFound(CommonResourceNotFound):
    """The requested avatar_id has no preprocessed data."""


# ─────────────────────────────────────────────── avatar discovery


def discover_avatars(avatar_data_root: str) -> dict[str, AvatarAssets]:
    """Scan ``avatar_data_root`` for preprocessed avatar folders.

    Each subfolder must contain: coords.pkl, latents.pt, mask_coords.pkl,
    full_imgs/, mask/. Returns a dict keyed by ``avatar_id`` (folder name),
    sorted alphabetically.
    """
    root = Path(avatar_data_root)
    if not root.exists():
        return {}

    avatars: dict[str, AvatarAssets] = {}
    for avatar_dir in sorted(root.iterdir()):
        if not avatar_dir.is_dir():
            continue
        coords = avatar_dir / "coords.pkl"
        latents = avatar_dir / "latents.pt"
        mask_coords = avatar_dir / "mask_coords.pkl"
        full_imgs = avatar_dir / "full_imgs"
        mask_dir = avatar_dir / "mask"
        if not (coords.exists() and latents.exists() and mask_coords.exists()):
            continue
        avatars[avatar_dir.name] = AvatarAssets(
            avatar_id=avatar_dir.name,
            data_dir=str(avatar_dir).replace("\\", "/"),
            full_imgs_dir=str(full_imgs).replace("\\", "/"),
            coords_path=str(coords).replace("\\", "/"),
            latents_path=str(latents).replace("\\", "/"),
            mask_dir=str(mask_dir).replace("\\", "/"),
            mask_coords_path=str(mask_coords).replace("\\", "/"),
        )
    return avatars


# ────────────────────────────────────────────────────────── AvatarPool


class AvatarPool(WorkerPool[AvatarWorker, AvatarAssets]):
    """Singleton manager for avatar-pinned video inference workers.

    Parameters
    ----------
    config : AvatarPoolConfig
        Pool configuration.
    worker_factory : Callable[[AvatarAssets], AvatarWorker] | None
        Optional factory to create workers (for testing). When ``None``,
        workers are created via :meth:`_default_worker_factory` which loads
        real MuseTalk models.
    """

    _resource_not_found_error = AvatarNotFound
    _pool_exhausted_error = AvatarPoolExhausted
    _gpu_memory_exhausted_error = GpuMemoryExhausted

    def __init__(
        self,
        config: AvatarPoolConfig,
        *,
        worker_factory=None,
    ) -> None:
        super().__init__(
            config,
            worker_factory=worker_factory,
            resources=discover_avatars(config.avatar_data_root),
            logger=logger,
        )
        # Shared MuseTalk models (loaded once, reused across workers).
        self._shared_models: dict[str, Any] | None = None

    @property
    def _resource_kind(self) -> str:
        return "avatar"

    @property
    def _preloaded_resource_ids(self) -> list[str]:
        return list(self._config.preloaded_avatars)

    @property
    def _avatars(self) -> dict[str, AvatarAssets]:
        """Backward-compatible alias for the shared ``_resources`` map."""
        return self._resources

    @_avatars.setter
    def _avatars(self, value: dict[str, AvatarAssets]) -> None:
        self._resources = value

    def _create_lease(
        self,
        session_id: str,
        resource_id: str,
        worker: AvatarWorker,
        ttl: float,
    ) -> AvatarLease:
        return cast(AvatarLease, AvatarLease.create(
            session_id=session_id,
            resource_id=resource_id,
            worker=worker,
            ttl=ttl,
        ))

    @property
    def available_avatars(self) -> list[str]:
        """List discovered avatar IDs."""
        return sorted(self._resources.keys())

    def _default_worker_factory(self, assets: AvatarAssets) -> AvatarWorker:
        """Default factory: loads real MuseTalk models (shared across workers).

        Requires torch + MuseTalk weights. For testing, inject a fake
        ``worker_factory``. Models are loaded once and cached in
        ``self._shared_models`` — subsequent workers reuse them, saving
        GPU memory (mirrors the voice-pool model-sharing constraint).
        """
        from .musetalk_worker import MuseTalkAvatarWorker, load_musetalk_models

        if self._shared_models is None:
            self._shared_models = load_musetalk_models(
                device=self._config.device,
                is_half=self._config.is_half,
                whisper_model_path=self._config.whisper_model_path,
                musetalk_model_dir=self._config.musetalk_model_dir,
                vae_model_dir=self._config.vae_model_dir,
            )

        return MuseTalkAvatarWorker(
            assets,
            target_fps=self._config.target_fps,
            width=self._config.width,
            height=self._config.height,
            batch_size=self._config.batch_size,
            device=self._config.device,
            is_half=self._config.is_half,
            whisper_model_path=self._config.whisper_model_path,
            musetalk_model_dir=self._config.musetalk_model_dir,
            vae_model_dir=self._config.vae_model_dir,
            shared_models=self._shared_models,
        )

    def _on_worker_evicted(self, worker: AvatarWorker) -> None:
        """Release the evicted worker's per-avatar memory (frames/latents).

        Shared MuseTalk models are owned by the pool and stay loaded.
        """
        unload = getattr(worker, "unload_avatar_data", None)
        if callable(unload):
            unload()

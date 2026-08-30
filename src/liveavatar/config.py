"""Avatar pool configuration.

Loads from ``LIVEAVATAR_*`` environment variables, a local ``.env.local`` /
``.env`` file, or explicit keyword construction (used in tests).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_file_candidates() -> list[str] | None:
    """Resolve candidate env files relative to the current working directory."""
    candidates = [
        str(name)
        for name in (Path(".env.local"), Path(".env"))
        if name.exists()
    ]
    return candidates or None


class AvatarPoolConfig(BaseSettings):
    """Configuration for the avatar worker pool."""

    model_config = SettingsConfigDict(
        env_prefix="LIVEAVATAR_",
        env_file=_env_file_candidates(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    avatar_data_root: str = "./avatars"
    device: str = "cuda"
    is_half: bool = True
    max_workers: int = 1

    lease_ttl: float = 300.0
    reap_interval: float = 30.0
    acquire_timeout: float = 10.0

    target_fps: int = 25
    width: int = 512
    height: int = 512
    batch_size: int = 4

    whisper_model_path: str = ""
    musetalk_model_dir: str = ""
    vae_model_dir: str = ""

    preloaded_avatars: list[str] = []

    @field_validator("preloaded_avatars", mode="before")
    @classmethod
    def _parse_preloaded_avatars(cls, v):
        """Support comma-separated string or list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

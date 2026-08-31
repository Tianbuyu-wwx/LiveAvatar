"""VoicePool configuration.

Supports both explicit construction (used in tests) and environment-variable
loading from ``LIVEAVATAR_VOICE_*`` or a local ``.env.local`` / ``.env`` file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_file_candidates() -> list[str] | None:
    """Resolve candidate env files relative to the current working directory."""
    return [
        str(Path(name))
        for name in (Path(".env.local"), Path(".env"))
        if Path(name).exists()
    ] or None


class VoicePoolConfig(BaseSettings):
    """Configuration for the LiveAvatar voice worker pool."""

    model_config = SettingsConfigDict(
        env_prefix="LIVEAVATAR_VOICE_",
        env_file=_env_file_candidates(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    device: str = "cuda"
    is_half: bool = True
    max_workers: int = 2  # ge=1 le=16

    weights_root: str = "./weights"
    bert_path: str = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    cnhuhbert_path: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base"

    lease_ttl: float = 300.0  # ge=0
    reap_interval: float = 30.0  # ge=0
    acquire_timeout: float = 10.0  # ge=0

    target_sample_rate: int = 16000  # ge=8000 le=48000

    preloaded_chars: list[str] = []

    api_key: str = ""  # API key for sensitive control endpoints; empty disables remote access

    @field_validator("preloaded_chars", mode="before")
    @classmethod
    def _parse_preloaded_chars(cls, v):
        """Support comma-separated string or list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @classmethod
    def from_env(cls) -> VoicePoolConfig:
        """Construct from environment variables / .env files."""
        return cls()

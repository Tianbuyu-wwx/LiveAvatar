"""Plugin interface for ASR / TTS backends (草案).

Plugins are registered in an in-process registry, optionally discovered
from installed packages via importlib entry-points::

    # my_asr_plugin setup.cfg
    [options.entry_points]
    liveavatar.plugins =
        my_asr = my_package:MyASR

Plugin classes implement :class:`ASRPlugin` or :class:`TTSPlugin`
(structural protocols — no inheritance required).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("liveavatar.plugins")

ENTRY_POINT_GROUP = "liveavatar.plugins"

_REGISTRY: dict[str, Any] = {}
_DISCOVERED = False


@runtime_checkable
class ASRPlugin(Protocol):
    """Speech-to-text: PCM S16LE in, transcript out."""

    name: str

    async def transcribe(self, pcm_s16le: bytes, sample_rate: int) -> str: ...


@runtime_checkable
class TTSPlugin(Protocol):
    """Text-to-speech: text in, PCM S16LE + sample rate out."""

    name: str

    async def synthesize(self, text: str) -> tuple[bytes, int]: ...


def register(name: str, plugin: Any) -> None:
    """Register a plugin instance/class under ``name`` (last wins)."""
    _REGISTRY[name] = plugin
    logger.info("plugin_registered", extra={"plugin": name})


def get(name: str) -> Any:
    """Return the plugin registered under ``name`` (None when absent)."""
    return _REGISTRY.get(name)


def available() -> list[str]:
    """Names of all registered plugins."""
    return sorted(_REGISTRY)


def clear() -> None:
    """Drop all registrations (test helper)."""
    _REGISTRY.clear()


def discover_entry_points() -> list[str]:
    """Import ``liveavatar.plugins`` entry-points once; return new names.

    Entry-point values may be a plugin instance, class, or zero-arg
    factory — classes/factories are instantiated with no arguments.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return []
    _DISCOVERED = True
    loaded: list[str] = []
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group = eps.select(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - metadata unavailable
        return []
    for ep in group:
        try:
            obj = ep.load()
            if isinstance(obj, type):
                obj = obj()
            register(ep.name, obj)
            loaded.append(ep.name)
        except Exception as exc:
            logger.warning("plugin_entry_point_failed: %s (%s)", ep.name, exc)
    return loaded

"""Self-contained MuseTalk model package for LiveAvatar.

Re-exports the minimal surface required by ``musetalk_worker``:
``load_all_model``, ``Audio2Feature`` and ``get_image_blending``.

The heavy modules pull in torch (and cv2 via ``blending``), so they are
imported lazily via module-level ``__getattr__`` (PEP 562).  This keeps
``import liveavatar.musetalk.models.<mod>`` working in light environments
(CI installs no torch) while preserving the public import surface.

Adapted from the MuseTalk reference implementation (MIT).
"""

from __future__ import annotations

from typing import Any

__all__ = ["Audio2Feature", "get_image_blending", "load_all_model"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "Audio2Feature": (".audio2feature", "Audio2Feature"),
    "get_image_blending": (".blending", "get_image_blending"),
    "load_all_model": (".utils", "load_all_model"),
}


def __getattr__(name: str) -> Any:  # noqa: ANN001 - PEP 562 protocol
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        import importlib

        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_EXPORTS))

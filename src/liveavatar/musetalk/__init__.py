"""Self-contained MuseTalk model package for LiveAvatar.

Re-exports the minimal surface required by ``musetalk_worker``:
``load_all_model``, ``Audio2Feature`` and ``get_image_blending``.

Adapted from the MuseTalk reference implementation (MIT).
"""

from __future__ import annotations

from .audio2feature import Audio2Feature
from .blending import get_image_blending
from .utils import load_all_model

__all__ = ["Audio2Feature", "get_image_blending", "load_all_model"]

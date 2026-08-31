"""Per-session video publisher/encoder factories (service wiring).

S4: avatar ids become filesystem path segments (avatar_data_root/<id>) —
they are validated here before any path join.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ..pipeline import SessionState
from .state import state

_AVATAR_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _valid_avatar_id(avatar_id: str) -> bool:
    return bool(_AVATAR_ID_RE.fullmatch(avatar_id))


def _region_encoder_for(avatar_id: str) -> Any:
    """RegionFrameEncoder for the avatar, or None when region.json is
    missing / degenerate (caller falls back to full-frame MJPEG)."""
    from ..region_codec import (
        RegionFrameEncoder,
        load_region_json,
        region_spec_from_masks,
    )

    if not _valid_avatar_id(avatar_id):
        return None
    region_path = os.path.join(
        state.pool_config.avatar_data_root, avatar_id, "region.json"
    )
    try:
        spec = load_region_json(region_path)
        return RegionFrameEncoder(spec)
    except (OSError, ValueError, KeyError):
        # No region.json — try to derive it from the masks on the fly.
        mask_dir = os.path.join(
            state.pool_config.avatar_data_root, avatar_id, "mask"
        )
        cfg = state.pool_config
        mask_spec = region_spec_from_masks(mask_dir, cfg.width, cfg.height)
        if mask_spec is None:
            return None
        return RegionFrameEncoder(mask_spec)


def _ws_sink_for(avatar_id: str) -> Any:
    """WebSocketSink with the configured codec (region → fallback MJPEG)."""
    from ..ws_sink import MjpegFrameEncoder, WebSocketSink

    cfg = state.pool_config
    encoder: Any = None
    if state.settings.codec == "region":
        encoder = _region_encoder_for(avatar_id)
    return WebSocketSink(
        encoder=encoder if encoder is not None else MjpegFrameEncoder(),
        target_fps=cfg.target_fps,
        width=cfg.width,
        height=cfg.height,
        quality=80,
    )


def _service_publisher_factory(session: SessionState) -> Any:
    """Create the per-session video publisher (self-developed transport).

    Every session gets a WebSocketSink served at
    ``/v1/sessions/{sid}/video`` with the configured codec (region →
    fallback MJPEG when the avatar has no region.json).
    """
    return _ws_sink_for(session.avatar_id)

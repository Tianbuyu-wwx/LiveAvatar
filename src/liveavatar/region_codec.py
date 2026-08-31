"""Region-delta frame encoder (R2 M4 — self-developed codec).

A digital-human video stream is mostly static: only the mouth region
changes between frames. Instead of re-sending a full MJPEG every frame
(~35 kB @512² q80), this encoder sends:

- a **full-canvas patch** when the static background changed (rare) or a
  keyframe is requested (connect / epoch boundary / periodic refresh);
- a **mouth-region patch** (mask bounding box grown by 8 px) otherwise
  (~2-6 kB), exploiting the avatar's mouth-region prior.

Payloads use :func:`~liveavatar.video_protocol.pack_region_payload`, so
every wire frame is independently decodable given the receiver's cached
base image — matching the drop-stale-frames epoch semantics (a lost frame
never corrupts the stream; the next frame restores correctness).

The background hash is computed with the mouth rect zeroed, so lip motion
does not trigger full-frame sends.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from .video_protocol import CODEC_REGION_DELTA, Patch, pack_region_payload
from .worker import AvatarFrame

_REGION_FILE = "region.json"


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """Mouth-region bounding box in canvas pixels (pre-padding)."""

    x: int
    y: int
    w: int
    h: int

    def grown(self, pad: int, width: int, height: int) -> RegionSpec:
        """Expand by ``pad`` px, clamped to the canvas."""
        return RegionSpec(
            x=max(0, self.x - pad),
            y=max(0, self.y - pad),
            w=min(width, self.x + self.w + pad) - max(0, self.x - pad),
            h=min(height, self.y + self.h + pad) - max(0, self.y - pad),
        )


def load_region_json(path: str) -> RegionSpec:
    """Load a RegionSpec from an avatar's ``region.json``."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RegionSpec(
        x=int(data["x"]), y=int(data["y"]), w=int(data["w"]), h=int(data["h"])
    )


def write_region_json(path: str, spec: RegionSpec) -> None:
    """Persist a RegionSpec next to the avatar data."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"x": spec.x, "y": spec.y, "w": spec.w, "h": spec.h}, f
        )


def region_spec_from_masks(
    mask_dir: str,
    width: int,
    height: int,
    fallback_fraction: float = 0.30,
) -> RegionSpec | None:
    """Compute the union bounding box of the avatar's mouth masks.

    Returns ``None`` when no masks exist / contain any foreground, so the
    caller can fall back to full-frame MJPEG.

    ``fallback_fraction``: when the union box exceeds this fraction of the
    canvas area the mask is too coarse for region encoding — return None.
    """
    import cv2
    import numpy as np

    if not os.path.isdir(mask_dir):
        return None
    try:
        files = sorted(
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
    except OSError:
        return None
    x0, y0, x1, y1 = width, height, 0, 0
    found = False
    for path in files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ys, xs = np.nonzero(img)
        if xs.size == 0:
            continue
        found = True
        x0, x1 = min(x0, int(xs.min())), max(x1, int(xs.max()) + 1)
        y0, y1 = min(y0, int(ys.min())), max(y1, int(ys.max()) + 1)
    if not found:
        return None
    spec = RegionSpec(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
    if spec.w * spec.h > fallback_fraction * width * height:
        return None
    return spec


class RegionFrameEncoder:
    """FrameEncoder producing region_delta payloads (satisfies the
    :class:`~liveavatar.ws_sink.FrameEncoder` protocol).

    The encoder keeps a hash of the static background (everything outside
    the mouth rect). A frame is sent as:

    - full-canvas patch: keyframe requested OR background hash changed
      (first frame, scene change, canvas resize);
    - mouth-region patch: otherwise (the common case, ~10x smaller).
    """

    codec = CODEC_REGION_DELTA

    def __init__(
        self,
        spec: RegionSpec,
        *,
        padding: int = 8,
        full_frame_fraction: float = 0.9,
    ) -> None:
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np
        self._spec = spec
        self._padding = padding
        self._full_frame_fraction = full_frame_fraction
        self._base_hash: bytes | None = None
        self._rect: RegionSpec | None = None  # canvas-sized rect cache

    # ------------------------------------------------------------------ API

    def encode(
        self, frame: AvatarFrame, *, keyframe: bool, quality: int
    ) -> bytes:
        np = self._np
        img = np.frombuffer(frame.frame_data, dtype=np.uint8).reshape(
            frame.height, frame.width, 3
        )
        rect = self._spec.grown(self._padding, frame.width, frame.height)
        self._rect = rect

        # Background hash: mouth rect zeroed so lip motion stays "static".
        base = img.copy()
        base[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w] = 0
        digest = hashlib.blake2b(base.tobytes(), digest_size=16).digest()

        background_changed = digest != self._base_hash
        self._base_hash = digest

        use_full = keyframe or background_changed or self._is_mostly_face(
            rect, frame.width, frame.height
        )
        if use_full:
            jpeg = self._jpeg(img, quality)
            patches = [Patch(x=0, y=0, w=frame.width, h=frame.height, jpeg=jpeg)]
        else:
            crop = img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
            jpeg = self._jpeg(crop, quality)
            patches = [Patch(x=rect.x, y=rect.y, w=rect.w, h=rect.h, jpeg=jpeg)]
        return pack_region_payload(patches)

    # ------------------------------------------------------------ internals

    def _is_mostly_face(self, rect: RegionSpec, width: int, height: int) -> bool:
        """Full-frame send is cheaper than a patch covering most pixels."""
        return rect.w * rect.h >= self._full_frame_fraction * width * height

    def _jpeg(self, img: Any, quality: int) -> bytes:
        ok, buf = self._cv2.imencode(
            ".jpg", img, [int(self._cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:  # pragma: no cover - cv2 failure is effectively fatal
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Zero-lag residual echo detector reference."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from ..frame import PCMFrame
from ..interfaces import ResidualEchoDetector


@dataclass
class ZeroLagEchoDetector(ResidualEchoDetector):
    """Compute simple correlation between near-end and far-end reference."""

    window_us: int = 20000  # one frame

    def push_frame(self, frame: PCMFrame, far_end_pcm: bytes | None) -> dict[str, Any] | None:
        if not far_end_pcm or len(far_end_pcm) < 4:
            return None
        near = frame.to_int16_array()
        far = list(struct.unpack(f"<{len(far_end_pcm) // 2}h", far_end_pcm))
        min_len = min(len(near), len(far))
        if min_len == 0:
            return None
        near = near[:min_len]
        far = far[:min_len]
        n_mean = sum(near) / min_len
        f_mean = sum(far) / min_len
        num = sum((n - n_mean) * (f - f_mean) for n, f in zip(near, far, strict=False))
        den_n = sum((n - n_mean) ** 2 for n in near) ** 0.5
        den_f = sum((f - f_mean) ** 2 for f in far) ** 0.5
        correlation = num / (den_n * den_f) if den_n and den_f else 0.0
        return {
            "correlation": max(-1.0, min(1.0, correlation)),
            "far_end_reference": far_end_pcm,
        }

    def reset(self) -> None:
        pass

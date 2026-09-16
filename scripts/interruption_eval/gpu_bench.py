# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""GPU health probe for the LT matrix chain.

Times 20 FP32 2048x2048 matmuls (after one warmup) on the default CUDA
device and prints the per-iter average in ms. A healthy GPU finishes in
single-digit ms; when another process thrashes VRAM/compute (game, video
call, a second MuseTalk server) the same op takes hundreds of ms to
seconds — the regime where MuseTalk UNET batch-16 forwards stretched to
5-10s, the audio pipeline starved (py-spy: render thread blocked at
whisper.feat_queue.put, inference pinned inside UNET forward for 8s+),
and every session died with "no SSE start event for utterance 2".

Exit code is always 0 when a number was produced; parse stdout's last
line. A failed probe (no CUDA etc.) prints nothing and exits 1 — callers
should treat that as "cannot judge" rather than unhealthy.
"""
from __future__ import annotations

import sys
import time

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("no-cuda", file=sys.stderr)
        return 1
    a = torch.randn(2048, 2048, device="cuda")
    b = a @ a  # warmup: cublas handle + kernel load
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        b = a @ a
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / 20 * 1000.0
    print(f"{ms:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

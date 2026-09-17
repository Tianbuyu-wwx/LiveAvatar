# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
"""GPU clock pre-warm for P-E4 data collection.

Laptop GPUs (RTX 5080 Laptop) drop to P8/180MHz when idle and never ramp on
bursty small-kernel inference workloads (whisper encoder + UNET batch), which
crawls the whole LiveTalking pipeline for the first ~20s of every session on
a warm server process. A sustained CUDA load for ~12s ramps the governor to
P0; the session's own 25fps load then keeps clocks up. Apply identically to
both arms (LiveTalking and LiveAvatar) for a fair environment policy.

Usage: python gpu_warm.py [--seconds 12]
"""
import argparse
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA required for gpu_warm"
    dev = torch.device("cuda")
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float32)
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < args.seconds:
        for _ in range(20):
            a = a @ a * 1e-3  # keep values bounded
        iters += 20
        torch.cuda.synchronize()
    clocks = torch.cuda.get_device_properties(0)
    print(
        f"gpu_warm: {iters} matmuls in {time.perf_counter()-t0:.1f}s on "
        f"{clocks.name}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

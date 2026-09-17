# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
"""GPU clock keeper for P-E4 data collection.

The RTX 5080 Laptop governor collapses to P8/180MHz within ~2s of idle and
never ramps on the bursty small-kernel inference pattern (whisper encoder +
batch-16 UNET, util 0-10%), which crawls LiveTalking's first ~20s on every
session after the first of a server process. `nvidia-smi -lgc` needs admin,
so instead this keeper runs a light duty-cycled CUDA matmul (~10% util) to
hold the power state up while a measurement session runs. Policy is applied
identically to both arms (LiveTalking / LiveAvatar) for fairness.

Usage: python gpu_keeper.py [--duty-ms 12] [--sleep-ms 25]
Kill with Ctrl+C or taskkill.
"""
import argparse
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duty-ms", type=float, default=12.0)
    ap.add_argument("--sleep-ms", type=float, default=25.0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA required for gpu_keeper"
    dev = torch.device("cuda")
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    print(f"gpu_keeper: duty={args.duty_ms}ms sleep={args.sleep_ms}ms", flush=True)
    t0 = time.perf_counter()
    while True:
        t_burst = time.perf_counter()
        while (time.perf_counter() - t_burst) * 1000 < args.duty_ms:
            for _ in range(8):
                a = a @ a * 1e-3
            torch.cuda.synchronize()
        time.sleep(args.sleep_ms / 1000.0)
        if time.perf_counter() - t0 > 3600:
            t0 = time.perf_counter()  # keep loop counters sane
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

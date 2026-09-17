# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-F survey rehearsal driver (SYNTHETIC — never for the paper).

Simulates N participants through the REAL survey UI (Playwright) and
collects their auto-downloaded ``pf_ratings_P##.json`` files into a
separate ``ratings_sim/`` directory so they can never be confused with
human data under ``ratings/``.

Ratings are scripted: an arm-biased scheme (trans > hard > lt on the
1-5 scale, read from pack_key.json) plus per-participant harshness and
jitter, so the downstream pf_analyze rehearsal should show the
expected directions. This validates the collect -> aggregate chain
end to end; it does NOT produce publishable data.

Usage:
    python pf_sim.py --url http://127.0.0.1:8777/ --n 20 --workers 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from playwright.async_api import async_playwright

BIAS = {"hard": 2.5, "trans": 4.0, "lt": 2.0}
TRIALS = 24  # 12 groups x 2 clips


def load_clip_arms(pack_path: Path) -> dict[str, str]:
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for p in pack["pairs"]:
        for arm, info in p["arms"].items():
            out[info["clip"]] = arm
    return out


def rate(rng: random.Random, arm: str, harsh: float) -> int:
    v = round(BIAS[arm] + harsh + rng.uniform(-0.8, 0.8))
    return max(1, min(5, v))


async def run_participant(
    browser, url: str, pid: str, seed: int, clip_arm: dict[str, str],
    out_dir: Path,
) -> tuple[str, str, float]:
    t0 = time.time()
    ctx = await browser.new_context(accept_downloads=True)
    ctx.set_default_timeout(30_000)
    page = await ctx.new_page()
    try:
        await page.goto(url)
        await page.fill("#pid", pid)
        await page.click("#btn-start")

        rng = random.Random(seed)
        harsh = rng.uniform(-0.7, 0.7)
        clip_seen: list[str] = []

        for k in range(TRIALS):
            pos = "A" if k % 2 == 0 else "B"
            want = f"第 {k // 2 + 1} / 12 组 · 片段 {pos}"
            await page.wait_for_function(
                "t => document.getElementById('prog').textContent === t",
                arg=want)
            await page.wait_for_function(
                "() => { const v = document.getElementById('player');"
                " return v && v.readyState >= 1 && isFinite(v.duration)"
                " && v.duration > 0; }")
            src = await page.evaluate(
                "document.getElementById('player').getAttribute('src')")
            clip_seen.append(src.split("/")[-1])
            await page.evaluate(
                "() => { const v = document.getElementById('player');"
                " v.muted = true;"
                " v.currentTime = Math.max(0, v.duration - 0.5);"
                " return v.play(); }")
            await page.wait_for_function(
                "() => !document.getElementById('panel')"
                ".classList.contains('lock')")

            arm = clip_arm[clip_seen[-1]]
            nat, com = rate(rng, arm, harsh), rate(rng, arm, harsh)
            await page.locator("#sc-nat button").nth(nat - 1).click()
            await page.locator("#sc-com button").nth(com - 1).click()
            if k == TRIALS - 1:
                async with page.expect_download() as dl_info:
                    await page.locator("#btn-next").click()
                dl = await dl_info.value
                dest = out_dir / f"pf_ratings_{pid}.json"
                await dl.save_as(str(dest))
            else:
                await page.locator("#btn-next").click()

        data = json.loads(
            (out_dir / f"pf_ratings_{pid}.json").read_text(encoding="utf-8"))
        assert data["participant_id"] == pid, f"{pid}: wrong id in file"
        assert len(data["trials"]) == TRIALS, f"{pid}: {len(data['trials'])} trials"
        assert len({t['clip'] for t in data['trials']}) == TRIALS, \
            f"{pid}: duplicate clips rated"
        return pid, "OK", time.time() - t0
    except Exception as e:  # noqa: BLE001 - report and keep other sims going
        return pid, f"FAIL: {type(e).__name__}: {e}", time.time() - t0
    finally:
        await ctx.close()


async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_arm = load_clip_arms(Path(args.pack))
    if len(clip_arm) != TRIALS:
        raise SystemExit(f"pack_key maps {len(clip_arm)} clips, expected {TRIALS}")

    pids = [f"P{i:02d}" for i in range(1, args.n + 1)]
    queue: asyncio.Queue[str] = asyncio.Queue()
    for p in pids:
        queue.put_nowait(p)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headed)

        async def worker() -> list[tuple[str, str, float]]:
            results = []
            while not queue.empty():
                pid = queue.get_nowait()
                r = await run_participant(
                    browser, args.url, pid, 1000 + int(pid[1:]),
                    clip_arm, out_dir)
                print(f"[{r[0]}] {r[1]} ({r[2]:.1f}s)", flush=True)
                results.append(r)
            return results

        packs = await asyncio.gather(*(worker() for _ in range(args.workers)))
        await browser.close()

    rows = [r for pack in packs for r in pack]
    ok = sum(1 for r in rows if r[1] == "OK")
    print(f"\nsim done: {ok}/{len(rows)} participants OK -> {out_dir}")
    return 0 if ok == len(rows) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8777/")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/interruption_eval/pf_pack/ratings_sim")
    ap.add_argument("--pack", default="data/interruption_eval/pf_pack/pack_key.json")
    ap.add_argument("--headed", action="store_true")
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

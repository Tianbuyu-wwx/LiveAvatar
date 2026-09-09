# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-E: download the model assets required by the LiveTalking baseline.

Assets land in THIS repo (canonical) and are shared into the LiveTalking
checkout via directory junctions created by :mod:`link_livetalking_models`:

1. models/musetalkV15/    <- TMElyralab/MuseTalk (unet.pth + musetalk.json)
2. models/sd-vae-ft-mse/  <- stabilityai/sd-vae-ft-mse (LiveTalking mounts it
                             as models/sd-vae via junction)
3. models/whisper/        <- reused from download_models.py (must exist)
4. LiveTalking/models/face-parse-bisent/ <- resnet18 + 79999_iter (LiveTalking-only)
5. data/video/yongen.mp4  <- MuseTalk demo avatar (shared avatar source)

Usage:
    python scripts/download_livetalking_assets.py \
        [--root <project_root>] [--lt-root <LiveTalking_root>] [--skip-video]
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path


def _download_url(url: str, target_path: Path, *, timeout: int = 60) -> None:
    """Download a raw URL with Range resume (same protocol as download_models)."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 1000:
        print(f"[URL] exists, skip: {target_path}")
        return
    print(f"[URL] downloading {url} -> {target_path}")
    headers: dict[str, str] = {}
    mode = "wb"
    if target_path.exists():
        headers["Range"] = f"bytes={target_path.stat().st_size}-"
        mode = "ab"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with open(target_path, mode) as f:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    print(f"[URL] done: {target_path} ({target_path.stat().st_size / 1e6:.1f} MB)")


def _download_hf_file(repo_id: str, filename: str, target_path: Path) -> None:
    from huggingface_hub import hf_hub_download

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 1000:
        print(f"[HF] exists, skip: {target_path}")
        return
    print(f"[HF] {repo_id}/{filename} -> {target_path}")
    local = hf_hub_download(repo_id=repo_id, filename=filename)
    if not Path(local).exists():
        raise FileNotFoundError(local)
    if Path(local).resolve() != target_path.resolve():
        _copy_resumable(Path(local), target_path)
    print(f"[HF] done: {target_path} ({target_path.stat().st_size / 1e6:.1f} MB)")


def _copy_resumable(src: Path, dst: Path) -> None:
    """Copy src to dst, resuming if a partial dst already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    done = dst.stat().st_size if dst.exists() else 0
    total = src.stat().st_size
    if done >= total:
        return
    with open(src, "rb") as fin, open(dst, "ab" if done else "wb") as fout:
        if done:
            fin.seek(done)
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)


def _snapshot_musetalk(root: Path) -> None:
    """unet.pth + musetalk.json via snapshot (reuses download_models layout)."""
    from huggingface_hub import snapshot_download

    target = root / "models" / "musetalkV15"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "unet.pth").exists() and (target / "musetalk.json").exists():
        print("[HF] musetalkV15 already complete")
        return
    print("[HF] downloading TMElyralab/MuseTalk (musetalkV15 subset)")
    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=str(target),
        allow_patterns=["musetalkV15/*"],
    )
    nested = target / "musetalkV15"
    if nested.exists():
        for f in nested.iterdir():
            dest = target / f.name
            if not dest.exists():
                f.rename(dest)
        nested.rmdir()
    print(f"[HF] done: {target}")


def _snapshot_sd_vae(root: Path) -> None:
    from huggingface_hub import snapshot_download

    target = root / "models" / "sd-vae-ft-mse"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "config.json").exists() and any(
        (target / n).exists()
        for n in ("diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors")
    ):
        print("[HF] sd-vae-ft-mse already complete")
        return
    print("[HF] downloading stabilityai/sd-vae-ft-mse")
    snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir=str(target))
    print(f"[HF] done: {target}")


# 79999_iter.pth mirrors on HF (checked in order; first that works wins).
_FACE_PARSE_REPOS = [
    "ManyOtherFunctions/face-parse-bisent",
    "camenduru/face-parse-bisent",
]


def download_face_parse(lt_root: Path) -> None:
    out = lt_root / "models" / "face-parse-bisent"
    _download_url(
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        out / "resnet18-5c106cde.pth",
    )
    target = out / "79999_iter.pth"
    if target.exists() and target.stat().st_size > 1000:
        print(f"[HF] exists, skip: {target}")
        return
    last_err: Exception | None = None
    for repo in _FACE_PARSE_REPOS:
        try:
            _download_hf_file(repo, "79999_iter.pth", target)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[HF] mirror failed: {repo}: {exc}")
            last_err = exc
    raise RuntimeError(f"no 79999_iter.pth mirror worked: {last_err}")


def download_demo_video(root: Path) -> None:
    base = "https://raw.githubusercontent.com/TMElyralab/MuseTalk/main"
    _download_url(f"{base}/data/video/yongen.mp4", root / "data" / "video" / "yongen.mp4")


def link_livetalking_models(root: Path, lt_root: Path) -> None:
    """Expose this repo's model dirs inside LiveTalking via junctions.

    LiveTalking expects models/sd-vae (we store sd-vae-ft-mse) and the same
    musetalkV15/whisper layout. Junctions avoid a second multi-GB copy.
    """
    lt_models = lt_root / "models"
    lt_models.mkdir(parents=True, exist_ok=True)
    pairs = [
        (root / "models" / "musetalkV15", lt_models / "musetalkV15"),
        (root / "models" / "sd-vae-ft-mse", lt_models / "sd-vae"),
        (root / "models" / "whisper", lt_models / "whisper"),
    ]
    for src, dst in pairs:
        if not src.exists():
            print(f"[LINK] source missing, skip: {src}")
            continue
        if dst.is_symlink() or dst.exists():
            # A junction/symlink already in place (or a real dir) — leave it.
            print(f"[LINK] exists: {dst}")
            continue
        subprocess_checked_junction(src, dst)


def subprocess_checked_junction(src: Path, dst: Path) -> None:
    import subprocess  # noqa: PLC0415

    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dst), str(src)], capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"[LINK] FAILED {dst} -> {src}: {r.stdout} {r.stderr}")
    else:
        print(f"[LINK] {dst} -> {src}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent.parent
    ap.add_argument("--root", default=str(default_root), help="LiveAvatar root")
    ap.add_argument("--lt-root", default=str(default_root.parent / "LiveTalking"))
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--no-links", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    lt_root = Path(args.lt_root).resolve()
    # China-network fallback for the HF endpoint unless explicitly set.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"root={root} lt_root={lt_root} HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    if not (root / "models" / "whisper" / "config.json").exists():
        print("[WARN] models/whisper missing — run scripts/download_models.py first")

    _snapshot_musetalk(root)
    _snapshot_sd_vae(root)
    download_face_parse(lt_root)
    if not args.skip_video:
        download_demo_video(root)
    if not args.no_links:
        link_livetalking_models(root, lt_root)
    print("=== livetalking assets ready ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

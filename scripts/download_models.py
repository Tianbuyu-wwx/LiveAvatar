"""Download all model weights and demo data for LiveAvatar.

Downloads:
1. models/musetalkV15/  ← TMElyralab/MuseTalk (unet.pth + musetalk.json)
2. models/sd-vae-ft-mse/ ← stabilityai/sd-vae-ft-mse (VAE)
3. models/whisper/       ← openai/whisper-tiny (audio2feature)
4. models/face_detection_yunet_2023mar.onnx ← OpenCV Zoo (face detection)
5. models/mediapipe/face_landmarker.task ← MediaPipe (5-point alignment)
6. data/video/yongen.mp4 + data/audio/*.wav ← MuseTalk GitHub demo

Usage:
    python scripts/download_models.py [--root <project_root>] [--skip-models] [--skip-demo]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


def _download_hf_repo(repo_id: str, target_dir: Path, *, allow_patterns: list[str] | None = None) -> None:
    """Download a HuggingFace repo to a local directory via snapshot_download."""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[HF] downloading {repo_id} → {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        allow_patterns=allow_patterns,
    )
    print(f"[HF] done: {target_dir}")


def _download_url(url: str, target_path: Path, *, timeout: int = 60) -> None:
    """Download a raw URL (resumable via Range header)."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 1000:
        print(f"[URL] exists, skip: {target_path}")
        return
    print(f"[URL] downloading {url} → {target_path}")
    headers: dict[str, str] = {}
    mode = "wb"
    if target_path.exists():
        existing = target_path.stat().st_size
        headers["Range"] = f"bytes={existing}-"
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


def download_models(root: Path) -> None:
    """Download all model directories."""
    models = root / "models"

    # 1. MuseTalk UNet (vae/pe are NOT separate — load_all_model builds pe
    #    in-code; only unet.pth + musetalk.json are needed from HF).
    musetalk_dir = models / "musetalkV15"
    _download_hf_repo("TMElyralab/MuseTalk", musetalk_dir, allow_patterns=["musetalkV15/*"])
    # snapshot_download preserves the repo subdir; flatten it.
    nested = musetalk_dir / "musetalkV15"
    if nested.exists():
        for f in nested.iterdir():
            dest = musetalk_dir / f.name
            if not dest.exists():
                f.rename(dest)
        nested.rmdir()

    # 2. sd-vae-ft-mse (full repo — AutoencoderKL needs config.json + weights).
    _download_hf_repo("stabilityai/sd-vae-ft-mse", models / "sd-vae-ft-mse")

    # 3. whisper-tiny (WhisperModel + AutoFeatureExtractor).
    _download_hf_repo(
        "openai/whisper-tiny",
        models / "whisper",
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "normalizer.json",
            "added_tokens.json",
            "special_tokens_map.json",
        ],
    )

    # 4. YuNet face detection (OpenCV Zoo).
    _download_url(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        models / "face_detection_yunet_2023mar.onnx",
    )

    # 5. MediaPipe face landmarker (5-point alignment).
    _download_url(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        models / "mediapipe" / "face_landmarker.task",
    )

    print("\n=== model summary ===")
    for name in ("musetalkV15", "sd-vae-ft-mse", "whisper", "mediapipe"):
        d = models / name
        if d.exists():
            files = sorted(f.name for f in d.iterdir())
            total = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            print(f"  {name}: {total / 1e6:.1f} MB, files={files}")
        else:
            print(f"  {name}: MISSING")
    for f in ("face_detection_yunet_2023mar.onnx",):
        p = models / f
        print(f"  {f}: {p.stat().st_size / 1e6:.1f} MB" if p.exists() else f"  {f}: MISSING")


def download_demo_data(root: Path) -> None:
    """Download the MuseTalk demo avatar video + audio from GitHub."""
    base = "https://raw.githubusercontent.com/TMElyralab/MuseTalk/main"
    data = root / "data"
    targets = [
        (f"{base}/data/video/yongen.mp4", data / "video" / "yongen.mp4"),
        (f"{base}/data/audio/yongen.wav", data / "audio" / "yongen.wav"),
        (f"{base}/data/audio/eng.wav", data / "audio" / "eng.wav"),
    ]
    for url, path in targets:
        try:
            _download_url(url, path)
        except Exception as exc:  # noqa: BLE001
            print(f"[URL] FAILED {url}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent.parent
    ap.add_argument("--root", default=str(default_root))
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"project root: {root}")

    if not args.skip_models:
        download_models(root)
    if not args.skip_demo:
        download_demo_data(root)

    print("\n=== all downloads complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

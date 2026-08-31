"""Download GPT-SoVITS pretrained models into third_party/GPT_SoVITS/pretrained_models.

Downloads (from HF repo ``lj1995/GPT-SoVITS`` unless noted):
1. chinese-hubert-base/          ← HuBERT audio encoder
2. chinese-roberta-wwm-ext-large/← BERT text encoder
3. s1v3.ckpt                     ← GPT (t2s) base model
4. v2Pro/                        ← SoVITS v2Pro base (s2Gv2Pro.pth / s2Dv2Pro.pth)
5. sv/pretrained_eres2netv2w24s4ep4.ckpt ← speaker-verifier used by v2Pro
6. fast_langdetect/lid.176.bin   ← from fasttext CDN (language segmenter)

Total ≈ 1.4 GB. For faster downloads in China set HF_ENDPOINT=https://hf-mirror.com.

Usage:
    python scripts/download_gptsovits.py [--root <project_root>]

Vendoring provenance and upgrade policy: third_party/GPT_SoVITS/README_SELF.md
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HF_REPO = "lj1995/GPT-SoVITS"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

PATTERNS = [
    "chinese-hubert-base/*",
    "chinese-roberta-wwm-ext-large/*",
    "s1v3.ckpt",
    "v2Pro/*",
    "sv/pretrained_eres2netv2w24s4ep4.ckpt",
]


def _download_url(url: str, target: Path, *, timeout: int = 120) -> None:
    if target.exists() and target.stat().st_size > 1000:
        print(f"[URL] exists, skip: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[URL] downloading {url} → {target}")
    urllib.request.urlretrieve(url, str(target))
    print(f"[URL] done: {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="project root (default: repo root)")
    args = parser.parse_args()

    root = args.root
    if root is None:
        root = Path(__file__).resolve().parents[1]
    target_dir = root / "third_party" / "GPT_SoVITS" / "pretrained_models"

    if not (root / "third_party" / "GPT_SoVITS" / "TTS_infer_pack").exists() and not (
        root / "third_party" / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
    ).exists():
        print(
            "warning: GPT-SoVITS engine code not found under third_party/GPT_SoVITS — "
            "pretrained models will still be downloaded.",
            file=sys.stderr,
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: huggingface_hub is required (pip install huggingface_hub)", file=sys.stderr)
        return 2

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[HF] downloading {HF_REPO} → {target_dir}")
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(target_dir),
        allow_patterns=PATTERNS,
    )
    print(f"[HF] done: {target_dir}")

    _download_url(LID_URL, target_dir / "fast_langdetect" / "lid.176.bin")
    print("all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

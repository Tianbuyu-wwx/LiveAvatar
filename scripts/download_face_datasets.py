"""Download the R1 face training datasets (WIDER FACE + 300W).

Pending-item #1 of the R1 plan (see docs/自研人脸检测与对齐方案_2026-08-31.md
§M5 worklist). Downloads:

- WIDER FACE: WIDER_train.zip + WIDER_val.zip + wider_face_split.zip
  from the canonical HuggingFace ``wider_face`` dataset repo (via a mirror
  endpoint by default, since huggingface.co is unreachable from CN networks).
- 300W: ``300w_dataset.zip`` from the HuggingFace mirror repo
  ``quoctai219/300W`` (no login required — the iBUG 4-part split needs an
  account, and dlib.net throttles at ~25 KB/s). Contains the 300W subsets
  with per-image ``.pts`` ground truth, exactly what
  ``scripts/make_face_dataset.py`` consumes via rglob.

Layout produced:

    data/face_ds/raw/wider/
        wider_face_split/wider_face_train_bbx_gt.txt
        WIDER_train/images/...
        WIDER_val/images/...
    data/face_ds/raw/300w/
        <afw|helen|ibug|lfpw|Test>/... .pts + images

Downloads use curl with resume (-C -) — urllib hangs on some of these
endpoints. Archives are integrity-checked before extraction. Both datasets
are research-only licensed (non-commercial).

Usage:
    python scripts/download_face_datasets.py [--wider] [--w300] [--root <dir>]
        [--hf-endpoint https://hf-mirror.com]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

W300_HF_REPO = "quoctai219/300W"
W300_ZIP = "300w_dataset.zip"


def _curl_download(url: str, target: Path, *, max_attempts: int = 30) -> None:
    """Download with curl, resumable (-C -), retrying until complete.

    Mirror connections drop frequently; error 18 (partial transfer) is not
    covered by curl's --retry, so loop here — -C - resumes where it left off.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    curl = "curl.exe" if sys.platform == "win32" else "curl"
    cmd = [
        curl, "-sSL", "-C", "-", "--speed-time", "60",
        "--speed-limit", "10240", "-o", str(target), url,
    ]
    print(f"[curl] {url} -> {target}")
    for attempt in range(1, max_attempts + 1):
        proc = subprocess.run(cmd)
        if proc.returncode == 0 and target.exists() and target.stat().st_size > 0:
            size_mb = target.stat().st_size / 1e6
            print(f"[curl] done: {target} ({size_mb:.1f} MB, {attempt} attempt(s))")
            return
        print(f"[curl] attempt {attempt} failed (rc={proc.returncode}), resuming ...")
    raise RuntimeError(f"curl still incomplete after {max_attempts} attempts: {url}")


def _verify_and_extract_zip(zip_path: Path, dest: Path) -> None:
    print(f"[zip] verifying {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad!r} in {zip_path}")
        print(f"[zip] ok ({len(zf.namelist())} members), extracting -> {dest}")
        zf.extractall(dest)
    print(f"[zip] done: {zip_path.name}")


def _verify_and_extract_tar(tar_path: Path, dest: Path) -> None:
    print(f"[tar] verifying {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        print(f"[tar] ok ({len(members)} members), extracting -> {dest}")
        tf.extractall(dest)
    print(f"[tar] done: {tar_path.name}")


def download_wider(root: Path, hf_endpoint: str) -> None:
    """WIDER FACE train/val images + bbox split annotations."""
    out = root / "raw" / "wider"
    for name in ("WIDER_train.zip", "WIDER_val.zip", "wider_face_split.zip"):
        url = f"{hf_endpoint}/datasets/wider_face/resolve/main/data/{name}"
        _curl_download(url, out / name)
        _verify_and_extract_zip(out / name, out)


def download_300w(root: Path, hf_endpoint: str) -> None:
    """300W (HF mirror, no login required): images + 68-point .pts."""
    out = root / "raw" / "300w"
    url = f"{hf_endpoint}/datasets/{W300_HF_REPO}/resolve/main/{W300_ZIP}"
    _curl_download(url, out / W300_ZIP)
    _verify_and_extract_zip(out / W300_ZIP, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent.parent / "data" / "face_ds"
    ap.add_argument("--root", default=str(default_root))
    ap.add_argument("--wider", action="store_true", help="download WIDER FACE")
    ap.add_argument("--w300", action="store_true", help="download 300W")
    ap.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="HuggingFace mirror endpoint (set to https://huggingface.co if reachable)",
    )
    args = ap.parse_args()
    if not (args.wider or args.w300):
        ap.error("nothing to do: pass --wider and/or --w300")

    root = Path(args.root)
    if args.wider:
        download_wider(root, args.hf_endpoint.rstrip("/"))
    if args.w300:
        download_300w(root, args.hf_endpoint.rstrip("/"))
    print("\n=== face dataset download complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""R1 M1: train the self-developed face detector on an M0 manifest.

Reads ``data/face_ds/manifest.jsonl`` (see scripts/make_face_dataset.py),
resizes images to a fixed square and optimizes the TinyFaceDetector with
the focal + smooth-L1 detection loss.  CPU-trainable (small model), GPU
auto-used when available and free.

Usage::

    python scripts/train_face_det.py --manifest data/face_ds/manifest.jsonl \
        --epochs 40 --out weights/self/facedet_256.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liveavatar.face_self import (  # noqa: E402
    STRIDE,
    TinyFaceDetector,
    detection_loss,
)


def _imread_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_entries(manifest: Path, size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize (image_tensor_float01, gt_boxes_norm_cxcywh) pairs."""
    entries: list[tuple[np.ndarray, np.ndarray]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        img_path = Path(e["image"])
        if not img_path.exists() or not e.get("boxes"):
            continue
        rgb = _imread_rgb(img_path)
        h, w = rgb.shape[:2]
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        sx, sy = size / w, size / h
        boxes = []
        for x, y, bw, bh in e["boxes"]:
            px, py, pw, ph = x * w * sx, y * h * sy, bw * w * sx, bh * h * sy
            boxes.append(
                [(px + pw / 2) / size, (py + ph / 2) / size, pw / size, ph / size]
            )
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        entries.append((tensor.numpy(), np.asarray(boxes, np.float32)))
    return entries


def train(args: argparse.Namespace) -> float:
    manifest = Path(args.manifest)
    entries = load_entries(manifest, args.size)
    if not entries:
        raise SystemExit(f"no usable entries in {manifest}")
    rng = random.Random(0)
    rng.shuffle(entries)
    n_val = max(1, int(len(entries) * args.val_frac)) if len(entries) > 4 else 0
    val, train_set = entries[:n_val], entries[n_val:]

    torch.manual_seed(0)
    model = TinyFaceDetector(width=args.width)
    device = torch.device(
        "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    anchors = TinyFaceDetector.make_anchors(
        args.size // STRIDE, args.size // STRIDE
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    last_loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_set)
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(train_set), args.batch):
            chunk = train_set[start : start + args.batch]
            imgs = torch.stack([torch.from_numpy(e[0]) for e in chunk]).to(device)
            gts = [torch.from_numpy(e[1]) for e in chunk]
            cls, box = model(imgs)
            loss, _ = detection_loss(cls, box, anchors, gts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            batches += 1
        last_loss = epoch_loss / max(batches, 1)
        if args.val:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for start in range(0, len(val), args.batch):
                    chunk = val[start : start + args.batch]
                    imgs = torch.stack([torch.from_numpy(e[0]) for e in chunk]).to(device)
                    gts = [torch.from_numpy(e[1]) for e in chunk]
                    cls, box = model(imgs)
                    vloss, _ = detection_loss(cls, box, anchors, gts)
                    val_loss += float(vloss)
            print(
                f"[train] epoch {epoch}: loss={last_loss:.4f} "
                f"val={val_loss / max(len(val) // args.batch, 1):.4f}",
                flush=True,
            )
        elif epoch % max(args.epochs // 10, 1) == 0 or epoch == args.epochs:
            print(f"[train] epoch {epoch}: loss={last_loss:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "width": args.width,
            "input_size": args.size,
            "stride": STRIDE,
        },
        out,
    )
    print(f"[train] saved checkpoint -> {out}")
    return last_loss


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/face_ds/manifest.jsonl")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=256, help="square input (mult of 16)")
    ap.add_argument("--width", type=int, default=48, help="head channel width")
    ap.add_argument("--val", action="store_true", help="print val loss each epoch")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--out", default="weights/self/facedet_256.pt")
    args = ap.parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

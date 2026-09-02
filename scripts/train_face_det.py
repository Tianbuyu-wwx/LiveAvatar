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
    ANCHOR_SIZES,
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


def load_entries(
    manifest: Path, size: int
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Materialize (image_uint8_chw, gt_boxes_norm_cxcywh, source) triples.

    Images stay uint8 (÷4 RAM vs float32) — at size 320 the full WIDER set
    would otherwise need ~16 GB resident.  Batches convert to float on GPU.
    Source tags drive --oversample AFTER the train/val split so duplicated
    entries never leak into the val set.
    """
    entries: list[tuple[np.ndarray, np.ndarray, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        img_path = Path(e["image"])
        if not img_path.exists() or not e.get("boxes"):
            continue
        img, boxes = _entry(e, img_path, size)
        entries.append((img, boxes, e.get("source", "")))
    return entries


def _entry(e: dict, img_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
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
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return tensor.numpy(), np.asarray(boxes, np.float32)


def train(args: argparse.Namespace) -> float:
    manifest = Path(args.manifest)
    entries = load_entries(manifest, args.size)
    if not entries:
        raise SystemExit(f"no usable entries in {manifest}")
    rng = random.Random(0)
    rng.shuffle(entries)
    n_val = max(1, int(len(entries) * args.val_frac)) if len(entries) > 4 else 0
    val, train_set = entries[:n_val], entries[n_val:]
    # Oversample AFTER the split: duplicated entries stay out of val.
    train_set = [
        e for e in train_set for _ in range(args._oversample.get(e[2], 1))
    ]

    torch.manual_seed(0)
    model = TinyFaceDetector(width=args.width)
    if args.init:
        # Warm start: continue a previous run (anchor layout must match).
        init_ckpt = torch.load(args.init, map_location="cpu", weights_only=True)
        if tuple(init_ckpt.get("anchor_sizes") or ()) != model.anchor_sizes:
            raise SystemExit(
                f"--init anchor layout mismatch: ckpt={init_ckpt.get('anchor_sizes')} "
                f"model={model.anchor_sizes}"
            )
        model.load_state_dict(init_ckpt["model"])
        print(f"[train] warm start from {args.init}", flush=True)
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
            imgs = imgs.float() / 255.0
            gts = [torch.from_numpy(e[1]) for e in chunk]
            if args.flip_aug:
                # Horizontal flip with mirrored cxcywh boxes.
                flip = torch.rand(imgs.shape[0]) < 0.5
                imgs[flip] = imgs[flip].flip(-1)
                for k in torch.nonzero(flip).flatten().tolist():
                    gts[k] = gts[k].clone()
                    gts[k][:, 0] = 1.0 - gts[k][:, 0]
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
                    imgs = imgs.float() / 255.0
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
            "anchor_sizes": ANCHOR_SIZES,
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
    ap.add_argument("--flip-aug", action="store_true",
                    help="horizontal flip + mirrored boxes augmentation")
    ap.add_argument("--oversample", default="",
                    help="repeat under-represented sources, e.g. 'w300:8' appends "
                         "8 copies of every w300 entry (comma-separated list)")
    ap.add_argument("--init", default="",
                    help="warm-start from an existing checkpoint (same anchor layout)")
    ap.add_argument("--out", default="weights/self/facedet_256.pt")
    args = ap.parse_args(argv)
    args._oversample = {
        src: int(count) for src, count in (
            part.split(":", 1) for part in args.oversample.split(",") if part
        )
    }
    train(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

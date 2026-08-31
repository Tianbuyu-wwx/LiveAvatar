"""R1 M2: train the self-developed 5-point landmark student on an M0 manifest.

Reads ``data/face_ds/manifest.jsonl`` entries that carry ``points5``
(300W ground truth + own-material MediaPipe teacher pseudo labels), crops
each face box with the shared ``FaceCropTransform`` and optimizes
``LandmarkNet5Self`` with focal heatmap + offset + coordinate losses.

CPU-trainable by default (project rule: never assume a free GPU):

    python scripts/train_face_landmarks.py \
        --manifest data/face_ds/manifest.jsonl --epochs 60 \
        --out weights/self/landmarks5_128.pt
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

from liveavatar.face_landmarks import (  # noqa: E402
    LM_STRIDE,
    FaceCropTransform,
    LandmarkNet5Self,
    landmark_loss,
)


def _imread_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_entries(manifest: Path, size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize (crop_image_float01, points5_crop_norm) training pairs."""
    entries: list[tuple[np.ndarray, np.ndarray]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if not e.get("points5"):
            continue
        img_path = Path(e["image"])
        if not img_path.exists() or not e.get("boxes"):
            continue
        rgb = _imread_rgb(img_path)
        h, w = rgb.shape[:2]
        box = e["boxes"][0]  # primary face
        # Denormalize the box to pixels, then crop with the shared transform.
        transform = FaceCropTransform(box, out_size=size)
        crop = transform.crop(rgb)
        pts_full = np.asarray(e["points5"], np.float32)  # (5, 2) full-image norm
        pts_crop = transform.points(pts_full)
        crop_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        entries.append((crop_t.numpy(), pts_crop))
    return entries


def train(args: argparse.Namespace) -> float:
    manifest = Path(args.manifest)
    entries = load_entries(manifest, args.size)
    if not entries:
        raise SystemExit(f"no entries with points5 in {manifest}")
    rng = random.Random(0)
    rng.shuffle(entries)

    torch.manual_seed(0)
    model = LandmarkNet5Self(width=args.width)
    device = torch.device(args.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    last_loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(entries)
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(entries), args.batch):
            chunk = entries[start : start + args.batch]
            imgs = torch.stack([torch.from_numpy(e[0]) for e in chunk]).to(device)
            pts = torch.stack([torch.from_numpy(e[1]) for e in chunk]).to(device)
            hm, off = model(imgs)
            loss, _ = landmark_loss(hm, off, pts, coord_weight=args.coord_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            batches += 1
        last_loss = epoch_loss / max(batches, 1)
        if epoch % max(args.epochs // 10, 1) == 0 or epoch == args.epochs:
            print(f"[train] epoch {epoch}: loss={last_loss:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "width": args.width,
            "input_size": args.size,
            "stride": LM_STRIDE,
        },
        out,
    )
    print(f"[train] saved checkpoint -> {out}")
    return last_loss


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/face_ds/manifest.jsonl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=128, help="square crop size")
    ap.add_argument("--width", type=int, default=32, help="head channel width")
    ap.add_argument("--coord-weight", type=float, default=0.25)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--out", default="weights/self/landmarks5_128.pt")
    args = ap.parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

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


# Horizontal flip swaps left/right eye and mouth identities; nose (2) stays.
_FLIP_PERM = (1, 0, 2, 4, 3)


def augment_batch(
    imgs: np.ndarray, pts: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """In-place horizontal flip + brightness/contrast jitter for one batch.

    imgs: (B, 3, H, W) float01; pts: (B, 5, 2) crop-normalized. The crop is
    a square, so flipping the image maps x → 1−x and the point identities
    swap (eye corners, mouth corners) while the nose stays. Brightness is a
    per-sample gain/offset — points are unchanged.
    """
    for i in range(imgs.shape[0]):
        if rng.random() < 0.5:
            imgs[i] = imgs[i, :, :, ::-1]
            pts[i, :, 0] = 1.0 - pts[i, :, 0]
            pts[i] = pts[i][list(_FLIP_PERM)]
        scale = rng.uniform(0.85, 1.15)
        bias = rng.uniform(-0.05, 0.05)
        imgs[i] = np.clip(imgs[i] * scale + bias, 0.0, 1.0)
    return imgs, pts


def load_entries(
    manifest: Path,
    size: int,
    det_model: object | None = None,
    det_size: int = 256,
    det_conf: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize (crop_image_float01, points5_crop_norm) training pairs.

    When the M1 detector is available, crops use the *detector* box — the
    exact convention used at inference time (``face_backend._landmarks5_self``).
    Training on teacher-style boxes while serving detector boxes is a
    train/serve skew that the tiny heatmap net does not survive.
    """
    from liveavatar.face_self import detect as det_detect

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
        box = e["boxes"][0]  # primary face (manifest fallback)
        if det_model is not None:
            hits = det_detect(det_model, rgb, conf_threshold=det_conf, input_size=det_size)
            if hits:
                x1, y1, x2, y2 = max(
                    hits, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])
                )[0]
                box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
        # Crop with the shared transform (normalized xywh box).
        transform = FaceCropTransform(box, out_size=size)
        crop = transform.crop(rgb)
        pts_full = np.asarray(e["points5"], np.float32)  # (5, 2) full-image norm
        pts_crop = transform.points(pts_full)
        crop_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        entries.append((crop_t.numpy(), pts_crop))
    return entries


def train(args: argparse.Namespace) -> float:
    manifest = Path(args.manifest)
    det_model = None
    det_size = 256
    if not args.no_det_box and Path(args.det_ckpt).exists():
        from liveavatar.face_backend import load_det_model

        det_model, det_size = load_det_model(args.det_ckpt)
        print(f"[train] crops from detector boxes: {args.det_ckpt} (size={det_size})")
    else:
        print("[train] detector ckpt unavailable — falling back to manifest boxes")
    entries = load_entries(
        manifest, args.size, det_model=det_model, det_size=det_size, det_conf=args.det_conf
    )
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
            imgs_np = np.stack([e[0] for e in chunk])
            pts_np = np.stack([e[1] for e in chunk])
            if args.augment:
                imgs_np, pts_np = augment_batch(imgs_np, pts_np, rng)
            imgs = torch.from_numpy(np.ascontiguousarray(imgs_np)).to(device)
            pts = torch.from_numpy(pts_np).to(device)
            hm, off = model(imgs)
            loss, _ = landmark_loss(
                hm, off, pts, coord_weight=args.coord_weight, decode_mode=args.decode
            )
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
            "decode_mode": args.decode,
            "pool_before_argmax": args.pool,
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
    ap.add_argument("--det-ckpt", default="weights/self/facedet_256.pt",
                    help="M1 detector ckpt — crops use its boxes (inference convention)")
    ap.add_argument("--no-det-box", action="store_true",
                    help="disable detector-box crops (use manifest boxes)")
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--decode", choices=("argmax", "soft"), default="argmax",
                    help="decode mode: soft-argmax (DSNT) avoids cell-jump keypoint "
                         "instability; stored in the ckpt and used at inference")
    ap.add_argument("--pool", action="store_true",
                    help="3x3 avg-pool the heatmap before argmax (anti-jitter); "
                         "stored in the ckpt, overridable via LANDMARK_POOL env")
    ap.add_argument("--augment", action="store_true",
                    help="horizontal flip + brightness jitter (on-the-fly)")
    ap.add_argument("--out", default="weights/self/landmarks5_128.pt")
    args = ap.parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

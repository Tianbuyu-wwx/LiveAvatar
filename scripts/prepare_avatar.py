"""Prepare avatar preprocessing data from a reference video/image.

Generates the avatar data directory expected by ``MuseTalkAvatarWorker``:
- full_imgs/  : original frames (0.jpg, 1.jpg, ...)
- coords.pkl  : face bounding boxes [(x1,y1,x2,y2), ...]
- latents.pt  : VAE-encoded reference latents [tensor(1,8,32,32), ...]
- mask/       : blending masks (grayscale, white=face region)
- mask_coords.pkl : mask crop boxes [(x_s,y_s,x_e,y_e), ...]

Face detection uses a switchable backend (R1 M3):
``--face-backend yunet|self`` (env ``FACE_BACKEND``) — OpenCV YuNet ONNX
(legacy default) or the self-developed TinyFaceDetector. By default a
5-point face alignment step is applied first (``--landmark-backend
mediapipe|self``, env ``LANDMARK_BACKEND``), which significantly improves
MuseTalk lip-sync quality.

Usage:
    python scripts/prepare_avatar.py \\
        --input data/video/yongen.mp4 \\
        --avatar-id yongen \\
        --avatar-data-root data/avatars \\
        --max-frames 8
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _imread_utf8(path: Path) -> np.ndarray | None:
    """cv2.imread with unicode path support on Windows."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_utf8(path: Path, img: np.ndarray) -> bool:
    """cv2.imwrite with unicode path support on Windows."""
    import cv2

    ok, buf = cv2.imencode(str(path.suffix), img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _extract_frames(video_path: str, out_dir: Path, max_frames: int, fps: int) -> list[Path]:
    """Extract frames from a video at the given fps → out_dir/0.jpg, 1.jpg, ..."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = max(1, int(round(src_fps / fps)))
    paths: list[Path] = []
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_skip == 0:
            p = out_dir / f"{saved}.jpg"
            _imwrite_utf8(p, frame)
            paths.append(p)
            saved += 1
            if saved >= max_frames:
                break
        idx += 1
    cap.release()
    print(f"[prepare] extracted {len(paths)} frames @ {fps}fps (src {src_fps:.1f})")
    return paths


def _detect_faces(
    frames: list,
    coords_path: Path,
    mask_coords_path: Path,
    *,
    backend: str | None = None,
    yunet_model_path: str = "models/face_detection_yunet_2023mar.onnx",
    det_ckpt_path: str = "weights/self/facedet_256.pt",
    bbox_shift: int = 5,
    conf_threshold: float = 0.7,
) -> tuple[list, list]:
    """Detect face bboxes → coords.pkl + mask_coords.pkl.

    ``backend``: "yunet" (OpenCV ONNX, legacy default) or "self" (R1
    self-developed detector); resolved via the ``FACE_BACKEND`` env var by
    the face_backend factory when None.
    """
    from liveavatar.face_backend import detect_faces

    boxes_per_frame: list[list] = [
        detect_faces(
            frame,
            backend,
            yunet_model_path=yunet_model_path,
            det_ckpt_path=det_ckpt_path,
            conf_threshold=conf_threshold,
        )
        for frame in frames
    ]

    coords_list: list[tuple[int, int, int, int]] = []
    mask_coords_list: list[tuple[int, int, int, int]] = []
    placeholder = (0, 0, 0, 0)

    for frame, boxes in zip(frames, boxes_per_frame):
        fh, fw = frame.shape[:2]
        if not boxes:
            coords_list.append(placeholder)
            mask_coords_list.append(placeholder)
            continue
        # Pick the largest face.
        best = max(boxes, key=lambda b: b.area)
        x1, y1 = max(0, int(round(best.x1))), max(0, int(round(best.y1)))
        x2, y2 = min(fw, int(round(best.x2))), min(fh, int(round(best.y2)))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            coords_list.append(placeholder)
            mask_coords_list.append(placeholder)
            continue
        # Expand bbox downward by bbox_shift fraction (MuseTalk convention).
        y2 = min(fh, int(y2 + bh * bbox_shift / 20.0))
        coords_list.append((x1, y1, x2, y2))

        # mask_coords: padded crop box (~1.25x face bbox) for blending.
        pad = int(bw * 0.25)
        mx1 = max(0, x1 - pad)
        my1 = max(0, y1 - pad)
        mx2 = min(fw, x2 + pad)
        my2 = min(fh, y2 + pad)
        mask_coords_list.append((mx1, my1, mx2, my2))

    with open(coords_path, "wb") as f:
        pickle.dump(coords_list, f)
    with open(mask_coords_path, "wb") as f:
        pickle.dump(mask_coords_list, f)
    print(
        f"[prepare] coords: {len(coords_list)} entries → {coords_path.name} "
        f"+ {mask_coords_path.name} (backend={backend or 'env'})"
    )
    return coords_list, mask_coords_list


def _generate_masks(
    frames: list, coords_list: list, mask_coords_list: list, mask_dir: Path
) -> None:
    """Generate blending masks: white in face bbox, feathered, within crop box."""
    import cv2

    mask_dir.mkdir(parents=True, exist_ok=True)
    for i, (frame, coord, mcoord) in enumerate(
        zip(frames, coords_list, mask_coords_list)
    ):
        x1, y1, x2, y2 = coord
        mx1, my1, mx2, my2 = mcoord
        cw, ch = mx2 - mx1, my2 - my1
        mask = np.zeros((ch, cw), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            # Face region in crop-box-local coords.
            fx1 = max(0, x1 - mx1)
            fy1 = max(0, y1 - my1)
            fx2 = min(cw, x2 - mx1)
            fy2 = min(ch, y2 - my1)
            mask[fy1:fy2, fx1:fx2] = 255
            # Feather edges for smoother blending.
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
        _imwrite_utf8(mask_dir / f"{i}.jpg", mask)
    print(f"[prepare] masks: {len(frames)} → {mask_dir.name}/")


def _encode_latents(
    frames: list,
    coords_list: list,
    latents_path: Path,
    vae_model_dir: str,
    device: str,
    is_half: bool,
) -> None:
    """Encode face crops via the VAE → latents.pt.

    Each latent is [1, 8, 32, 32] (masked + ref concatenated), matching
    ``vae.get_latents_for_unet``.
    """
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from liveavatar.musetalk.models.vae import VAE

    vae = VAE(model_path=vae_model_dir, use_float16=is_half)
    if device.startswith("cuda"):
        vae.vae = vae.vae.to(device)
        if is_half:
            vae.vae = vae.vae.half()

    latents: list = []
    placeholder = torch.zeros(1, 8, 32, 32, dtype=vae.vae.dtype)
    for i, (frame, coord) in enumerate(zip(frames, coords_list)):
        x1, y1, x2, y2 = coord
        if x2 <= x1 or y2 <= y1:
            latents.append(placeholder.cpu())
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            latents.append(placeholder.cpu())
            continue
        # VAE expects a 256x256 face crop (its fixed resized_img).
        crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        # Pass the numpy crop directly to VAE.preprocess_img (else-branch).
        # This avoids temp files and unicode path issues on Windows.
        try:
            latent = vae.get_latents_for_unet(crop_256)  # [1, 8, 32, 32]
            latents.append(latent.cpu())
        except Exception as exc:  # noqa: BLE001
            print(f"[prepare] latent {i} failed: {exc}; using placeholder")
            latents.append(placeholder.cpu())
        print(f"[prepare] latent {i}: shape {tuple(latents[-1].shape)}")

    torch.save(latents, str(latents_path))
    print(f"[prepare] latents: {len(latents)} → {latents_path.name}")


def prepare_avatar(
    input_path: str,
    avatar_id: str,
    avatar_data_root: str,
    *,
    vae_model_dir: str = "models/sd-vae-ft-mse",
    face_model_path: str = "models/face_detection_yunet_2023mar.onnx",
    device: str = "cuda",
    is_half: bool = True,
    max_frames: int = 8,
    fps: int = 25,
    bbox_shift: int = 5,
    align: bool = True,
    align_size: int = 768,
    face_backend: str | None = None,
    landmark_backend: str | None = None,
    det_ckpt: str | None = None,
    lm_ckpt: str | None = None,
) -> str:
    """Run the full preprocessing pipeline; return the avatar data dir.

    ``face_backend``: "yunet" (default) or "self"; ``landmark_backend``:
    "mediapipe" (default) or "self". None falls back to the ``FACE_BACKEND``
    / ``LANDMARK_BACKEND`` env vars, then the legacy defaults.
    """

    data_dir = Path(avatar_data_root) / avatar_id
    full_imgs_dir = data_dir / "full_imgs"
    mask_dir = data_dir / "mask"
    coords_path = data_dir / "coords.pkl"
    latents_path = data_dir / "latents.pt"
    mask_coords_path = data_dir / "mask_coords.pkl"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 0. Optional 5-point face alignment.
    input_p = Path(input_path)
    aligned_input = str(input_p)
    if align:
        # Lazy import so prepare_avatar runs without mediapipe when --no-align.
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from face_align import align_image, align_video

        if input_p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            aligned_path = (
                Path(tempfile.gettempdir()) / f"prepare_avatar_aligned_{avatar_id}.mp4"
            )
            print(f"[prepare] aligning video with 5-point landmarks → {aligned_path}")
            align_video(
                str(input_p),
                str(aligned_path),
                output_size=align_size,
                max_frames=max_frames,
                landmark_backend=landmark_backend,
                det_ckpt=det_ckpt,
                lm_ckpt=lm_ckpt,
            )
            aligned_input = str(aligned_path)
        elif input_p.is_file() and input_p.suffix.lower() in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        ):
            aligned_path = (
                Path(tempfile.gettempdir()) / f"prepare_avatar_aligned_{avatar_id}.jpg"
            )
            print(f"[prepare] aligning image with 5-point landmarks → {aligned_path}")
            if align_image(
                str(input_p),
                str(aligned_path),
                output_size=align_size,
                landmark_backend=landmark_backend,
                det_ckpt=det_ckpt,
                lm_ckpt=lm_ckpt,
            ):
                aligned_input = str(aligned_path)
            else:
                print("[prepare] warning: face alignment failed, using original image")
        elif input_p.is_dir():
            print("[prepare] warning: image-directory input is not aligned; pass --no-align")

    # 1. Extract / load frames.
    input_p = Path(aligned_input)
    if input_p.is_dir():
        img_paths = sorted(
            glob.glob(str(input_p / "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: (
                int(os.path.splitext(os.path.basename(x))[0])
                if os.path.splitext(os.path.basename(x))[0].isdigit()
                else x
            ),
        )[:max_frames]
        paths = [Path(p) for p in img_paths]
    elif input_p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        paths = _extract_frames(aligned_input, full_imgs_dir, max_frames, fps)
    else:  # single image
        full_imgs_dir.mkdir(parents=True, exist_ok=True)
        dst = full_imgs_dir / "0.jpg"
        shutil.copy(aligned_input, dst)
        paths = [dst]

    # Load frames into memory using unicode-safe reader.
    frames = [_imread_utf8(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError(f"no readable frames from {input_path}")

    # 2. Face detection → coords.pkl + mask_coords.pkl.
    coords_list, mask_coords_list = _detect_faces(
        frames,
        coords_path,
        mask_coords_path,
        backend=face_backend,
        yunet_model_path=face_model_path,
        det_ckpt_path=det_ckpt or "weights/self/facedet_256.pt",
        bbox_shift=bbox_shift,
    )
    if all(c == (0, 0, 0, 0) for c in coords_list):
        raise RuntimeError(
            f"no faces detected in any frame of {input_path} — "
            "use a clearer front-face reference"
        )

    # 3. Masks.
    _generate_masks(frames, coords_list, mask_coords_list, mask_dir)

    # 4. Latents (real VAE).
    _encode_latents(
        frames, coords_list, latents_path,
        vae_model_dir=vae_model_dir, device=device, is_half=is_half,
    )

    # 5. Region spec for the self-developed region-delta transport (M4):
    #    union bounding box of the mouth masks → region.json. Missing /
    #    degenerate masks simply skip the file (transport falls back to
    #    full-frame MJPEG).
    try:
        from liveavatar.region_codec import (
            region_spec_from_masks,
            write_region_json,
        )

        spec = region_spec_from_masks(str(mask_dir), *frames[0].shape[1::-1])
        if spec is None:
            print("[prepare] warning: no usable mouth masks — region.json skipped")
        else:
            write_region_json(str(data_dir / "region.json"), spec)
            print(
                f"  region: ({spec.x},{spec.y}) {spec.w}x{spec.h}"
                f" (region-delta transport ready)"
            )
    except ImportError:
        print("[prepare] warning: liveavatar not importable — region.json skipped")

    print(f"\n[prepare] avatar '{avatar_id}' ready at {data_dir}")
    print(f"  full_imgs: {len(paths)} frames")
    print(f"  coords: {sum(1 for c in coords_list if c != (0, 0, 0, 0))} valid")
    return str(data_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="reference video/image/dir")
    ap.add_argument("--avatar-id", required=True)
    ap.add_argument("--avatar-data-root", default="data/avatars")
    ap.add_argument("--vae-model-dir", default="models/sd-vae-ft-mse")
    ap.add_argument("--face-model-path", default="models/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--bbox-shift", type=int, default=5)
    ap.add_argument("--no-align", action="store_true", help="skip 5-point face alignment")
    ap.add_argument("--align-size", type=int, default=768)
    ap.add_argument(
        "--face-backend",
        choices=("yunet", "self"),
        default=None,
        help="face detection backend (default: FACE_BACKEND env or yunet)",
    )
    ap.add_argument(
        "--landmark-backend",
        choices=("mediapipe", "self"),
        default=None,
        help="5-point landmark backend (default: LANDMARK_BACKEND env or mediapipe)",
    )
    ap.add_argument(
        "--det-ckpt",
        default=None,
        help="self detector checkpoint (default: weights/self/facedet_256.pt)",
    )
    ap.add_argument(
        "--lm-ckpt",
        default=None,
        help="self landmark checkpoint (default: weights/self/landmarks5_128.pt)",
    )
    args = ap.parse_args()

    prepare_avatar(
        args.input,
        args.avatar_id,
        args.avatar_data_root,
        vae_model_dir=args.vae_model_dir,
        face_model_path=args.face_model_path,
        device=args.device,
        is_half=not args.no_half,
        max_frames=args.max_frames,
        fps=args.fps,
        bbox_shift=args.bbox_shift,
        align=not args.no_align,
        align_size=args.align_size,
        face_backend=args.face_backend,
        landmark_backend=args.landmark_backend,
        det_ckpt=args.det_ckpt,
        lm_ckpt=args.lm_ckpt,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

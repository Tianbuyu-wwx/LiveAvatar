"""MuseTalk streaming avatar worker — PCM → video frames inference.

Implements ``AvatarWorker._infer_batch`` using the MuseTalk pipeline:
1. PCM S16LE → float32 numpy → Whisper ``audio2feat`` → per-frame audio features.
2. ``feature2chunks`` slices features into a ``batch_size`` batch.
3. Positional encoding → UNet forward → VAE decode → BGR face crops.
4. ``get_image_blending`` pastes face crops back onto the original full image.

Models (VAE / UNet / PE / Whisper) are shared across workers via the
``shared_models`` parameter — each worker only loads its own avatar data
(latents, coords, masks, full images). This conserves GPU memory while
keeping avatar data isolated (cross-talk prevention).

Silence detection: when PCM RMS energy falls below a threshold, the worker
skips inference and returns the avatar's default full image frames with
``is_speaking=False``. This is the first link in the degradation chain
(MuseTalk → static → audio-only).
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
from typing import Any

from .worker import AvatarAssets, AvatarWorker

logger = logging.getLogger("liveavatar.musetalk_worker")


# ────────────────────────────────────────────────────────── helpers


def _mirror_index(size: int, index: int) -> int:
    """Back-and-forth indexing for cycling through avatar reference frames.

    Matches ``utils.image.mirror_index`` from the MuseTalk reference
    implementation but inlined here to avoid importing extra modules at
    load time.
    """
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    return size - res - 1


def _imread_utf8(path: str):
    """cv2.imread with unicode path support on Windows."""
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _read_imgs_utf8(img_list: list[str]) -> list:
    """Read a list of image paths in parallel; unicode-safe on Windows."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    frames = [None] * len(img_list)
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(lambda i, p: (i, _imread_utf8(p)), idx, path): idx
            for idx, path in enumerate(img_list)
        }
        for future in as_completed(futures):
            idx, img = future.result()
            frames[idx] = img
    return frames


def _pcm_s16le_to_float32(pcm: bytes):
    """Convert PCM S16LE bytes → float32 numpy array in [-1, 1]."""
    import numpy as np

    if len(pcm) < 2:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


# ──────────────────────────────────────────────── model cache helper


def load_musetalk_models(
    *,
    device: str = "cuda",
    is_half: bool = True,
    whisper_model_path: str = "models/whisper",
    musetalk_model_dir: str = "models/musetalkV15",
    vae_model_dir: str = "models/sd-vae-ft-mse",
) -> dict[str, Any]:
    """Load MuseTalk models once and return a shared model dict.

    Returns a dict with keys: ``vae``, ``unet``, ``pe``, ``timesteps``,
    ``audio_processor``. Callers should cache the result (e.g. on the pool)
    and pass it to every ``MuseTalkAvatarWorker`` via ``shared_models``.
    """
    import torch

    from .musetalk import Audio2Feature, load_all_model

    unet_path = os.path.join(musetalk_model_dir, "unet.pth")
    unet_config = os.path.join(musetalk_model_dir, "musetalk.json")

    vae, unet, pe = load_all_model(
        unet_model_path=unet_path,
        vae_model_path=vae_model_dir or os.path.join("models", "sd-vae-ft-mse"),
        unet_config=unet_config,
        device=device,
    )
    timesteps = torch.tensor([0], device=device)

    if is_half:
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
    else:
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)

    audio_processor = Audio2Feature(model_path=whisper_model_path)

    logger.info(
        "musetalk_models_loaded",
        extra={
            "device": device,
            "is_half": is_half,
            "unet_path": unet_path,
        },
    )
    return {
        "vae": vae,
        "unet": unet,
        "pe": pe,
        "timesteps": timesteps,
        "audio_processor": audio_processor,
    }


# ────────────────────────────────────────────────── MuseTalkAvatarWorker


class MuseTalkAvatarWorker(AvatarWorker):
    """Streaming video avatar worker backed by MuseTalk.

    Each worker is pinned to one avatar (fixed latents/coords/masks loaded
    at construction). Models are shared via ``shared_models`` to conserve
    GPU memory; avatar data is never shared (cross-talk isolation).

    Parameters
    ----------
    assets : AvatarAssets
        Preprocessed avatar data paths (coords.pkl, latents.pt, etc.).
    shared_models : dict | None
        Pre-loaded model dict from :func:`load_musetalk_models`. When
        ``None``, models are loaded in-process (expensive — prefer sharing).
    device, is_half, whisper_model_path :
        Used only when ``shared_models is None``.
    silence_rms_threshold : float
        PCM RMS threshold (in float32 [-1,1] units) below which audio is
        treated as silence. Default ~0.01 (~−40 dBFS).
    """

    def __init__(
        self,
        assets: AvatarAssets,
        *,
        target_fps: int = 25,
        width: int = 720,
        height: int = 1280,
        batch_size: int = 4,
        device: str = "cuda",
        is_half: bool = True,
        whisper_model_path: str = "models/whisper",
        musetalk_model_dir: str = "models/musetalkV15",
        vae_model_dir: str = "models/sd-vae-ft-mse",
        shared_models: dict[str, Any] | None = None,
        silence_rms_threshold: float = 0.01,
    ) -> None:
        super().__init__(
            assets,
            target_fps=target_fps,
            width=width,
            height=height,
            batch_size=batch_size,
        )
        self._device = device
        self._is_half = is_half
        self._silence_rms_threshold = silence_rms_threshold

        # ── Models (shared or self-loaded) ──
        if shared_models is not None:
            self._vae = shared_models["vae"]
            self._unet = shared_models["unet"]
            self._pe = shared_models["pe"]
            self._timesteps = shared_models["timesteps"]
            self._audio_processor = shared_models["audio_processor"]
        else:
            models = load_musetalk_models(
                device=device,
                is_half=is_half,
                whisper_model_path=whisper_model_path,
                musetalk_model_dir=musetalk_model_dir,
                vae_model_dir=vae_model_dir,
            )
            self._vae = models["vae"]
            self._unet = models["unet"]
            self._pe = models["pe"]
            self._timesteps = models["timesteps"]
            self._audio_processor = models["audio_processor"]

        # ── Avatar data ──
        self._load_avatar_data()

        # Frame index cycles through reference frames via mirror_index.
        self._frame_index = 0

        logger.info(
            "musetalk_worker_init",
            extra={
                "avatar_id": self.avatar_id,
                "avatar_length": self._avatar_length,
                "batch_size": batch_size,
                "target_fps": target_fps,
                "shared_models": shared_models is not None,
            },
        )

    # ------------------------------------------------- avatar data

    def _load_avatar_data(self) -> None:
        """Load preprocessed avatar data from disk.

        Loads:
        - ``coords.pkl`` : face bounding boxes per frame.
        - ``mask_coords.pkl`` : mask crop coordinates per frame.
        - ``latents.pt`` : VAE-encoded reference latents per frame.
        - ``full_imgs/*.jpg`` : original full images (sorted by name).
        - ``mask/*.jpg`` : blending masks per frame (sorted by name).
        """
        import torch

        assets = self.assets

        with open(assets.coords_path, "rb") as f:
            self._coord_list = pickle.load(f)
        with open(assets.mask_coords_path, "rb") as f:
            self._mask_coords_list = pickle.load(f)
        self._input_latent_list = torch.load(assets.latents_path, map_location="cpu")

        img_list = sorted(
            glob.glob(os.path.join(assets.full_imgs_dir, "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        self._frame_list = _read_imgs_utf8(img_list)

        mask_list = sorted(
            glob.glob(os.path.join(assets.mask_dir, "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        self._mask_list = _read_imgs_utf8(mask_list)

        self._avatar_length = len(self._input_latent_list)
        if self._avatar_length == 0:
            raise ValueError(
                f"avatar '{self.avatar_id}' has no reference latents in "
                f"{assets.latents_path}"
            )

    # --------------------------------------------------- inference

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        """Run MuseTalk inference on one PCM chunk.

        When the incoming PCM is short (less than one ``batch_size`` worth of
        audio), the legacy behaviour is preserved and exactly ``batch_size``
        frames are returned. This keeps the streaming adapter and unit tests
        working unchanged.

        When the PCM is long (e.g. an entire segment fed by an offline batch
        renderer), the whole audio is processed at once. Whisper features are
        extracted once and sliced with the correct per-batch ``start`` offset,
        so every video frame gets audio features from its real temporal
        neighbourhood instead of the clamped/repeated tail of a 160ms chunk.
        This fixes the lip-sync drift in offline renders.
        """
        import numpy as np
        import torch

        audio_np = _pcm_s16le_to_float32(pcm_s16le)
        frames: list[tuple[bytes, bool]] = []

        samples_per_batch = self.batch_size * 16000 // self.target_fps
        total_samples = audio_np.size

        if total_samples < samples_per_batch:
            # Legacy / streaming path: keep the original batch_size contract.
            expected_frames = self.batch_size
        else:
            # Offline path: render as many frames as the audio duration dictates.
            expected_frames = max(
                self.batch_size,
                round(total_samples * self.target_fps / 16000),
            )

        # ── Silence path: skip inference, return default pose ──
        if self._is_silence(audio_np):
            for i in range(expected_frames):
                idx = _mirror_index(self._avatar_length, self._frame_index + i)
                full_frame = self._frame_list[idx]
                resized = self._resize_to_target(full_frame)
                frames.append((resized.tobytes(), False))
            self._frame_index += expected_frames
            return frames

        # ── Speaking path: full MuseTalk inference ──
        # 1. Audio → Whisper features once for the whole segment.
        whisper_feature = self._audio_processor.audio2feat(audio_np)

        # Prefer self._device; fall back to the first model parameter's device
        # so fake test models without ``parameters()`` still work.
        params = getattr(self._unet.model, "parameters", None)
        target_device = self._device if params is None else next(params()).device

        # 2. Process video frames in batches, advancing ``start`` correctly.
        for start in range(0, expected_frames, self.batch_size):
            current_batch_size = min(self.batch_size, expected_frames - start)

            whisper_chunks = self._audio_processor.feature2chunks(
                whisper_feature,
                self.target_fps,
                current_batch_size,
                audio_feat_length=[2, 2],
                start=start,
            )
            whisper_batch = np.stack(whisper_chunks)

            # 3. Gather reference latents via mirror_index.
            latent_list = []
            for i in range(current_batch_size):
                idx = _mirror_index(self._avatar_length, self._frame_index + i)
                latent_list.append(self._input_latent_list[idx])
            latent_batch = torch.cat(latent_list, dim=0)

            # 4. Positional encoding + UNet forward.
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(
                device=target_device,
                dtype=self._unet.model.dtype,
            )
            audio_feature_batch = self._pe(audio_feature_batch)
            latent_batch = latent_batch.to(
                device=target_device,
                dtype=self._unet.model.dtype,
            )

            pred_latents = self._unet.model(
                latent_batch,
                self._timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample

            # 5. VAE decode → BGR face crops.
            pred = self._vae.decode_latents(pred_latents)

            # 6. Paste back onto original full image.
            for i, res_frame in enumerate(pred):
                idx = _mirror_index(self._avatar_length, self._frame_index + i)
                combine = self._paste_back(res_frame, idx)
                resized = self._resize_to_target(combine)
                frames.append((resized.tobytes(), True))

            self._frame_index += current_batch_size

        return frames

    # ---------------------------------------------------- helpers

    def _paste_back(self, pred_frame, idx: int):
        """Composite a predicted face crop onto the original full image.

        Mirrors ``MuseReal.paste_back_frame`` from the MuseTalk reference
        implementation. Imports cv2 and the blending utility lazily so tests
        can override this method without requiring cv2.
        """
        import copy

        import cv2
        import numpy as np

        from .musetalk import get_image_blending

        bbox = self._coord_list[idx]
        ori_frame = copy.deepcopy(self._frame_list[idx])
        x1, y1, x2, y2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        mask = self._mask_list[idx]
        mask_crop_box = self._mask_coords_list[idx]
        return get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

    def _resize_to_target(self, frame):
        """Resize a BGR frame to the worker's target (width, height)."""
        import cv2

        h, w = frame.shape[:2]
        if w == self.width and h == self.height:
            return frame
        return cv2.resize(frame, (self.width, self.height))

    def _is_silence(self, audio_np) -> bool:
        """Detect silence via RMS energy."""
        import numpy as np

        if audio_np.size == 0:
            return True
        rms = float(np.sqrt(np.mean(audio_np**2)))
        return rms < self._silence_rms_threshold

    # ---------------------------------------------------- cleanup

    def unload_avatar_data(self) -> None:
        """Release avatar data tensors to free memory.

        Models are shared and should NOT be released here — only the
        per-worker avatar data (frames, masks, latents).
        """
        self._frame_list = []
        self._mask_list = []
        self._input_latent_list = []
        self._coord_list = []
        self._mask_coords_list = []
        logger.info(
            "avatar_data_unloaded", extra={"avatar_id": self.avatar_id}
        )

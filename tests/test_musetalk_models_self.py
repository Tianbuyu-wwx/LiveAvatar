# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""A1 (C4): self-written AutoencoderKLCompat / UNet2DConditionCompat tests.

The real sd-vae-ft-mse / musetalkV15 weights are NOT available in CI, so
these tests validate architecture shapes, the diffusers state-dict key
mapping (lossless round-trip + legacy/modern attention naming) and
determinism, using synthetic weights only.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover — CI light env
    torch = None

try:
    from liveavatar.musetalk.models.unet2d_compat import UNet2DConditionCompat
    from liveavatar.musetalk.models.vae_kl import AutoencoderKLCompat
except ImportError:  # pragma: no cover — torch-dependent model modules
    UNet2DConditionCompat = None  # type: ignore[assignment]
    AutoencoderKLCompat = None  # type: ignore[assignment]

requires_torch = unittest.skipUnless(torch is not None, "torch not installed")

# Small VAE config (ch=32, ch_mult 1-2-4-4): same architecture as
# sd-vae-ft-mse but ~30x fewer parameters so the suite stays cheap on CPU.
SMALL_VAE_CHANNELS = [32, 64, 128, 128]
SMALL_VAE_CFG = {
    "scaling_factor": 0.18215,
    "latent_channels": 4,
    "block_out_channels": SMALL_VAE_CHANNELS,
}

MUNETALK_UNET_CONFIG = {
    "_class_name": "UNet2DConditionModel",
    "in_channels": 8,
    "out_channels": 4,
    "block_out_channels": [320, 640, 1280, 1280],
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ],
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ],
    "layers_per_block": 2,
    "attention_head_dim": 8,
    "cross_attention_dim": 384,
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "flip_sin_to_cos": True,
    "downscale_freq_shift": 0,
}


# ------------------------------------------------------------------- VAE


@requires_torch
def test_vae_encode_decode_shapes() -> None:
    vae = AutoencoderKLCompat(
        ch=SMALL_VAE_CHANNELS[0],
        ch_mult=tuple(c // SMALL_VAE_CHANNELS[0] for c in SMALL_VAE_CHANNELS),
    )
    vae.eval()
    with torch.no_grad():
        enc = vae.encode(torch.randn(1, 3, 64, 64))
        latent = enc.latent_dist.sample()
        assert latent.shape == (1, 4, 8, 8)

        img = vae.decode(torch.randn(1, 4, 8, 8)).sample
        assert img.shape == (1, 3, 64, 64)


@requires_torch
def test_vae_decode_to_uint8_range() -> None:
    """Mirror the decode_latents post-processing of vae.VAE."""
    vae = AutoencoderKLCompat(
        ch=SMALL_VAE_CHANNELS[0],
        ch_mult=tuple(c // SMALL_VAE_CHANNELS[0] for c in SMALL_VAE_CHANNELS),
    )
    vae.eval()
    with torch.no_grad():
        latents = torch.randn(1, 4, 8, 8) / vae.scaling_factor
        image = vae.decode(latents).sample
    import numpy as np

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    assert image.dtype == np.uint8
    assert image.shape == (1, 64, 64, 3)
    assert float(image.astype(np.float32).std()) > 0.0  # not flat


@requires_torch
def test_vae_pretrained_kl_lossless(tmp_path: Path) -> None:
    """State-dict round-trip through from_pretrained_kl is lossless."""
    vae = AutoencoderKLCompat(
        ch=SMALL_VAE_CHANNELS[0],
        ch_mult=tuple(c // SMALL_VAE_CHANNELS[0] for c in SMALL_VAE_CHANNELS),
    )
    vae.eval()
    path = tmp_path / "sd-vae-ft-mse"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(SMALL_VAE_CFG), encoding="utf-8")
    torch.save(dict(vae.state_dict()), path / "diffusion_pytorch_model.bin")

    loaded = AutoencoderKLCompat.from_pretrained_kl(str(path))
    assert loaded.scaling_factor == 0.18215

    torch.manual_seed(0)
    latents = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        out_a = vae.decode(latents).sample
        out_b = loaded.decode(latents).sample
    assert torch.equal(out_a, out_b)


@requires_torch
def test_vae_modern_attention_key_remap(tmp_path: Path) -> None:
    """Modern diffusers to_q/to_k/to_v/to_out.0 keys remap to legacy names."""
    vae = AutoencoderKLCompat(
        ch=SMALL_VAE_CHANNELS[0],
        ch_mult=tuple(c // SMALL_VAE_CHANNELS[0] for c in SMALL_VAE_CHANNELS),
    )
    state = {k: v.clone() for k, v in vae.state_dict().items()}

    def to_modern(key: str) -> str:
        return (
            key.replace(".query.", ".to_q.")
            .replace(".key.", ".to_k.")
            .replace(".value.", ".to_v.")
            .replace(".proj_attn.", ".to_out.0.")
        )

    modern = {to_modern(k): v for k, v in state.items()}
    path = tmp_path / "vae-modern"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(SMALL_VAE_CFG), encoding="utf-8")
    torch.save(modern, path / "diffusion_pytorch_model.bin")

    loaded = AutoencoderKLCompat.from_pretrained_kl(str(path))
    torch.manual_seed(1)
    latents = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        assert torch.equal(vae.decode(latents).sample, loaded.decode(latents).sample)


@requires_torch
def test_vae_incomplete_mapping_raises(tmp_path: Path) -> None:
    """A truncated weight file must be a hard error, never a silent load."""
    vae = AutoencoderKLCompat(
        ch=SMALL_VAE_CHANNELS[0],
        ch_mult=tuple(c // SMALL_VAE_CHANNELS[0] for c in SMALL_VAE_CHANNELS),
    )
    state = dict(vae.state_dict())
    del state["encoder.conv_in.weight"]
    path = tmp_path / "vae-truncated"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(SMALL_VAE_CFG), encoding="utf-8")
    torch.save(state, path / "diffusion_pytorch_model.bin")
    try:
        AutoencoderKLCompat.from_pretrained_kl(str(path))
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for incomplete mapping")


# ------------------------------------------------------------------- UNet


def _build_unet() -> UNet2DConditionCompat:
    torch.manual_seed(0)
    return UNet2DConditionCompat.from_config_dict(MUNETALK_UNET_CONFIG)


@requires_torch
def test_unet_from_config_matches_musetalk() -> None:
    unet = _build_unet()
    keys = set(unet.state_dict())
    # diffusers state-dict key structure for the MuseTalk layout
    for probe in (
        "conv_in.weight",
        "time_embedding.linear_1.weight",
        "time_embedding.linear_2.weight",
        "down_blocks.0.resnets.0.norm1.weight",
        "down_blocks.0.resnets.0.time_emb_proj.weight",
        "down_blocks.0.attentions.0.norm.weight",
        "down_blocks.0.attentions.0.proj_in.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.weight",
        "down_blocks.0.downsamplers.0.conv.weight",
        "down_blocks.3.resnets.1.conv2.weight",  # DownBlock2D: no attentions
        "mid_block.resnets.0.conv1.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight",
        "up_blocks.0.resnets.0.conv_shortcut.weight",  # 2560 → 1280
        "up_blocks.1.resnets.2.conv_shortcut.weight",  # 1920 → 1280
        "up_blocks.2.resnets.0.conv_shortcut.weight",  # 1920 → 640
        "up_blocks.3.resnets.2.conv_shortcut.weight",  # 640 → 320
        "up_blocks.1.upsamplers.0.conv.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.norm3.weight",
        "conv_norm_out.weight",
        "conv_out.weight",
    ):
        assert probe in keys, f"missing expected diffusers key: {probe}"


@requires_torch
def test_unet_forward_shape_and_dtype_surface() -> None:
    unet = _build_unet()
    unet.eval()
    sample = torch.randn(1, 8, 32, 32)
    context = torch.randn(1, 4, 384)  # one frame of whisper features
    with torch.no_grad():
        out = unet(sample, torch.tensor([0]), encoder_hidden_states=context)
    assert out.sample.shape == (1, 4, 32, 32)
    assert unet.dtype == next(unet.parameters()).dtype
    assert len(list(unet.parameters())) > 0


@requires_torch
def test_unet_forward_deterministic() -> None:
    unet = _build_unet()
    unet.eval()
    torch.manual_seed(3)
    sample = torch.randn(1, 8, 32, 32)
    context = torch.randn(1, 4, 384)
    with torch.no_grad():
        a = unet(sample, torch.tensor([0]), encoder_hidden_states=context).sample
        b = unet(sample, torch.tensor([0]), encoder_hidden_states=context).sample
    assert torch.equal(a, b)


@requires_torch
def test_unet_rejects_unknown_config() -> None:
    cfg = dict(MUNETALK_UNET_CONFIG)
    cfg["future_key"] = True
    try:
        UNet2DConditionCompat.from_config_dict(cfg)
    except ValueError as exc:
        assert "future_key" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown config key")


@requires_torch
def test_unet_skips_fully_consumed() -> None:
    """Down-path states (1 + 2*levels + downsamplers) must match up-path pops."""
    unet = _build_unet()
    sample = torch.randn(1, 8, 32, 32)
    context = torch.randn(1, 2, 384)
    with torch.no_grad():
        # runs the full forward; if skip accounting were wrong the cat would
        # raise a channel-mismatch error.
        out = unet(sample, torch.tensor([0]), encoder_hidden_states=context)
    assert out.sample.shape == (1, 4, 32, 32)


# ------------------------------------------------- package import hygiene


def test_package_import_is_torch_free() -> None:
    """``import liveavatar.musetalk`` must not pull torch/cv2 (PEP 562 lazy
    re-exports).  Guards against a regression that would break test
    collection in CI light environments."""
    import os
    import subprocess
    import sys

    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src_dir)
    code = (
        "import sys\n"
        "for _m in ('torch', 'torchvision', 'cv2'):\n"
        "    sys.modules[_m] = None\n"  # makes `import torch` raise ImportError
        "import liveavatar.musetalk\n"
        "from liveavatar.musetalk.models.mel_frontend import mel_filterbank\n"
        "import numpy as np\n"
        "assert mel_filterbank().shape == (80, 201)\n"
        "assert liveavatar.musetalk.__all__ == [\n"
        "    'Audio2Feature', 'get_image_blending', 'load_all_model']\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")

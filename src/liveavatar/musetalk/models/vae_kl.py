"""Self-written SD-VAE (AutoencoderKL-compatible) loader — A1 of the
self-replace plan (C4).

Replaces the ``diffusers.AutoencoderKL`` dependency used by
``musetalk/models/vae.py`` with a self-contained torch module:

* same architecture as ``stabilityai/sd-vae-ft-mse``
  (ch=128, ch_mult=(1, 2, 4, 4), attention in the mid block only,
  latent_channels=4);
* same call surface the existing ``VAE`` wrapper relies on:
  ``encode(x).latent_dist.sample()`` and ``decode(z).sample``;
* module attribute naming mirrors diffusers state-dict keys
  (``down_blocks.N.resnets.M...``) so the *original* diffusers-format
  weight files load unchanged, with a strict completeness assertion.
"""

from __future__ import annotations

import json
import math
import os
import re
import types
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------- blocks


class _ResnetBlock(nn.Module):
    """Pre-activation resnet (GroupNorm-SiLU-Conv ×2) without timestep.

    Attribute names match diffusers ``ResnetBlock2D`` state-dict keys.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut: nn.Conv2d | None = None
        if in_ch != out_ch:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class _AttentionBlock(nn.Module):
    """Single-head self-attention over spatial positions (diffusers
    ``AttentionBlock`` semantics: GroupNorm → QKV linear → softmax → proj,
    with residual). Attribute names match the legacy diffusers keys;
    modern ``to_q``-style keys are remapped by :func:`_remap_keys`.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, ch, eps=1e-6)
        self.query = nn.Linear(ch, ch)
        self.key = nn.Linear(ch, ch)
        self.value = nn.Linear(ch, ch)
        self.proj_attn = nn.Linear(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.group_norm(x).view(b, c, h * w).transpose(1, 2)  # B, HW, C
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(c), dim=-1)
        out = self.proj_attn(attn @ v)  # B, HW, C
        return residual + out.transpose(1, 2).reshape(b, c, h, w)


class _Downsample2D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample2D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class _DownEncoderBlock(nn.Module):
    """``resnets`` (+ optional ``downsamplers``) — diffusers naming."""

    def __init__(self, in_ch: int, out_ch: int, add_downsample: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([_ResnetBlock(in_ch, out_ch), _ResnetBlock(out_ch, out_ch)])
        self.downsamplers = nn.ModuleList([_Downsample2D(out_ch)]) if add_downsample else nn.ModuleList([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x


class _UpDecoderBlock(nn.Module):
    """``resnets`` (+ optional ``upsamplers``) — diffusers naming."""

    def __init__(self, in_ch: int, out_ch: int, add_upsample: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [_ResnetBlock(in_ch, out_ch), _ResnetBlock(out_ch, out_ch), _ResnetBlock(out_ch, out_ch)]
        )
        self.upsamplers = nn.ModuleList([_Upsample2D(out_ch)]) if add_upsample else nn.ModuleList([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x


class _MidBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([_ResnetBlock(ch, ch), _ResnetBlock(ch, ch)])
        self.attentions = nn.ModuleList([_AttentionBlock(ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


# ------------------------------------------------------------- encoder/decoder


class _Encoder(nn.Module):
    def __init__(self, in_ch: int, ch: int, ch_mult: tuple[int, ...], z_ch: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        curr = ch
        for level, mult in enumerate(ch_mult):
            out = ch * mult
            self.down_blocks.append(
                _DownEncoderBlock(curr, out, add_downsample=level < len(ch_mult) - 1)
            )
            curr = out
        self.mid_block = _MidBlock(curr)
        self.conv_norm_out = nn.GroupNorm(32, curr, eps=1e-6)
        self.conv_out = nn.Conv2d(curr, 2 * z_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class _Decoder(nn.Module):
    def __init__(self, z_ch: int, ch: int, ch_mult: tuple[int, ...], out_ch: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(z_ch, ch * ch_mult[-1], 3, padding=1)
        self.mid_block = _MidBlock(ch * ch_mult[-1])
        # up_blocks are named from the bottleneck outward (up_blocks.0 is the
        # deepest level) — mirrors diffusers state_dict naming.
        self.up_blocks = nn.ModuleList()
        curr = ch * ch_mult[-1]
        for level in range(len(ch_mult) - 1, -1, -1):
            out = ch * ch_mult[level]
            self.up_blocks.append(_UpDecoderBlock(curr, out, add_upsample=level > 0))
            curr = out
        self.conv_norm_out = nn.GroupNorm(32, ch, eps=1e-6)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


# ------------------------------------------------------------------ wrapper


@dataclass(frozen=True)
class _LatentDist:
    mean: torch.Tensor
    logvar: torch.Tensor

    def sample(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        return self.mean + std * torch.randn_like(std)


@dataclass(frozen=True)
class _EncodeResult:
    latent_dist: _LatentDist


@dataclass(frozen=True)
class _DecodeResult:
    sample: torch.Tensor


class AutoencoderKLCompat(nn.Module):
    """Diffusers ``AutoencoderKL`` drop-in for the MuseTalk VAE wrapper.

    Only the exact call surface used by ``musetalk/models/vae.py`` is
    provided: ``encode(x).latent_dist.sample()``, ``decode(z).sample``,
    ``scaling_factor`` and ``dtype``/``device``.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        z_ch: int = 4,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 4, 4),
        scaling_factor: float = 0.18215,
    ) -> None:
        super().__init__()
        self.scaling_factor = scaling_factor
        self.encoder = _Encoder(in_ch, ch, ch_mult, z_ch)
        self.decoder = _Decoder(z_ch, ch, ch_mult, out_ch)
        self.quant_conv = nn.Conv2d(2 * z_ch, 2 * z_ch, 1)
        self.post_quant_conv = nn.Conv2d(z_ch, z_ch, 1)

    # ----------------------------------------------------------- inference

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return next(self.parameters()).device

    @property
    def config(self) -> types.SimpleNamespace:
        """diffusers-style ``config`` namespace (scaling_factor read by VAE)."""
        return types.SimpleNamespace(scaling_factor=self.scaling_factor)

    def encode(self, x: torch.Tensor) -> _EncodeResult:
        h = self.quant_conv(self.encoder(x))
        mean, logvar = h.chunk(2, dim=1)
        return _EncodeResult(_LatentDist(mean, logvar))

    def decode(self, z: torch.Tensor) -> _DecodeResult:
        return _DecodeResult(self.decoder(self.post_quant_conv(z)))

    # -------------------------------------------------------------- loader

    @classmethod
    def from_pretrained_kl(cls, model_path: str) -> "AutoencoderKLCompat":
        """Load diffusers-format weights from a local directory.

        Accepts both legacy (``query/key/value/proj_attn``) and modern
        (``to_q/to_k/to_v/to_out.0``) attention key naming. Asserts the
        mapping is complete — missing/unexpected keys are a hard error so
        a silently half-loaded model can never reach inference.
        """
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found under {model_path}")
        with open(config_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        scaling = float(cfg.get("scaling_factor", 0.18215))
        z_ch = int(cfg.get("latent_channels", 4))
        # diffusers ``block_out_channels`` holds the absolute channel count of
        # each level (e.g. [128, 256, 512, 512]); derive ch / ch_mult from it.
        channels = [int(c) for c in cfg.get("block_out_channels", (128, 256, 512, 512))]
        ch = channels[0]
        if ch <= 0 or any(c % ch for c in channels):
            raise ValueError(f"invalid block_out_channels: {channels}")
        ch_mult = tuple(c // ch for c in channels)

        model = cls(z_ch=z_ch, ch=ch, ch_mult=ch_mult, scaling_factor=scaling)

        state = _load_state_dict(model_path)
        state = _remap_keys(state)

        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "incomplete AutoencoderKL weight mapping: "
                f"missing={result.missing_keys[:8]}... "
                f"unexpected={result.unexpected_keys[:8]}..."
            )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model


def _load_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load ``diffusion_pytorch_model.safetensors`` (preferred) or ``.bin``."""
    safe = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
    if os.path.exists(safe):
        return _load_safetensors(safe)
    bin_path = os.path.join(model_path, "diffusion_pytorch_model.bin")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"no diffusion_pytorch_model.safetensors/.bin under {model_path}"
    )


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Minimal safetensors reader (header + raw buffer), no external deps."""
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len).decode("utf-8"))
        buf = fh.read()
    dtypes = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F64": torch.float64,
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        tensor = torch.frombuffer(bytearray(buf[start:end]), dtype=dtypes[meta["dtype"]])
        tensors[name] = tensor.reshape(meta["shape"])
    return tensors


# ------------------------------------------------------------------ key map

_RE_ATT_QUERY = re.compile(r"\.to_q\.")
_RE_ATT_KEY = re.compile(r"\.to_k\.")
_RE_ATT_VALUE = re.compile(r"\.to_v\.")
_RE_ATT_PROJ = re.compile(r"\.to_out\.0\.")


def _remap_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map modern diffusers attention keys onto legacy attribute names."""
    renamed: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = _RE_ATT_PROJ.sub(".proj_attn.", _RE_ATT_VALUE.sub(".value.", _RE_ATT_KEY.sub(".key.", _RE_ATT_QUERY.sub(".query.", key))))
        renamed[new_key] = value
    return renamed

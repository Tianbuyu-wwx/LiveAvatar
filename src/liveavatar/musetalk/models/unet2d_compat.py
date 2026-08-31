"""Self-written UNet2DConditionModel-compatible module — A1 (C4).

Replaces the ``diffusers.UNet2DConditionModel`` dependency used by
``musetalk/models/unet.py`` with a self-contained torch module for the
exact configuration MuseTalk 1.5 uses (SD-1.5 shaped UNet with
cross_attention_dim=384 and in_channels=8):

* module attribute naming mirrors diffusers state-dict keys, so the
  original ``unet.pth`` (diffusers key layout) loads unchanged;
* strict completeness assertion on load — a half-mapped model can
  never reach inference;
* call surface used by ``musetalk_worker.py``:
  ``model(sample, t, encoder_hidden_states=...).sample`` plus
  ``.dtype`` / ``.parameters()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class _UNetOutput:
    sample: torch.Tensor


# ----------------------------------------------------------------- attention


class _Attention(nn.Module):
    """Multi-head attention with diffusers ``Attention`` key layout
    (``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``)."""

    def __init__(self, query_dim: int, cross_dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(cross_dim, inner, bias=False)
        self.to_v = nn.Linear(cross_dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, query_dim)])

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = x if context is None else context
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(context).view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(context).view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out[0](out)


class _GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return F.gelu(a) * b


class _BasicTransformerBlock(nn.Module):
    """diffusers ``BasicTransformerBlock`` key layout."""

    def __init__(self, dim: int, heads: int, dim_head: int, cross_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn1 = _Attention(dim, dim, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.attn2 = _Attention(dim, cross_dim, heads, dim_head)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5)
        self.ff = nn.Module()
        self.ff.net = nn.Module()
        self.ff.net._modules["0"] = _GEGLU(dim, dim * 4)
        self.ff.net._modules["2"] = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        net0 = self.ff.net._modules["0"]
        net2 = self.ff.net._modules["2"]
        h = net0(self.norm3(x))
        return x + net2(h)


class _Transformer2DModel(nn.Module):
    """GroupNorm → proj_in(1×1 conv) → transformer block → proj_out,
    with outer residual — diffusers ``Transformer2DModel`` (UNet usage,
    non-linear projection) key layout."""

    def __init__(self, in_ch: int, heads: int, dim_head: int, cross_dim: int, num_groups: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_ch, eps=eps)
        self.proj_in = nn.Conv2d(in_ch, in_ch, 1)
        self.transformer_blocks = nn.ModuleList(
            [_BasicTransformerBlock(in_ch, heads, dim_head, cross_dim)]
        )
        self.proj_out = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.view(b, c, h * w).transpose(1, 2)
        for block in self.transformer_blocks:
            x = block(x, context)
        x = x.transpose(1, 2).view(b, c, h, w)
        return self.proj_out(x) + residual


# -------------------------------------------------------------------- blocks


class _ResnetBlock2D(nn.Module):
    """diffusers ``ResnetBlock2D`` (with ``time_emb_proj``) key layout."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, num_groups: int, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch, eps=eps)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch, eps=eps)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut: nn.Conv2d | None = None
        if in_ch != out_ch:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class _Downsample2D(nn.Module):
    def __init__(self, ch: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample2D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class _CrossAttnDownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        layers: int,
        heads: int,
        dim_head: int,
        cross_dim: int,
        temb_dim: int,
        num_groups: int,
        eps: float,
        padding: int,
        add_downsample: bool,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        curr = in_ch
        for _ in range(layers):
            self.resnets.append(_ResnetBlock2D(curr, out_ch, temb_dim, num_groups, eps))
            self.attentions.append(
                _Transformer2DModel(out_ch, heads, dim_head, cross_dim, num_groups, eps)
            )
            curr = out_ch
        self.downsamplers = nn.ModuleList([_Downsample2D(out_ch, padding)]) if add_downsample else None

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        states: list[torch.Tensor] = []
        for resnet, attn in zip(self.resnets, self.attentions, strict=True):
            x = resnet(x, temb)
            x = attn(x, context)
            states.append(x)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
            states.append(x)
        return x, states


class _DownBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, layers: int, temb_dim: int, num_groups: int, eps: float
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        curr = in_ch
        for _ in range(layers):
            self.resnets.append(_ResnetBlock2D(curr, out_ch, temb_dim, num_groups, eps))
            curr = out_ch
        self.downsamplers = None

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        states: list[torch.Tensor] = []
        for resnet in self.resnets:
            x = resnet(x, temb)
            states.append(x)
        return x, states


class _MidBlock(nn.Module):
    def __init__(
        self, ch: int, heads: int, dim_head: int, cross_dim: int, temb_dim: int, num_groups: int, eps: float
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [_ResnetBlock2D(ch, ch, temb_dim, num_groups, eps), _ResnetBlock2D(ch, ch, temb_dim, num_groups, eps)]
        )
        self.attentions = nn.ModuleList(
            [_Transformer2DModel(ch, heads, dim_head, cross_dim, num_groups, eps)]
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)


class _CrossAttnUpBlock(nn.Module):
    def __init__(
        self,
        in_chs: tuple[int, ...],  # per-resnet input channels (after skip concat)
        out_ch: int,
        heads: int,
        dim_head: int,
        cross_dim: int,
        temb_dim: int,
        num_groups: int,
        eps: float,
        add_upsample: bool,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [_ResnetBlock2D(c, out_ch, temb_dim, num_groups, eps) for c in in_chs]
        )
        self.attentions = nn.ModuleList(
            [_Transformer2DModel(out_ch, heads, dim_head, cross_dim, num_groups, eps) for _ in in_chs]
        )
        self.upsamplers = nn.ModuleList([_Upsample2D(out_ch)]) if add_upsample else None

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor], temb: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        popped: list[torch.Tensor] = []
        for resnet, attn in zip(self.resnets, self.attentions, strict=True):
            x = torch.cat([x, skips.pop()], dim=1)
            x = resnet(x, temb)
            x = attn(x, context)
            popped.append(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x, popped


class _UpBlock(nn.Module):
    def __init__(
        self,
        in_chs: tuple[int, ...],
        out_ch: int,
        temb_dim: int,
        num_groups: int,
        eps: float,
        add_upsample: bool,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [_ResnetBlock2D(c, out_ch, temb_dim, num_groups, eps) for c in in_chs]
        )
        self.upsamplers = nn.ModuleList([_Upsample2D(out_ch)]) if add_upsample else None

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor], temb: torch.Tensor
    ) -> torch.Tensor:
        for resnet in self.resnets:
            x = torch.cat([x, skips.pop()], dim=1)
            x = resnet(x, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


# ---------------------------------------------------------------------- UNet


def _timestep_embedding(timesteps: torch.Tensor, dim: int, flip_sin_to_cos: bool, downscale_freq_shift: float) -> torch.Tensor:
    """diffusers ``get_timestep_embedding`` semantics."""
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(start=0, end=half, dtype=torch.float32) / (
        half - downscale_freq_shift
    )
    emb = torch.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=1)
    return emb.to(dtype=timesteps.dtype if timesteps.is_floating_point() else torch.float32)


class UNet2DConditionCompat(nn.Module):
    """Diffusers ``UNet2DConditionModel`` drop-in for the MuseTalk UNet."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        down_block_types: tuple[str, ...],
        up_block_types: tuple[str, ...],
        layers_per_block: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        if len(block_out_channels) != len(down_block_types) or len(block_out_channels) != len(up_block_types):
            raise ValueError("block lists must have equal length")
        if down_block_types != ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"):
            raise ValueError(f"unsupported down_block_types: {down_block_types}")
        if up_block_types != ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"):
            raise ValueError(f"unsupported up_block_types: {up_block_types}")

        temb_dim = 4 * block_out_channels[0]
        ch0 = block_out_channels[0]
        self.conv_in = nn.Conv2d(in_channels, ch0, 3, padding=1)
        self.time_embedding = nn.Module()
        self.time_embedding.linear_1 = nn.Linear(ch0, temb_dim)
        self.time_embedding.linear_2 = nn.Linear(temb_dim, temb_dim)
        self._flip_sin_to_cos = flip_sin_to_cos
        self._downscale_freq_shift = downscale_freq_shift
        self._temb_in_dim = ch0  # sinusoidal width (diffusers Timesteps(ch0))
        self._temb_dim = temb_dim

        common = dict(temb_dim=temb_dim, num_groups=norm_num_groups, eps=norm_eps)

        # down path: channels 320 → 640 → 1280 → 1280, attention on 0-2
        self.down_blocks = nn.ModuleList()
        curr = ch0
        for level, ch_out in enumerate(block_out_channels):
            if down_block_types[level] == "DownBlock2D":
                self.down_blocks.append(
                    _DownBlock(curr, ch_out, layers_per_block, **common)
                )
            else:
                self.down_blocks.append(
                    _CrossAttnDownBlock(
                        curr,
                        ch_out,
                        layers_per_block,
                        heads=ch_out // attention_head_dim,
                        dim_head=attention_head_dim,
                        cross_dim=cross_attention_dim,
                        padding=downsample_padding,
                        add_downsample=level < len(block_out_channels) - 1,
                        **common,
                    )
                )
            curr = ch_out
        ch_mid = block_out_channels[-1]

        # mid block (UNetMidBlock2DCrossAttn)
        self.mid_block = _MidBlock(
            ch_mid,
            heads=ch_mid // attention_head_dim,
            dim_head=attention_head_dim,
            cross_dim=cross_attention_dim,
            **common,
        )

        # up path (bottom-up naming); per-resnet inputs include skip concat
        chs = list(block_out_channels)
        # skip channels bottom-up: d3r1, d3r0, d2ds, d2r1, d2r0, d1ds, d1r1, d1r0, d0ds, d0r1, d0r0, conv_in
        skips_bottom_up = [
            chs[3],
            chs[3],
            chs[2],
            chs[2],
            chs[2],
            chs[1],
            chs[1],
            chs[1],
            chs[0],
            chs[0],
            chs[0],
            ch0,
        ]
        self.up_blocks = nn.ModuleList()
        prev = ch_mid
        idx = 0
        for level in range(len(up_block_types)):
            # up_blocks are named bottom-up: up_blocks.0 sits at the
            # bottleneck, so output channels run over reversed(block_out_channels).
            ch_out = block_out_channels[len(block_out_channels) - 1 - level]
            in_chs: list[int] = []
            for _ in range(layers_per_block + 1):
                in_chs.append(prev + skips_bottom_up[idx])
                prev = ch_out
                idx += 1
            if up_block_types[level] == "UpBlock2D":
                self.up_blocks.append(_UpBlock(tuple(in_chs), ch_out, add_upsample=level < len(up_block_types) - 1, **common))
            else:
                self.up_blocks.append(
                    _CrossAttnUpBlock(
                        tuple(in_chs),
                        ch_out,
                        heads=ch_out // attention_head_dim,
                        dim_head=attention_head_dim,
                        cross_dim=cross_attention_dim,
                        add_upsample=level < len(up_block_types) - 1,
                        **common,
                    )
                )

        self.conv_norm_out = nn.GroupNorm(norm_num_groups, ch0, eps=norm_eps)
        self.conv_out = nn.Conv2d(ch0, out_channels, 3, padding=1)

    # ----------------------------------------------------------- inference

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        return next(self.parameters()).dtype

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        encoder_hidden_states: torch.Tensor,
    ) -> _UNetOutput:
        temb = _timestep_embedding(
            torch.as_tensor([timestep]) if not torch.is_tensor(timestep) else timestep.reshape(-1),
            self._temb_in_dim,
            self._flip_sin_to_cos,
            self._downscale_freq_shift,
        )
        temb = temb.to(device=sample.device, dtype=sample.dtype)
        temb = self.time_embedding.linear_2(F.silu(self.time_embedding.linear_1(temb)))

        states: list[torch.Tensor] = [sample]
        x = self.conv_in(sample)
        states[-1] = x

        for block in self.down_blocks:
            if isinstance(block, _CrossAttnDownBlock):
                x, new_states = block(x, temb, encoder_hidden_states)
            else:
                x, new_states = block(x, temb)
            states.extend(new_states)

        x = self.mid_block(x, temb, encoder_hidden_states)

        for block in self.up_blocks:
            if isinstance(block, _CrossAttnUpBlock):
                x, _ = block(x, states, temb, encoder_hidden_states)
            else:
                x = block(x, states, temb)

        x = self.conv_out(F.silu(self.conv_norm_out(x)))
        return _UNetOutput(sample=x)

    # -------------------------------------------------------------- loader

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "UNet2DConditionCompat":
        """Build from a diffusers ``unet_config`` JSON dict."""
        supported = {
            "_class_name",
            "in_channels",
            "out_channels",
            "block_out_channels",
            "down_block_types",
            "up_block_types",
            "layers_per_block",
            "attention_head_dim",
            "cross_attention_dim",
            "norm_num_groups",
            "norm_eps",
            "flip_sin_to_cos",
            "downscale_freq_shift",
            "downsample_padding",
            "sample_size",
            "mid_block_type",
            "only_cross_attention",
            "upcast_attention",
            "act_fn",
            "center_input_sample",
            "freq_shift",
            "transformer_layers_per_block",
        }
        unknown = set(cfg) - supported
        if unknown:
            raise ValueError(f"unsupported unet config keys: {sorted(unknown)}")
        if cfg.get("transformer_layers_per_block", 1) != 1:
            raise ValueError("transformer_layers_per_block > 1 unsupported")
        if cfg.get("mid_block_type", "UNetMidBlock2DCrossAttn") != "UNetMidBlock2DCrossAttn":
            raise ValueError(f"unsupported mid_block_type: {cfg.get('mid_block_type')}")
        if cfg.get("only_cross_attention", False):
            raise ValueError("only_cross_attention unsupported")
        return cls(
            in_channels=int(cfg["in_channels"]),
            out_channels=int(cfg["out_channels"]),
            block_out_channels=tuple(int(c) for c in cfg["block_out_channels"]),
            down_block_types=tuple(cfg["down_block_types"]),
            up_block_types=tuple(cfg["up_block_types"]),
            layers_per_block=int(cfg["layers_per_block"]),
            attention_head_dim=int(cfg["attention_head_dim"]),
            cross_attention_dim=int(cfg["cross_attention_dim"]),
            norm_num_groups=int(cfg.get("norm_num_groups", 32)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            flip_sin_to_cos=bool(cfg.get("flip_sin_to_cos", True)),
            downscale_freq_shift=float(cfg.get("downscale_freq_shift", cfg.get("freq_shift", 0))),
            downsample_padding=int(cfg.get("downsample_padding", 1)),
        )

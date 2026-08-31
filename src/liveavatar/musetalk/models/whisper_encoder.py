"""Self-written Whisper encoder (tiny) + log-mel feature extractor — A2 (C5).

Replaces the ``transformers.WhisperModel`` / ``AutoFeatureExtractor``
dependency used by ``audio2feature.py`` with self-contained torch/numpy
code for the exact configuration MuseTalk uses (openai/whisper-tiny
encoder, 80 mel bins, 4 layers, d_model=384):

* the log-mel frontend reproduces the transformers WhisperFeatureExtractor
  pipeline (Slaney mel filterbank, Hann STFT, log10 clamp + normalise);
* module attribute naming mirrors HF state-dict keys, so the *original*
  ``openai/whisper-tiny`` weight files load unchanged with a strict
  completeness assertion;
* call surface used by ``audio2feature.py``: ``extractor(wav).input_features``
  and ``encoder(x).hidden_states`` (5 tensors: conv+pos embedding and the
  4 raw layer outputs, PRE final layer norm — transformers 4.x semantics,
  which is what MuseTalk was built against; transformers 5.x quietly changed
  the last hidden_states entry to the post-LN state, so the reference must
  be pinned here).
"""

from __future__ import annotations

import json
import math
import os
import types
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# -------------------------------------------------------- mel filterbank
# (numpy-only implementation lives in mel_frontend; re-exported here to
# keep the historical ``whisper_encoder.mel_filterbank`` import surface)

from .mel_frontend import _hz_to_mel, _mel_to_hz, mel_filterbank  # noqa: F401


# --------------------------------------------------- log-mel feature extractor

CHUNK_S = 30
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80


class WhisperFeatureExtractorCompat:
    """transformers ``WhisperFeatureExtractor`` drop-in (log-mel, 30 s)."""

    def __init__(self, sr: int = SAMPLE_RATE) -> None:
        self.sr = sr
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS
        self.chunk_len = CHUNK_S * sr
        self.mel_filters = mel_filterbank(sr, N_FFT, N_MELS)

    def __call__(self, wav, return_tensors: str = "pt", sampling_rate: int = SAMPLE_RATE) -> types.SimpleNamespace:
        if sampling_rate != self.sr:
            raise ValueError(f"expected {self.sr} Hz input, got {sampling_rate}")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError("expected a mono waveform")
        if wav.size < self.chunk_len:
            wav = np.pad(wav, (0, self.chunk_len - wav.size))
        else:
            wav = wav[: self.chunk_len]

        window = torch.hann_window(self.n_fft)
        frames = torch.stft(
            torch.from_numpy(wav),
            self.n_fft,
            self.hop_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        magnitudes = frames.abs() ** 2
        mel = torch.from_numpy(self.mel_filters) @ magnitudes
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        features = log_spec[:, : self.chunk_len // self.hop_length]  # (80, 3000)
        return types.SimpleNamespace(input_features=features.unsqueeze(0))


# ----------------------------------------------------------- encoder blocks


class WhisperEncoderLayer(nn.Module):
    """HF ``WhisperEncoderLayer`` key layout (attention biases included)."""

    def __init__(self, d_model: int, heads: int, ffn_dim: int, eps: float) -> None:
        super().__init__()
        self.embed_dim = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.self_attn = nn.Module()
        # matches openai/whisper: key projection has no bias, q/v/out do
        self.self_attn.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.self_attn.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.self_attn.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.self_attn.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        residual = x
        x = self.self_attn_layer_norm(x)
        q = self.self_attn.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.self_attn.k_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.self_attn.v_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, self.embed_dim)
        x = residual + self.self_attn.out_proj(out)
        x = x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        return x


@dataclass(frozen=True)
class _EncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


class WhisperEncoderCompat(nn.Module):
    """HF ``WhisperEncoder`` drop-in for the whisper-tiny configuration."""

    def __init__(
        self,
        *,
        d_model: int = 384,
        encoder_layers: int = 4,
        encoder_attention_heads: int = 6,
        encoder_ffn_dim: int = 1536,
        num_mel_bins: int = 80,
        max_source_positions: int = 1500,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model % encoder_attention_heads:
            raise ValueError("d_model must be divisible by encoder_attention_heads")
        self.conv1 = nn.Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(max_source_positions, d_model)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(d_model, encoder_attention_heads, encoder_ffn_dim, layer_norm_eps) for _ in range(encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, input_features: torch.Tensor) -> _EncoderOutput:
        x = F.gelu(self.conv1(input_features))
        x = F.gelu(self.conv2(x)).transpose(1, 2)  # B, 1500, d_model
        x = x + self.embed_positions.weight

        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        return _EncoderOutput(last_hidden_state=self.layer_norm(x), hidden_states=tuple(hidden_states))

    # -------------------------------------------------------------- loader

    @classmethod
    def from_pretrained_whisper(cls, model_path: str) -> "WhisperEncoderCompat":
        """Load HF ``openai/whisper-tiny`` weights from a local directory.

        Accepts ``pytorch_model.bin`` or ``model.safetensors``; full-model
        checkpoints (with ``encoder.`` prefixed keys and decoder weights)
        are filtered automatically. Missing/unexpected encoder keys are a
        hard error so a half-loaded model can never reach inference.
        """
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found under {model_path}")
        with open(config_path, encoding="utf-8") as fh:
            cfg = json.load(fh)

        model = cls(
            d_model=int(cfg["d_model"]),
            encoder_layers=int(cfg["encoder_layers"]),
            encoder_attention_heads=int(cfg["encoder_attention_heads"]),
            encoder_ffn_dim=int(cfg["encoder_ffn_dim"]),
            num_mel_bins=int(cfg["num_mel_bins"]),
            max_source_positions=int(cfg.get("max_source_positions", 1500)),
            layer_norm_eps=float(cfg.get("layer_norm_epsilon", 1e-5)),
        )

        state = _load_state_dict(model_path)
        prefix = "encoder."
        # both the legacy ("encoder.*") and modern ("model.encoder.*") HF
        # checkpoint layouts must load unchanged
        enc_state = {
            k[k.rindex(prefix) + len(prefix) :]: v
            for k, v in state.items()
            if prefix in k
        }
        result = model.load_state_dict(enc_state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "incomplete Whisper encoder weight mapping: "
                f"missing={result.missing_keys[:8]}... "
                f"unexpected={result.unexpected_keys[:8]}..."
            )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model


def _load_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load ``model.safetensors`` (preferred) or ``pytorch_model.bin``."""
    from .vae_kl import _load_safetensors

    safe = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safe):
        return _load_safetensors(safe)
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no model.safetensors/pytorch_model.bin under {model_path}")

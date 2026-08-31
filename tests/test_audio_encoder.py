"""A2 (C5): self-written Whisper encoder + log-mel extractor tests.

The real openai/whisper-tiny weights are NOT available in CI, so structural
tests use synthetic weights. When ``transformers`` is installed the log-mel
frontend is compared against the reference implementation; when the real
``models/whisper`` directory exists the encoder outputs are compared against
``transformers.WhisperModel`` (network-free, local weights only).
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover — CI light env
    torch = None  # type: ignore[assignment]

try:
    from transformers.audio_utils import mel_filter_bank as reference_mel_filter_bank

    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover — CI light env
    reference_mel_filter_bank = None  # type: ignore[assignment]
    HAS_TRANSFORMERS = False

from liveavatar.musetalk.models.whisper_encoder import (
    WhisperEncoderCompat,
    WhisperFeatureExtractorCompat,
    mel_filterbank,
)

requires_torch = unittest.skipUnless(torch is not None, "torch not installed")
requires_transformers = unittest.skipUnless(
    torch is not None and HAS_TRANSFORMERS, "torch/transformers not installed"
)
requires_real_weights = unittest.skipUnless(
    torch is not None
    and HAS_TRANSFORMERS
    and os.path.isdir(os.path.join("models", "whisper")),
    "real whisper weights / transformers not available",
)

TINY_CONFIG = {
    "d_model": 384,
    "encoder_layers": 4,
    "encoder_attention_heads": 6,
    "encoder_ffn_dim": 1536,
    "num_mel_bins": 80,
    "max_source_positions": 1500,
    "layer_norm_epsilon": 1e-5,
}


# --------------------------------------------------------- mel filterbank


def test_mel_filterbank_sanity() -> None:
    fb = mel_filterbank()
    assert fb.shape == (80, 201)
    assert np.isfinite(fb).all()
    assert (fb >= 0).all()
    # every filter peaks at 1 (before norm the peak is 1; slaney norm scales
    # by 2/(upper-lower) which keeps the peak at enorm[i])
    assert (fb.max(axis=1) > 0).all()


@requires_transformers
def test_mel_filterbank_matches_reference() -> None:
    ref = reference_mel_filter_bank(
        num_frequency_bins=201,
        num_mel_filters=80,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    np.testing.assert_allclose(mel_filterbank(), ref.T, atol=1e-6)


def test_feature_extractor_shapes_and_determinism() -> None:
    extractor = WhisperFeatureExtractorCompat()
    rng = np.random.default_rng(7)
    wav = rng.standard_normal(16000).astype(np.float32) * 0.1

    out = extractor(wav, sampling_rate=16000)
    assert out.input_features.shape == (1, 80, 3000)
    assert torch.isfinite(out.input_features).all()

    again = extractor(wav, sampling_rate=16000).input_features
    assert torch.equal(out.input_features, again)

    # silence must differ from noise (not clamped to a constant spectrogram)
    silent = extractor(np.zeros(16000, np.float32), sampling_rate=16000).input_features
    assert not torch.allclose(out.input_features, silent, atol=1e-3)


def test_feature_extractor_rejects_bad_rate() -> None:
    extractor = WhisperFeatureExtractorCompat()
    try:
        extractor(np.zeros(100, np.float32), sampling_rate=8000)
    except ValueError as exc:
        assert "16000" in str(exc)
    else:
        raise AssertionError("expected ValueError for wrong sampling rate")


@requires_transformers
def test_feature_extractor_matches_reference() -> None:
    """Full log-mel pipeline must match transformers (no network needed)."""
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

    ref = WhisperFeatureExtractor(
        chunk_length=30,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=False,
    )
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(32000).astype(np.float32) * 0.1

    ref_feat = ref(wav, return_tensors="pt", sampling_rate=16000).input_features
    mine_feat = WhisperFeatureExtractorCompat()(wav, sampling_rate=16000).input_features
    assert ref_feat.shape == mine_feat.shape == (1, 80, 3000)
    torch.testing.assert_close(mine_feat, ref_feat, atol=1e-5, rtol=0)


# ---------------------------------------------------------------- encoder


def _build_encoder() -> WhisperEncoderCompat:
    torch.manual_seed(0)
    kwargs = dict(TINY_CONFIG)
    kwargs["layer_norm_eps"] = kwargs.pop("layer_norm_epsilon")  # HF config name
    return WhisperEncoderCompat(**kwargs)


@requires_torch
def test_encoder_forward_shapes() -> None:
    encoder = _build_encoder()
    encoder.eval()
    x = torch.randn(1, 80, 3000)
    with torch.no_grad():
        out = encoder(x)
    assert len(out.hidden_states) == TINY_CONFIG["encoder_layers"] + 1
    for h in out.hidden_states:
        assert h.shape == (1, 1500, 384)
    assert out.last_hidden_state.shape == (1, 1500, 384)
    # last_hidden_state passes through the final layer norm (unit RMS-ish)
    assert out.last_hidden_state.std() > 0


@requires_torch
def test_encoder_deterministic() -> None:
    encoder = _build_encoder()
    encoder.eval()
    x = torch.randn(1, 80, 3000)
    with torch.no_grad():
        a = encoder(x).last_hidden_state
        b = encoder(x).last_hidden_state
    assert torch.equal(a, b)


@requires_torch
def test_encoder_state_dict_roundtrip(tmp_path: Path) -> None:
    """HF full-model layout (encoder.* prefix + decoder keys) loads lossless."""
    encoder = _build_encoder()
    encoder.eval()
    full_state = {f"encoder.{k}": v.clone() for k, v in encoder.state_dict().items()}
    # decoder weights must be ignored by the loader
    full_state["decoder.embed_tokens.weight"] = torch.randn(51864, 384)

    path = tmp_path / "whisper"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(TINY_CONFIG), encoding="utf-8")
    torch.save(full_state, path / "pytorch_model.bin")

    loaded = WhisperEncoderCompat.from_pretrained_whisper(str(path))
    x = torch.randn(1, 80, 3000)
    with torch.no_grad():
        assert torch.equal(encoder(x).last_hidden_state, loaded(x).last_hidden_state)


@requires_torch
def test_encoder_incomplete_mapping_raises(tmp_path: Path) -> None:
    encoder = _build_encoder()
    state = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
    del state["encoder.conv1.weight"]
    path = tmp_path / "whisper-truncated"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(TINY_CONFIG), encoding="utf-8")
    torch.save(state, path / "pytorch_model.bin")
    try:
        WhisperEncoderCompat.from_pretrained_whisper(str(path))
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for incomplete mapping")


@requires_torch
def test_encoder_rejects_bad_config() -> None:
    kwargs = dict(TINY_CONFIG)
    kwargs["layer_norm_eps"] = kwargs.pop("layer_norm_epsilon")
    kwargs["d_model"] = 385  # not divisible by 6 heads
    try:
        WhisperEncoderCompat(**kwargs)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid head split")


# ------------------------------------------------- real weights (optional)


@requires_real_weights
def test_real_weights_match_transformers() -> None:
    """With real openai/whisper-tiny weights: outputs match HF WhisperModel.

    Semantics note: MuseTalk was built against transformers 4.x, where
    ``hidden_states`` are the raw layer outputs (pre final layer norm).
    transformers 5.x captures the post-layer-norm state as the last entry,
    so the last channel is compared against the raw layer-3 output instead.
    """
    from transformers import WhisperModel

    hf = WhisperModel.from_pretrained("models/whisper")
    hf.eval()
    mine = WhisperEncoderCompat.from_pretrained_whisper("models/whisper")
    mine.eval()

    rng = np.random.default_rng(3)
    wav = rng.standard_normal(16000 * 4).astype(np.float32) * 0.1
    features = WhisperFeatureExtractorCompat()(wav, sampling_rate=16000).input_features

    with torch.no_grad():
        ref = hf.encoder(features, output_hidden_states=True).hidden_states
        out = mine(features).hidden_states
    assert len(ref) == len(out) == TINY_CONFIG["encoder_layers"] + 1
    for i, (r, o) in enumerate(zip(ref, out, strict=True)):
        if i == len(ref) - 1:
            r = hf.encoder.layers[-1](ref[-2], None)  # raw (pre-LN) output
        # tiny deviations (≤ ~1e-3) come from matmul accumulation order
        # (SDPA vs manual softmax attention) — well below model noise
        torch.testing.assert_close(o, r, atol=2e-3, rtol=1e-3)

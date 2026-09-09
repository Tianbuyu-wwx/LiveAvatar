# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for MuseTalkAvatarWorker with fake shared models (torch-gated).

Runs where torch+cv2 exist; CI (no torch) skips the whole module.
Covers: silence path, speaking path, offline long-audio path, unload,
mirror_index — all without real model weights.
"""

from __future__ import annotations

import pickle
import tempfile
import types
import unittest
from pathlib import Path

from liveavatar.worker import AvatarAssets

torch = None
try:
    import torch as _torch

    torch = _torch
    import cv2  # noqa: F401
except ImportError:
    torch = None

if torch is not None:
    from liveavatar.musetalk_worker import (
        MuseTalkAvatarWorker,
        _mirror_index,
        _pcm_s16le_to_float32,
    )

    def _loud_pcm(samples: int = 640) -> bytes:
        """Loud square-ish PCM (RMS well above silence threshold)."""
        return b"\x00\x40" * samples

    def _silent_pcm(samples: int = 640) -> bytes:
        return b"\x00\x00" * samples

    class _FakeUNetModel:
        dtype = torch.float32

        def __call__(self, latent, timesteps, encoder_hidden_states=None):
            return types.SimpleNamespace(sample=latent)

    class _FakeUnet:
        def __init__(self) -> None:
            self.model = _FakeUNetModel()

    class _FakePe:
        def __call__(self, x):
            return x

    class _FakeVae:
        def decode_latents(self, latents):
            import numpy as np

            batch = latents.shape[0]
            return np.zeros((batch, 64, 64, 3), dtype=np.uint8)

    class _RecordingVae:
        """_FakeVae that also captures the decoded latent batches (P-D)."""

        def __init__(self) -> None:
            self.captured: list = []

        def decode_latents(self, latents):
            import numpy as np

            self.captured.append(latents.detach().clone())
            return np.zeros((latents.shape[0], 64, 64, 3), dtype=np.uint8)

    class _FakeAudioProcessor:
        def audio2feat(self, audio_np):
            import numpy as np

            return np.zeros((max(1, audio_np.size // 320), 32), dtype=np.float32)

        def feature2chunks(self, feature, fps, batch_size, audio_feat_length, start):
            import numpy as np

            return [np.zeros((8, 32), dtype=np.float32) for _ in range(batch_size)]

    def _fake_shared_models() -> dict:
        return {
            "vae": _FakeVae(),
            "unet": _FakeUnet(),
            "pe": _FakePe(),
            "timesteps": torch.tensor([0]),
            "audio_processor": _FakeAudioProcessor(),
        }

    def _make_avatar(tmp: str) -> AvatarAssets:
        import numpy as np

        root = Path(tmp)
        (root / "full_imgs").mkdir()
        (root / "mask").mkdir()
        with open(root / "coords.pkl", "wb") as f:
            pickle.dump([(0, 0, 64, 64)], f)
        with open(root / "mask_coords.pkl", "wb") as f:
            pickle.dump([(0, 0, 64, 64)], f)
        torch.save([torch.zeros(1, 4, 8, 8)], root / "latents.pt")
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :] = (9, 9, 9)
        import cv2

        cv2.imwrite(str(root / "full_imgs" / "0.jpg"), img)
        cv2.imwrite(str(root / "mask" / "0.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
        return AvatarAssets(
            avatar_id="fake",
            data_dir=str(root),
            full_imgs_dir=str(root / "full_imgs"),
            coords_path=str(root / "coords.pkl"),
            latents_path=str(root / "latents.pt"),
            mask_dir=str(root / "mask"),
            mask_coords_path=str(root / "mask_coords.pkl"),
        )


@unittest.skipIf(torch is None, "torch not installed")
class TestMirrorIndex(unittest.TestCase):
    def test_forward_and_backward(self):
        self.assertEqual(_mirror_index(5, 0), 0)
        self.assertEqual(_mirror_index(5, 4), 4)
        self.assertEqual(_mirror_index(5, 5), 4)  # turn 1 → mirrored
        self.assertEqual(_mirror_index(5, 9), 0)

    def test_always_in_range(self):
        for i in range(50):
            self.assertTrue(0 <= _mirror_index(7, i) < 7)


@unittest.skipIf(torch is None, "torch not installed")
class TestPcmToFloat(unittest.TestCase):
    def test_conversion(self):

        out = _pcm_s16le_to_float32(b"\x00\x80")  # -32768
        self.assertAlmostEqual(float(out[0]), -1.0, places=3)

    def test_short_input(self):
        out = _pcm_s16le_to_float32(b"\x00")
        self.assertEqual(out.size, 0)


@unittest.skipIf(torch is None, "torch not installed")
class TestMuseTalkWorkerPaths(unittest.TestCase):
    def _make_worker(self, batch_size: int = 4):
        tmp_obj = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_obj.cleanup)
        worker = MuseTalkAvatarWorker(
            _make_avatar(tmp_obj.name),
            target_fps=25,
            width=64,
            height=64,
            batch_size=batch_size,
            device="cpu",
            is_half=False,
            shared_models=_fake_shared_models(),
        )
        # Bypass cv2 blending: predicted crops are already full-frame.
        worker._paste_back = lambda res_frame, idx: res_frame
        return worker

    def test_silence_path_returns_default_frames(self):
        w = self._make_worker()
        frames = w._infer_batch(_silent_pcm(640))
        self.assertEqual(len(frames), w.batch_size)
        for data, speaking in frames:
            self.assertFalse(speaking)
            self.assertEqual(len(data), 64 * 64 * 3)
            self.assertEqual(data[0:3], b"\x09\x09\x09")  # original image

    def test_speaking_path_produces_frames(self):
        w = self._make_worker()
        frames = w._infer_batch(_loud_pcm(640))
        self.assertEqual(len(frames), w.batch_size)
        for data, speaking in frames:
            self.assertTrue(speaking)
            self.assertEqual(len(data), 64 * 64 * 3)
        self.assertEqual(w._frame_index, w.batch_size)

    def test_offline_long_audio_frame_count(self):
        w = self._make_worker(batch_size=4)
        # 2 seconds @16kHz → 50 frames @25fps.
        frames = w._infer_batch(_loud_pcm(32000))
        self.assertEqual(len(frames), 50)
        self.assertEqual(w._frame_index, 50)

    def test_empty_pcm_is_silence(self):
        w = self._make_worker()
        frames = w._infer_batch(b"")
        self.assertEqual(len(frames), w.batch_size)
        self.assertFalse(frames[0][1])

    def test_unload_avatar_data_clears_buffers(self):
        w = self._make_worker()
        w.unload_avatar_data()
        self.assertEqual(w._frame_list, [])
        self.assertEqual(w._input_latent_list, [])
        self.assertEqual(w._coord_list, [])
        # Models (shared) are untouched.
        self.assertIsNotNone(w._unet)


@unittest.skipIf(torch is None, "torch not installed")
class TestMuseTalkLatentTransition(unittest.TestCase):
    """P-D latent interpolation backend (paper §4.4, torch-gated)."""

    def _make_worker(self):
        tmp_obj = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_obj.cleanup)
        vae = _RecordingVae()
        models = _fake_shared_models()
        models["vae"] = vae
        worker = MuseTalkAvatarWorker(
            _make_avatar(tmp_obj.name),
            target_fps=25,
            width=64,
            height=64,
            batch_size=4,
            device="cpu",
            is_half=False,
            shared_models=models,
        )
        worker._paste_back = lambda res_frame, idx: res_frame
        return worker, vae

    def test_no_transition_before_first_inference(self):
        w, vae = self._make_worker()
        self.assertEqual(w.render_transition_frames(3), [])
        self.assertEqual(vae.captured, [])

    def test_disabled_length_renders_nothing(self):
        w, vae = self._make_worker()
        w._last_pred_latent = torch.ones(1, 4, 8, 8)
        self.assertEqual(w.render_transition_frames(0), [])
        self.assertEqual(vae.captured, [])

    def test_latent_interpolation_math(self):
        import numpy as np

        w, vae = self._make_worker()
        # z_now = last predicted latent (spoken), z_closed = neutral
        # reference latent. Values 1.0 / 3.0 make the blend arithmetic
        # easy to verify: z_i = (1-α_i)·1 + α_i·3.
        w._last_pred_latent = torch.full((1, 4, 8, 8), 1.0)
        w._input_latent_list[0] = torch.full((1, 4, 8, 8), 3.0)

        frames = w.render_transition_frames(3)
        self.assertEqual(len(frames), 3)
        for data, speaking in frames:
            self.assertFalse(speaking)
            self.assertEqual(len(data), 64 * 64 * 3)

        self.assertEqual(len(vae.captured), 1)  # one batched VAE decode
        zs = vae.captured[0]
        self.assertEqual(tuple(zs.shape), (3, 4, 8, 8))
        expected = [1.0 + a * 2.0 for a in (1 / 3, 2 / 3, 1.0)]
        for i, exp in enumerate(expected):
            np.testing.assert_allclose(
                zs[i].numpy(), np.full((4, 8, 8), exp), atol=1e-6
            )
        # The transition does not consume reference-frame indices.
        self.assertEqual(w._frame_index, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

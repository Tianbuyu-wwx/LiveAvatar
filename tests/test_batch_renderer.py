"""Tests for the offline batch renderer (render_to_file with a fake worker)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from liveavatar.batch_renderer import render_to_file
from liveavatar.worker import AvatarAssets, AvatarWorker


class _RenderWorker(AvatarWorker):
    """4x4 BGR24 worker; each PCM chunk yields batch_size frames."""

    def __init__(self) -> None:
        super().__init__(
            AvatarAssets(
                avatar_id="fake",
                data_dir="avatars/fake",
                full_imgs_dir="avatars/fake/full_imgs",
                coords_path="avatars/fake/coords.pkl",
                latents_path="avatars/fake/latents.pt",
                mask_dir="avatars/fake/mask",
                mask_coords_path="avatars/fake/mask_coords.pkl",
            ),
            target_fps=25,
            width=4,
            height=4,
            batch_size=4,
        )

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        return [(bytes((i, 0, 0)) * (4 * 4), True) for i in range(self.batch_size)]


class TestRenderToFile(unittest.IsolatedAsyncioTestCase):
    async def test_renders_all_frames_to_mp4(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not installed")
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.mp4")
            worker = _RenderWorker()
            pcm = b"\x01\x00" * 640  # one chunk → batch_size=4 frames
            n = await render_to_file(worker, pcm, 16000, out)
            self.assertEqual(n, 4)
            self.assertTrue(Path(out).exists())

    async def test_tiny_pcm_still_yields_one_batch(self):
        """Streaming contract: even a tiny/empty chunk yields one batch."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not installed")
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "tiny.mp4")
            n = await render_to_file(_RenderWorker(), b"", 16000, out)
            self.assertEqual(n, 4)
            self.assertTrue(Path(out).exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

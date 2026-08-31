"""Tests for the offline batch renderer (render_to_file / _run / main)."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from liveavatar import batch_renderer
from liveavatar.batch_renderer import render_to_file
from liveavatar.config import AvatarPoolConfig
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


class _FakeLease:
    worker = _RenderWorker()


class _FakePool:
    """Records the config and avatar requested; hands out a fake lease."""

    instances: list[_FakePool] = []

    def __init__(self, config: AvatarPoolConfig) -> None:
        self.config = config
        _FakePool.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def acquire(self, session: str, avatar_id: str, timeout: float = 0.0):
        self.acquired_avatar = avatar_id
        return _FakeLease()

    async def release_async(self, session: str) -> None:
        self.released = True

    async def stop(self) -> None:
        self.stopped = True


class TestRun(unittest.IsolatedAsyncioTestCase):
    async def test_run_acquires_renders_releases(self):
        _FakePool.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.mp4")
            with (
                mock.patch.object(batch_renderer, "AvatarPool", _FakePool),
                mock.patch(
                    "liveavatar.preview.load_wav_mono_16k",
                    return_value=(b"\x01\x00" * 640, 0.04),
                ),
            ):
                args = argparse.Namespace(
                    audio="a.wav",
                    avatar="yongen",
                    avatar_data_root=str(Path(tmp) / "avatars"),
                    device="cpu",
                    batch_size=4,
                    acquire_timeout=1.0,
                    save=out,
                )
                n = await batch_renderer._run(args)
            self.assertEqual(n, 4)
            pool = _FakePool.instances[0]
            self.assertEqual(pool.acquired_avatar, "yongen")
            self.assertTrue(pool.released and pool.stopped)
            self.assertEqual(pool.config.device, "cpu")
            self.assertTrue(Path(out).exists())


class TestMain(unittest.TestCase):
    def test_main_invokes_run_and_prints_frames(self):
        with (
            mock.patch.object(
                batch_renderer, "_run", new=mock.AsyncMock(return_value=7)
            ) as run_mock,
            mock.patch("sys.argv", ["batch", "--audio", "a.wav", "--avatar", "y"]),
            mock.patch("builtins.print") as print_mock,
        ):
            batch_renderer.main(
                ["--audio", "a.wav", "--avatar", "y", "--save", "out.mp4"]
            )
        run_mock.assert_awaited_once()
        self.assertTrue(any("7" in str(c) for c in print_mock.call_args_list))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

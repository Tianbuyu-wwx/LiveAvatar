"""Tests for preview helpers: load_wav_mono_16k, PreviewSink, run_preview."""

from __future__ import annotations

import argparse
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from liveavatar.preview import PreviewSink, load_wav_mono_16k
from liveavatar.worker import AvatarAssets, AvatarFrame, AvatarWorker


def _write_wav(path, samples: list[int], framerate: int, nchannels: int = 1,
               sampwidth: int = 2) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        frames = b"".join(struct.pack("<h", s) for s in samples)
        if nchannels > 1:
            frames = frames * nchannels  # interleaved identical channels
        wf.writeframes(frames)


class TestLoadWavMono16k(unittest.TestCase):
    def test_mono_16k_passthrough(self):
        import os
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            _write_wav(tmp.name, [0, 16384, -16384, 32767], 16000)
            pcm, duration = load_wav_mono_16k(tmp.name)
            self.assertEqual(pcm, struct.pack("<4h", 0, 16384, -16384, 32767))
            self.assertAlmostEqual(duration, 4 / 16000, places=6)
        finally:
            os.unlink(tmp.name)

    def test_stereo_downmix(self):
        import os
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            # Stereo: L=16384, R=0 → mean 8192.
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(struct.pack("<2h", 16384, 0))
            pcm, duration = load_wav_mono_16k(tmp.name)
            self.assertEqual(len(pcm), 2)
            self.assertEqual(struct.unpack("<h", pcm), (8192,))
        finally:
            os.unlink(tmp.name)

    def test_resample_8k_to_16k(self):
        import os
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            _write_wav(tmp.name, [0, 16000], 8000)
            pcm, duration = load_wav_mono_16k(tmp.name)
            n_out = len(pcm) // 2
            self.assertEqual(n_out, 4)  # 2 samples @8k → 4 @16k
            self.assertAlmostEqual(duration, 2 / 8000, places=6)
        finally:
            os.unlink(tmp.name)

    def test_8bit_rejected(self):
        import os
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(1)
                wf.setframerate(16000)
                wf.writeframes(b"\x80\x80")
            with self.assertRaises(SystemExit):
                load_wav_mono_16k(tmp.name)
        finally:
            os.unlink(tmp.name)


class TestPreviewSink(unittest.IsolatedAsyncioTestCase):
    def _frame(self, pts_us: int = 0) -> AvatarFrame:
        return AvatarFrame(
            frame_data=bytes(4 * 4 * 3),
            pts_us=pts_us,
            epoch=0,
            width=4,
            height=4,
            is_speaking=True,
        )

    async def test_publish_accepts_and_buffers(self):
        sink = PreviewSink(maxlen=4)
        self.assertTrue(await sink.publish_frame(self._frame(), epoch=0))
        self.assertEqual(len(sink.frames), 1)

    async def test_publish_rejects_stale_epoch(self):
        sink = PreviewSink()
        sink.cancel_epoch(new_epoch=3)
        self.assertEqual(sink.current_epoch, 3)
        self.assertFalse(await sink.publish_frame(self._frame(), epoch=2))
        self.assertTrue(await sink.publish_frame(self._frame(), epoch=3))

    async def test_cancel_epoch_only_advances(self):
        sink = PreviewSink()
        sink.cancel_epoch(new_epoch=5)
        sink.cancel_epoch(new_epoch=2)  # must not rewind
        self.assertEqual(sink.current_epoch, 5)


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


class _FakeLease:
    worker = _RenderWorker()


class _FakePool:
    def __init__(self, config) -> None:
        self.config = config

    async def start(self) -> None:
        pass

    async def acquire(self, session: str, avatar_id: str, timeout: float = 0.0):
        return _FakeLease()

    async def release_async(self, session: str) -> None:
        pass

    async def stop(self) -> None:
        pass


class TestRunPreview(unittest.IsolatedAsyncioTestCase):
    async def test_run_preview_writes_mp4(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not installed")
        with tempfile.TemporaryDirectory() as tmp:
            wav = str(Path(tmp) / "a.wav")
            # 0.2 s @ 16 kHz mono — two batch-sized chunks.
            _write_wav(wav, [1000] * 3200, 16000)
            out = str(Path(tmp) / "out.mp4")
            args = argparse.Namespace(
                audio=wav,
                avatar="fake",
                avatar_data_root=str(Path(tmp) / "avatars"),
                device="cpu",
                acquire_timeout=2.0,
                save=out,
            )
            with mock.patch("liveavatar.preview.AvatarPool", _FakePool):
                from liveavatar.preview import run_preview

                await run_preview(args)
            self.assertTrue(Path(out).exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

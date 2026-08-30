"""Tests for preview helpers: load_wav_mono_16k (pure wave+numpy)."""

from __future__ import annotations

import struct
import unittest
import wave

from liveavatar.preview import load_wav_mono_16k


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

import unittest

from liveavatar.audio_in.frame import PCMFrame
from liveavatar.audio_in.reference.asr import ScriptedAsrAdapter
from liveavatar.audio_in.reference.echo import ZeroLagEchoDetector
from liveavatar.audio_in.reference.eou import SilenceEouDetector
from liveavatar.audio_in.reference.vad import EnergyVad


class TestReferenceAdapters(unittest.TestCase):
    def _make_frame(self, samples, seq: int = 1, pts_us: int = 0) -> PCMFrame:
        return PCMFrame.from_int16_array(
            samples, "s1", 0, seq, pts_us, pts_us + 1_000_000
        )

    def test_energy_vad(self):
        vad = EnergyVad(threshold_db=-50.0, release_db=-55.0)
        silent = self._make_frame([0] * 320)
        loud = self._make_frame([30000] * 320, seq=2, pts_us=20000)
        self.assertEqual(len(vad.push_frame(silent)), 0)
        events = vad.push_frame(loud)
        self.assertEqual(events[0]["kind"], "speech_start")

    def test_silence_eou(self):
        eou = SilenceEouDetector(silence_needed_us=60000)
        frame1 = self._make_frame([30000] * 320, seq=1, pts_us=0)
        events = eou.push_frame(frame1, vad_active=True)
        self.assertEqual(len(events), 0)
        frame2 = self._make_frame([0] * 320, seq=2, pts_us=80000)
        events = eou.push_frame(frame2, vad_active=False)
        self.assertEqual(len(events), 0)
        frame3 = self._make_frame([0] * 320, seq=3, pts_us=140000)
        events = eou.push_frame(frame3, vad_active=False)
        self.assertEqual(events[0]["confidence"], 1.0)

    def test_scripted_asr(self):
        asr = ScriptedAsrAdapter()
        loud = self._make_frame([30000] * 320, seq=1, pts_us=0)
        silent = self._make_frame([0] * 320, seq=2, pts_us=20000)
        events = asr.push_frame(loud)
        self.assertEqual(events[0]["phase"], "partial")
        # 4 silent frames accumulate silence_count; the 5th triggers final.
        for _ in range(4):
            asr.push_frame(silent)
        final = asr.push_frame(silent)
        self.assertEqual(final[0]["phase"], "final")

    def test_zero_lag_echo(self):
        detector = ZeroLagEchoDetector()
        # Non-constant identical signals yield a correlation of 1.0.
        frame = self._make_frame([0] * 160 + [32767] * 160, seq=1, pts_us=0)
        ref = frame.pcm_s16le
        event = detector.push_frame(frame, ref)
        self.assertIsNotNone(event)
        self.assertGreater(event["correlation"], 0.9)


if __name__ == "__main__":
    unittest.main()

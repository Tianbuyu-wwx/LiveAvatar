import unittest

from liveavatar.audio_in.frame import FRAME_DURATION_US, SAMPLE_RATE, PCMFrame


class TestPCMFrame(unittest.TestCase):
    def test_silence_frame_valid(self):
        frame = PCMFrame.silence("s1", 0, 1, 0, 1_000_000)
        self.assertEqual(frame.sample_rate, SAMPLE_RATE)
        self.assertEqual(frame.channels, 1)
        self.assertEqual(frame.frame_duration_us, FRAME_DURATION_US)
        self.assertEqual(frame.energy_db(), -96.0)

    def test_energy_calculation(self):
        # 40 ms frame needs 640 samples.
        samples = [0] * 320 + [32767] * 320
        frame = PCMFrame.from_int16_array(
            samples, "s1", 0, 1, 0, 1_000_000, frame_duration_us=40000
        )
        self.assertGreater(frame.energy_db(), -20.0)

    def test_invalid_duration(self):
        with self.assertRaises(ValueError):
            PCMFrame.silence("s1", 0, 1, 0, 1_000_000, frame_duration_us=30000)

    def test_wrong_buffer_size(self):
        samples = [0] * 100  # wrong for 20ms
        with self.assertRaises(ValueError):
            PCMFrame.from_int16_array(samples, "s1", 0, 1, 0, 1_000_000)


if __name__ == "__main__":
    unittest.main()

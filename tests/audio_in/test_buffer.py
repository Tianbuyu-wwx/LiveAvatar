import unittest

from liveavatar.audio_in.buffer import BoundedFrameBuffer, DropPolicy
from liveavatar.audio_in.frame import PCMFrame


class TestBoundedFrameBuffer(unittest.TestCase):
    def make_frame(self, seq: int, epoch: int = 0, pts_us: int = 0) -> PCMFrame:
        return PCMFrame.silence("s1", epoch, seq, pts_us, pts_us + 1_000_000)

    def test_fifo_order(self):
        buf = BoundedFrameBuffer(capacity=3)
        for i in range(3):
            self.assertTrue(buf.push(self.make_frame(i + 1, pts_us=i * 20000)))
        self.assertEqual(len(buf), 3)
        self.assertEqual(buf.pop().seq, 1)

    def test_drop_oldest(self):
        buf = BoundedFrameBuffer(capacity=2, drop_policy=DropPolicy.DROP_OLDEST)
        for i in range(4):
            buf.push(self.make_frame(i + 1, pts_us=i * 20000))
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.pop().seq, 3)
        self.assertEqual(buf.stats.dropped_full, 2)

    def test_drop_newest(self):
        buf = BoundedFrameBuffer(capacity=2, drop_policy=DropPolicy.DROP_NEWEST)
        for i in range(4):
            buf.push(self.make_frame(i + 1, pts_us=i * 20000))
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.pop().seq, 1)

    def test_epoch_advance(self):
        buf = BoundedFrameBuffer(capacity=5)
        buf.push(self.make_frame(1, epoch=0))
        buf.push(self.make_frame(2, epoch=1))
        buf.advance_epoch(1)
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf.stats.dropped_epoch, 1)

    def test_deadline_drop(self):
        clock = {"t": 0}
        buf = BoundedFrameBuffer(capacity=5, clock_now_us=lambda: clock["t"])
        frame = PCMFrame.silence("s1", 0, 1, 0, 100)
        clock["t"] = 200
        self.assertFalse(buf.push(frame))
        self.assertEqual(buf.stats.dropped_deadline, 1)

    def test_duplicate_drop(self):
        buf = BoundedFrameBuffer(capacity=5)
        buf.push(self.make_frame(1))
        self.assertFalse(buf.push(self.make_frame(1)))
        self.assertEqual(buf.stats.duplicates, 1)


if __name__ == "__main__":
    unittest.main()

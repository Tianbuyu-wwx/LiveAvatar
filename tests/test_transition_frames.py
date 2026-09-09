# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-D barge-in transition integration tests (paper §4.4).

Adapter level: the transition sentinel enqueued by ``cancel_epoch`` must
publish the worker's mouth-close easing frames as the FIRST frames of the
new epoch (before any utterance-2 chunk), with PTS continuing the old
clock. Workers without a transition implementation (or with a raising
one) degrade to the hard cut end to end.

Worker level: the MouthProxyWorker openness-scheduled backend (the
faithful analog of latent interpolation) eases the mouth to the closed
neutral state and lands its internal openness there.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from liveavatar.adapter import AvatarStreamingAdapter
from liveavatar.runtime.transition import CLOSED_OPENNESS
from liveavatar.worker import AvatarFrame, AvatarWorker
from tests.conftest import make_assets as _make_assets
from tests.conftest import pcm, wait_until

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "interruption_eval"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from mouth_proxy import MouthProxyWorker  # noqa: E402

_FRAME_BYTES = 4 * 4 * 3  # width * height * 3 for the fake 4x4 workers


class _TransitionWorker(AvatarWorker):
    """Fake worker with a programmable P-D transition implementation."""

    def __init__(
        self,
        assets,
        *,
        n_transition: int = 3,
        fail: bool = False,
        batch_size: int = 2,
    ) -> None:
        super().__init__(
            assets, target_fps=25, width=4, height=4, batch_size=batch_size
        )
        self.n_transition = n_transition
        self.fail = fail
        self.chunks_seen = 0

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        self.chunks_seen += 1
        return [(bytes([0x10 + self.chunks_seen]) * _FRAME_BYTES, True)] * (
            self.batch_size
        )

    def render_transition_frames(
        self, num_frames: int = 3
    ) -> list[tuple[bytes, bool]]:
        if self.fail:
            raise RuntimeError("transition render boom")
        n = min(num_frames, self.n_transition)
        return [(bytes([0xEE]) * _FRAME_BYTES, False)] * n


class _BareWorker(AvatarWorker):
    """Fake worker WITHOUT a transition override (base-class hard cut)."""

    def __init__(self, assets) -> None:
        super().__init__(assets, target_fps=25, width=4, height=4, batch_size=2)
        self.chunks_seen = 0

    def _infer_batch(self, pcm_s16le: bytes) -> list[tuple[bytes, bool]]:
        self.chunks_seen += 1
        return [(b"\x20" * _FRAME_BYTES, True)] * self.batch_size


class TestAdapterTransitionFrames(unittest.IsolatedAsyncioTestCase):
    async def _drive(
        self, adapter: AvatarStreamingAdapter, worker: _TransitionWorker
    ) -> list[AvatarFrame]:
        await adapter.start()
        try:
            await adapter.push_pcm(pcm(320, 100), 0, 0)  # utterance 1, epoch 0
            await wait_until(lambda: len(adapter.published_frames) >= 2)
            adapter.cancel_epoch(1)
            await adapter.push_pcm(pcm(320, 100), 1_000_000, 1)  # utterance 2
            # Wait for utt2 frames to be PUBLISHED (chunks_seen ticks before
            # publication; stop() would truncate the publish loop).
            await wait_until(
                lambda: len(
                    [
                        f
                        for f in adapter.published_frames
                        if f.epoch == 1 and f.is_speaking
                    ]
                )
                >= 2
            )
        finally:
            await adapter.stop()
        return adapter.published_frames

    async def test_transition_frames_published_before_utt2(self) -> None:
        worker = _TransitionWorker(_make_assets("trans"))
        adapter = AvatarStreamingAdapter(
            worker=worker, publisher=None, session_id="s", transition_frames=3
        )
        frames = await self._drive(adapter, worker)

        epoch0 = [f for f in frames if f.epoch == 0]
        trans = [f for f in frames if f.epoch == 1 and not f.is_speaking]
        utt2 = [f for f in frames if f.epoch == 1 and f.is_speaking]

        self.assertTrue(all(f.is_speaking for f in epoch0))
        self.assertEqual(len(trans), 3)
        self.assertEqual(
            adapter.stats.transition_frames_published, 3
        )
        # Transition frames precede the first utterance-2 frame.
        first_trans = frames.index(trans[0])
        first_utt2 = frames.index(utt2[0])
        self.assertLess(first_trans, first_utt2)
        # PTS continues the old clock and never collides with utterance 2.
        self.assertGreater(trans[0].pts_us, epoch0[-1].pts_us)
        self.assertTrue(
            all(
                b.pts_us > a.pts_us
                for a, b in zip(trans, trans[1:], strict=False)
            )
        )
        self.assertLess(trans[-1].pts_us, utt2[0].pts_us)

    async def test_transition_disabled_keeps_hard_cut(self) -> None:
        worker = _TransitionWorker(_make_assets("trans"))
        adapter = AvatarStreamingAdapter(
            worker=worker, publisher=None, session_id="s", transition_frames=0
        )
        frames = await self._drive(adapter, worker)
        self.assertEqual(
            [f for f in frames if not f.is_speaking], []
        )
        self.assertEqual(adapter.stats.transition_frames_published, 0)
        # Utterance-2 frames still flow (hard cut preserved).
        self.assertTrue(any(f.epoch == 1 and f.is_speaking for f in frames))

    async def test_worker_without_override_hard_cuts(self) -> None:
        worker = _BareWorker(_make_assets("bare"))
        adapter = AvatarStreamingAdapter(
            worker=worker, publisher=None, session_id="s", transition_frames=3
        )
        frames = await self._drive(adapter, worker)
        self.assertEqual([f for f in frames if not f.is_speaking], [])
        self.assertTrue(any(f.epoch == 1 and f.is_speaking for f in frames))

    async def test_raising_worker_degrades_to_hard_cut(self) -> None:
        worker = _TransitionWorker(_make_assets("trans"), fail=True)
        adapter = AvatarStreamingAdapter(
            worker=worker, publisher=None, session_id="s", transition_frames=3
        )
        frames = await self._drive(adapter, worker)
        self.assertEqual([f for f in frames if not f.is_speaking], [])
        self.assertEqual(adapter.stats.transition_frames_published, 0)
        self.assertTrue(any(f.epoch == 1 and f.is_speaking for f in frames))

    async def test_second_barge_in_drops_pending_transition(self) -> None:
        worker = _TransitionWorker(_make_assets("trans"))
        adapter = AvatarStreamingAdapter(
            worker=worker, publisher=None, session_id="s", transition_frames=3
        )
        await adapter.start()
        try:
            await adapter.push_pcm(pcm(320, 100), 0, 0)
            await wait_until(lambda: len(adapter.published_frames) >= 2)
            # Two sync cancel_epoch calls — the consumer cannot interleave:
            # the epoch-1 sentinel is drained, only the epoch-2 one remains.
            adapter.cancel_epoch(1)
            adapter.cancel_epoch(2)
            await adapter.push_pcm(pcm(320, 100), 2_000_000, 2)
            await wait_until(
                lambda: len(
                    [
                        f
                        for f in adapter.published_frames
                        if f.epoch == 2 and f.is_speaking
                    ]
                )
                >= 2
            )
        finally:
            await adapter.stop()
        frames = adapter.published_frames
        self.assertEqual([f for f in frames if f.epoch == 1], [])
        trans = [f for f in frames if f.epoch == 2 and not f.is_speaking]
        self.assertEqual(len(trans), 3)
        self.assertTrue(
            all(f.epoch == 2 for f in frames if not f.is_speaking)
        )


class TestMouthProxyTransition(unittest.TestCase):
    """Openness-scheduled backend: mouth eases shut, state lands closed."""

    def _cavity_pixels(self, frame_bytes: bytes) -> int:
        img = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(256, 256, 3)
        # Mouth cavity color (70, 50, 60) BGR — see render_face.
        return int(np.all(img == np.array([70, 50, 60], np.uint8), axis=2).sum())

    def test_no_transition_before_first_render(self) -> None:
        worker = MouthProxyWorker("mouth_t0")
        self.assertEqual(worker.render_transition_frames(3), [])

    def test_transition_eases_mouth_shut(self) -> None:
        worker = MouthProxyWorker("mouth_t1")
        # Loud PCM drives the mouth wide open (RMS → openness well > 0.5).
        worker._infer_batch(b"\x10\x40" * 640)
        self.assertGreater(worker.last_openness, 0.5)
        before = worker._last_img
        cavity_before = self._cavity_pixels(before.tobytes())

        trans = worker.render_transition_frames(3)
        self.assertEqual(len(trans), 3)
        self.assertTrue(all(not speaking for _, speaking in trans))
        # Internal state lands on the neutral so utterance 2 starts closed.
        self.assertAlmostEqual(worker.last_openness, CLOSED_OPENNESS)

        imgs = [
            np.frombuffer(d, dtype=np.uint8).reshape(256, 256, 3)
            for d, _ in trans
        ]
        # Openness-scheduled ease toward the α=1 closed frame ⇒ distance
        # to the final frame strictly decreases along the schedule.
        last = imgs[-1].astype(np.float64)
        dists = [
            float(np.linalg.norm(f.astype(np.float64) - last)) for f in imgs
        ]
        for a, b in zip(dists, dists[1:], strict=False):
            self.assertLess(b, a)
        self.assertEqual(dists[-1], 0.0)
        # The closed final frame shows a smaller mouth cavity than the
        # open-mouth frame the transition started from.
        self.assertLess(self._cavity_pixels(trans[-1][0]), cavity_before)

    def test_silence_after_transition_stays_closed(self) -> None:
        worker = MouthProxyWorker("mouth_t2")
        worker._infer_batch(b"\x10\x40" * 640)
        worker.render_transition_frames(3)
        worker._infer_batch(b"\x00\x00" * 640)  # silence batch (utterance 2 gap)
        self.assertLessEqual(worker.last_openness, CLOSED_OPENNESS + 0.01)


if __name__ == "__main__":
    unittest.main()

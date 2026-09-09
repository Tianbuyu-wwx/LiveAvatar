# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Reference fake TTS that generates deterministic PCM chunks."""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncGenerator
from dataclasses import dataclass


@dataclass
class FakeTtsSegment:
    segment_seq: int
    text: str
    epoch: int
    pcm_s16le: bytes
    pts_us: int
    duration_us: int


class FakeTts:
    """Deterministic TTS for control-flow testing.

    Each text produces a short chirp. Supports cancel by epoch.

    Streaming mode (``synthesize_stream``) mirrors ``NvcWorker`` so that the
    RealtimeWorker routes through the async background-task path, enabling
    PCM fan-out to the Avatar adapter for video generation.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.segment_counter: int = 0
        self.active_segments: list[FakeTtsSegment] = []

    def synthesize(self, text: str, epoch: int, pts_us: int) -> list[FakeTtsSegment]:
        self.segment_counter += 1
        duration_us = 200000  # 200 ms per segment
        samples = int(self.sample_rate * duration_us / 1_000_000)
        # Simple chirp based on text length.
        freq = 440 + (len(text) % 10) * 50
        pcm = bytearray()
        for i in range(samples):
            t = i / self.sample_rate
            sample = int(8000 * (1 if (t * freq * 2) % 2 < 1 else -1))
            pcm.extend(struct.pack("<h", sample))
        segment = FakeTtsSegment(
            segment_seq=self.segment_counter,
            text=text,
            epoch=epoch,
            pcm_s16le=bytes(pcm),
            pts_us=pts_us,
            duration_us=duration_us,
        )
        self.active_segments.append(segment)
        return [segment]

    async def synthesize_stream(
        self, text: str, epoch: int, pts_us: int
    ) -> AsyncGenerator[FakeTtsSegment, None]:
        """Async streaming variant used by the Avatar demo path.

        Yields the same single chirp segment as ``synthesize`` so that
        ``RealtimeWorker._run_tts_stream`` fans the PCM out to the avatar
        adapter while also publishing the audio track.
        """
        # Simulate a tiny yield to keep the async generator semantics.
        await asyncio.sleep(0)
        for segment in self.synthesize(text, epoch, pts_us):
            yield segment

    def cancel_epoch(self, epoch: int) -> int:
        before = len(self.active_segments)
        self.active_segments = [s for s in self.active_segments if s.epoch >= epoch]
        return before - len(self.active_segments)

    def pop_played_segments(self, consumed_pts_us: int) -> list[FakeTtsSegment]:
        played = [s for s in self.active_segments if s.pts_us + s.duration_us <= consumed_pts_us]
        self.active_segments = [
            s for s in self.active_segments if s.pts_us + s.duration_us > consumed_pts_us
        ]
        return played

    def pending_audio_tail(self, consumed_pts_us: int) -> bytes | None:
        """P-C protocol: PCM of the not-yet-played audio tail.

        Returns the concatenated s16le PCM of active segments that have
        not finished playing by ``consumed_pts_us``, trimmed so the tail
        starts at the current playback position; ``None`` when nothing
        is pending. Consumed by the energy-valley cut selector
        (``runtime.valley``) on barge-in. TTS adapters that keep no
        server-side pending buffer simply omit this method — the worker
        degrades to the hard cut.
        """
        parts: list[bytes] = []
        for seg in sorted(self.active_segments, key=lambda s: s.pts_us):
            if seg.pts_us + seg.duration_us <= consumed_pts_us:
                continue
            skip = int(max(0, consumed_pts_us - seg.pts_us) * self.sample_rate / 1_000_000)
            parts.append(seg.pcm_s16le[skip * 2:])
        if not parts:
            return None
        return b"".join(parts)

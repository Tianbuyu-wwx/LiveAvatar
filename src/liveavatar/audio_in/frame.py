"""Canonical PCM frame definition and sample clock utilities."""

from __future__ import annotations

import struct
from dataclasses import dataclass

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_US = 20000  # 20 ms
BYTES_PER_SAMPLE = 2
VALID_DURATIONS_US = (20000, 40000, 60000)


@dataclass(frozen=True, slots=True)
class SampleClock:
    """Monotonic sample clock in microseconds."""

    pts_us: int = 0

    def advance(self, duration_us: int) -> SampleClock:
        return SampleClock(pts_us=self.pts_us + duration_us)

    def samples_to_us(self, samples: int) -> int:
        return int(samples * 1_000_000 / SAMPLE_RATE)

    def us_to_samples(self, us: int) -> int:
        return int(us * SAMPLE_RATE / 1_000_000)


@dataclass(frozen=True, slots=True)
class PCMFrame:
    """Canonical audio frame.

    Constraints:
    - sample_rate == 16000
    - channels == 1
    - frame_duration_us in {20000, 40000, 60000}
    - pcm_s16le length matches duration and sample rate
    - sample_clock_pts_us monotonic within session/epoch
    """

    session_id: str
    epoch: int
    seq: int
    pts_us: int
    deadline_us: int
    pcm_s16le: bytes
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    frame_duration_us: int = FRAME_DURATION_US
    discontinuity: bool = False
    sample_clock_pts_us: int = 0

    def __post_init__(self) -> None:
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SAMPLE_RATE}")
        if self.channels != CHANNELS:
            raise ValueError(f"channels must be {CHANNELS}")
        if self.frame_duration_us not in VALID_DURATIONS_US:
            raise ValueError(f"frame_duration_us must be one of {VALID_DURATIONS_US}")
        expected_bytes = int(
            self.sample_rate * self.channels * self.frame_duration_us / 1_000_000 * BYTES_PER_SAMPLE
        )
        if len(self.pcm_s16le) != expected_bytes:
            raise ValueError(
                f"pcm_s16le length {len(self.pcm_s16le)} != expected {expected_bytes} "
                f"for {self.frame_duration_us}us frame"
            )

    @property
    def num_samples(self) -> int:
        return len(self.pcm_s16le) // (self.channels * BYTES_PER_SAMPLE)

    def energy_db(self) -> float:
        """Compute RMS energy in dB relative to full scale."""
        if not self.pcm_s16le:
            return -96.0
        samples = struct.unpack(f"<{self.num_samples}h", self.pcm_s16le)
        rms = sum(s * s for s in samples) / len(samples)
        if rms == 0:
            return -96.0
        import math

        return 20 * math.log10(math.sqrt(rms) / 32768.0)

    def to_int16_array(self) -> list[int]:
        return list(struct.unpack(f"<{self.num_samples}h", self.pcm_s16le))

    @classmethod
    def from_int16_array(
        cls,
        samples: list[int],
        session_id: str,
        epoch: int,
        seq: int,
        pts_us: int,
        deadline_us: int,
        sample_clock_pts_us: int = 0,
        discontinuity: bool = False,
        frame_duration_us: int = FRAME_DURATION_US,
    ) -> PCMFrame:
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        return cls(
            session_id=session_id,
            epoch=epoch,
            seq=seq,
            pts_us=pts_us,
            deadline_us=deadline_us,
            pcm_s16le=pcm,
            sample_clock_pts_us=sample_clock_pts_us,
            discontinuity=discontinuity,
            frame_duration_us=frame_duration_us,
        )

    @classmethod
    def silence(
        cls,
        session_id: str,
        epoch: int,
        seq: int,
        pts_us: int,
        deadline_us: int,
        sample_clock_pts_us: int = 0,
        frame_duration_us: int = FRAME_DURATION_US,
    ) -> PCMFrame:
        expected_samples = int(
            SAMPLE_RATE * CHANNELS * frame_duration_us / 1_000_000
        )
        pcm = b"\x00" * (expected_samples * BYTES_PER_SAMPLE)
        return cls(
            session_id=session_id,
            epoch=epoch,
            seq=seq,
            pts_us=pts_us,
            deadline_us=deadline_us,
            pcm_s16le=pcm,
            sample_clock_pts_us=sample_clock_pts_us,
            frame_duration_us=frame_duration_us,
        )

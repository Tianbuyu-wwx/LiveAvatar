"""Tutor audio publisher: stream TTS PCM chunks to a LiveKit audio track.

Consumes canonical 16kHz mono S16LE chunks produced by the worker's TTS
(currently ``FakeTts``, later NVC streaming TTS), splits them into 20ms
``AudioFrame``s, and feeds a LiveKit ``AudioSource`` that backs a published
``LocalAudioTrack``. Supports epoch-aware cancellation so a confirmed
interrupt stops playback of stale Tutor audio within one frame (<=20ms)
plus encoder/transport latency.

Scope (Sprint 1, step 2): publish Tutor audio only. The orchestration loop
that drains the worker's ``tts_audio`` output events and calls
``publish_chunk`` is wired in the loopback step (4). Control data-channel
mapping is step 3.

Cancellation model
------------------
A confirmed interrupt bumps the worker epoch. The publisher mirrors that via
``cancel_epoch(new_epoch)``. ``publish_chunk`` checks ``chunk_epoch <
current_epoch`` before every 20ms frame and aborts the remainder on mismatch,
so at most one already-captured frame (<=20ms) leaks through after an
interrupt. Final audio flushing is the browser Thin Client's job (playback
ACK / flush), matching the Phase 1 architecture.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

# Optional LiveKit RTC SDK (heavy native dependency).
try:
    from livekit import rtc  # type: ignore

    _HAS_LIVEKIT = True
except Exception:  # pragma: no cover - exercised only when livekit absent
    _HAS_LIVEKIT = False
    rtc = None  # type: ignore

# Sibling import; RealtimeAudio path is added by paths.py / PYTHONPATH.
try:
    from liveavatar.audio_in.frame import (
        BYTES_PER_SAMPLE,
        CHANNELS,
        FRAME_DURATION_US,
        SAMPLE_RATE,
    )

    _HAS_AUDIO = True
except Exception:  # pragma: no cover - mirrors worker.py fallback
    _HAS_AUDIO = False
    SAMPLE_RATE = 16000
    CHANNELS = 1
    FRAME_DURATION_US = 20000
    BYTES_PER_SAMPLE = 2

logger = logging.getLogger("liveavatar.runtime.tutor_publisher")

# Bytes per canonical 20ms frame at 16kHz mono S16LE: 320 samples * 2 bytes.
_FRAME_BYTES = int(
    SAMPLE_RATE * CHANNELS * FRAME_DURATION_US / 1_000_000
) * BYTES_PER_SAMPLE


@dataclass
class TutorPublisherStats:
    """Counters for the Tutor audio publisher."""

    chunks_seen: int = 0
    chunks_published: int = 0
    chunks_cancelled: int = 0
    chunks_timed_out: int = 0
    frames_published: int = 0
    frames_dropped_epoch: int = 0
    frames_dropped_timeout: int = 0
    bytes_published: int = 0
    track_published: bool = False


def _split_pcm_into_frames(pcm_s16le: bytes, frame_bytes: int = _FRAME_BYTES) -> list[bytes]:
    """Split a PCM chunk into ``frame_bytes``-sized frames.

    The final partial frame is right-padded with silence so every emitted
    frame is exactly ``frame_bytes``. An empty input yields no frames.
    """
    if not pcm_s16le:
        return []
    out: list[bytes] = []
    for i in range(0, len(pcm_s16le), frame_bytes):
        chunk = pcm_s16le[i : i + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
        out.append(chunk)
    return out


class TutorAudioPublisher:
    """Publish Tutor TTS audio to a LiveKit room as a cancellable track.

    Lifecycle::

        publisher = TutorAudioPublisher(room.local_participant, session_id)
        await publisher.start()                  # create + publish track
        await publisher.publish_chunk(pcm, epoch)  # feed a TTS chunk
        publisher.cancel_epoch(new_epoch)        # interrupt: stop stale audio
        await publisher.stop()                   # unpublish + cleanup

    The publisher is single-session and single-track. Chunk epoch must be the
    worker epoch at which the TTS was synthesized; stale-epoch chunks are
    dropped frame-by-frame for prompt interrupt response.
    """

    def __init__(
        self,
        local_participant: Any,
        session_id: str,
        *,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        frame_duration_us: int = FRAME_DURATION_US,
        capture_timeout_s: float = 0.1,
        metrics: Any = None,
    ) -> None:
        # NOTE: livekit is only required in start(); __init__ stays livekit-free
        # so the pure publish/cancel logic is unit-testable with a fake source.
        self.local_participant = local_participant
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_us = frame_duration_us
        self.capture_timeout_s = capture_timeout_s
        self.metrics = metrics
        self._frame_bytes = int(
            sample_rate * channels * frame_duration_us / 1_000_000
        ) * BYTES_PER_SAMPLE
        self._current_epoch = 0
        self._source: Any = None  # rtc.AudioSource
        self._track: Any = None  # rtc.LocalAudioTrack
        self._publication: Any = None  # rtc.LocalTrackPublication
        self.stats = TutorPublisherStats()

    # ------------------------------------------------------------------ start

    async def start(self) -> None:
        """Create the AudioSource, LocalAudioTrack and publish it."""
        if not _HAS_LIVEKIT:
            raise RuntimeError(
                "livekit package is not installed; "
                "install the 'livekit' extra to use TutorAudioPublisher"
            )
        if self._track is not None:
            return
        self._source = rtc.AudioSource(self.sample_rate, self.channels)
        self._track = rtc.LocalAudioTrack.create_audio_track(
            f"tutor-{self.session_id}", self._source
        )
        options = rtc.TrackPublishOptions()
        # Convention: published bot audio uses SOURCE_MICROPHONE. The worker's
        # own LiveKitParticipantAdapter only subscribes to *remote* tracks, so
        # it never loops this track back to itself.
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        self._publication = await self.local_participant.publish_track(
            self._track, options
        )
        self.stats.track_published = True
        logger.info(
            "tutor_publisher_started",
            extra={
                "session_id": self.session_id,
                "track_sid": str(getattr(self._publication, "sid", None)),
            },
        )

    # -------------------------------------------------------------- publish

    async def publish_chunk(self, pcm_s16le: bytes, epoch: int) -> bool:
        """Feed one TTS PCM chunk to the track, respecting the current epoch.

        Returns True if the whole chunk was published, False if it was
        cancelled (stale epoch) or aborted (capture backpressure timeout).
        """
        self.stats.chunks_seen += 1
        if self._source is None:
            raise RuntimeError("publisher not started; call start() first")
        frames = _split_pcm_into_frames(pcm_s16le, self._frame_bytes)
        published = 0
        for frame_bytes in frames:
            # Check the live epoch before every frame so an interrupt that
            # bumps _current_epoch mid-chunk aborts within one frame.
            if epoch < self._current_epoch:
                self.stats.frames_dropped_epoch += len(frames) - published
                self.stats.chunks_cancelled += 1
                return False
            try:
                await self._capture_frame(frame_bytes)
            except asyncio.TimeoutError:
                self.stats.frames_dropped_timeout += len(frames) - published
                self.stats.chunks_timed_out += 1
                return False
            published += 1
            self.stats.frames_published += 1
            if (
                self.metrics is not None
                and self.metrics.record_first_playback()
            ):
                logger.info(
                    "trace_first_playback",
                    extra={
                        "session_id": self.session_id,
                        "frames_published": self.stats.frames_published,
                        "first_to_playback_ms": self.metrics.first_to_playback_ms,
                    },
                )
        self.stats.chunks_published += 1
        self.stats.bytes_published += len(pcm_s16le)
        return True

    async def _capture_frame(self, frame_bytes: bytes) -> None:
        """Create an AudioFrame from raw PCM and feed it to the source.

        Override in tests to avoid the livekit dependency. The default impl
        assigns PCM via a zero-copy int16 memoryview cast (no numpy needed).
        """
        samples_per_channel = len(frame_bytes) // (BYTES_PER_SAMPLE * self.channels)
        frame = rtc.AudioFrame.create(
            self.sample_rate, self.channels, samples_per_channel
        )
        # frame.data is an int16 ('h') memoryview; cast the bytes view to
        # match so the buffer-protocol assignment succeeds without numpy.
        frame.data[:] = memoryview(frame_bytes).cast("h")
        await asyncio.wait_for(
            self._source.capture_frame(frame), timeout=self.capture_timeout_s
        )

    # -------------------------------------------------------------- cancel

    def cancel_epoch(self, new_epoch: int) -> None:
        """Advance the cancellation epoch; stale-epoch chunks stop playing.

        Monotonic: a lower or equal value is ignored.
        """
        if new_epoch > self._current_epoch:
            self._current_epoch = new_epoch

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    # ------------------------------------------------------------------ stop

    async def stop(self) -> None:
        """Unpublish the track and release resources."""
        if self._publication is not None and self.local_participant is not None:
            track_sid = getattr(self._publication, "sid", None) or getattr(
                self._track, "sid", None
            )
            if track_sid is not None:
                try:
                    await self.local_participant.unpublish_track(track_sid)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("tutor_publisher_unpublish_error: %s", exc)
        self._publication = None
        self._track = None
        self._source = None
        self.stats.track_published = False

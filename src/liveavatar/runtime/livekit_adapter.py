"""LiveKit remote participant adapter for student_mic ingestion.

Subscribes to the browser Thin Client's published microphone track over
LiveKit WebRTC, decodes/resamples to canonical 16kHz mono S16LE, segments
into fixed 20ms PCMFrames, and pushes them into a RealtimeWorker's input
queue with epoch/seq/PTS/deadline metadata.

Scope (Sprint 1, step 1): receive student_mic only. Tutor audio publishing
and control data-channel mapping are implemented in sibling follow-ups.

Design notes
------------
- ``livekit`` (the RTC SDK) is a heavy native dependency. It is imported
  optionally so that the pure segmentation logic stays unit-testable on CI
  without native libs (mirrors the existing ``_HAS_AUDIO`` pattern).
- ``rtc.AudioStream(track, sample_rate=16000, num_channels=1)`` requests
  LiveKit to resample the publisher's Opus stream to canonical PCM. A numpy
  safety-net resampler handles any frame that still arrives at a foreign
  rate/channel layout.
- Token provisioning is decoupled: the adapter accepts a pre-issued token.
  ``issue_worker_token`` is provided as a convenience for self-issuing with
  the shared LiveKit API secret (mirrors RealtimeCore's HS256 claims).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

# Optional LiveKit RTC SDK (heavy native dependency).
try:
    from livekit import rtc  # type: ignore

    _HAS_LIVEKIT = True
except Exception:  # pragma: no cover - exercised only when livekit absent
    _HAS_LIVEKIT = False
    rtc = None  # type: ignore

# Optional numpy for the resampling safety net.
try:
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False

# Sibling import; RealtimeAudio path is added by paths.py / PYTHONPATH.
try:
    from liveavatar.audio_in.frame import (
        BYTES_PER_SAMPLE,
        CHANNELS,
        FRAME_DURATION_US,
        SAMPLE_RATE,
        PCMFrame,
    )

    _HAS_AUDIO = True
except Exception:  # pragma: no cover - mirrors worker.py fallback
    _HAS_AUDIO = False
    PCMFrame = Any  # type: ignore
    SAMPLE_RATE = 16000
    CHANNELS = 1
    FRAME_DURATION_US = 20000
    BYTES_PER_SAMPLE = 2

logger = logging.getLogger("liveavatar.runtime.livekit_adapter")

# Bytes per canonical 20ms frame at 16kHz mono S16LE: 320 samples * 2 bytes.
_FRAME_BYTES = int(
    SAMPLE_RATE * CHANNELS * FRAME_DURATION_US / 1_000_000
) * BYTES_PER_SAMPLE


@dataclass
class SegmenterStats:
    """Counters for the PCM segmentation stage."""

    frames_emitted: int = 0
    bytes_consumed: int = 0
    remainder_bytes: int = 0
    discontinuities: int = 0


@dataclass
class AdapterStats:
    """Counters for the LiveKit adapter."""

    frames_received: int = 0  # raw AudioFrames from LiveKit
    frames_pushed: int = 0  # canonical PCMFrames pushed to worker
    resampled: int = 0  # frames that hit the numpy safety net
    tracks_subscribed: int = 0
    push_dropped: int = 0  # frames rejected by worker queue (capacity/epoch)
    connected: bool = False


class _FrameSegmenter:
    """Accumulate raw 16kHz mono S16LE PCM and emit fixed 20ms PCMFrames.

    Pure and synchronous so it can be unit-tested without LiveKit. The caller
    reads the current epoch from the worker when feeding so emitted frames
    carry the right epoch (interrupt-purged frames then get dropped by the
    worker's epoch-aware queue).
    """

    def __init__(
        self,
        session_id: str,
        frame_duration_us: int = FRAME_DURATION_US,
        deadline_budget_us: int = 2_000_000,
    ) -> None:
        self.session_id = session_id
        self.frame_duration_us = frame_duration_us
        self.deadline_budget_us = deadline_budget_us
        self._buffer = bytearray()
        self._seq = 0
        self._pts_us = 0
        self._discontinuity = False
        self.stats = SegmenterStats()
        self._frame_bytes = int(
            SAMPLE_RATE * CHANNELS * frame_duration_us / 1_000_000
        ) * BYTES_PER_SAMPLE

    def mark_discontinuity(self) -> None:
        """Flag the next emitted frame as discontinuous (e.g. after a stall)."""
        self._discontinuity = True

    def feed(self, pcm_s16le: bytes, epoch: int) -> list[Any]:
        """Append PCM and return zero or more canonical PCMFrames.

        ``pcm_s16le`` must already be 16kHz mono S16LE; use
        ``_resample_to_16k_mono`` to normalize upstream frames first.
        """
        if not _HAS_AUDIO:
            return []
        self._buffer.extend(pcm_s16le)
        out: list[Any] = []
        while len(self._buffer) >= self._frame_bytes:
            chunk = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            frame = PCMFrame(
                session_id=self.session_id,
                epoch=epoch,
                seq=self._seq,
                pts_us=self._pts_us,
                deadline_us=self._pts_us + self.deadline_budget_us,
                pcm_s16le=chunk,
                frame_duration_us=self.frame_duration_us,
                discontinuity=self._discontinuity,
                sample_clock_pts_us=self._pts_us,
            )
            if self._discontinuity:
                self.stats.discontinuities += 1
                self._discontinuity = False
            self._seq += 1
            self._pts_us += self.frame_duration_us
            self.stats.frames_emitted += 1
            self.stats.bytes_consumed += self._frame_bytes
            out.append(frame)
        self.stats.remainder_bytes = len(self._buffer)
        return out

    def flush_remainder(self, epoch: int) -> Any | None:
        """Pad any leftover buffer to a full frame with silence.

        Returns None when the buffer is empty. The emitted frame is marked
        discontinuous because it contains padded samples.
        """
        if not _HAS_AUDIO or len(self._buffer) == 0:
            return None
        chunk = bytes(self._buffer) + b"\x00" * (
            self._frame_bytes - len(self._buffer)
        )
        del self._buffer[:]
        frame = PCMFrame(
            session_id=self.session_id,
            epoch=epoch,
            seq=self._seq,
            pts_us=self._pts_us,
            deadline_us=self._pts_us + self.deadline_budget_us,
            pcm_s16le=chunk,
            frame_duration_us=self.frame_duration_us,
            discontinuity=True,
            sample_clock_pts_us=self._pts_us,
        )
        self._seq += 1
        self._pts_us += self.frame_duration_us
        self.stats.frames_emitted += 1
        self.stats.discontinuities += 1
        self.stats.remainder_bytes = 0
        return frame


def _resample_to_16k_mono(data: bytes, sample_rate: int, num_channels: int) -> bytes:
    """Normalize arbitrary PCM S16LE to 16kHz mono S16LE via numpy.

    This is a safety net. When ``rtc.AudioStream`` is constructed with
    ``sample_rate=16000, num_channels=1`` LiveKit resamples upstream, so this
    path should rarely fire. Linear interpolation is acceptable here because
    the common path never enters it; a proper anti-aliased decimator belongs
    in Sprint 2 (real ASR/AEC calibration).
    """
    if not _HAS_NUMPY:
        # Without numpy we can only trust the caller already gave us 16k mono.
        return data
    if len(data) < 2:
        return b""
    # Truncate to a whole number of (sample, channel) units.
    usable = (len(data) // (BYTES_PER_SAMPLE * max(1, num_channels))) * (
        BYTES_PER_SAMPLE * max(1, num_channels)
    )
    if usable == 0:
        return b""
    samples = np.frombuffer(data[:usable], dtype=np.int16)
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1).astype(np.int16)
    if sample_rate == SAMPLE_RATE:
        return samples.tobytes()
    n_in = len(samples)
    if n_in == 0:
        return b""
    n_out = max(1, int(round(n_in * SAMPLE_RATE / sample_rate)))
    idx = np.linspace(0, n_in - 1, n_out)
    resampled = np.interp(idx, np.arange(n_in), samples.astype(np.float32))
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def issue_worker_token(
    api_key: str,
    api_secret: str,
    session_id: str,
    identity: str | None = None,
    ttl_seconds: int = 300,
) -> tuple[str, str, int]:
    """Issue a LiveKit-compatible HS256 JWT for the worker participant.

    Mirrors RealtimeCore's token claims so the same LiveKit server accepts it.
    Grants publish + subscribe + data so the worker can later publish Tutor
    audio and exchange control data without re-issuing. Returns
    ``(token, room_name, expires_at)``.
    """
    import jwt  # PyJWT; declared here so the module imports without it.

    ttl_seconds = max(30, min(3600, ttl_seconds))
    now = int(time.time())
    room_name = f"wisdomvii-{session_id}"
    worker_identity = identity or f"worker-{session_id}"
    claims = {
        "iss": api_key,
        "sub": worker_identity,
        "nbf": now,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": f"{session_id}-worker-{now}",
        "video": {
            "roomJoin": True,
            "room": room_name,
        },
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
    }
    token = jwt.encode(claims, api_secret, algorithm="HS256")
    return token, room_name, now + ttl_seconds


class LiveKitParticipantAdapter:
    """Receive a student_mic track from LiveKit and feed a RealtimeWorker.

    Lifecycle::

        adapter = LiveKitParticipantAdapter(worker, session_id, url, token)
        await adapter.start()      # connect + subscribe
        ...
        await adapter.stop()       # cancel streams + disconnect

    The adapter is single-session: one student per worker. It subscribes to
    every remote audio track (the student is the only other participant in a
    1:1 session) and pushes canonical PCMFrames into ``worker.input_queue``
    via ``worker.push_frame``.
    """

    def __init__(
        self,
        worker: Any,
        session_id: str,
        room_url: str,
        token: str,
        *,
        frame_duration_us: int = FRAME_DURATION_US,
        deadline_budget_us: int = 2_000_000,
        auto_subscribe: bool = True,
        metrics: Any = None,
    ) -> None:
        # NOTE: livekit is only required in start(); __init__ stays livekit-free
        # so the runtime and pure drain tests can construct the adapter without
        # the native dependency (mirrors TutorAudioPublisher).
        self.worker = worker
        self.session_id = session_id
        self.room_url = room_url
        self.token = token
        self.auto_subscribe = auto_subscribe
        self.metrics = metrics
        self.stats = AdapterStats()
        self._segmenter = _FrameSegmenter(
            session_id=session_id,
            frame_duration_us=frame_duration_us,
            deadline_budget_us=deadline_budget_us,
        )
        self._room: Any = None  # rtc.Room
        self._streams: dict[str, Any] = {}  # track_sid -> rtc.AudioStream
        self._tasks: set[asyncio.Task] = set()
        self._stopping = False
        # Wall-clock timestamp (monotonic, us) of the last received raw frame,
        # used to detect capture stalls and flag discontinuity.
        self._last_recv_monotonic_us: float = 0.0
        self._stall_threshold_us = frame_duration_us * 3

    @property
    def local_participant(self) -> Any:
        """Expose the room's local participant for sibling publishers.

        Returns None before ``start()`` or after ``stop()``. Used by the
        Tutor audio publisher (Sprint 1 step 2) to publish on the same room.
        """
        if self._room is None:
            return None
        return self._room.local_participant

    @property
    def room(self) -> Any:
        """Expose the livekit Room for sibling adapters (e.g. control channel).

        Returns None before ``start()`` or after ``stop()``.
        """
        return self._room

    # ------------------------------------------------------------------ start

    async def start(self) -> None:
        """Connect to the LiveKit room and subscribe to existing/new tracks."""
        if not _HAS_LIVEKIT:
            raise RuntimeError(
                "livekit package is not installed; "
                "install the 'livekit' extra to use LiveKitParticipantAdapter"
            )
        if self._room is not None:
            return
        self._stopping = False
        self._room = rtc.Room()

        @self._room.on("track_subscribed")
        def _on_track_subscribed(
            track: Any, publication: Any, participant: Any
        ) -> None:
            self._on_track_subscribed(track, publication, participant)

        @self._room.on("track_unsubscribed")
        def _on_track_unsubscribed(track: Any, publication: Any) -> None:
            self._on_track_unsubscribed(track, publication)

        await self._room.connect(
            self.room_url,
            self.token,
            rtc.RoomOptions(auto_subscribe=self.auto_subscribe),
        )
        self.stats.connected = True
        logger.info(
            "livekit_adapter_connected",
            extra={
                "session_id": self.session_id,
                "room": getattr(self._room, "name", None),
            },
        )

        # Subscribe to tracks published before the worker joined.
        for participant in self._room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = getattr(publication, "track", None)
                if track is not None:
                    self._on_track_subscribed(track, publication, participant)

    # -------------------------------------------------------------- subscribe

    def _on_track_subscribed(
        self, track: Any, publication: Any, participant: Any
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        # Ignore tracks we own (future Tutor audio) by checking source.
        source = getattr(publication, "source", None)
        if source is not None and source == rtc.TrackSource.SOURCE_MICROPHONE:
            # Student mic — keep subscribing. (Worker's own Tutor track uses a
            # different source, so this filter is safe.)
            pass
        track_sid = str(getattr(publication, "sid", None) or id(track))
        if track_sid in self._streams:
            return
        self.stats.tracks_subscribed += 1
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=CHANNELS)
        self._streams[track_sid] = stream
        task = asyncio.create_task(self._consume_stream(track_sid, stream))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.info(
            "livekit_adapter_track_subscribed",
            extra={
                "session_id": self.session_id,
                "track_sid": str(track_sid),
                "participant": getattr(participant, "identity", None),
            },
        )

    def _on_track_unsubscribed(self, track: Any, publication: Any) -> None:
        track_sid = str(getattr(publication, "sid", None) or id(track))
        stream = self._streams.pop(track_sid, None)
        if stream is not None:
            asyncio.create_task(self._aclose_stream(stream))

    async def _aclose_stream(self, stream: Any) -> None:
        try:
            await stream.aclose()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("livekit_adapter_stream_close_error: %s", exc)

    # -------------------------------------------------------------- consume

    async def _consume_stream(self, track_sid: str, stream: Any) -> None:
        try:
            async for frame_event in stream:
                if self._stopping:
                    break
                frame = frame_event.frame
                self.stats.frames_received += 1
                pcm = frame.data
                # Safety net: normalize any non-canonical frame.
                if (
                    frame.sample_rate != SAMPLE_RATE
                    or frame.num_channels != CHANNELS
                ):
                    self.stats.resampled += 1
                    pcm = _resample_to_16k_mono(
                        pcm, frame.sample_rate, frame.num_channels
                    )
                self._maybe_mark_stall()
                epoch = getattr(self.worker, "epoch", 0)
                frames = self._segmenter.feed(pcm, epoch)
                for canonical in frames:
                    pushed = await self.worker.push_frame(canonical)
                    if pushed:
                        self.stats.frames_pushed += 1
                        if (
                            self.metrics is not None
                            and self.metrics.record_first_packet()
                        ):
                            logger.info(
                                "trace_first_packet",
                                extra={
                                    "session_id": self.session_id,
                                    "track_sid": track_sid,
                                    "seq": canonical.seq,
                                    "pts_us": canonical.pts_us,
                                },
                            )
                    else:
                        self.stats.push_dropped += 1
                self._last_recv_monotonic_us = time.monotonic_ns() / 1000.0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._stopping:
                logger.warning(
                    "livekit_adapter_stream_error",
                    extra={"session_id": self.session_id, "error": str(exc)},
                )

    def _maybe_mark_stall(self) -> None:
        now_us = time.monotonic_ns() / 1000.0
        if (
            self._last_recv_monotonic_us
            and now_us - self._last_recv_monotonic_us > self._stall_threshold_us
        ):
            self._segmenter.mark_discontinuity()

    # ------------------------------------------------------------------ stop

    async def stop(self) -> None:
        """Cancel stream consumers, close streams, and disconnect the room."""
        self._stopping = True
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for stream in list(self._streams.values()):
            await self._aclose_stream(stream)
        self._streams.clear()

        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("livekit_adapter_disconnect_error: %s", exc)
            self._room = None
        self.stats.connected = False

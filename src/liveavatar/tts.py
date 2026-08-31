"""NVC streaming TTS adapter — drop-in replacement for FakeTts.

Provides the same sync interface as ``FakeTts`` (``synthesize``,
``cancel_epoch``, ``pop_played_segments``) so it can be injected into
the current ``RealtimeWorker`` without code changes.

Additionally exposes ``synthesize_stream()`` — an async generator that
yields ``TtsSegment`` objects incrementally. This is the primary
interface for the Step 4 worker modification that replaces the blocking
``synthesize()`` call with a background streaming task.

Two construction modes
----------------------
1. **Direct mode**: pass a pre-built ``NvcWorker`` (or any duck-typed
   object with ``_build_request``, ``_to_canonical_pcm``,
   ``synthesize_stream`` and ``_tts.run``). Used for testing and when
   the caller manages the VoicePool lease externally.

2. **Pool mode**: pass a ``VoicePool`` + ``session_id`` + ``char_id``.
   The adapter acquires/releases the lease via ``acquire()`` / ``release()``.
   Used by ``LiveKitWorkerRuntime`` which owns the session lifecycle.

Sync vs async trade-off
-----------------------
The sync ``synthesize()`` calls ``tts.run()`` directly and blocks the
event loop during inference. This is acceptable for the transitional
period (Step 3) where the worker still calls ``synthesize()``
synchronously.

The async ``synthesize_stream()`` delegates to
``NvcWorker.synthesize_stream()`` which runs torch inference in a
thread via ``asyncio.to_thread``, keeping the event loop responsive.
This is the production path after the Step 4 worker modification.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any

# Optional: voice_pool imports (LiveAvatar voice package)
try:
    from liveavatar.voice.lease import CancelToken
    from liveavatar.voice.pool import VoicePool
    from liveavatar.voice.worker import NvcWorker

    _HAS_VOICE_POOL = True
except ImportError:
    _HAS_VOICE_POOL = False

    # Fallback definitions so the module is importable without voice_pool.
    class CancelToken:  # type: ignore[no-redef]
        """Stub CancelToken when voice_pool is not available."""

        def __init__(self) -> None:
            self._cancelled = False

        def cancel(self) -> None:
            self._cancelled = True

        @property
        def cancelled(self) -> bool:
            return self._cancelled

    VoicePool = Any  # type: ignore[misc, assignment]
    NvcWorker = Any  # type: ignore[misc, assignment]

logger = logging.getLogger("liveavatar.tts")


@dataclass(slots=True)
class TtsSegment:
    """A single TTS audio segment.

    Field-compatible with ``FakeTtsSegment`` so the worker can consume
    either interchangeably via duck typing.
    """

    segment_seq: int
    text: str
    epoch: int
    pcm_s16le: bytes
    pts_us: int
    duration_us: int


# ────────────────────────────────────────────── canonical constants

_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # S16LE
_FRAME_DURATION_US = 1_000_000  # for duration calc


class NvcStreamingTtsAdapter:
    """Drop-in replacement for ``FakeTts`` backed by an NVC TTS worker.

    Parameters
    ----------
    worker : NvcWorker | None
        Pre-built worker (direct mode). If ``None``, ``pool`` must be
        provided and ``acquire()`` called before synthesis.
    pool : VoicePool | None
        Voice pool for lease management (pool mode). If ``None``,
        ``worker`` must be provided.
    session_id : str
        Session ID for pool mode lease management.
    char_id : str
        Character ID for pool mode lease management.
    sample_rate : int
        Canonical output sample rate (default 16000).
    text_lang : str
        Default language for synthesis (default ``"zh"``).
    """

    def __init__(
        self,
        *,
        worker: Any = None,
        pool: Any = None,
        session_id: str = "",
        char_id: str = "",
        sample_rate: int = _SAMPLE_RATE,
        text_lang: str = "zh",
    ) -> None:
        if worker is None and pool is None:
            raise ValueError("must provide either 'worker' or 'pool'")

        self._pool = pool
        self._session_id = session_id
        self._char_id = char_id
        self._sample_rate = sample_rate
        self._default_text_lang = text_lang

        # In direct mode, worker is set immediately.
        # In pool mode, worker is set after acquire().
        self._worker: Any = worker
        self._lease: Any = None

        # Epoch tracking — cancels segments from old epochs.
        self._current_epoch: int = 0
        self._cancel_token: CancelToken = CancelToken()

        # Segment tracking (mirrors FakeTts for pop_played_segments).
        self._segment_counter: int = 0
        self._active_segments: list[TtsSegment] = []

    # ----------------------------------------------------- lifecycle

    async def acquire(self) -> None:
        """Acquire a lease from the VoicePool (pool mode only).

        In direct mode this is a no-op.
        """
        if self._pool is None:
            return
        if self._worker is not None and self._lease is not None:
            # Already acquired — renew.
            self._lease = await self._pool.acquire(
                self._session_id, self._char_id
            )
            return
        self._lease = await self._pool.acquire(self._session_id, self._char_id)
        self._worker = self._lease.worker
        logger.info(
            "adapter_acquired",
            extra={
                "session_id": self._session_id,
                "char_id": self._char_id,
            },
        )

    async def release(self) -> None:
        """Release the VoicePool lease (pool mode only)."""
        if self._pool is not None and self._lease is not None:
            await self._pool.release_async(self._session_id)
            self._lease = None
            # Keep self._worker for reference, but mark as released.
            logger.info(
                "adapter_released",
                extra={"session_id": self._session_id},
            )

    @property
    def char_id(self) -> str:
        """The character ID of the bound worker."""
        if self._worker is not None:
            return getattr(self._worker, "char_id", self._char_id)
        return self._char_id

    @property
    def has_worker(self) -> bool:
        return self._worker is not None

    @property
    def active_segment_count(self) -> int:
        return len(self._active_segments)

    # ────────────────────────────── sync interface (FakeTts compat)

    def synthesize(
        self,
        text: str,
        epoch: int,
        pts_us: int,
        *,
        text_lang: str = "",
    ) -> list[TtsSegment]:
        """Synchronous synthesis — blocks during TTS inference.

        Returns a list of ``TtsSegment`` objects (one per TTS chunk).
        PTS advances across chunks so the worker can enqueue them
        sequentially.

        .. note::
            This call blocks the event loop. For non-blocking streaming,
            use :meth:`synthesize_stream` instead.
        """
        if epoch < self._current_epoch:
            logger.debug("synthesize_skipped_old_epoch", extra={"epoch": epoch})
            return []

        worker = self._require_worker()
        lang = text_lang or self._default_text_lang

        # Fresh cancel token for this synthesis.
        self._cancel_token = CancelToken()

        # Build the TTS request using the worker's method.
        req = worker._build_request(
            text=text,
            text_lang=lang,
            speed_factor=1.0,
            top_k=15,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.35,
        )

        # Run the synchronous TTS generator directly.
        # This is safe because the event loop is blocked (no concurrency).
        gen: Generator = worker._tts.run(req)

        segments: list[TtsSegment] = []
        current_pts = pts_us
        for sr, audio_np in gen:
            if self._cancel_token.cancelled:
                logger.info(
                    "synthesize_cancelled_mid_stream",
                    extra={"epoch": epoch, "segments_so_far": len(segments)},
                )
                break
            pcm = worker._to_canonical_pcm(audio_np, sr)
            self._segment_counter += 1
            duration_us = len(pcm) * _FRAME_DURATION_US // (
                self._sample_rate * _BYTES_PER_SAMPLE
            )
            seg = TtsSegment(
                segment_seq=self._segment_counter,
                text=text,
                epoch=epoch,
                pcm_s16le=pcm,
                pts_us=current_pts,
                duration_us=duration_us,
            )
            segments.append(seg)
            self._active_segments.append(seg)
            current_pts += duration_us

        return segments

    def cancel_epoch(self, epoch: int) -> int:
        """Cancel all segments with ``epoch < given`` and stop active synthesis.

        Returns the number of segments removed.
        """
        self._current_epoch = epoch
        self._cancel_token.cancel()

        before = len(self._active_segments)
        self._active_segments = [
            s for s in self._active_segments if s.epoch >= epoch
        ]
        removed = before - len(self._active_segments)

        if removed:
            logger.info(
                "cancel_epoch",
                extra={"epoch": epoch, "removed": removed},
            )
        return removed

    def pop_played_segments(self, consumed_pts_us: int) -> list[TtsSegment]:
        """Remove and return segments that have been fully played."""
        played = [
            s
            for s in self._active_segments
            if s.pts_us + s.duration_us <= consumed_pts_us
        ]
        self._active_segments = [
            s
            for s in self._active_segments
            if s.pts_us + s.duration_us > consumed_pts_us
        ]
        return played

    # ──────────────────────── async streaming interface (Step 4)

    async def synthesize_stream(
        self,
        text: str,
        epoch: int,
        pts_us: int,
        *,
        text_lang: str = "",
        speed_factor: float = 1.0,
    ) -> AsyncGenerator[TtsSegment, None]:
        """Async streaming synthesis — yields segments without blocking.

        Delegates to ``NvcWorker.synthesize_stream()`` which runs torch
        inference in a thread via ``asyncio.to_thread``. The event loop
        stays responsive for ASR/VAD/EOU processing and control events.

        Yields
        ------
        TtsSegment
            One per TTS chunk, with monotonically increasing PTS.
        """
        if epoch < self._current_epoch:
            return

        worker = self._require_worker()
        lang = text_lang or self._default_text_lang

        # Fresh cancel token for this synthesis.
        self._cancel_token = CancelToken()

        current_pts = pts_us
        async for pcm in worker.synthesize_stream(
            text,
            cancel_token=self._cancel_token,
            text_lang=lang,
            speed_factor=speed_factor,
        ):
            # Check if epoch advanced while we were waiting for the next chunk.
            if epoch < self._current_epoch:
                logger.info(
                    "stream_cancelled_epoch_advanced",
                    extra={"epoch": epoch, "current": self._current_epoch},
                )
                break

            self._segment_counter += 1
            duration_us = len(pcm) * _FRAME_DURATION_US // (
                self._sample_rate * _BYTES_PER_SAMPLE
            )
            seg = TtsSegment(
                segment_seq=self._segment_counter,
                text=text,
                epoch=epoch,
                pcm_s16le=pcm,
                pts_us=current_pts,
                duration_us=duration_us,
            )
            self._active_segments.append(seg)
            yield seg
            current_pts += duration_us

    # ──────────────────────────────────────────────── internals

    def _require_worker(self) -> Any:
        """Return the bound worker or raise."""
        if self._worker is None:
            raise RuntimeError(
                "no worker available; call acquire() first (pool mode) "
                "or pass worker= in the constructor (direct mode)"
            )
        return self._worker

    def to_dict(self) -> dict:
        """Snapshot for logging/debugging."""
        return {
            "char_id": self.char_id,
            "has_worker": self.has_worker,
            "current_epoch": self._current_epoch,
            "segment_counter": self._segment_counter,
            "active_segments": len(self._active_segments),
            "pool_mode": self._pool is not None,
        }

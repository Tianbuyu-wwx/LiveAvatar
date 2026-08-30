"""AvatarStreamingAdapter: PCM chunk → AvatarWorker → AvatarVideoPublisher.

Sits between the audio producer (e.g. a streaming TTS worker) and the
AvatarVideoPublisher (which captures frames into a LiveKit video track).
It owns the AvatarLease lifecycle and converts the worker's async generator
into background-task consumption so the caller's event loop stays free.

Lifecycle::

    adapter = AvatarStreamingAdapter(
        pool=avatar_pool, publisher=video_publisher,
        session_id=..., avatar_id=...,
    )
    await adapter.start()                  # acquire lease + start consumer
    await adapter.push_pcm(pcm, pts_us, epoch)  # feed a TTS PCM chunk
    adapter.cancel_epoch(new_epoch)        # interrupt: stop stale-epoch frames
    await adapter.stop()                   # cancel consumer + release lease

Cancellation model
------------------
Each ``push_pcm`` call carries the epoch at which the PCM was synthesized.
The adapter forwards the epoch to ``AvatarWorker.synthesize_video_stream``
and to ``AvatarVideoPublisher.publish_frame``. When ``cancel_epoch`` is
called (typically when the caller advances its epoch on an interrupt), the
adapter:

1. Cancels the in-flight ``CancelToken`` — the worker's
   ``synthesize_video_stream`` breaks out of its generator loop promptly.
2. Advances the publisher's epoch — stale-epoch frames still in flight
   through the publisher are dropped before ``capture_frame``.

This two-pronged cancellation ensures that an interrupt stops video within
one frame (≤ 40ms at 25 fps) of the confirmed interrupt, matching the
interrupt latency on the audio side.

Degradation chain
-----------------
If the worker's ``_infer_batch`` raises an exception or consistently times
out, the adapter switches to a fallback worker (typically a
``StaticAvatarWorker``) after ``degrade_after_errors`` consecutive errors.
A subsequent ``reset_degradation()`` (called on epoch advance) restores
the primary worker.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .lease import CancelToken
from .video_publisher import AvatarVideoPublisher
from .worker import AvatarFrame, AvatarWorker

logger = logging.getLogger("liveavatar.adapter")


@dataclass
class AvatarAdapterStats:
    """Counters for the AvatarStreamingAdapter."""

    pcm_chunks_pushed: int = 0
    frames_produced: int = 0
    frames_published: int = 0
    frames_dropped_epoch: int = 0
    frames_dropped_error: int = 0
    cancel_count: int = 0
    inference_errors: int = 0
    consecutive_errors: int = 0
    degradation_count: int = 0
    degraded: bool = False
    queue_high_water: int = 0
    lease_acquired: bool = False


@dataclass
class _PendingChunk:
    """A PCM chunk pending inference, queued by ``push_pcm``."""

    pcm_s16le: bytes
    pts_us: int
    epoch: int


class AvatarStreamingAdapter:
    """Stream TTS PCM chunks through an AvatarWorker into a video track.

    Parameters
    ----------
    pool : Any | None
        AvatarPool for lease management. When ``None``, ``worker`` must be
        supplied directly (direct mode — caller owns the worker).
    publisher : AvatarVideoPublisher | None
        The LiveKit video publisher. When ``None``, frames are produced but
        not published (testing mode — capture them via ``published_frames``).
    session_id, avatar_id :
        Lease identification.
    worker : AvatarWorker | None
        Pre-bound worker (direct mode — skips pool acquire/release).
    fallback_worker : AvatarWorker | None
        Degradation fallback. When the primary worker errors out, the
        adapter switches to this worker automatically.
    degrade_after_errors : int
        Number of consecutive errors before switching to the fallback.
    queue_capacity : int
        Maximum pending PCM chunks. ``push_pcm`` blocks (with timeout)
        when full to apply backpressure on the TTS stream.
    """

    def __init__(
        self,
        *,
        pool: Any = None,
        publisher: AvatarVideoPublisher | None = None,
        session_id: str = "",
        avatar_id: str = "",
        worker: AvatarWorker | None = None,
        fallback_worker: AvatarWorker | None = None,
        degrade_after_errors: int = 3,
        queue_capacity: int = 32,
        push_timeout_s: float = 0.05,
    ) -> None:
        if pool is None and worker is None:
            raise ValueError(
                "AvatarStreamingAdapter requires either pool= or worker="
            )
        self._pool = pool
        self._publisher = publisher
        self._session_id = session_id
        self._avatar_id = avatar_id
        self._fallback_worker = fallback_worker
        self._degrade_after_errors = max(1, degrade_after_errors)
        self._push_timeout_s = push_timeout_s

        # Direct mode: caller-supplied worker. Pool mode: leased at start().
        self._direct_worker = worker
        self._leased_worker: AvatarWorker | None = None
        self._lease: Any = None

        # Pending PCM queue — consumed by the background inference task.
        self._queue: asyncio.Queue[_PendingChunk] = asyncio.Queue(
            maxsize=queue_capacity
        )
        self._consumer_task: asyncio.Task | None = None
        self._consumer_loop: asyncio.AbstractEventLoop | None = None
        self._stopped = False
        self._cancel_token: CancelToken | None = None

        # Test/observability hooks.
        self.stats = AvatarAdapterStats()
        # When publisher is None, capture produced frames here for tests.
        self.published_frames: list[AvatarFrame] = []

        # Track the latest epoch seen for stale-chunk filtering.
        self._current_epoch = 0

    # ---------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Acquire the lease (pool mode) and start the background consumer."""
        if self._consumer_task is not None:
            return  # already started

        if self._direct_worker is None:
            if self._pool is None:
                raise RuntimeError(
                    "AvatarStreamingAdapter: pool required when no direct worker"
                )
            self._lease = await self._pool.acquire(
                self._session_id, self._avatar_id
            )
            self._leased_worker = self._lease.worker
            self.stats.lease_acquired = True
        else:
            self._leased_worker = self._direct_worker

        self._stopped = False
        self._cancel_token = CancelToken()
        self._spawn_consumer()
        logger.info(
            "avatar_adapter_started",
            extra={
                "session_id": self._session_id,
                "avatar_id": self._avatar_id,
                "worker": self._active_worker.avatar_id,
                "pool_mode": self._pool is not None,
            },
        )

    def _spawn_consumer(self) -> None:
        """Start the background consumer on the current event loop.

        Called from ``start()`` and re-spawned lazily from ``push_pcm`` when
        the adapter is driven from a different loop than the one it started
        on (e.g. embedded in a request-per-loop test client).
        """
        self._consumer_task = asyncio.create_task(self._consume_loop())
        self._consumer_loop = asyncio.get_running_loop()

    async def stop(self) -> None:
        """Cancel the consumer and release the lease (pool mode)."""
        self._stopped = True
        if self._cancel_token is not None:
            self._cancel_token.cancel()
        task = self._consumer_task
        self._consumer_task = None
        if task is not None:
            try:
                task.cancel()
            except Exception:  # pragma: no cover - cross-loop edge cases
                pass
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            # Only await tasks owned by the current loop.
            if (
                current_loop is not None
                and task.get_loop() is current_loop
                and not task.done()
            ):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._lease is not None and self._pool is not None:
            try:
                await self._pool.release_async(self._session_id)
            except Exception:
                logger.exception(
                    "avatar_adapter_release_failed",
                    extra={"session_id": self._session_id},
                )
            self._lease = None
            self._leased_worker = None
            self.stats.lease_acquired = False

        logger.info(
            "avatar_adapter_stopped",
            extra={
                "session_id": self._session_id,
                "frames_published": self.stats.frames_published,
                "degraded": self.stats.degraded,
            },
        )

    # ----------------------------------------------------------- push_pcm

    async def push_pcm(
        self, pcm_s16le: bytes, pts_us: int, epoch: int
    ) -> bool:
        """Enqueue one PCM chunk for avatar inference.

        Returns True if enqueued, False if dropped due to:
        - Stale epoch (chunk's epoch < current epoch).
        - Queue full (backpressure — TTS is outpacing Avatar inference).

        Stale-epoch chunks are dropped here so the consumer doesn't waste
        GPU cycles on cancelled speech. The publisher also drops stale-epoch
        frames as a second line of defense.
        """
        if epoch < self._current_epoch:
            self.stats.frames_dropped_epoch += 1
            return False

        if self._stopped:
            return False

        # Re-spawn the consumer if it died or was started on a different
        # event loop than the one pushing right now.
        task = self._consumer_task
        if (
            task is None
            or task.done()
            or self._consumer_loop is not asyncio.get_running_loop()
        ):
            self._spawn_consumer()

        chunk = _PendingChunk(pcm_s16le=pcm_s16le, pts_us=pts_us, epoch=epoch)
        try:
            await asyncio.wait_for(
                self._queue.put(chunk), timeout=self._push_timeout_s
            )
        except asyncio.TimeoutError:
            # Queue full — drop this chunk rather than blocking TTS.
            # Audio is the master clock; Avatar must never slow it down.
            self.stats.frames_dropped_error += 1
            return False

        self.stats.pcm_chunks_pushed += 1
        # Track high-water mark for observability.
        qsize = self._queue.qsize()
        if qsize > self.stats.queue_high_water:
            self.stats.queue_high_water = qsize
        return True

    # ------------------------------------------------------- cancel_epoch

    def cancel_epoch(self, new_epoch: int) -> None:
        """Advance the cancellation epoch.

        1. Filter future ``push_pcm`` calls with stale epochs.
        2. Cancel the in-flight ``CancelToken`` — the consumer's current
           ``synthesize_video_stream`` generator breaks out promptly.
        3. Forward to the publisher so stale frames still in capture are
           dropped.
        4. Reset the degradation flag — a new epoch is a fresh start.
        """
        if new_epoch <= self._current_epoch:
            return
        self._current_epoch = new_epoch

        if self._cancel_token is not None:
            self._cancel_token.cancel()
            self.stats.cancel_count += 1
        # Replace the token for the next chunk.
        self._cancel_token = CancelToken()

        if self._publisher is not None:
            self._publisher.cancel_epoch(new_epoch)

        # Drain any queued stale chunks.
        drained = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            self.stats.frames_dropped_epoch += drained

        # Reset degradation — the next epoch starts fresh.
        if self.stats.degraded:
            self.stats.degraded = False
            self.stats.consecutive_errors = 0
            logger.info(
                "avatar_degradation_reset",
                extra={"session_id": self._session_id, "epoch": new_epoch},
            )

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def _active_worker(self) -> AvatarWorker:
        """Return the worker currently in use (primary or fallback)."""
        if self.stats.degraded and self._fallback_worker is not None:
            return self._fallback_worker
        assert self._leased_worker is not None, "start() not called"
        return self._leased_worker

    # -------------------------------------------------------- consume loop

    async def _consume_loop(self) -> None:
        """Background task: drain the PCM queue → infer → publish frames."""
        try:
            while True:
                chunk = await self._queue.get()
                # Re-check epoch after dequeue — chunk may have gone stale
                # while waiting in the queue.
                if chunk.epoch < self._current_epoch:
                    self.stats.frames_dropped_epoch += 1
                    continue

                await self._process_chunk(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "avatar_consume_loop_crashed",
                extra={"session_id": self._session_id},
            )

    async def _process_chunk(self, chunk: _PendingChunk) -> None:
        """Infer one PCM chunk and publish its frames.

        MuseTalk's ``_infer_batch`` consumes ``batch_size / target_fps``
        seconds of audio per call (e.g. 4 frames @ 25 fps = 160 ms). Split the
        incoming TTS PCM into those fixed-size sub-chunks and feed them
        sequentially so the full duration is rendered.
        """
        worker = self._active_worker
        token = self._cancel_token
        assert token is not None

        sample_rate = 16000
        samples_per_batch = int(worker.batch_size * sample_rate / worker.target_fps)
        bytes_per_batch = samples_per_batch * 2
        pts_increment_us = worker.batch_size * 1_000_000 // worker.target_fps
        current_pts = chunk.pts_us

        try:
            for offset in range(0, len(chunk.pcm_s16le), bytes_per_batch):
                sub = chunk.pcm_s16le[offset : offset + bytes_per_batch]
                # Pad final partial chunk with silence so MuseTalk always sees
                # a full audio batch.
                if len(sub) < bytes_per_batch:
                    sub = sub + b"\x00" * (bytes_per_batch - len(sub))

                async for frame in worker.synthesize_video_stream(
                    sub,
                    pts_us=current_pts,
                    epoch=chunk.epoch,
                    cancel_token=token,
                ):
                    # Re-check epoch before publishing — interrupt may have
                    # fired mid-stream.
                    if chunk.epoch < self._current_epoch:
                        self.stats.frames_dropped_epoch += 1
                        break

                    self.stats.frames_produced += 1
                    await self._publish_frame(frame, chunk.epoch)

                current_pts += pts_increment_us

            # Successful inference resets the error streak.
            self.stats.consecutive_errors = 0
        except asyncio.CancelledError:
            raise
        except Exception:
            self.stats.inference_errors += 1
            self.stats.consecutive_errors += 1
            self.stats.frames_dropped_error += 1
            logger.exception(
                "avatar_inference_error",
                extra={
                    "session_id": self._session_id,
                    "avatar_id": worker.avatar_id,
                    "consecutive_errors": self.stats.consecutive_errors,
                    "epoch": chunk.epoch,
                },
            )
            # Trigger degradation if the threshold is hit and a fallback exists.
            if (
                not self.stats.degraded
                and self._fallback_worker is not None
                and self.stats.consecutive_errors >= self._degrade_after_errors
            ):
                self.stats.degraded = True
                self.stats.degradation_count += 1
                logger.warning(
                    "avatar_degraded_to_fallback",
                    extra={
                        "session_id": self._session_id,
                        "primary": worker.avatar_id,
                        "fallback": self._fallback_worker.avatar_id,
                        "consecutive_errors": self.stats.consecutive_errors,
                    },
                )

    async def _publish_frame(self, frame: AvatarFrame, epoch: int) -> None:
        """Publish one frame via the publisher, or capture it for tests."""
        if self._publisher is None:
            self.published_frames.append(frame)
            self.stats.frames_published += 1
            return
        published = await self._publisher.publish_frame(frame, epoch)
        if published:
            self.stats.frames_published += 1
        elif self._publisher.current_epoch > epoch:
            self.stats.frames_dropped_epoch += 1

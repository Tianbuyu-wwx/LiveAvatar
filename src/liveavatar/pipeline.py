"""AvatarPipeline — high-level orchestrator for one or many sessions.

Wires together the pieces the RealtimeWorker runtime used to manage:

    audio source (WS / file)
        → AvatarStreamingAdapter.push_pcm(pcm, pts_us, epoch)
            → AvatarPool.acquire() lease → AvatarWorker (MuseTalk)
            → publisher sink (self-developed WebSocketSink / capture)

The pipeline owns:
- the AvatarPool lifecycle (start/stop, worker loading),
- per-session adapter + publisher creation,
- epoch advancement on interrupts,
- session stats for observability.

Example (pool mode, WS publishing)::

    pipeline = AvatarPipeline(AvatarPoolConfig())
    await pipeline.start()
    await pipeline.open_session("s1", "yongen")
    await pipeline.push_pcm("s1", pcm_chunk, pts_us=0, epoch=0)
    pipeline.cancel_epoch("s1", 1)          # interrupt
    await pipeline.close_session("s1")
    await pipeline.stop()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .adapter import AvatarStreamingAdapter
from .config import AvatarPoolConfig
from .pool import AvatarPool

logger = logging.getLogger("liveavatar.pipeline")


@dataclass
class SessionState:
    """Per-session runtime objects owned by the pipeline."""

    session_id: str
    avatar_id: str
    adapter: AvatarStreamingAdapter | None = None
    # PublishSink-compatible (WebSocketSink); None = capture mode (tests).
    publisher: Any | None = None
    # Monotonic PTS clock for sources that don't carry their own timestamps
    # (e.g. WebSocket PCM): advanced by the chunk duration on every push.
    next_pts_us: int = 0
    epoch: int = 0
    samples_pushed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class AvatarPipeline:
    """Orchestrates the avatar pool and per-session streaming adapters.

    Parameters
    ----------
    config : AvatarPoolConfig
        Pool configuration (avatar_data_root, device, batch_size, ...).
    pool : AvatarPool | None
        Pre-built pool (testing). When ``None`` one is created from config.
    publisher_factory : Callable[[SessionState], Publisher-like] | None
        Optional factory overriding publisher creation — used by tests and
        by the local preview sink. The factory receives the session state
        and must return an object with ``publish_frame`` / ``cancel_epoch``
        / ``current_epoch`` (i.e. AvatarVideoPublisher-compatible).
    """

    def __init__(
        self,
        config: AvatarPoolConfig,
        *,
        pool: AvatarPool | None = None,
        publisher_factory: Callable[[SessionState], Any] | None = None,
    ) -> None:
        self._config = config
        self._pool = pool if pool is not None else AvatarPool(config)
        self._publisher_factory = publisher_factory
        self._sessions: dict[str, SessionState] = {}
        self._started = False

    # ------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Start the avatar pool (loads preloaded avatars, starts reaper)."""
        if self._started:
            return
        await self._pool.start()
        self._started = True
        logger.info(
            "pipeline_started",
            extra={"avatars": self._pool.available_avatars},
        )

    async def stop(self) -> None:
        """Close all sessions and stop the pool."""
        for session_id in list(self._sessions.keys()):
            await self.close_session(session_id)
        await self._pool.stop()
        self._started = False
        logger.info("pipeline_stopped")

    # -------------------------------------------------------- sessions

    async def open_session(
        self,
        session_id: str,
        avatar_id: str,
        *,
        width: int | None = None,
        height: int | None = None,
        target_fps: int | None = None,
        queue_capacity: int = 32,
        degrade_after_errors: int = 3,
    ) -> SessionState:
        """Create the publisher + adapter for a session.

        Without a ``publisher_factory`` the session runs in capture mode
        (publisher None) for testing / local tooling.
        """
        if session_id in self._sessions:
            raise ValueError(f"session '{session_id}' already open")

        state = SessionState(session_id=session_id, avatar_id=avatar_id)

        # Publisher: custom factory or capture-mode (None).
        if self._publisher_factory is not None:
            state.publisher = self._publisher_factory(state)

        if state.publisher is not None and hasattr(state.publisher, "start"):
            await state.publisher.start()

        state.adapter = AvatarStreamingAdapter(
            pool=self._pool,
            publisher=state.publisher,
            session_id=session_id,
            avatar_id=avatar_id,
            queue_capacity=queue_capacity,
            degrade_after_errors=degrade_after_errors,
        )
        await state.adapter.start()
        self._sessions[session_id] = state
        logger.info(
            "pipeline_session_open",
            extra={"session_id": session_id, "avatar_id": avatar_id},
        )
        return state

    async def close_session(self, session_id: str) -> bool:
        """Stop the session adapter and unpublish its video track."""
        state = self._sessions.pop(session_id, None)
        if state is None:
            return False
        if state.adapter is not None:
            await state.adapter.stop()
        if state.publisher is not None and hasattr(state.publisher, "stop"):
            try:
                await state.publisher.stop()
            except Exception:
                logger.exception(
                    "pipeline_publisher_stop_failed",
                    extra={"session_id": session_id},
                )
        logger.info("pipeline_session_closed", extra={"session_id": session_id})
        return True

    @property
    def pool(self) -> AvatarPool:
        """The shared avatar pool (reused by duplex sessions)."""
        return self._pool

    @property
    def sessions(self) -> dict[str, SessionState]:
        return dict(self._sessions)

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    # ------------------------------------------------------------ audio

    async def push_pcm(
        self,
        session_id: str,
        pcm_s16le: bytes,
        *,
        pts_us: int | None = None,
        epoch: int | None = None,
        sample_rate: int = 16000,
    ) -> bool:
        """Push one PCM chunk into the session's avatar stream.

        When ``pts_us`` is None the session's monotonic PTS clock is used
        (advanced by the chunk's duration). When ``epoch`` is None the
        session's current epoch is used.
        """
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"session '{session_id}' not open")
        assert state.adapter is not None, "session opened without adapter"

        if pts_us is None:
            pts_us = state.next_pts_us
            state.next_pts_us += (
                len(pcm_s16le) // 2
            ) * 1_000_000 // sample_rate
        if epoch is None:
            epoch = state.epoch

        pushed = await state.adapter.push_pcm(pcm_s16le, pts_us, epoch)
        if pushed:
            state.samples_pushed += len(pcm_s16le) // 2
        return pushed

    def cancel_epoch(self, session_id: str, new_epoch: int) -> None:
        """Interrupt: advance the session's epoch (drops stale audio+frames)."""
        state = self._sessions.get(session_id)
        if state is None:
            return
        assert state.adapter is not None, "session opened without adapter"
        state.epoch = max(state.epoch, new_epoch)
        state.adapter.cancel_epoch(state.epoch)
        logger.info(
            "pipeline_epoch_cancelled",
            extra={"session_id": session_id, "epoch": state.epoch},
        )

    # ------------------------------------------------------- inspection

    def session_stats(self, session_id: str) -> dict[str, Any]:
        """Snapshot of one session's adapter + publisher stats."""
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"session '{session_id}' not open")
        assert state.adapter is not None, "session opened without adapter"
        stats: dict[str, Any] = {
            "session_id": session_id,
            "avatar_id": state.avatar_id,
            "epoch": state.epoch,
            "samples_pushed": state.samples_pushed,
            "adapter": vars(state.adapter.stats),
        }
        pub = state.publisher
        if pub is not None and hasattr(pub, "stats"):
            pub_stats = pub.stats
            # PublishSink protocol defines stats() as a method; legacy
            # publishers expose a dataclass attribute.
            stats["publisher"] = pub_stats() if callable(pub_stats) else vars(
                pub_stats
            )
        return stats

    def stats(self) -> dict[str, Any]:
        """Snapshot of the whole pipeline (pool + sessions)."""
        return {
            "pool": self._pool.stats(),
            "sessions": [sid for sid in self._sessions],
        }

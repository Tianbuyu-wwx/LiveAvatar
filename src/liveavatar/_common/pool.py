# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Generic worker pool with lease management and fair queuing.

A pool manages a set of resource-pinned workers. Each worker is loaded once
and never switched, which prevents cross-talk between concurrent sessions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .lease import Lease

WorkerT = TypeVar("WorkerT")
AssetsT = TypeVar("AssetsT")


class PoolError(Exception):
    """Base exception for worker pool errors."""


class PoolExhausted(PoolError):
    """All workers for a resource are busy and the acquire timed out."""


class GpuMemoryExhausted(PoolError):
    """Cannot load another worker because ``max_workers`` has been reached."""


class ResourceNotFound(PoolError):
    """The requested resource ID has no assets available."""


@dataclass
class Waiter(Generic[WorkerT]):
    """A pending acquire request waiting in a FIFO queue."""

    session_id: str
    resource_id: str
    deadline: float
    future: asyncio.Future[Lease[WorkerT]] = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )

    @property
    def expired(self) -> bool:
        return time.monotonic() > self.deadline


class WorkerPool(Generic[WorkerT, AssetsT]):
    """Generic resource-pinned worker pool.

    Subclasses provide resource discovery and worker factory logic.

    Parameters
    ----------
    config:
        Pool configuration object. Common attributes read: ``max_workers``,
        ``lease_ttl``, ``reap_interval``, ``acquire_timeout``.
    worker_factory:
        Callable taking assets and returning a worker. When ``None``, the
        subclass ``_default_worker_factory`` is used.
    resources:
        Optional pre-built resource map. When ``None``,
        ``_discover_resources`` is called.
    logger:
        Optional logger instance.
    """

    _resource_not_found_error: type[ResourceNotFound] = ResourceNotFound
    _pool_exhausted_error: type[PoolExhausted] = PoolExhausted
    _gpu_memory_exhausted_error: type[GpuMemoryExhausted] = GpuMemoryExhausted

    def __init__(
        self,
        config: Any,
        *,
        worker_factory: Callable[[AssetsT], WorkerT] | None = None,
        resources: dict[str, AssetsT] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._worker_factory = worker_factory or self._default_worker_factory
        self._resources = resources if resources is not None else self._discover_resources()
        self._workers: dict[str, WorkerT] = {}
        self._loaded_at: dict[str, float] = {}
        self._leases: dict[str, Lease[WorkerT]] = {}
        self._waiters: dict[str, deque[Waiter[WorkerT]]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._reaper_task: asyncio.Task | None = None
        self._started = False
        self._logger = logger or logging.getLogger("liveavatar.pool")

    # ----------------------------------------------------- subclass hooks

    def _discover_resources(self) -> dict[str, AssetsT]:
        """Discover available resources. Subclasses must override."""
        raise NotImplementedError

    def _default_worker_factory(self, assets: AssetsT) -> WorkerT:
        """Create a worker from assets. Subclasses must override."""
        raise NotImplementedError

    @property
    def _resource_kind(self) -> str:
        """Human-readable resource kind for logs (e.g. ``avatar``)."""
        return "resource"

    @property
    def _preloaded_resource_ids(self) -> list[str]:
        """Resource IDs to preload on startup."""
        return []

    @property
    def _max_workers(self) -> int:
        return int(getattr(self._config, "max_workers", 2))

    @property
    def _lease_ttl(self) -> float:
        return float(getattr(self._config, "lease_ttl", 60.0))

    @property
    def _reap_interval(self) -> float:
        return float(getattr(self._config, "reap_interval", 10.0))

    @property
    def _acquire_timeout(self) -> float:
        return float(getattr(self._config, "acquire_timeout", 5.0))

    @property
    def _max_loaded_workers(self) -> int:
        """Cap on concurrently loaded workers (0 = unlimited)."""
        return int(getattr(self._config, "max_loaded_workers", 0))

    # ----------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Start the pool: preload configured resources and launch reaper."""
        if self._started:
            return
        self._started = True

        for resource_id in self._preloaded_resource_ids:
            if resource_id in self._resources:
                try:
                    await self._load_worker(resource_id)
                    self._logger.info(
                        "pool_resource_preloaded",
                        extra={"resource_id": resource_id, "kind": self._resource_kind},
                    )
                except Exception:
                    self._logger.exception(
                        "pool_preload_failed",
                        extra={"resource_id": resource_id, "kind": self._resource_kind},
                    )

        self._reaper_task = asyncio.create_task(self._reap_loop())
        self._logger.info(
            "pool_started",
            extra={
                "kind": self._resource_kind,
                "resources": sorted(self._resources.keys()),
                "max_workers": self._max_workers,
                "preloaded": sorted(self._workers.keys()),
            },
        )

    async def stop(self) -> None:
        """Stop the pool: cancel reaper, fail waiters, clear leases."""
        if not self._started:
            return
        self._started = False

        if self._reaper_task:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None

        for waiters in self._waiters.values():
            for waiter in waiters:
                if not waiter.future.done():
                    waiter.future.set_exception(PoolError("pool stopped"))
        self._waiters.clear()
        self._leases.clear()
        self._logger.info("pool_stopped", extra={"kind": self._resource_kind})

    # ------------------------------------------------------- acquire

    def _create_lease(
        self,
        session_id: str,
        resource_id: str,
        worker: WorkerT,
        ttl: float,
    ) -> Lease[WorkerT]:
        """Create a lease instance. Subclasses may override for typed leases."""
        return Lease[WorkerT].create(
            session_id=session_id,
            resource_id=resource_id,
            worker=worker,
            ttl=ttl,
        )

    async def acquire(
        self,
        session_id: str,
        resource_id: str,
        *,
        timeout: float | None = None,
    ) -> Lease[WorkerT]:
        """Acquire a lease on a resource worker for ``session_id``.

        If ``session_id`` already holds a lease for the same resource, the
        lease is renewed and returned. If it holds a lease for a different
        resource, the old lease is released first.
        """
        if resource_id not in self._resources:
            raise self._resource_not_found_error(
                f"{self._resource_kind} '{resource_id}' not found; "
                f"available: {sorted(self._resources.keys())}"
            )

        wait_timeout = timeout if timeout is not None else self._acquire_timeout

        async with self._lock:
            existing = self._leases.get(session_id)
            if existing:
                if existing.resource_id == resource_id:
                    existing.renew(self._lease_ttl)
                    return existing
                self._release_locked(session_id)

            worker = self._workers.get(resource_id)
            if worker is None:
                worker = await self._load_worker(resource_id)

            leased = any(
                lease.worker is worker and not lease.is_expired()
                for lease in self._leases.values()
            )
            if not leased:
                lease = self._create_lease(
                    session_id=session_id,
                    resource_id=resource_id,
                    worker=worker,
                    ttl=self._lease_ttl,
                )
                self._leases[session_id] = lease
                self._logger.info(
                    "pool_lease_acquired",
                    extra={
                        "session_id": session_id,
                        "resource_id": resource_id,
                        "lease_id": lease.lease_id,
                        "kind": self._resource_kind,
                    },
                )
                return lease

            waiter = Waiter[WorkerT](
                session_id=session_id,
                resource_id=resource_id,
                deadline=time.monotonic() + wait_timeout,
            )
            self._waiters[resource_id].append(waiter)

        try:
            return await asyncio.wait_for(waiter.future, timeout=wait_timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                try:
                    self._waiters[resource_id].remove(waiter)
                except ValueError:
                    pass
            raise self._pool_exhausted_error(
                f"acquire {self._resource_kind} '{resource_id}' "
                f"timed out after {wait_timeout}s"
            ) from None

    # ------------------------------------------------------- release

    def release(self, session_id: str) -> bool:
        """Release the lease held by ``session_id`` (fire-and-forget)."""
        if session_id not in self._leases:
            return False
        asyncio.create_task(self.release_async(session_id))
        return True

    async def release_async(self, session_id: str) -> bool:
        """Release the lease and wait for the release to complete."""
        if session_id not in self._leases:
            return False
        async with self._lock:
            self._release_locked(session_id)
        return True

    def _release_locked(self, session_id: str) -> None:
        """Release lease and dispatch to the next waiter (caller holds lock)."""
        lease = self._leases.pop(session_id, None)
        if lease is None:
            return
        self._logger.info(
            "pool_lease_released",
            extra={
                "session_id": session_id,
                "resource_id": lease.resource_id,
                "lease_id": lease.lease_id,
                "kind": self._resource_kind,
            },
        )
        self._dispatch_waiter_locked(lease.resource_id)

    def _dispatch_waiter_locked(self, resource_id: str) -> None:
        """Assign the worker to the next eligible waiter (caller holds lock)."""
        queue = self._waiters.get(resource_id)
        if not queue:
            return
        worker = self._workers.get(resource_id)
        if worker is None:
            return

        while queue:
            waiter = queue.popleft()
            if waiter.expired:
                if not waiter.future.done():
                    waiter.future.set_exception(
                        self._pool_exhausted_error("waiter expired")
                    )
                continue
            lease = self._create_lease(
                session_id=waiter.session_id,
                resource_id=resource_id,
                worker=worker,
                ttl=self._lease_ttl,
            )
            self._leases[waiter.session_id] = lease
            if not waiter.future.done():
                waiter.future.set_result(lease)
                self._logger.info(
                    "pool_waiter_fulfilled",
                    extra={
                        "session_id": waiter.session_id,
                        "resource_id": resource_id,
                        "kind": self._resource_kind,
                    },
                )
            return

    # --------------------------------------------------------- renew

    def renew(self, session_id: str) -> Lease[WorkerT] | None:
        """Renew the lease for ``session_id``."""
        lease = self._leases.get(session_id)
        if lease is None:
            return None
        lease.renew(self._lease_ttl)
        return lease

    # ---------------------------------------------------------- reap

    async def _reap_loop(self) -> None:
        """Background task that reclaims expired leases and evicts overflow."""
        while self._started:
            try:
                await asyncio.sleep(self._reap_interval)
                count = await self._reap_expired()
                if count:
                    self._logger.info(
                        "pool_reap_completed",
                        extra={"expired_count": count, "kind": self._resource_kind},
                    )
                evicted = await self._evict_overflow()
                if evicted:
                    self._logger.info(
                        "pool_overflow_evicted",
                        extra={"evicted": evicted, "kind": self._resource_kind},
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                self._logger.exception("pool_reap_error", extra={"kind": self._resource_kind})

    async def _reap_expired(self) -> int:
        """Release all expired leases and dispatch waiters (thread-safe)."""
        now = time.monotonic()
        expired_sessions = [
            sid for sid, lease in self._leases.items() if lease.is_expired(now)
        ]
        for sid in expired_sessions:
            async with self._lock:
                lease = self._leases.pop(sid, None)
                if lease is None:
                    continue
                self._logger.warning(
                    "pool_lease_reaped",
                    extra={
                        "session_id": sid,
                        "resource_id": lease.resource_id,
                        "kind": self._resource_kind,
                    },
                )
                self._dispatch_waiter_locked(lease.resource_id)
        return len(expired_sessions)

    # ---------------------------------------------------- worker load

    async def evict_worker(self, resource_id: str) -> bool:
        """Unload a loaded worker to free its resources.

        Fails (returns False) when the resource is unknown, has an active
        lease, or has pending waiters. After removal,
        :meth:`_on_worker_evicted` releases per-worker resources.
        """
        async with self._lock:
            if resource_id not in self._workers:
                return False
            if any(
                lease.resource_id == resource_id
                for lease in self._leases.values()
            ):
                return False
            if self._waiters.get(resource_id):
                return False
            worker = self._workers.pop(resource_id)
            self._loaded_at.pop(resource_id, None)
        self._on_worker_evicted(worker)
        self._logger.info(
            "pool_worker_evicted",
            extra={"resource_id": resource_id, "kind": self._resource_kind},
        )
        return True

    def _on_worker_evicted(self, worker: WorkerT) -> None:
        """Hook: release per-worker resources after eviction."""

    async def _evict_overflow(self) -> int:
        """Evict least-recently-loaded workers above ``max_loaded_workers``."""
        cap = self._max_loaded_workers
        if cap <= 0 or len(self._workers) <= cap:
            return 0
        async with self._lock:
            if len(self._workers) <= cap:
                return 0
            candidates = sorted(
                self._workers,
                key=lambda rid: self._loaded_at.get(rid, 0.0),
            )
            evicted = 0
            for rid in candidates:
                if len(self._workers) <= cap:
                    break
                if any(
                    lease.resource_id == rid for lease in self._leases.values()
                ):
                    continue
                if self._waiters.get(rid):
                    continue
                worker = self._workers.pop(rid)
                self._loaded_at.pop(rid, None)
                self._on_worker_evicted(worker)
                evicted += 1
                self._logger.info(
                    "pool_worker_evicted",
                    extra={"resource_id": rid, "kind": self._resource_kind},
                )
        return evicted

    async def _load_worker(self, resource_id: str) -> WorkerT:
        """Load (or return existing) worker for ``resource_id``.

        Must be called while holding ``self._lock``.
        """
        if resource_id in self._workers:
            return self._workers[resource_id]

        if len(self._workers) >= self._max_workers:
            raise self._gpu_memory_exhausted_error(
                f"max_workers ({self._max_workers}) reached; "
                f"cannot load {self._resource_kind} '{resource_id}'. "
                f"Loaded: {sorted(self._workers.keys())}"
            )

        assets = self._resources[resource_id]
        worker = await asyncio.to_thread(self._worker_factory, assets)
        self._workers[resource_id] = worker
        self._loaded_at[resource_id] = time.monotonic()
        self._logger.info(
            "pool_worker_loaded",
            extra={
                "resource_id": resource_id,
                "kind": self._resource_kind,
                "workers_count": len(self._workers),
            },
        )
        return worker

    # ------------------------------------------------------- inspect

    @property
    def resources(self) -> dict[str, AssetsT]:
        return dict(self._resources)

    @property
    def loaded_workers(self) -> list[str]:
        return sorted(self._workers.keys())

    @property
    def active_leases(self) -> dict[str, Lease[WorkerT]]:
        return dict(self._leases)

    @property
    def pending_waiters(self) -> dict[str, int]:
        return {rid: len(q) for rid, q in self._waiters.items() if q}

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of pool state for observability."""
        now = time.monotonic()
        return {
            "started": self._started,
            "kind": self._resource_kind,
            "max_workers": self._max_workers,
            "loaded_workers": len(self._workers),
            "loaded_worker_ids": sorted(self._workers.keys()),
            "available_resources": sorted(self._resources.keys()),
            "active_leases": len(self._leases),
            "active_lease_sessions": sorted(self._leases.keys()),
            "pending_waiters": self.pending_waiters,
            "leases": {
                sid: lease.to_dict()
                for sid, lease in self._leases.items()
                if not lease.is_expired(now)
            },
        }

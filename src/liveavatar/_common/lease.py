# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Lease and cooperative cancellation primitives.

A :class:`Lease` binds a worker/resource to a session for a TTL. The
:class:`CancelToken` provides cooperative cancellation for streaming inference
or generation (e.g. TTS chunks, avatar video frames).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Generic, TypeVar

WorkerT = TypeVar("WorkerT")


class CancelToken:
    """Cooperative cancellation flag.

    Safe to call ``cancel()`` from any coroutine; ``cancelled`` and ``wait()``
    are safe to check from a streaming generator.
    """

    __slots__ = ("_cancelled", "_event")

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._event: asyncio.Event = asyncio.Event()

    def cancel(self) -> None:
        """Mark as cancelled and wake any waiter."""
        self._cancelled = True
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True if ``cancel()`` has been called."""
        return self._cancelled

    async def wait(self) -> None:
        """Block until cancelled."""
        await self._event.wait()

    def reset(self) -> None:
        """Clear cancellation state (for reuse)."""
        self._cancelled = False
        self._event.clear()


@dataclass(slots=True)
class Lease(Generic[WorkerT]):
    """Exclusive lease granting a session access to a worker/resource.

    Attributes
    ----------
    lease_id:
        Unique lease identifier.
    session_id:
        Session that owns the lease.
    resource_id:
        Resource identifier (character ID, avatar ID, etc.).
    worker:
        The actual worker instance.
    acquired_at:
        ``time.monotonic()`` when the lease was created.
    deadline:
        ``time.monotonic()`` when the lease expires.
    renew_count:
        Number of times ``renew()`` has been called.
    """

    lease_id: str
    session_id: str
    resource_id: str
    worker: WorkerT
    acquired_at: float
    deadline: float
    renew_count: int = 0

    # Per-lease cancellation token, created lazily.
    _cancel_token: CancelToken | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(
        cls,
        session_id: str,
        resource_id: str,
        worker: WorkerT,
        ttl: float,
    ) -> Lease[WorkerT]:
        """Create a new lease with ``ttl`` seconds until expiry."""
        now = time.monotonic()
        return cls(
            lease_id=f"lease_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            resource_id=resource_id,
            worker=worker,
            acquired_at=now,
            deadline=now + ttl,
        )

    def is_expired(self, now: float | None = None) -> bool:
        """True if the lease deadline has passed."""
        if now is None:
            now = time.monotonic()
        return now > self.deadline

    def renew(self, ttl: float) -> None:
        """Extend the deadline to ``now + ttl``."""
        self.deadline = time.monotonic() + ttl
        self.renew_count += 1

    @property
    def remaining(self) -> float:
        """Seconds remaining before expiry (never negative)."""
        return max(0.0, self.deadline - time.monotonic())

    def cancel(self) -> None:
        """Cancel any streaming work associated with this lease."""
        if self._cancel_token is None:
            self._cancel_token = CancelToken()
        self._cancel_token.cancel()

    def cancel_token(self) -> CancelToken:
        """Return (and optionally create) the lease's cancel token."""
        if self._cancel_token is None:
            self._cancel_token = CancelToken()
        return self._cancel_token

    def to_dict(self) -> dict:
        """Serialize lease metadata (worker itself is not serialized)."""
        return {
            "lease_id": self.lease_id,
            "session_id": self.session_id,
            "resource_id": self.resource_id,
            "acquired_at": self.acquired_at,
            "deadline": self.deadline,
            "remaining": round(self.remaining, 3),
            "renew_count": self.renew_count,
        }

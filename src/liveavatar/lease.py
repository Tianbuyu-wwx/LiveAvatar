"""Lease management for avatar workers.

Re-exports :class:`liveavatar._common.lease.CancelToken` and defines
:class:`AvatarLease` as a thin wrapper around the shared :class:`Lease`
primitive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._common.lease import CancelToken, Lease

if TYPE_CHECKING:
    pass

__all__ = ["CancelToken", "AvatarLease"]


class AvatarLease(Lease["AvatarWorker"]):  # type: ignore[name-defined]
    """Exclusive lease on an AvatarWorker for a session.

    The lease grants the session exclusive access to the worker's GPU
    inference pipeline. The worker's avatar data (face coords, latents,
    masks) is loaded once at construction and never switched — mirroring
    the voice-pool cross-talk isolation guarantee.
    """

    @property
    def avatar_id(self) -> str:
        return self.resource_id

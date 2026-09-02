# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Lease and cancellation primitives for the voice pool.

Re-exports :class:`liveavatar._common.lease.CancelToken` and defines
:class:`VoiceLease` as a thin wrapper around the shared :class:`Lease`
primitive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from liveavatar._common.lease import CancelToken, Lease

if TYPE_CHECKING:
    from .worker import NvcWorker  # noqa: F401  (used in quoted base class)

__all__ = ["CancelToken", "VoiceLease"]


class VoiceLease(Lease["NvcWorker"]):
    """A lease granting a session exclusive access to a character worker."""

    @property
    def char_id(self) -> str:
        return self.resource_id

    def to_dict(self) -> dict:
        """Serialize lease metadata with character-specific aliases."""
        d = super().to_dict()
        d["char_id"] = self.resource_id
        d["worker_char_id"] = getattr(self.worker, "char_id", self.resource_id)
        return d

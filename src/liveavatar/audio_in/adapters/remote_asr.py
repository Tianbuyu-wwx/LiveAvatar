# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Remote ASR adapter — delegates to the RealtimeAsr microservice.

This adapter implements :class:`~realtime_audio.interfaces.StreamingAsrAdapter`
by returning buffered ``partial``/``final`` ASR events from the shared
:class:`RealtimeAsrClient`.

``flush()`` sends a ``flush`` control message to the service. The final
ASR event arrives asynchronously and is returned on the next ``push_frame``
call. ``advance_epoch()`` sends an ``advance_epoch`` control message which
also resets the service-side VAD and EOU state.
"""

from __future__ import annotations

from typing import Any

from ..frame import PCMFrame
from ..interfaces import StreamingAsrAdapter
from .realtime_asr_client import RealtimeAsrClient


class RemoteAsrAdapter(StreamingAsrAdapter):
    """Streaming ASR adapter backed by the remote RealtimeAsr service.

    Parameters
    ----------
    client : RealtimeAsrClient
        Shared WebSocket client (same instance as the VAD/EOU adapters).
    """

    def __init__(self, client: RealtimeAsrClient) -> None:
        self._client = client

    def push_frame(self, frame: PCMFrame) -> list[dict[str, Any]]:
        """Return buffered ASR events.

        The PCM frame was already sent by the VAD adapter's ``push_frame``
        call (deduplicated by ``seq`` in the shared client).
        """
        return self._client.drain_asr()

    def flush(self) -> list[dict[str, Any]]:
        """Send a flush control message and return any buffered ASR events.

        The final ASR result arrives asynchronously and will be returned on
        the next ``push_frame`` call. The returned list may be empty if no
        events have arrived yet.
        """
        self._client.send_control("flush")
        return self._client.drain_asr()

    def advance_epoch(self, epoch: int) -> None:
        """Send an ``advance_epoch`` control message to the service.

        This triggers a service-side reset of ASR cache, VAD state, and EOU
        state — the full pipeline is purged for the new epoch.
        """
        self._client.send_control("advance_epoch", epoch=epoch)

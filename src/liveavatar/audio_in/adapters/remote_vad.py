# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Remote VAD adapter — delegates to the RealtimeAsr microservice.

This adapter implements :class:`~realtime_audio.interfaces.VoiceActivityDetector`
by forwarding PCM frames to a shared :class:`RealtimeAsrClient` and returning
buffered ``speech_start``/``speech_end`` events.

The client deduplicates frames by ``seq``, so calling ``send_frame`` from
multiple adapters for the same frame sends the PCM only once.
"""

from __future__ import annotations

from typing import Any

from ..frame import PCMFrame
from ..interfaces import VoiceActivityDetector
from .realtime_asr_client import RealtimeAsrClient


class RemoteVadAdapter(VoiceActivityDetector):
    """VAD adapter backed by the remote RealtimeAsr service.

    Parameters
    ----------
    client : RealtimeAsrClient
        Shared WebSocket client. The same instance must be shared with
        :class:`RemoteEouAdapter` and :class:`RemoteAsrAdapter`.
    """

    def __init__(self, client: RealtimeAsrClient) -> None:
        self._client = client

    def push_frame(self, frame: PCMFrame) -> list[dict[str, Any]]:
        """Send the PCM frame to the service and return buffered VAD events.

        Events from frame *N* typically arrive while frame *N+1* is being
        processed (≈ 20 ms latency). If the connection is down, returns an
        empty list.
        """
        self._client.send_frame(frame)
        return self._client.drain_vad()

    def reset(self) -> None:
        """No-op — the service resets VAD state on ``advance_epoch``.

        The worker's ``advance_epoch`` sends an ``advance_epoch`` control
        message to the service via :meth:`RemoteAsrAdapter.advance_epoch`,
        which triggers the service-side VAD reset.
        """

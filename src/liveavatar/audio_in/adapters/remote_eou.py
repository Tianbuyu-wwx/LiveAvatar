# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Remote EOU adapter — delegates to the RealtimeAsr microservice.

This adapter implements
:class:`~realtime_audio.interfaces.EndOfUtteranceDetector` by returning
buffered ``eou`` events from the shared :class:`RealtimeAsrClient`.

The ``vad_active`` parameter is intentionally **ignored**: the remote
service runs its own Silero VAD internally and drives the EOU state machine
from that. Passing the worker's local (energy-based) VAD state would be
less accurate and could conflict with the service's decision.
"""

from __future__ import annotations

from typing import Any

from ..frame import PCMFrame
from ..interfaces import EndOfUtteranceDetector
from .realtime_asr_client import RealtimeAsrClient


class RemoteEouAdapter(EndOfUtteranceDetector):
    """EOU adapter backed by the remote RealtimeAsr service.

    Parameters
    ----------
    client : RealtimeAsrClient
        Shared WebSocket client (same instance as the VAD/ASR adapters).
    """

    def __init__(self, client: RealtimeAsrClient) -> None:
        self._client = client

    def push_frame(self, frame: PCMFrame, vad_active: bool) -> list[dict[str, Any]]:
        """Return buffered EOU events.

        ``vad_active`` is ignored — the service computes EOU from its own
        internal Silero VAD, which is more accurate than the worker's
        energy-based VAD.

        The PCM frame is **not** re-sent here; it was already sent by the
        VAD adapter's ``push_frame`` call (deduplicated by ``seq`` in the
        shared client).
        """
        return self._client.drain_eou()

    def reset(self) -> None:
        """No-op — the service resets EOU state on ``advance_epoch``."""

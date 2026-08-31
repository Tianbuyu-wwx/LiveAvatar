"""Remote adapters for the RealtimeAsr microservice.

This subpackage provides drop-in replacements for the reference
``EnergyVad`` / ``SilenceEouDetector`` / ``ScriptedAsrAdapter`` that
delegate to a remote ``RealtimeAsr`` service over WebSocket.

All three adapters share a single :class:`~realtime_asr_client.RealtimeAsrClient`
(= one WebSocket connection per session). The client sends each PCM frame
once and buffers the returned ``vad``/``asr``/``eou`` events; each adapter
drains only the events of its own type.
"""

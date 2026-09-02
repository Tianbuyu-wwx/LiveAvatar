# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for the remote RealtimeAsr adapters.

These tests verify the adapter logic (buffer/drain, frame deduplication,
event stripping, control message dispatch) without a live WebSocket
connection. The ``RealtimeAsrClient._read_loop`` is tested by directly
manipulating the internal buffers, simulating what the reader task would do.
"""

from __future__ import annotations

import unittest
from unittest import mock

from liveavatar.audio_in.adapters.realtime_asr_client import RealtimeAsrClient
from liveavatar.audio_in.adapters.remote_asr import RemoteAsrAdapter
from liveavatar.audio_in.adapters.remote_eou import RemoteEouAdapter
from liveavatar.audio_in.adapters.remote_vad import RemoteVadAdapter
from liveavatar.audio_in.frame import PCMFrame


def _make_frame(seq: int = 1, pts_us: int = 0) -> PCMFrame:
    """Create a 20ms PCM frame filled with silence."""
    return PCMFrame.silence(
        session_id="test_session",
        epoch=0,
        seq=seq,
        pts_us=pts_us,
        deadline_us=pts_us + 1_000_000,
    )


class TestRealtimeAsrClientBuffers(unittest.TestCase):
    """Test the client's buffer/drain logic without a real connection."""

    def test_drain_empty_returns_empty_list(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        self.assertEqual(client.drain_vad(), [])
        self.assertEqual(client.drain_asr(), [])
        self.assertEqual(client.drain_eou(), [])
        self.assertEqual(client.drain_acks(), [])

    def test_drain_returns_buffered_events(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        # Simulate events arriving from the reader loop.
        client._vad_buf.append({"kind": "speech_start", "confidence": 0.95})
        client._asr_buf.append({"phase": "partial", "text": "你好", "revision": 1})
        client._eou_buf.append({"confidence": 0.85, "silence_us": 500000})

        vad = client.drain_vad()
        asr = client.drain_asr()
        eou = client.drain_eou()

        self.assertEqual(len(vad), 1)
        self.assertEqual(vad[0]["kind"], "speech_start")
        self.assertEqual(len(asr), 1)
        self.assertEqual(asr[0]["text"], "你好")
        self.assertEqual(len(eou), 1)
        self.assertEqual(eou[0]["silence_us"], 500000)

        # Second drain is empty (buffers were cleared).
        self.assertEqual(client.drain_vad(), [])
        self.assertEqual(client.drain_asr(), [])
        self.assertEqual(client.drain_eou(), [])

    def test_strip_event_removes_type_and_ts_us(self):
        raw = {"type": "vad", "kind": "speech_start", "confidence": 0.9, "ts_us": 12345}
        stripped = RealtimeAsrClient._strip_event(raw)
        self.assertNotIn("type", stripped)
        self.assertNotIn("ts_us", stripped)
        self.assertEqual(stripped["kind"], "speech_start")
        self.assertEqual(stripped["confidence"], 0.9)

    def test_strip_event_preserves_all_other_fields(self):
        raw = {
            "type": "asr",
            "phase": "final",
            "text": "你好世界",
            "stability": 1.0,
            "revision": 3,
            "words": [{"word": "你好", "start_us": 0, "end_us": 500000}],
            "ts_us": 99999,
        }
        stripped = RealtimeAsrClient._strip_event(raw)
        self.assertEqual(stripped["phase"], "final")
        self.assertEqual(stripped["text"], "你好世界")
        self.assertEqual(stripped["stability"], 1.0)
        self.assertEqual(stripped["revision"], 3)
        self.assertEqual(len(stripped["words"]), 1)
        self.assertNotIn("type", stripped)
        self.assertNotIn("ts_us", stripped)


class TestRealtimeAsrClientSendFrame(unittest.TestCase):
    """Test frame deduplication and graceful degradation."""

    def test_send_frame_no_op_when_disconnected(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        self.assertFalse(client.is_connected)
        frame = _make_frame(seq=1)
        # Should not raise even though not connected.
        client.send_frame(frame)

    def test_send_frame_deduplicates_by_seq(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        client._connected = True
        client._ws = mock.MagicMock()

        frame1 = _make_frame(seq=1)
        frame2 = _make_frame(seq=1)  # same seq
        frame3 = _make_frame(seq=2)  # different seq

        with mock.patch("asyncio.create_task") as mock_task:
            client.send_frame(frame1)
            client.send_frame(frame2)  # deduplicated, no new task
            client.send_frame(frame3)  # different seq, new task

            # Only 2 tasks created (frame1 and frame3; frame2 deduplicated).
            self.assertEqual(mock_task.call_count, 2)


class TestRealtimeAsrClientSendControl(unittest.TestCase):
    """Test control message dispatch."""

    def test_send_control_no_op_when_disconnected(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        # Should not raise.
        client.send_control("advance_epoch", epoch=1)
        client.send_control("flush")
        client.send_control("close")

    def test_send_control_schedules_task_when_connected(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        client._connected = True
        client._ws = mock.MagicMock()

        with mock.patch("asyncio.create_task") as mock_task:
            client.send_control("advance_epoch", epoch=5)
            self.assertEqual(mock_task.call_count, 1)


class TestRemoteVadAdapter(unittest.TestCase):
    """Test the RemoteVadAdapter."""

    def test_push_frame_sends_and_drains(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        client._connected = True
        client._ws = mock.MagicMock()
        client._vad_buf.append({"kind": "speech_start", "confidence": 0.9})

        adapter = RemoteVadAdapter(client)
        frame = _make_frame(seq=1)

        with mock.patch("asyncio.create_task"):
            events = adapter.push_frame(frame)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "speech_start")

    def test_push_frame_returns_empty_when_no_events(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        adapter = RemoteVadAdapter(client)
        frame = _make_frame(seq=1)
        events = adapter.push_frame(frame)
        self.assertEqual(events, [])

    def test_reset_is_noop(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        adapter = RemoteVadAdapter(client)
        adapter.reset()  # must not raise


class TestRemoteEouAdapter(unittest.TestCase):
    """Test the RemoteEouAdapter."""

    def test_push_frame_drains_eou_ignores_vad_active(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        client._eou_buf.append({"confidence": 0.85, "silence_us": 500000})

        adapter = RemoteEouAdapter(client)
        frame = _make_frame(seq=1)

        # vad_active=True should be ignored; EOU events are still returned.
        events = adapter.push_frame(frame, vad_active=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["silence_us"], 500000)

    def test_push_frame_returns_empty_when_no_events(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        adapter = RemoteEouAdapter(client)
        frame = _make_frame(seq=1)
        events = adapter.push_frame(frame, vad_active=False)
        self.assertEqual(events, [])

    def test_reset_is_noop(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        adapter = RemoteEouAdapter(client)
        adapter.reset()  # must not raise


class TestRemoteAsrAdapter(unittest.TestCase):
    """Test the RemoteAsrAdapter."""

    def test_push_frame_drains_asr(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        client._asr_buf.append({"phase": "partial", "text": "你好", "revision": 1})

        adapter = RemoteAsrAdapter(client)
        frame = _make_frame(seq=1)
        events = adapter.push_frame(frame)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "partial")
        self.assertEqual(events[0]["text"], "你好")

    def test_flush_sends_control_and_drains(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        client._connected = True
        client._ws = mock.MagicMock()
        client._asr_buf.append({"phase": "final", "text": "最终结果", "revision": 2})

        adapter = RemoteAsrAdapter(client)

        with mock.patch("asyncio.create_task"):
            events = adapter.flush()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "final")

    def test_advance_epoch_sends_control(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        client._connected = True
        client._ws = mock.MagicMock()

        adapter = RemoteAsrAdapter(client)

        with mock.patch("asyncio.create_task"):
            adapter.advance_epoch(5)


class TestAdapterSharingClient(unittest.TestCase):
    """Verify that all three adapters share the same client instance."""

    def test_shared_client_buffers_are_independent(self):
        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        vad = RemoteVadAdapter(client)
        eou = RemoteEouAdapter(client)
        asr = RemoteAsrAdapter(client)

        # All point to the same client.
        self.assertIs(vad._client, client)
        self.assertIs(eou._client, client)
        self.assertIs(asr._client, client)

        # Buffer events of each type.
        client._vad_buf.append({"kind": "speech_start", "confidence": 0.9})
        client._eou_buf.append({"confidence": 0.85, "silence_us": 300000})
        client._asr_buf.append({"phase": "partial", "text": "test", "revision": 1})

        frame = _make_frame(seq=1)

        # VAD adapter sends frame + drains VAD events.
        with mock.patch("asyncio.create_task"):
            vad_events = vad.push_frame(frame)
        self.assertEqual(len(vad_events), 1)

        # EOU adapter drains EOU events (no send).
        eou_events = eou.push_frame(frame, vad_active=True)
        self.assertEqual(len(eou_events), 1)

        # ASR adapter drains ASR events (no send, frame deduplicated).
        asr_events = asr.push_frame(frame)
        self.assertEqual(len(asr_events), 1)

        # All buffers now empty.
        self.assertEqual(client.drain_vad(), [])
        self.assertEqual(client.drain_eou(), [])
        self.assertEqual(client.drain_asr(), [])


class TestWorkerInjection(unittest.TestCase):
    """Verify the worker accepts and uses injected adapters."""

    def test_worker_uses_injected_adapters(self):
        from liveavatar.runtime.worker import RealtimeWorker

        client = RealtimeAsrClient("ws://localhost:8300/asr/stream", "s1")
        vad = RemoteVadAdapter(client)
        eou = RemoteEouAdapter(client)
        asr = RemoteAsrAdapter(client)

        worker = RealtimeWorker("s1", vad=vad, eou=eou, asr=asr)

        # The injected adapters are used, not the reference implementations.
        self.assertIs(worker.vad, vad)
        self.assertIs(worker.eou, eou)
        self.assertIs(worker.asr, asr)

    def test_worker_falls_back_to_reference_when_not_injected(self):
        from liveavatar.runtime.worker import RealtimeWorker

        worker = RealtimeWorker("s1")

        # Reference adapters are used (if realtime_audio is importable).
        # Just verify they exist and have the right interface.
        if worker.vad is not None:
            self.assertTrue(hasattr(worker.vad, "push_frame"))
            self.assertTrue(hasattr(worker.vad, "reset"))
        if worker.eou is not None:
            self.assertTrue(hasattr(worker.eou, "push_frame"))
            self.assertTrue(hasattr(worker.eou, "reset"))
        if worker.asr is not None:
            self.assertTrue(hasattr(worker.asr, "push_frame"))
            self.assertTrue(hasattr(worker.asr, "flush"))
            self.assertTrue(hasattr(worker.asr, "advance_epoch"))

    def test_advance_epoch_resets_vad_and_eou(self):
        """advance_epoch must call reset() on VAD and EOU adapters."""
        from liveavatar.runtime.worker import RealtimeWorker

        vad = mock.MagicMock()
        eou = mock.MagicMock()
        asr = mock.MagicMock()

        worker = RealtimeWorker("s1", vad=vad, eou=eou, asr=asr)
        worker.advance_epoch()

        vad.reset.assert_called_once()
        eou.reset.assert_called_once()
        asr.advance_epoch.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for CancelToken and VoiceLease."""

from __future__ import annotations

import asyncio
import time
import unittest

from liveavatar.voice.lease import CancelToken, VoiceLease


# ── stub for NvcWorker (avoids importing torch) ──
class _FakeWorker:
    char_id = "nahida"


class TestCancelToken(unittest.TestCase):
    def test_initial_state_not_cancelled(self):
        token = CancelToken()
        self.assertFalse(token.cancelled)

    def test_cancel_sets_flag(self):
        token = CancelToken()
        token.cancel()
        self.assertTrue(token.cancelled)

    def test_cancel_is_idempotent(self):
        token = CancelToken()
        token.cancel()
        token.cancel()
        self.assertTrue(token.cancelled)

    def test_reset_clears_flag(self):
        token = CancelToken()
        token.cancel()
        token.reset()
        self.assertFalse(token.cancelled)

    def test_wait_returns_after_cancel(self):
        async def _run():
            token = CancelToken()
            # Cancel after a short delay.
            asyncio.get_event_loop().call_later(0.01, token.cancel)
            await asyncio.wait_for(token.wait(), timeout=1.0)

        asyncio.run(_run())

    def test_wait_returns_immediately_if_already_cancelled(self):
        async def _run():
            token = CancelToken()
            token.cancel()
            await asyncio.wait_for(token.wait(), timeout=1.0)

        asyncio.run(_run())


class TestVoiceLeaseCreate(unittest.TestCase):
    def test_create_sets_fields(self):
        worker = _FakeWorker()
        lease = VoiceLease.create("sess_1", "nahida", worker, ttl=60.0)  # type: ignore
        self.assertEqual(lease.session_id, "sess_1")
        self.assertEqual(lease.char_id, "nahida")
        self.assertIs(lease.worker, worker)
        self.assertEqual(lease.renew_count, 0)
        self.assertTrue(lease.lease_id.startswith("lease_"))

    def test_create_lease_id_is_unique(self):
        worker = _FakeWorker()
        l1 = VoiceLease.create("s1", "a", worker, 60.0)  # type: ignore
        l2 = VoiceLease.create("s2", "a", worker, 60.0)  # type: ignore
        self.assertNotEqual(l1.lease_id, l2.lease_id)

    def test_deadline_is_in_future(self):
        worker = _FakeWorker()
        before = time.monotonic()
        lease = VoiceLease.create("s1", "a", worker, 30.0)  # type: ignore
        after = time.monotonic()
        self.assertGreaterEqual(lease.deadline, before + 30.0)
        self.assertLessEqual(lease.deadline, after + 30.0 + 0.1)


class TestVoiceLeaseExpiry(unittest.TestCase):
    def test_not_expired_when_future(self):
        worker = _FakeWorker()
        lease = VoiceLease.create("s1", "a", worker, 60.0)  # type: ignore
        self.assertFalse(lease.is_expired())

    def test_expired_when_past(self):
        worker = _FakeWorker()
        lease = VoiceLease.create("s1", "a", worker, 60.0)  # type: ignore
        # Simulate expiry.
        lease.deadline = time.monotonic() - 1.0
        self.assertTrue(lease.is_expired())

    def test_expired_with_explicit_now(self):
        worker = _FakeWorker()
        lease = VoiceLease.create("s1", "a", worker, 60.0)  # type: ignore
        self.assertFalse(lease.is_expired(now=lease.acquired_at))
        self.assertTrue(lease.is_expired(now=lease.deadline + 0.001))

    def test_remaining_is_positive(self):
        worker = _FakeWorker()
        lease = VoiceLease.create("s1", "a", worker, 60.0)  # type: ignore
        self.assertGreater(lease.remaining, 59.0)
        # Allow tiny float tolerance: (now + ttl) - now2 may exceed ttl by
        # a few ULPs due to IEEE-754 rounding.
        self.assertLessEqual(lease.remaining, 60.0 + 1e-6)

    def test_remaining_is_zero_when_expired(self):
        worker = _FakeWorker()
        lease = VoiceLease.create("s1", "a", worker, 60.0)  # type: ignore
        lease.deadline = time.monotonic() - 10.0
        self.assertEqual(lease.remaining, 0.0)


class TestVoiceLeaseRenew(unittest.TestCase):
    def test_renew_extends_deadline(self):
        worker = _FakeWorker()
        lease = VoiceLease.create("s1", "a", worker, 10.0)  # type: ignore
        old_deadline = lease.deadline
        time.sleep(0.01)
        lease.renew(60.0)
        self.assertGreater(lease.deadline, old_deadline)
        self.assertEqual(lease.renew_count, 1)

    def test_renew_multiple_times(self):
        worker = _FakeWorker()
        lease = VoiceLease.create("s1", "a", worker, 10.0)  # type: ignore
        lease.renew(30.0)
        lease.renew(30.0)
        lease.renew(30.0)
        self.assertEqual(lease.renew_count, 3)


class TestVoiceLeaseToDict(unittest.TestCase):
    def test_to_dict_contains_all_fields(self):
        worker = _FakeWorker()
        lease = VoiceLease.create("s1", "nahida", worker, 60.0)  # type: ignore
        d = lease.to_dict()
        self.assertIn("lease_id", d)
        self.assertIn("session_id", d)
        self.assertIn("char_id", d)
        self.assertIn("acquired_at", d)
        self.assertIn("deadline", d)
        self.assertIn("remaining", d)
        self.assertIn("renew_count", d)
        self.assertIn("worker_char_id", d)
        self.assertEqual(d["session_id"], "s1")
        self.assertEqual(d["char_id"], "nahida")
        self.assertEqual(d["renew_count"], 0)


if __name__ == "__main__":
    unittest.main()

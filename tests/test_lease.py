"""Tests for CancelToken and AvatarLease.

Aligned to the current source API in ``liveavatar/lease.py``:
- ``CancelToken`` exposes ``cancelled`` / ``cancel()`` / ``wait()`` only.
- ``AvatarLease.create`` is keyword-only; ``lease_id`` is a 12-char hex string.
- ``AvatarLease`` exposes ``renew()`` / ``is_expired()`` / ``remaining``.
"""

from __future__ import annotations

import asyncio
import time
import unittest

from liveavatar.lease import AvatarLease, CancelToken


# ── stub for AvatarWorker (avoids importing torch) ──
class _FakeWorker:
    avatar_id = "nahida"


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


class TestAvatarLeaseCreate(unittest.TestCase):
    def test_create_sets_fields(self):
        worker = _FakeWorker()
        lease = AvatarLease.create(
            session_id="sess_1",
            resource_id="nahida",
            worker=worker,  # type: ignore[arg-type]
            ttl=60.0,
        )
        self.assertEqual(lease.session_id, "sess_1")
        self.assertEqual(lease.avatar_id, "nahida")
        self.assertIs(lease.worker, worker)
        self.assertEqual(lease.renew_count, 0)

    def test_create_lease_id_is_unique(self):
        worker = _FakeWorker()
        l1 = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=60.0  # type: ignore[arg-type]
        )
        l2 = AvatarLease.create(
            session_id="s2", resource_id="a", worker=worker, ttl=60.0  # type: ignore[arg-type]
        )
        self.assertNotEqual(l1.lease_id, l2.lease_id)

    def test_lease_id_is_hex_string(self):
        """lease_id is ``lease_`` + 12-char hex string (uuid4 hex prefix)."""
        worker = _FakeWorker()
        lease = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=60.0  # type: ignore[arg-type]
        )
        self.assertTrue(lease.lease_id.startswith("lease_"))
        hex_part = lease.lease_id[6:]
        self.assertEqual(len(hex_part), 12)
        # Must be valid hexadecimal.
        int(hex_part, 16)

    def test_deadline_is_in_future(self):
        worker = _FakeWorker()
        before = time.monotonic()
        lease = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=30.0  # type: ignore[arg-type]
        )
        after = time.monotonic()
        self.assertGreaterEqual(lease.deadline, before + 30.0)
        self.assertLessEqual(lease.deadline, after + 30.0 + 0.1)

    def test_acquired_at_set_to_now(self):
        worker = _FakeWorker()
        before = time.monotonic()
        lease = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=60.0  # type: ignore[arg-type]
        )
        after = time.monotonic()
        self.assertGreaterEqual(lease.acquired_at, before)
        self.assertLessEqual(lease.acquired_at, after)


class TestAvatarLeaseExpiry(unittest.TestCase):
    def test_not_expired_when_future(self):
        worker = _FakeWorker()
        lease = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=60.0  # type: ignore[arg-type]
        )
        self.assertFalse(lease.is_expired())

    def test_expired_when_past(self):
        worker = _FakeWorker()
        lease = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=60.0  # type: ignore[arg-type]
        )
        # Simulate expiry.
        lease.deadline = time.monotonic() - 1.0
        self.assertTrue(lease.is_expired())

    def test_remaining_is_positive(self):
        worker = _FakeWorker()
        lease = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=60.0  # type: ignore[arg-type]
        )
        self.assertGreater(lease.remaining, 59.0)
        # Allow tiny float tolerance: (now + ttl) - now2 may exceed ttl by
        # a few ULPs due to IEEE-754 rounding.
        self.assertLessEqual(lease.remaining, 60.0 + 1e-6)

    def test_remaining_is_zero_when_expired(self):
        worker = _FakeWorker()
        lease = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=60.0  # type: ignore[arg-type]
        )
        lease.deadline = time.monotonic() - 10.0
        self.assertEqual(lease.remaining, 0.0)


class TestAvatarLeaseRenew(unittest.TestCase):
    def test_renew_extends_deadline(self):
        worker = _FakeWorker()
        lease = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=10.0  # type: ignore[arg-type]
        )
        old_deadline = lease.deadline
        time.sleep(0.01)
        lease.renew(60.0)
        self.assertGreater(lease.deadline, old_deadline)
        self.assertEqual(lease.renew_count, 1)

    def test_renew_multiple_times(self):
        worker = _FakeWorker()
        lease = AvatarLease.create(
            session_id="s1", resource_id="a", worker=worker, ttl=10.0  # type: ignore[arg-type]
        )
        lease.renew(30.0)
        lease.renew(30.0)
        lease.renew(30.0)
        self.assertEqual(lease.renew_count, 3)


if __name__ == "__main__":
    unittest.main()

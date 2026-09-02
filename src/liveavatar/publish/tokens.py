# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Session-token signing (stdlib HMAC-SHA256 HS256, no extra deps).

Short-lived bearer tokens scoped to a single session (M-C task 17): the
shared ``LIVEAVATAR_API_KEY`` stays server-side and is exchanged for a
per-session token that expires automatically, so leaked credentials have a
bounded blast radius instead of a permanent shared secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwt_sign(payload: dict[str, Any], secret: str) -> str:
    """Sign a compact HS256 JWT using only the stdlib."""
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )
    sig = hmac.new(
        secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return signing_input + "." + _b64url(sig)


def make_session_token(
    *,
    api_key: str,
    api_secret: str,
    session_id: str,
    ttl_s: int = 300,
    scope: str = "session",
) -> str:
    """Mint a short-lived session token (HS256) for ``session_id``.

    Claims: ``iss`` = api key id, ``sub`` = session id, ``scope`` limits
    what the bearer may do, ``exp`` enforces the short TTL. Verify with
    :func:`verify_session_token`.
    """
    now = int(time.time())
    payload = {
        "iss": api_key,
        "sub": session_id,
        "iat": now,
        "nbf": now - 5,
        "exp": now + ttl_s,
        "scope": scope,
    }
    return _jwt_sign(payload, api_secret)


def verify_session_token(token: str, api_secret: str) -> dict[str, Any] | None:
    """Verify signature + expiry; returns the claims or None."""
    try:
        head_b64, payload_b64, sig_b64 = token.split(".")
        signing_input = f"{head_b64}.{payload_b64}"
        expected = hmac.new(
            api_secret.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        actual = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
        if not hmac.compare_digest(actual, expected):
            return None
        claims: dict[str, Any] = json.loads(
            base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4))
        )
        if int(claims.get("exp", 0)) < int(time.time()):
            return None
        return claims
    except (ValueError, KeyError, TypeError):
        return None

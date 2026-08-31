"""LiveKit-compatible JWT signing (stdlib HMAC-SHA256, no extra deps)."""

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
    """Sign a LiveKit-compatible JWT (HS256) using only the stdlib."""
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


def make_access_token(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    ttl_s: int = 3600,
    can_publish: bool = True,
) -> str:
    """Mint a LiveKit room-join token for ``identity``.

    ``can_publish=False`` for browser tokens — viewers must only subscribe;
    the avatar publisher bot keeps its own can_publish=True token.
    """
    now = int(time.time())
    payload = {
        "iss": api_key,
        "sub": identity,
        "iat": now,
        "nbf": now - 5,
        "exp": now + ttl_s,
        "name": identity,
        "video": {
            "roomJoin": True,
            "room": room,
            "canPublish": can_publish,
            "canSubscribe": True,
            "canPublishData": can_publish,
        },
    }
    return _jwt_sign(payload, api_secret)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""REST + WS authentication (REF-3) — one shared credential core (Q4).

The static-key comparison and session-token verification used to live
inline in ``routes.py`` in two parallel blocks (REST handler prologue +
WS handshake). Both transports now delegate here:

- REST endpoints declare ``Depends(require_api_key)``; a bad key raises
  :class:`Unauthorized`, rendered by the app-level handler into the
  legacy ``{"error": "unauthorized"}`` 401 body (byte-identical).
- The WebSocket handshake calls :func:`require_ws_auth` and closes with
  the legacy ``4401`` code — WS stays a plain bool check because the
  close code/reason must be sent by the endpoint itself.

Response bodies and close codes are unchanged by this refactor.
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import Request

from .state import state


class Unauthorized(Exception):
    """Raised by :func:`require_api_key`; rendered as the legacy 401 body."""


def _key_matches(provided: str | None, key: str) -> bool:
    """Constant-time static-key comparison shared by REST and WS."""
    return provided is not None and hmac.compare_digest(provided, key)


def require_api_key(request: Request) -> None:
    """FastAPI dependency (REST): static-key gate (401 on mismatch).

    Authorized — or auth disabled via unset ``api_key`` — returns None.
    """
    key = state.settings.api_key
    if not key:
        return
    if _key_matches(request.headers.get("X-API-Key"), key):
        return
    raise Unauthorized()


def _extract_bearer(websocket: Any) -> str | None:
    """Session token from WS query params / headers (handshake only)."""
    token = websocket.query_params.get("token") or websocket.headers.get(
        "x-session-token"
    )
    if token:
        return token
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def require_ws_auth(websocket: Any, session_id: str | None = None) -> bool:
    """True when the WS handshake is authorized (or auth is disabled).

    Two credential paths: the static ``api_key`` (query param ``api_key``
    or ``X-API-Key`` header) or a short-lived HS256 session token (query
    param ``token``, ``X-Session-Token`` or ``Authorization: Bearer``)
    whose ``sub`` claim matches ``session_id``.
    """
    key = state.settings.api_key
    secret = state.settings.api_secret
    if not key and not secret:
        return True
    provided = websocket.query_params.get("api_key") or websocket.headers.get(
        "x-api-key"
    )
    if key and _key_matches(provided, key):
        return True
    if secret and session_id is not None:
        token = _extract_bearer(websocket)
        if token is not None:
            from .tokens import verify_session_token

            claims = verify_session_token(token, secret)
            if (
                claims is not None
                and claims.get("sub") == session_id
                and claims.get("scope") == "session"
            ):
                return True
    return False

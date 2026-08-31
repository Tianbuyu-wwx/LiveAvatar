"""LiveAvatar publish service — FastAPI + WebSocket entry point.

Minimal surface (self-developed WS transport only):

    POST   /v1/sessions              create session → {session_id, video_ws}
    DELETE /v1/sessions/{sid}        close session
    GET    /v1/sessions/{sid}/stats  adapter + publisher counters
    GET    /v1/avatars               list available avatars
    WS     /v1/sessions/{sid}/audio  stream PCM / send control messages
    WS     /v1/sessions/{sid}/video  avatar video (self-developed transport)
    GET    /metrics                  Prometheus exposition (LIVEAVATAR_METRICS=on)
    GET    /health                   liveness probe
    GET    /                         web demo (served from web/)

Layout (split from the former single publish.py, A1):
- ``settings``        PublishSettings (env parsing)
- ``tokens``          stdlib HS256 session-token signing/verification
- ``state``           AppState + the process-wide ``state`` singleton
- ``encoders``        per-session publisher/encoder factories + avatar_id
- ``session_manager`` pipeline/voice-pool lifecycle + duplex sessions
- ``routes``          REST endpoints + app assembly
- ``ws_routes``       audio/video WebSocket endpoints

Run::

    uvicorn liveavatar.publish:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

from fastapi.staticfiles import StaticFiles

# Import order matters: routes assembles the app, ws_routes registers the
# WS endpoints on the same instance, then the static mount goes last.
from . import ws_routes  # noqa: F401  (registers WS routes on app)
from .encoders import (  # noqa: F401
    _region_encoder_for,
    _service_publisher_factory,
    _valid_avatar_id,
)
from .routes import (  # noqa: F401
    CreateSessionBody,
    _check_auth,
    _check_ws_auth,
    _web_dir,
    app,
)
from .session_manager import (  # noqa: F401
    _default_avatar_id,
    _ensure_pipeline,
    _open_duplex_session,
)
from .settings import PublishSettings  # noqa: F401
from .state import AppState, state  # noqa: F401
from .tokens import make_session_token, verify_session_token  # noqa: F401

# Static web demo: mounted LAST (catch-all at "/") so every API/WS route
# registered above matches first. Starlette StaticFiles handles
# traversal/404 safely.
if os.path.isdir(_web_dir()):
    app.mount("/", StaticFiles(directory=_web_dir(), html=True), name="web")

__all__ = [
    "AppState",
    "CreateSessionBody",
    "PublishSettings",
    "app",
    "make_session_token",
    "state",
    "verify_session_token",
]

"""Regression tests for the httpx2 compatibility fix (CI, 2026-08-31).

The dev environment ships ``httpx2`` — Pydantic's maintained continuation
of httpx, imported as ``import httpx2`` — and does **not** provide the
legacy ``httpx`` module (CI run for b2ac0e5 failed with
``ModuleNotFoundError: No module named 'httpx'`` in all three matrix
versions). The acceptance probes and the e2e bench therefore alias it as
``import httpx2 as httpx``. These tests keep it that way:

1. the httpx2 distribution is installed (dev extra);
2. no first-party module imports the bare ``httpx`` module — a static
   guard so a regression fails here with a clear message instead of
   mid-suite deep in a real-socket probe;
3. the exact client surface used by the acceptance probes (AsyncClient
   over ASGITransport, post/delete/raise_for_status/json) works on the
   installed httpx2 version.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import unittest
from pathlib import Path

import httpx2 as httpx

_ROOT = Path(__file__).resolve().parents[1]
_SCAN_DIRS = ("src", "tests", "scripts")
_HIDDEN_DIRS = {".venv", "__pycache__", "node_modules", ".git"}
_BARE_HTTPX = re.compile(r"\b(?:import|from)\s+httpx\b")


class Httpx2CompatTests(unittest.TestCase):
    def test_httpx2_is_installed(self) -> None:
        spec = importlib.util.find_spec("httpx2")
        self.assertIsNotNone(
            spec, "httpx2 missing: install with `pip install -e .[server,dev]`"
        )

    def test_no_first_party_bare_httpx_import(self) -> None:
        offenders: list[str] = []
        for sub in _SCAN_DIRS:
            for path in (_ROOT / sub).rglob("*.py"):
                if any(part in _HIDDEN_DIRS for part in path.parts):
                    continue
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if _BARE_HTTPX.search(line):
                        offenders.append(f"{path.relative_to(_ROOT)}:{lineno}")
        self.assertEqual([], offenders)

    def test_acceptance_client_surface_works(self) -> None:
        """The probe calls (post/delete over ASGI, raise_for_status,
        .json()) must work on the installed httpx2 version."""

        class _App:
            """Minimal ASGI app: POST echoes JSON, DELETE returns 204."""

            async def __call__(self, scope, receive, send) -> None:
                assert scope["type"] == "http"
                body = b""
                while True:
                    message = await receive()
                    body += message.get("body", b"")
                    if not message.get("more_body", False):
                        break
                if scope["method"] == "POST":
                    payload = json.loads(body or b"{}")
                    response, status = json.dumps({"echo": payload}).encode(), 200
                else:
                    response, status = b"", 204
                await send(
                    {
                        "type": "http.response.start",
                        "status": status,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": response,
                    }
                )

        async def exercise() -> None:
            transport = httpx.ASGITransport(app=_App())
            async with httpx.AsyncClient(
                transport=transport, base_url="http://probe"
            ) as client:
                resp = await client.post("/x", json={"a": 1})
                resp.raise_for_status()
                self.assertEqual({"echo": {"a": 1}}, resp.json())
                resp = await client.delete("/x")
                resp.raise_for_status()
                self.assertEqual(204, resp.status_code)

        asyncio.run(exercise())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

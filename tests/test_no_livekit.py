# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""M-C guard: LiveKit is fully removed from the runtime.

Fails if:
- any ``src/liveavatar/**/*.py`` mentions livekit in any casing;
- ``pyproject.toml`` declares a livekit dependency / extra;
- ``web/`` references the livekit-client SDK;
- the removed transport switch (``LIVEAVATAR_TRANSPORT``) comes back.

Historical docs (docs/*.md reports, CHANGELOG) are intentionally exempt.
"""

from __future__ import annotations

import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "liveavatar"
_WEB = _ROOT / "web"


class NoLivekitGuardTests(unittest.TestCase):
    def test_no_livekit_in_source_tree(self) -> None:
        offenders: list[str] = []
        for path in sorted(_SRC.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "livekit" in text.lower():
                offenders.append(str(path.relative_to(_ROOT)))
        self.assertEqual(offenders, [], "livekit references found in src/")

    def test_no_livekit_dependency_declared(self) -> None:
        pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("livekit", pyproject.lower())

    def test_no_livekit_client_in_web(self) -> None:
        for path in sorted(_WEB.glob("*")):
            if path.is_file():
                self.assertNotIn(
                    "livekit",
                    path.read_text(encoding="utf-8").lower(),
                    f"livekit reference in {path.name}",
                )

    def test_transport_switch_env_not_read(self) -> None:
        from liveavatar.publish import PublishSettings

        settings = PublishSettings()
        self.assertFalse(hasattr(settings, "transport"))
        self.assertFalse(hasattr(settings, "livekit_url"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""Repo hygiene guards (C1 of the self-replace implementation plan).

These tests prevent a repeat of the P0 defect where a bare ``models/``
pattern in .gitignore silently excluded the self-written source files
under ``src/liveavatar/musetalk/models/`` from version control.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_GIT_ROOT = __file__


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr}"
    return proc.stdout


@pytest.mark.skipif(
    subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True).returncode
    != 0,
    reason="not a git checkout",
)
def test_musetalk_models_sources_tracked() -> None:
    """Source files under nested models/ dirs must be committed."""
    tracked = _git("ls-files", "src/liveavatar/musetalk/models/", "third_party/GPT_SoVITS/AR/models/")
    for path in (
        "src/liveavatar/musetalk/models/__init__.py",
        "src/liveavatar/musetalk/models/vae.py",
        "src/liveavatar/musetalk/models/unet.py",
        "third_party/GPT_SoVITS/AR/models/__init__.py",
        "third_party/GPT_SoVITS/AR/models/t2s_model.py",
        "third_party/GPT_SoVITS/AR/models/t2s_lightning_module.py",
        "third_party/GPT_SoVITS/AR/models/utils.py",
    ):
        assert path in tracked.split(), (
            f"{path} is not tracked by git — check .gitignore anchoring"
        )


@pytest.mark.skipif(
    subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True).returncode
    != 0,
    reason="not a git checkout",
)
def test_no_ignored_source_files() -> None:
    """No .py file anywhere in the repo may be gitignored."""
    out = _git(
        "ls-files",
        "-o",
        "-i",
        "--exclude-standard",
    )
    offenders = [
        line
        for line in out.splitlines()
        if line.endswith(".py") and "__pycache__" not in line
    ]
    assert not offenders, f"source files are gitignored: {offenders}"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))

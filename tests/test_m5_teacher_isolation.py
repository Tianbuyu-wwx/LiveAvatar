"""M5 guard tests: MediaPipe must stay OUT of the runtime dependency set.

R1 M5 demoted MediaPipe to a training-time teacher (optional ``teacher``
extra + ``download_models.py --teacher``). These guards prevent regressions:

- pyproject [project].dependencies must not mention mediapipe
- the ``teacher`` extra must declare it (so training setups stay one flag away)
- ``download_models.download_models`` (the default set) must not fetch the
  MediaPipe asset; only ``download_teacher_assets`` may
- no module-level ``import mediapipe`` anywhere in src/ or scripts/
  (lazy imports inside functions are the required pattern)
"""

from __future__ import annotations

import ast
import inspect
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _dependencies_block(text: str) -> str:
    """Return the [project] ``dependencies = [...]`` block as text."""
    start = text.index("dependencies = [")
    end = text.index("]", start)
    return text[start:end]


def _extra_block(text: str, extra: str) -> str:
    """Return the ``<extra> = [...]`` optional-dependency block as text."""
    start = text.index(f"{extra} = [")
    end = text.index("]", start)
    return text[start:end]


class PyprojectGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _PYPROJECT.read_text(encoding="utf-8")

    def test_runtime_dependencies_exclude_mediapipe(self) -> None:
        block = _dependencies_block(self.text)
        self.assertNotIn("mediapipe", block)

    def test_teacher_extra_declares_mediapipe(self) -> None:
        self.assertIn("mediapipe", _extra_block(self.text, "teacher"))


class DownloadScriptGuardTests(unittest.TestCase):
    def test_default_download_set_excludes_mediapipe_asset(self) -> None:
        import download_models as dm

        src = inspect.getsource(dm.download_models)
        self.assertNotIn("face_landmarker", src)
        self.assertIn("face_detection_yunet_2023mar.onnx", src)  # still default

    def test_teacher_assets_function_downloads_mediapipe(self) -> None:
        import download_models as dm

        src = inspect.getsource(dm.download_teacher_assets)
        self.assertIn("face_landmarker.task", src)


class NoModuleLevelMediapipeImportTests(unittest.TestCase):
    def test_no_module_level_import_anywhere(self) -> None:
        targets = sorted(
            (_REPO_ROOT / "src" / "liveavatar").rglob("*.py")
        ) + sorted((_REPO_ROOT / "scripts").glob("*.py"))
        offenders: list[str] = []
        for path in targets:
            if "musetalk" in path.parts:
                continue  # vendored upstream code, excluded from self-research
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:  # module level only
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n == "mediapipe" or n.startswith("mediapipe.") for n in names):
                    offenders.append(str(path.relative_to(_REPO_ROOT)))
        self.assertEqual(offenders, [], "module-level mediapipe import found")


if __name__ == "__main__":
    unittest.main()

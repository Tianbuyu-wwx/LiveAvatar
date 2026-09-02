"""Tests for the face backend factory (R1 M3).

Covers backend resolution (arg > env > default), the yunet/mediapipe legacy
paths (via fakes, so they run in the light CI env without the model files),
checkpoint loading for the self backend, error messages, and the
``scripts/face_align.py`` wiring. All CPU-only.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from liveavatar import face_backend as fb
from liveavatar.face_backend import (
    FaceBox,
    detect_faces,
    landmarks5,
    reset_backend_caches,
    resolve_backend,
    resolve_det_conf,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))  # for scripts/face_align.py

try:
    import torch

    from liveavatar.face_landmarks import LandmarkNet5Self
    from liveavatar.face_self import ANCHOR_SIZES, TinyFaceDetector

    _HAVE_TORCH = True
except ImportError:  # CI installs light extras only; torch is optional there
    _HAVE_TORCH = False

_TORCH = unittest.skipUnless(_HAVE_TORCH, "torch not installed (light CI env)")


def _synthetic_image(size: int = 200) -> np.ndarray:
    img = np.full((size, size, 3), 30, np.uint8)
    img[60:140, 70:150] = 220
    return img


class _FakeYuNet:
    """Stands in for cv2.FaceDetectorYN (rows: x, y, w, h, ..., score)."""

    def __init__(self, faces: np.ndarray) -> None:
        self.faces = faces
        self.sizes: list[tuple[int, int]] = []

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802
        self.sizes.append(tuple(size))

    def detect(self, _img: np.ndarray):
        return True, self.faces


class tempfile_dir:
    """Minimal tempdir context manager."""

    def __enter__(self) -> str:
        import tempfile

        self._tdir = tempfile.TemporaryDirectory()
        return self._tdir.__enter__()

    def __exit__(self, *exc) -> None:
        self._tdir.__exit__(*exc)


class ResolveBackendTests(unittest.TestCase):
    def test_default_when_no_arg_no_env(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FACE_BACKEND", None)
            self.assertEqual(
                resolve_backend(None, "FACE_BACKEND", "yunet", ("yunet", "self")),
                "yunet",
            )

    def test_env_fallback_and_arg_override(self) -> None:
        with mock.patch.dict(os.environ, {"FACE_BACKEND": "self"}):
            self.assertEqual(
                resolve_backend(None, "FACE_BACKEND", "yunet", ("yunet", "self")),
                "self",
            )
            self.assertEqual(
                resolve_backend("yunet", "FACE_BACKEND", "yunet", ("yunet", "self")),
                "yunet",
            )

    def test_invalid_arg_lists_choices(self) -> None:
        with self.assertRaisesRegex(ValueError, "yunet"):
            resolve_backend("bogus", "FACE_BACKEND", "yunet", ("yunet", "self"))

    def test_invalid_env(self) -> None:
        with mock.patch.dict(os.environ, {"FACE_BACKEND": "bogus"}):
            with self.assertRaises(ValueError):
                resolve_backend(None, "FACE_BACKEND", "yunet", ("yunet", "self"))


class ResolveDetConfTests(unittest.TestCase):
    def test_per_backend_defaults(self) -> None:
        self.assertEqual(resolve_det_conf("self", None), fb.DEFAULT_SELF_DET_CONF)
        self.assertEqual(resolve_det_conf("yunet", None), fb.DEFAULT_DET_CONF)

    def test_explicit_value_overrides(self) -> None:
        self.assertEqual(resolve_det_conf("self", 0.7), 0.7)
        self.assertEqual(resolve_det_conf("yunet", 0.2), 0.2)


class DetectFacesTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_backend_caches()

    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_faces(_synthetic_image(), backend="bogus")

    def test_self_backend_missing_checkpoint_message(self) -> None:
        # Runs without torch: the checkpoint existence check must fire before
        # any torch import so the light CI env gets the actionable error.
        with self.assertRaisesRegex(FileNotFoundError, "train_face_det.py"):
            detect_faces(
                _synthetic_image(), backend="self", det_ckpt_path="nowhere/none.pt"
            )

    def test_env_switch_end_to_end_missing_checkpoint(self) -> None:
        with mock.patch.dict(os.environ, {"FACE_BACKEND": "self"}):
            with self.assertRaisesRegex(FileNotFoundError, "train_face_det.py"):
                detect_faces(_synthetic_image(), det_ckpt_path="nowhere/none.pt")

    def test_yunet_backend_with_fake_detector(self) -> None:
        # Real YuNet rows: x, y, w, h, 10 landmark cols, score (15 total).
        faces = np.array(
            [[
                10.0, 20.0, 50.0, 60.0,
                20.0, 35.0,  # right eye
                40.0, 35.0,  # left eye
                30.0, 50.0,  # nose tip
                22.0, 65.0,  # right mouth corner
                38.0, 65.0,  # left mouth corner
                0.9,
            ]]
        )
        fake = _FakeYuNet(faces)
        img = _synthetic_image()
        with tempfile_dir() as tmp:
            model_path = str(Path(tmp) / "yunet.onnx")
            Path(model_path).write_bytes(b"")
            with mock.patch.object(
                fb, "_YUNET_CACHE", {}, create=True
            ), mock.patch("cv2.FaceDetectorYN_create", return_value=fake) as create:
                boxes = detect_faces(
                    img, backend="yunet", yunet_model_path=model_path
                )
                # Second call exercises the detector cache.
                detect_faces(img, backend="yunet", yunet_model_path=model_path)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(
            (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2, boxes[0].score),
            (10.0, 20.0, 60.0, 80.0, 0.9),
        )
        # The regressed 5 points land on the FaceBox in pixel coords.
        assert boxes[0].points5 is not None
        self.assertEqual(boxes[0].points5.shape, (5, 2))
        self.assertEqual(boxes[0].points5[0].tolist(), [20.0, 35.0])
        # First call creates (constructor sets size); second call reuses the
        # cached detector → exactly one setInputSize and one create call.
        self.assertEqual(create.call_count, 1)
        self.assertEqual(len(fake.sizes), 1)
        self.assertEqual(fake.sizes[0], (img.shape[1], img.shape[0]))

    def test_yunet_no_faces(self) -> None:
        fake = _FakeYuNet(np.zeros((0, 15)))
        with tempfile_dir() as tmp:
            model_path = str(Path(tmp) / "yunet.onnx")
            Path(model_path).write_bytes(b"")
            with mock.patch.object(
                fb, "_YUNET_CACHE", {}, create=True
            ), mock.patch("cv2.FaceDetectorYN_create", return_value=fake):
                boxes = detect_faces(
                    _synthetic_image(), backend="yunet", yunet_model_path=model_path
                )
        self.assertEqual(boxes, [])

    def test_yunet_missing_model_message(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "download_models"):
            detect_faces(
                _synthetic_image(),
                backend="yunet",
                yunet_model_path="nowhere/yunet.onnx",
            )


class Landmarks5MediapipeTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_backend_caches()

    def test_mediapipe_backend_with_fakes(self) -> None:
        # Fake mediapipe module so the branch runs in the light CI env.
        fake_mp = types.SimpleNamespace(
            Image=lambda image_format, data: ("mp_image", data),
            ImageFormat=types.SimpleNamespace(SRGB="srgb"),
        )
        pts = {i: SimpleNamespace(x=i / 500.0, y=i / 1000.0) for i in range(478)}
        result = SimpleNamespace(face_landmarks=[[pts[i] for i in range(478)]])
        fake_landmarker = SimpleNamespace(detect=lambda _img: result)

        img = _synthetic_image()
        with mock.patch.dict(sys.modules, {"mediapipe": fake_mp}), mock.patch.object(
            fb, "_MP_LANDMARKER_CACHE", {}, create=True
        ), mock.patch.object(fb, "_get_mp_landmarker", return_value=fake_landmarker):
            out = landmarks5(img, backend="mediapipe")

        self.assertIsNotNone(out)
        expected = np.array(
            [[pts[i].x, pts[i].y] for i in (33, 263, 1, 61, 291)], dtype=np.float32
        )
        np.testing.assert_allclose(out, expected)

    def test_mediapipe_backend_no_face(self) -> None:
        fake_mp = types.SimpleNamespace(
            Image=lambda image_format, data: ("mp_image", data),
            ImageFormat=types.SimpleNamespace(SRGB="srgb"),
        )
        result = SimpleNamespace(face_landmarks=[])
        fake_landmarker = SimpleNamespace(detect=lambda _img: result)
        with mock.patch.dict(sys.modules, {"mediapipe": fake_mp}), mock.patch.object(
            fb, "_get_mp_landmarker", return_value=fake_landmarker
        ):
            self.assertIsNone(landmarks5(_synthetic_image(), backend="mediapipe"))

    def test_unknown_landmark_backend(self) -> None:
        with self.assertRaises(ValueError):
            landmarks5(_synthetic_image(), backend="bogus")


@_TORCH
class SelfBackendTorchTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_backend_caches()

    @staticmethod
    def _save_ckpts(tmp: str) -> tuple[str, str]:
        det_path = str(Path(tmp) / "det.pt")
        lm_path = str(Path(tmp) / "lm.pt")
        torch.save(
            {
                "model": TinyFaceDetector(width=8).state_dict(),
                "width": 8,
                "input_size": 32,
                "anchor_sizes": ANCHOR_SIZES,
            },
            det_path,
        )
        torch.save(
            {
                "model": LandmarkNet5Self(width=8).state_dict(),
                "width": 8,
                "input_size": 32,
            },
            lm_path,
        )
        return det_path, lm_path

    def test_checkpoint_roundtrip_and_cache(self) -> None:
        with tempfile_dir() as tmp:
            det_path, lm_path = self._save_ckpts(tmp)
            det_model, det_size = fb.load_det_model(det_path)
            lm_model, lm_size = fb.load_lm_model(lm_path)
            self.assertIsInstance(det_model, TinyFaceDetector)
            self.assertIsInstance(lm_model, LandmarkNet5Self)
            self.assertEqual((det_size, lm_size), (32, 32))
            self.assertFalse(det_model.training)
            self.assertIs(fb.load_det_model(det_path)[0], det_model)  # cached
            self.assertIs(fb.load_lm_model(lm_path)[0], lm_model)

    def test_malformed_checkpoint_message(self) -> None:
        with tempfile_dir() as tmp:
            bad = Path(tmp) / "bad.pt"
            torch.save({"foo": 1}, bad)
            with self.assertRaisesRegex(ValueError, "malformed"):
                fb.load_det_model(str(bad))

    def test_detect_faces_self_with_stubbed_detect(self) -> None:
        with tempfile_dir() as tmp:
            det_path, _ = self._save_ckpts(tmp)
            with mock.patch.object(
                fb._face_self, "detect", return_value=[([10.0, 20.0, 60.0, 80.0], 0.9)]
            ) as det:
                boxes = detect_faces(
                    _synthetic_image(), backend="self", det_ckpt_path=det_path
                )
        self.assertEqual(det.call_count, 1)
        self.assertEqual(len(boxes), 1)
        self.assertIsInstance(boxes[0], FaceBox)
        self.assertEqual(boxes[0].x2 - boxes[0].x1, 50.0)

    def test_landmarks5_self_pipeline(self) -> None:
        img = _synthetic_image()
        with tempfile_dir() as tmp:
            det_path, lm_path = self._save_ckpts(tmp)
            stub_out = np.full((5, 2), 0.5, dtype=np.float32)
            with mock.patch.object(
                fb._face_self, "detect", return_value=[([20.0, 20.0, 100.0, 100.0], 0.95)]
            ), mock.patch.object(
                fb._face_landmarks, "landmarks5", return_value=stub_out
            ) as lm5:
                out = landmarks5(
                    img, backend="self", det_ckpt_path=det_path, lm_ckpt_path=lm_path
                )
        # The stubbed crop-landmark step got the largest box, normalized to
        # the full image, and the checkpoint's input size.
        np.testing.assert_allclose(out, stub_out)
        h, w = img.shape[:2]
        np.testing.assert_allclose(
            lm5.call_args.args[2],
            [20.0 / w, 20.0 / h, 80.0 / w, 80.0 / h],
            rtol=1e-6,
        )
        self.assertEqual(lm5.call_args.kwargs.get("input_size"), 32)

    def test_landmarks5_self_no_face(self) -> None:
        with tempfile_dir() as tmp:
            det_path, lm_path = self._save_ckpts(tmp)
            with mock.patch.object(fb._face_self, "detect", return_value=[]):
                self.assertIsNone(
                    landmarks5(
                        _synthetic_image(),
                        backend="self",
                        det_ckpt_path=det_path,
                        lm_ckpt_path=lm_path,
                    )
                )

    def test_facebox_xywh_norm(self) -> None:
        box = FaceBox(10.0, 20.0, 60.0, 100.0)
        self.assertEqual(box.as_xywh_norm(200, 200), [0.05, 0.1, 0.25, 0.4])


class FaceAlignWiringTests(unittest.TestCase):
    """scripts/face_align.py ↔ face_backend switch wiring."""

    def tearDown(self) -> None:
        reset_backend_caches()

    def test_align_image_self_backend(self) -> None:
        import cv2
        from face_align import align_image

        src = _synthetic_image()
        pts = np.array(
            [[0.4, 0.4], [0.6, 0.4], [0.5, 0.5], [0.42, 0.62], [0.58, 0.62]],
            dtype=np.float32,
        )
        with tempfile_dir() as tmp:
            src_path = str(Path(tmp) / "in.png")
            dst_path = str(Path(tmp) / "out.png")
            ok_buf, buf = cv2.imencode(".png", src)
            assert ok_buf
            buf.tofile(src_path)
            with mock.patch.object(
                fb, "landmarks5", return_value=pts
            ) as lm5:
                ok = align_image(
                    src_path,
                    dst_path,
                    output_size=128,
                    landmark_backend="self",
                    det_ckpt="d.pt",
                    lm_ckpt="l.pt",
                )
            self.assertTrue(ok)
            self.assertTrue(Path(dst_path).exists())
            out = cv2.imread(dst_path)
            self.assertEqual(out.shape, (128, 128, 3))
            self.assertEqual(lm5.call_args.kwargs["det_ckpt_path"], "d.pt")
            self.assertEqual(lm5.call_args.kwargs["lm_ckpt_path"], "l.pt")

    def test_align_image_self_backend_no_face(self) -> None:
        import cv2
        from face_align import align_image

        with tempfile_dir() as tmp:
            src_path = str(Path(tmp) / "in.png")
            dst_path = str(Path(tmp) / "out.png")
            ok_buf, buf = cv2.imencode(".png", _synthetic_image())
            assert ok_buf
            buf.tofile(src_path)
            with mock.patch.object(fb, "landmarks5", return_value=None):
                ok = align_image(
                    src_path, dst_path, output_size=128, landmark_backend="self"
                )
        self.assertFalse(ok)
        self.assertFalse(Path(dst_path).exists())

    def test_align_image_invalid_backend(self) -> None:
        import cv2
        from face_align import align_image

        with tempfile_dir() as tmp:
            src_path = str(Path(tmp) / "in.png")
            ok_buf, buf = cv2.imencode(".png", _synthetic_image())
            assert ok_buf
            buf.tofile(src_path)
            with self.assertRaises(ValueError):
                align_image(src_path, str(Path(tmp) / "o.png"), landmark_backend="bogus")


if __name__ == "__main__":
    unittest.main()

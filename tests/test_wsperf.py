"""Tests for the wsperf measurement tool (scripts/wsperf.py, M0).

The statistics logic is exercised directly and via a short synthetic run
(pure CPU — real time pacing only, no torch/GPU/server).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from liveavatar.video_protocol import (
    CODEC_MJPEG_FULL,
    CODEC_REGION_DELTA,
    FLAG_EPOCH_BOUNDARY,
    FLAG_KEYFRAME,
    VideoFrameHeader,
)

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_wsperf():
    spec = importlib.util.spec_from_file_location("wsperf", _SCRIPTS / "wsperf.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass resolution requires the module to be registered in sys.modules
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wsperf = _load_wsperf()


def _frame(epoch: int = 0, seq: int = 0, flags: int = 0) -> VideoFrameHeader:
    return VideoFrameHeader(
        flags=flags,
        codec=CODEC_MJPEG_FULL,
        quality=80,
        seq=seq,
        epoch=epoch,
        pts_us=seq * 40_000,
        width=512,
        height=512,
    )


class TestStreamStats:
    def test_fps_and_bandwidth_from_known_schedule(self):
        stats = wsperf.StreamStats()
        t0 = 100.0
        for i in range(25):  # 25 frames at 40ms = ~1s
            stats.feed(_frame(seq=i), 1000 + i, t0 + i * 0.04)
        report = stats.report(codec=0)
        assert stats.frames == 25
        assert report["duration_s"] == pytest.approx(0.96, abs=0.001)
        assert report["fps"] == pytest.approx(25 / 0.96, abs=0.05)
        assert report["gap_ms_p50"] == pytest.approx(40.0, abs=0.5)
        assert report["keyframes"] == 0

    def test_keyframe_and_epoch_boundary_accounting(self):
        stats = wsperf.StreamStats()
        t = 0.0
        for i in range(10):
            flags = FLAG_KEYFRAME if i % 5 == 0 else 0
            stats.feed(_frame(seq=i, flags=flags), 100, t)
            t += 0.04
        report = stats.report(codec=0)
        assert report["keyframes"] == 2
        assert report["epoch_boundaries"] == 0

        stats2 = wsperf.StreamStats()
        for i in range(4):
            flags = FLAG_EPOCH_BOUNDARY if i == 3 else 0
            stats2.feed(_frame(seq=i, flags=flags), 100, i * 0.04)
        report2 = stats2.report(codec=0)
        assert report2["epoch_boundaries"] == 1
        assert report2["boundary_gap_ms_p50"] > 0

    def test_stale_epoch_dropped(self):
        stats = wsperf.StreamStats()
        stats.feed(_frame(epoch=5), 100, 0.0)
        assert stats.feed(_frame(epoch=5, seq=1), 100, 0.04)
        assert not stats.feed(_frame(epoch=4, seq=2), 100, 0.08)
        assert stats.stale_dropped == 1
        assert stats.frames == 2
        assert stats.max_epoch_seen == 5

    def test_empty_report_is_safe(self):
        report = wsperf.StreamStats().report(codec=0)
        assert report["frames"] == 0
        assert report["fps"] == 0.0
        assert report["gap_ms_p50"] == 0.0
        assert report["first_frame_ms"] == 0.0

    def test_percentile_helper(self):
        vals = list(range(1, 101))
        assert wsperf._percentile(vals, 50) == pytest.approx(50, abs=1)
        assert wsperf._percentile(vals, 95) == pytest.approx(95, abs=1)
        assert wsperf._percentile([], 50) == 0.0


class TestSyntheticRun:
    @pytest.mark.parametrize("codec", [CODEC_MJPEG_FULL, CODEC_REGION_DELTA])
    def test_short_run_reports(self, codec: int):
        report = wsperf.run_synthetic(
            n_frames=20, fps=100.0, codec=codec, quality=80, pace=True
        )
        assert report["mode"] == "synthetic"
        assert report["frames"] == 20
        assert report["fps_nominal"] == 100.0
        # 100fps → keyframe every 100 frames; short run has exactly 1.
        assert report["keyframes"] == 1
        assert report["wire_kbps"] > 0
        assert report["encode_ms_p50"] >= 0
        assert report["stale_dropped"] == 0

    def test_run_is_deterministic_in_bytes(self):
        a = wsperf.run_synthetic(n_frames=10, fps=1000.0, codec=CODEC_MJPEG_FULL, pace=False)
        b = wsperf.run_synthetic(n_frames=10, fps=1000.0, codec=CODEC_MJPEG_FULL, pace=False)
        assert a["wire_mb_total"] == b["wire_mb_total"]
        assert a["avg_frame_kb"] == b["avg_frame_kb"]


class TestCli:
    def test_main_synthetic_prints_json(self, capsys: pytest.CaptureFixture[str]):
        rc = wsperf.main(["--synthetic", "10", "--fps", "500", "--codec", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        report = json.loads(out)
        assert report["mode"] == "synthetic"
        assert report["codec"] == CODEC_REGION_DELTA
        assert report["frames"] == 10

    def test_main_requires_mode(self, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(SystemExit):
            wsperf.main([])

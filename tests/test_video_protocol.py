# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for the video wire protocol (docs/PROTOCOL.md, frozen at R2-M0).

Pure functions — no I/O, no torch, no GPU.
"""

from __future__ import annotations

import pytest

from liveavatar.video_protocol import (
    CODEC_MJPEG_FULL,
    CODEC_REGION_DELTA,
    FLAG_EOF,
    FLAG_EPOCH_BOUNDARY,
    FLAG_KEYFRAME,
    HEADER_SIZE,
    Patch,
    VideoFrameHeader,
    VideoProtocolError,
    has_flag,
    make_flags,
    pack_region_frame,
    pack_region_payload,
    pack_video_frame,
    unpack_region_frame,
    unpack_region_payload,
    unpack_video_frame,
)


def _header(**overrides) -> VideoFrameHeader:
    base = dict(
        flags=0,
        codec=CODEC_MJPEG_FULL,
        quality=80,
        seq=7,
        epoch=3,
        pts_us=1_234_567,
        width=512,
        height=512,
    )
    base.update(overrides)
    return VideoFrameHeader(**base)


class TestRoundtrip:
    def test_header_size_is_26(self):
        assert HEADER_SIZE == 26

    def test_full_frame_roundtrip(self):
        payload = b"\xff\xd8jpegdata\xff\xd9"
        wire = pack_video_frame(_header(flags=FLAG_KEYFRAME), payload)
        header, out = unpack_video_frame(wire)
        assert header == _header(flags=FLAG_KEYFRAME)
        assert out == payload
        assert len(wire) == HEADER_SIZE + len(payload)

    def test_extreme_field_values_roundtrip(self):
        # seq wraps at u16; pts_us is signed i64; epoch is u32.
        header = _header(seq=0xFFFF, epoch=0xFFFFFFFF, pts_us=-5)
        wire = pack_video_frame(header, b"x")
        out, payload = unpack_video_frame(wire)
        assert out.seq == 0xFFFF
        assert out.epoch == 0xFFFFFFFF
        assert out.pts_us == -5
        assert payload == b"x"

    def test_region_roundtrip_two_patches(self):
        patches = [
            Patch(x=10, y=20, w=200, h=200, jpeg=b"a" * 50),
            Patch(x=0, y=0, w=512, h=512, jpeg=b"b" * 100),
        ]
        wire = pack_region_frame(
            _header(codec=CODEC_REGION_DELTA, flags=FLAG_EPOCH_BOUNDARY), patches
        )
        header, out = unpack_region_frame(wire)
        assert header.codec == CODEC_REGION_DELTA
        assert out == patches

    def test_region_empty_patch_list_roundtrip(self):
        wire = pack_region_frame(_header(codec=CODEC_REGION_DELTA), [])
        _, out = unpack_region_frame(wire)
        assert out == []

    def test_flag_helpers(self):
        flags = make_flags(keyframe=True, epoch_boundary=True)
        assert has_flag(flags, FLAG_KEYFRAME)
        assert has_flag(flags, FLAG_EPOCH_BOUNDARY)
        assert not has_flag(flags, FLAG_EOF)
        assert make_flags(eof=True) == FLAG_EOF


class TestPackValidation:
    def test_rejects_reserved_flag_bits(self):
        with pytest.raises(VideoProtocolError, match="reserved flag"):
            pack_video_frame(_header(flags=0x08), b"")

    def test_rejects_unknown_codec(self):
        with pytest.raises(VideoProtocolError, match="unknown codec"):
            pack_video_frame(_header(codec=9), b"")

    def test_rejects_quality_out_of_range(self):
        for quality in (0, 101):
            with pytest.raises(VideoProtocolError, match="quality"):
                pack_video_frame(_header(quality=quality), b"")


class TestUnpackErrors:
    def test_rejects_truncated_header(self):
        with pytest.raises(VideoProtocolError, match="truncated header"):
            unpack_video_frame(b"\x01" * (HEADER_SIZE - 1))

    def test_rejects_unknown_msg_type(self):
        wire = bytearray(pack_video_frame(_header(), b"abc"))
        wire[0] = 2
        with pytest.raises(VideoProtocolError, match="unknown msg_type"):
            unpack_video_frame(bytes(wire))

    def test_rejects_reserved_flag_bits(self):
        wire = bytearray(pack_video_frame(_header(), b"abc"))
        wire[1] = 0x80
        with pytest.raises(VideoProtocolError, match="reserved flag"):
            unpack_video_frame(bytes(wire))

    def test_rejects_unknown_codec(self):
        wire = bytearray(pack_video_frame(_header(), b"abc"))
        wire[2] = 5
        with pytest.raises(VideoProtocolError, match="unknown codec"):
            unpack_video_frame(bytes(wire))

    def test_rejects_payload_len_mismatch(self):
        wire = bytearray(pack_video_frame(_header(), b"abc"))
        wire[22:26] = (99).to_bytes(4, "little")  # declared 99, actual 3
        with pytest.raises(VideoProtocolError, match="payload_len mismatch"):
            unpack_video_frame(bytes(wire))

    def test_rejects_payload_truncation(self):
        wire = pack_video_frame(_header(), b"abc")
        with pytest.raises(VideoProtocolError, match="payload_len mismatch"):
            unpack_video_frame(wire[:-1])


class TestRegionPayloadErrors:
    def test_rejects_truncated_count(self):
        with pytest.raises(VideoProtocolError, match="patch count"):
            unpack_region_payload(b"\x01")

    def test_rejects_truncated_patch_header(self):
        data = pack_region_payload([Patch(x=0, y=0, w=8, h=8, jpeg=b"z" * 10)])
        with pytest.raises(VideoProtocolError, match="truncated patch header"):
            unpack_region_payload(data[:-12])

    def test_rejects_zero_geometry(self):
        with pytest.raises(VideoProtocolError, match="geometry"):
            pack_region_payload([Patch(x=0, y=0, w=0, h=8, jpeg=b"z")])

    def test_rejects_empty_jpeg(self):
        with pytest.raises(VideoProtocolError, match="empty"):
            pack_region_payload([Patch(x=0, y=0, w=8, h=8, jpeg=b"")])

    def test_rejects_truncated_jpeg(self):
        data = pack_region_payload([Patch(x=0, y=0, w=8, h=8, jpeg=b"z" * 10)])
        with pytest.raises(VideoProtocolError, match="truncated jpeg"):
            unpack_region_payload(data[:-3])

    def test_unpack_region_frame_rejects_wrong_codec(self):
        wire = pack_video_frame(_header(codec=CODEC_MJPEG_FULL), b"not-a-region")
        with pytest.raises(VideoProtocolError, match="not a region frame"):
            unpack_region_frame(wire)

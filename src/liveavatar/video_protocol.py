"""LiveAvatar video transport wire protocol v1 — pure pack/unpack codecs.

Implements the framing defined in ``docs/PROTOCOL.md`` (frozen at R2-M0,
2026-08-30). No I/O lives here: pack/unpack are pure functions shared by
the server sink (M1) and the measurement tool (scripts/wsperf.py), so
every code path is unit-testable.

Wire layout (little-endian, 26-byte fixed header)::

    offset type  field
    0      u8    msg_type        1 = video_frame
    1      u8    flags           bit0 keyframe, bit1 epoch_boundary, bit2 eof
    2      u8    codec           0 = mjpeg_full, 1 = region_delta
    3      u8    quality         JPEG quality (1-100)
    4      u16   seq
    6      u32   epoch
    10     i64   pts_us
    18     u16   width
    20     u16   height
    22     u32   payload_len
    26     ...   payload

Region payload (codec=1): ``u16 patch_count`` then per patch
``u16 x | u16 y | u16 w | u16 h | u32 jpg_len | JPEG bytes``. Patches are
independent replacement crops (NOT residuals) — every frame decodes
standalone and may be dropped arbitrarily.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# ───────────────────────────────────────────────────── protocol constants

MSG_VIDEO = 1
KNOWN_MSG_TYPES = frozenset({MSG_VIDEO})

FLAG_KEYFRAME = 0x01
FLAG_EPOCH_BOUNDARY = 0x02
FLAG_EOF = 0x04
KNOWN_FLAG_BITS = FLAG_KEYFRAME | FLAG_EPOCH_BOUNDARY | FLAG_EOF

CODEC_MJPEG_FULL = 0
CODEC_REGION_DELTA = 1
KNOWN_CODECS = frozenset({CODEC_MJPEG_FULL, CODEC_REGION_DELTA})

_HEADER = struct.Struct("<BBBBHIqHHI")
HEADER_SIZE = _HEADER.size  # 26

_REGION_PREFIX = struct.Struct("<H")
_PATCH_HEAD = struct.Struct("<HHHHI")


class VideoProtocolError(ValueError):
    """Fatal wire-protocol violation (receiver must close the connection)."""


# ───────────────────────────────────────────────────────────── data model


@dataclass(frozen=True, slots=True)
class VideoFrameHeader:
    """Parsed/packable 26-byte frame header (payload excluded)."""

    flags: int
    codec: int
    quality: int
    seq: int
    epoch: int
    pts_us: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Patch:
    """One region_delta replacement crop (top-left origin, JPEG payload)."""

    x: int
    y: int
    w: int
    h: int
    jpeg: bytes


# ──────────────────────────────────────────────────────────── flag helpers


def make_flags(
    *, keyframe: bool = False, epoch_boundary: bool = False, eof: bool = False
) -> int:
    """Compose the flags byte from booleans."""
    flags = 0
    if keyframe:
        flags |= FLAG_KEYFRAME
    if epoch_boundary:
        flags |= FLAG_EPOCH_BOUNDARY
    if eof:
        flags |= FLAG_EOF
    return flags


def has_flag(flags: int, bit: int) -> bool:
    """True when ``bit`` (one of the FLAG_* constants) is set."""
    return bool(flags & bit)


# ────────────────────────────────────────────────────────── frame pack/unpack


def pack_video_frame(header: VideoFrameHeader, payload: bytes) -> bytes:
    """Pack header + payload into one wire frame.

    Raises :class:`VideoProtocolError` on invalid field values so that a
    bug in the sender cannot produce silently-corrupt traffic.
    """
    if header.flags & ~KNOWN_FLAG_BITS:
        raise VideoProtocolError(f"reserved flag bits set: {header.flags:#x}")
    if header.codec not in KNOWN_CODECS:
        raise VideoProtocolError(f"unknown codec: {header.codec}")
    if not 1 <= header.quality <= 100:
        raise VideoProtocolError(f"quality out of range: {header.quality}")
    if len(payload) >= 2**32:
        raise VideoProtocolError(f"payload too large: {len(payload)} bytes")
    return _HEADER.pack(
        MSG_VIDEO,
        header.flags,
        header.codec,
        header.quality,
        header.seq,
        header.epoch,
        header.pts_us,
        header.width,
        header.height,
        len(payload),
    ) + payload


def unpack_video_frame(buf: bytes | memoryview) -> tuple[VideoFrameHeader, bytes]:
    """Unpack one complete wire frame.

    Raises :class:`VideoProtocolError` on truncation or any unknown field
    value (see PROTOCOL.md §5) — receivers must treat these as fatal.
    """
    view = memoryview(buf)
    if len(view) < HEADER_SIZE:
        raise VideoProtocolError(f"truncated header: {len(view)} < {HEADER_SIZE}")
    (
        msg_type,
        flags,
        codec,
        _quality,
        seq,
        epoch,
        pts_us,
        width,
        height,
        payload_len,
    ) = _HEADER.unpack_from(view)

    if msg_type not in KNOWN_MSG_TYPES:
        raise VideoProtocolError(f"unknown msg_type: {msg_type}")
    if flags & ~KNOWN_FLAG_BITS:
        raise VideoProtocolError(f"reserved flag bits set: {flags:#x}")
    if codec not in KNOWN_CODECS:
        raise VideoProtocolError(f"unknown codec: {codec}")
    if len(view) - HEADER_SIZE != payload_len:
        raise VideoProtocolError(
            f"payload_len mismatch: declared {payload_len}, got {len(view) - HEADER_SIZE}"
        )

    header = VideoFrameHeader(
        flags=flags,
        codec=codec,
        quality=_quality,
        seq=seq,
        epoch=epoch,
        pts_us=pts_us,
        width=width,
        height=height,
    )
    payload = bytes(view[HEADER_SIZE:])
    return header, payload


# ──────────────────────────────────────────────── region payload pack/unpack


def pack_region_payload(patches: list[Patch]) -> bytes:
    """Pack region_delta payload (§3.2 of PROTOCOL.md)."""
    out = bytearray(_REGION_PREFIX.pack(len(patches)))
    for p in patches:
        if p.w <= 0 or p.h <= 0:
            raise VideoProtocolError(f"patch geometry invalid: {p.w}x{p.h}")
        if not p.jpeg:
            raise VideoProtocolError("patch jpeg payload is empty")
        out += _PATCH_HEAD.pack(p.x, p.y, p.w, p.h, len(p.jpeg))
        out += p.jpeg
    return bytes(out)


def unpack_region_payload(data: bytes) -> list[Patch]:
    """Unpack region_delta payload; raises on any geometry/truncation error."""
    view = memoryview(data)
    if len(view) < _REGION_PREFIX.size:
        raise VideoProtocolError("truncated region payload: no patch count")
    (count,) = _REGION_PREFIX.unpack_from(view)
    offset = _REGION_PREFIX.size
    patches: list[Patch] = []
    for i in range(count):
        if len(view) - offset < _PATCH_HEAD.size:
            raise VideoProtocolError(f"truncated patch header at index {i}")
        x, y, w, h, jpg_len = _PATCH_HEAD.unpack_from(view, offset)
        offset += _PATCH_HEAD.size
        if w == 0 or h == 0:
            raise VideoProtocolError(f"patch {i}: zero geometry {w}x{h}")
        if jpg_len == 0:
            raise VideoProtocolError(f"patch {i}: empty jpeg payload")
        if len(view) - offset < jpg_len:
            raise VideoProtocolError(f"patch {i}: truncated jpeg ({jpg_len} declared)")
        patches.append(
            Patch(x=x, y=y, w=w, h=h, jpeg=bytes(view[offset : offset + jpg_len]))
        )
        offset += jpg_len
    return patches


def pack_region_frame(header: VideoFrameHeader, patches: list[Patch]) -> bytes:
    """Convenience: pack a codec=1 frame from structured patches."""
    return pack_video_frame(header, pack_region_payload(patches))


def unpack_region_frame(
    buf: bytes | memoryview,
) -> tuple[VideoFrameHeader, list[Patch]]:
    """Convenience: unpack a codec=1 frame into header + structured patches."""
    header, payload = unpack_video_frame(buf)
    if header.codec != CODEC_REGION_DELTA:
        raise VideoProtocolError(f"not a region frame: codec={header.codec}")
    return header, unpack_region_payload(payload)

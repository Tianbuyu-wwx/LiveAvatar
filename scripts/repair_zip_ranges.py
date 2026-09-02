#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Repair a partially-corrupt zip by re-downloading only the bad byte ranges.

A resumed download over a flaky mirror can splice in garbage while keeping the
total file size correct (curl -C - sees a full-size file and passes). This tool
walks every zip member, decompresses it in place, finds corrupted byte ranges,
and refetches exactly those ranges with ``curl -r start-end`` until the whole
archive passes ``zipfile.testzip()``.

Usage:
    python scripts/repair_zip_ranges.py <zip-path> <url>
"""

from __future__ import annotations

import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path


def _member_range(f: object, info: zipfile.ZipInfo) -> tuple[int, int, int]:
    """Return (start, data_start, end) for a member's local header + payload."""
    f.seek(info.header_offset)  # type: ignore[attr-defined]
    header = f.read(30)  # type: ignore[attr-defined]
    if len(header) != 30 or struct.unpack("<I", header[:4])[0] != 0x04034B50:
        end = info.header_offset + 30 + info.compress_size - 1
        return info.header_offset, info.header_offset + 30, end
    fname_len, extra_len = struct.unpack("<HH", header[26:30])
    data_start = info.header_offset + 30 + fname_len + extra_len
    return info.header_offset, data_start, data_start + info.compress_size - 1


def _find_bad_ranges(zip_path: Path) -> list[tuple[int, int]]:
    """Scan all members; return merged absolute byte ranges that fail to decompress."""
    bad: list[tuple[int, int]] = []
    with open(zip_path, "rb") as f, zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            start, data_start, end = _member_range(f, info)
            f.seek(data_start)
            raw = f.read(end - data_start + 1)
            try:
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    out = zlib.decompressobj(-15).decompress(raw)
                    if len(out) != info.file_size:
                        raise zlib.error("size mismatch")
                elif info.compress_type != zipfile.ZIP_STORED:
                    continue
            except zlib.error:
                bad.append((start, end))
                print(f"[bad] {info.filename}  bytes {start}-{end}")
    # merge overlapping/adjacent ranges (small gaps are headers between members —
    # refetching them too is harmless and avoids hundreds of tiny requests)
    bad.sort()
    merged: list[list[int]] = []
    for s, e in bad:
        if merged and s <= merged[-1][1] + 1_000_000:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _refetch(zip_path: Path, url: str, start: int, end: int) -> None:
    curl = "curl.exe" if sys.platform == "win32" else "curl"
    tmp = zip_path.with_suffix(".part")
    expected = end - start + 1
    # manual resume: curl cannot combine -C - with -r, so track fetched bytes
    # and request only the remaining subrange, appending chunk by chunk.
    have = tmp.stat().st_size if tmp.exists() else 0
    if have > expected:
        tmp.unlink()
        have = 0
    while have < expected:
        sub_end = end
        proc = subprocess.run(
            [curl, "-sSL", "-r", f"{start + have}-{sub_end}",
             "-o", str(tmp.with_suffix(".chunk")), url]
        )
        chunk = tmp.with_suffix(".chunk")
        got = chunk.stat().st_size if chunk.exists() else 0
        if proc.returncode not in (0, 18) or got == 0:
            chunk.unlink(missing_ok=True)
            print(f"[curl] range fetch retry (rc={proc.returncode})")
            continue
        with open(tmp, "ab") as dst, open(chunk, "rb") as src:
            dst.write(src.read())
        chunk.unlink()
        have += got
        if have < expected:
            print(f"[curl] partial range: {have}/{expected} bytes, resuming ...")
    if have != expected:
        raise RuntimeError(f"range fetch failed: {start}-{end}")
    payload = tmp.read_bytes()
    tmp.unlink()
    with open(zip_path, "r+b") as f:
        f.seek(start)
        f.write(payload)


def main() -> int:
    zip_path = Path(sys.argv[1])
    url = sys.argv[2]
    for round_no in range(1, 6):
        print(f"=== repair round {round_no}: scanning {zip_path.name} ===")
        ranges = _find_bad_ranges(zip_path)
        if not ranges:
            print("[scan] all members decompress cleanly, running full testzip ...")
            with zipfile.ZipFile(zip_path) as zf:
                bad_member = zf.testzip()
            if bad_member is None:
                print("[done] zip verified OK")
                return 0
            print(f"[warn] testzip still reports {bad_member!r}; full refetch needed")
            return 2
        for start, end in ranges:
            print(f"[fix] refetching bytes {start}-{end} ({(end - start + 1) / 1e6:.2f} MB)")
            _refetch(zip_path, url, start, end)
    print("[fail] still corrupt after 5 repair rounds")
    return 1


if __name__ == "__main__":
    sys.exit(main())

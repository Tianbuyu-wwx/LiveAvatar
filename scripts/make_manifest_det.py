# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""One-off: build data/face_ds/manifest_det.jsonl for M1 retraining.

- w300 entries: replace GT-derived tight boxes with YuNet detection boxes
  (full-face, human-annotation style — same family as WIDER labels and the
  acceptance reference), falling back to the GT box when YuNet misses.
- w300 entries duplicated x8 (under-represented hi-res source).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(_SCRIPTS_DIR), str(_SCRIPTS_DIR.parent / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from liveavatar import face_backend as fb  # noqa: E402

OVERSAMPLE = 8
OUT = Path("data/face_ds/manifest_det.jsonl")

_rows_text = Path("data/face_ds/manifest.jsonl").read_text(encoding="utf-8")
rows = [json.loads(line) for line in _rows_text.splitlines()]
out_lines: list[str] = []
replaced = 0
fallback = 0
for r in rows:
    if r.get("source") != "w300":
        out_lines.append(json.dumps(r, ensure_ascii=False))
        continue
    img = r["image"]
    bgr = cv2.imdecode(np.fromfile(img, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"skip unreadable: {img}")
        continue
    h, w = bgr.shape[:2]
    try:
        hits = fb.detect_faces(bgr, backend="yunet", conf_threshold=0.5)
    except Exception as exc:  # noqa: BLE001 - YuNet model missing etc.
        print(f"yunet failed on {img}: {exc}")
        hits = []
    if hits:
        best = max(hits, key=lambda f: (f.x2 - f.x1) * (f.y2 - f.y1))
        boxes = [[best.x1 / w, best.y1 / h, (best.x2 - best.x1) / w, (best.y2 - best.y1) / h]]
        replaced += 1
    else:
        boxes = r["boxes"]  # keep GT tight box as fallback
        fallback += 1
    for _ in range(OVERSAMPLE):
        out_lines.append(json.dumps({**r, "boxes": boxes}, ensure_ascii=False))

OUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
summary = (
    f"written {OUT}: {len(out_lines)} rows | "
    f"w300 yunet-replaced: {replaced}, GT-fallback: {fallback}"
)
print(summary)

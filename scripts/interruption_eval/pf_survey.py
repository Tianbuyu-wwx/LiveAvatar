# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""P-F2: self-contained paired-rating survey page generator.

``write_survey_page(out_dir, manifest)`` renders ``survey/index.html`` with
the (arm-free) manifest embedded, copies the referenced clips next to it,
so the whole ``survey/`` folder is portable: open index.html locally
(file:// works — no server needed) or drop it on any static host.

Page protocol (simplified ITU-ACR dual stimulus):
- participant id (P01..P20) → seeded RNG (mulberry32 over the id) shuffles
  group order AND the within-pair presentation order, so every participant
  sees a different sequence;
- per clip: watch, then rate naturalness and interruption comfort (1-5);
  rating panel unlocks when the video ends (re-watch allowed);
- progress autosaves to localStorage; on finish the ratings download as
  ``pf_ratings_<id>.json`` for the experimenter to collect.
The page never sees arm identities — results reference neutral group/clip
ids only; pf_analyze.py maps them back through pack_key.json.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_CLIPS_SUBDIR = "clips"


def write_survey_page(out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_out = out_dir / _CLIPS_SUBDIR
    clips_out.mkdir(exist_ok=True)
    src_root = out_dir.parent / "clips"
    for g in manifest["groups"]:
        for clip in g["clips"]:
            src = src_root / clip
            if not src.exists():
                raise FileNotFoundError(f"clip missing: {src}")
            shutil.copyfile(src, clips_out / clip)

    manifest_js = json.dumps(manifest, ensure_ascii=False)
    html = _HTML_TEMPLATE.replace("__MANIFEST__", manifest_js)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>虚拟主播打断体验评价</title>
<style>
  body { font-family: system-ui, sans-serif; background: #f5f6f8; color: #222;
         margin: 0; display: flex; justify-content: center; }
  .card { background: #fff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.08);
          max-width: 640px; width: 100%; margin: 24px; padding: 24px 28px; }
  h1 { font-size: 20px; } h2 { font-size: 17px; }
  video { width: 100%; max-width: 480px; display: block; margin: 12px auto;
          border-radius: 8px; background: #000; }
  .scale { display: flex; gap: 8px; margin: 6px 0 14px; }
  .scale button { flex: 1; padding: 10px 0; font-size: 15px; border: 1.5px solid #bbb;
                  border-radius: 8px; background: #fff; cursor: pointer; }
  .scale button.sel { background: #2563eb; color: #fff; border-color: #2563eb; }
  .dim { font-size: 14px; color: #555; margin-top: 10px; }
  .primary { display: block; width: 100%; padding: 12px; margin-top: 16px; font-size: 16px;
             border: 0; border-radius: 8px; background: #2563eb; color: #fff; cursor: pointer; }
  .primary:disabled { background: #9db7f0; cursor: not-allowed; }
  .muted { color: #777; font-size: 13px; }
  .prog { font-size: 13px; color: #888; margin-bottom: 8px; }
  .lock { opacity: .45; pointer-events: none; }
  .err { color: #c0392b; font-size: 14px; margin-top: 8px; }
</style>
</head>
<body>
<div class="card">
  <!-- ── screen 1: intro ── -->
  <section id="s-intro">
    <h1>虚拟主播打断体验评价</h1>
    <p>您将看到 <b>24 段短视频</b>（12 组、每组 2 段）。每段视频中，
    一个虚拟主播正在说话时被新的语音请求打断，随后开始回答新的内容。</p>
    <p>请从两个角度为<b>每段视频</b>打分（1–5 分）：</p>
    <ul>
      <li><b>口型自然度</b>：打断瞬间及之后的嘴部动作是否自然
      （有无冻结、跳变、口型与声音对不上）。</li>
      <li><b>打断舒适度</b>：整体上这次"话被打断"的体验是否舒适
      （停顿多久、反应快慢、是否突兀）。</li>
    </ul>
    <p class="muted">说明：每组内的两段视频可能来自不同的虚拟形象，请分别独立
    评价；每段视频可重看；全部约需 8–10 分钟。数据仅用于学术研究。</p>
    <label>被试编号（实验员提供，如 P01）：<br>
      <input id="pid" placeholder="P01" maxlength="8"
             style="font-size:16px;padding:8px;margin-top:6px;width:140px">
    </label>
    <div id="intro-err" class="err"></div>
    <button id="btn-start" class="primary">开始</button>
  </section>

  <!-- ── screen 2: rating ── -->
  <section id="s-rate" style="display:none">
    <div class="prog" id="prog"></div>
    <h2 id="clip-title"></h2>
    <video id="player" src="" controls playsinline preload="auto"></video>
    <div id="panel" class="lock">
      <div class="dim"><b>口型自然度</b>（1=非常不自然，5=完全自然）</div>
      <div class="scale" id="sc-nat"></div>
      <div class="dim"><b>打断舒适度</b>（1=非常不舒适，5=非常舒适）</div>
      <div class="scale" id="sc-com"></div>
      <button id="btn-next" class="primary" disabled>确认</button>
    </div>
  </section>

  <!-- ── screen 3: done ── -->
  <section id="s-done" style="display:none">
    <h1>完成，感谢参与！</h1>
    <p>已生成结果文件 <b id="done-file"></b>（浏览器已自动下载）。</p>
    <p>请将该文件发送给实验员。若下载未弹出，
    <a href="#" id="dl-again">点击此处重新下载</a>。</p>
  </section>
</div>
<script>
const MANIFEST = __MANIFEST__;
const DIMS = ["naturalness", "comfort"];
const LS_KEY = "pf_survey_v1";

// ── seeded RNG: deterministic per-participant presentation order ──
function mulberry32(a) {
  return function() {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function hashId(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function shuffled(arr, rnd) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ── state ──
let S = null;  // {pid, order:[{group, clips:[{clip, shown_as}]}], idx, trials:[]}

function save() { localStorage.setItem(LS_KEY, JSON.stringify(S)); }
function restore() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)); } catch { return null; }
}
const $ = id => document.getElementById(id);

function show(screen) {
  for (const s of ["s-intro", "s-rate", "s-done"]) $(s).style.display = "none";
  $(screen).style.display = "block";
}

function scaleButtons(el, onSel) {
  el.innerHTML = "";
  for (let v = 1; v <= 5; v++) {
    const b = document.createElement("button");
    b.textContent = v;
    b.onclick = () => { el.dataset.val = v; onSel(); };
    el.appendChild(b);
  }
}

function markScale(el) {
  [...el.children].forEach(b =>
    b.classList.toggle("sel", Number(b.textContent) === Number(el.dataset.val)));
}

function currentTrial() { return S.order[S.idx]; }

function renderCurrent() {
  const t = currentTrial();
  const posName = t.pos === 0 ? "A" : "B";
  $("prog").textContent =
    `第 ${S.idx + 1} / ${S.order.length} 组 · 片段 ${posName}`;
  $("clip-title").textContent = "请完整观看后打分";
  $("panel").classList.add("lock");
  delete $("sc-nat").dataset.val; delete $("sc-com").dataset.val;
  markScale($("sc-nat")); markScale($("sc-com"));
  $("btn-next").disabled = true;
  $("btn-next").textContent = t.pos === 0 ? "确认，看组内下一段" : "确认，下一组";
  const p = $("player");
  p.src = "clips/" + t.clips[t.pos];
  p.load();
}

function unlockIfEnded() { $("panel").classList.remove("lock"); }

function commitRating() {
  const t = currentTrial();
  const clip = t.clips[t.pos];
  const prev = S.trials.find(x => x.group === t.group && x.clip === clip);
  const rec = {
    group: t.group, clip: clip, shown_as: t.pos === 0 ? "A" : "B",
    naturalness: Number($("sc-nat").dataset.val),
    comfort: Number($("sc-com").dataset.val),
  };
  if (prev) Object.assign(prev, rec); else S.trials.push(rec);
  if (t.pos === 0) { t.pos = 1; save(); renderCurrent(); }
  else {
    t.pos = 0; S.idx++;
    save();
    if (S.idx >= S.order.length) finish(); else renderCurrent();
  }
}

function finish() {
  const out = {
    participant_id: S.pid,
    finished_at: new Date().toISOString(),
    user_agent: navigator.userAgent,
    trials: S.trials,
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const name = `pf_ratings_${S.pid}.json`;
  $("done-file").textContent = name;
  const dl = () => {
    const a = document.createElement("a");
    a.href = url; a.download = name; a.click();
  };
  $("dl-again").onclick = e => { e.preventDefault(); dl(); };
  dl();
  show("s-done");
}

// ── wiring ──
$("btn-start").onclick = () => {
  const pid = $("pid").value.trim().toUpperCase();
  if (!/^P\\d{2}$/.test(pid)) {
    $("intro-err").textContent = "编号格式应为 P + 两位数字，如 P07";
    return;
  }
  const saved = restore();
  // resume only if the saved plan still matches the CURRENT manifest
  // (clip ids change when the pack is rebuilt) and work remains
  const valid = saved && saved.pid === pid && Array.isArray(saved.order)
    && saved.idx <= saved.order.length
    && saved.order.every(t => MANIFEST.groups.some(g => g.id === t.group
        && t.clips.every(c => g.clips.includes(c))));
  if (valid && saved.idx < saved.order.length) {
    S = saved;                       // resume
  } else {
    const rnd = mulberry32(hashId(pid));
    const groups = shuffled(MANIFEST.groups, rnd).map(g => {
      const clips = shuffled(g.clips, rnd);
      return {group: g.id, clips: clips, pos: 0};
    });
    S = {pid: pid, order: groups, idx: 0, trials: [],
         started_at: new Date().toISOString()};
    save();
  }
  show("s-rate");
  renderCurrent();
};

$("player").addEventListener("ended", unlockIfEnded);
scaleButtons($("sc-nat"), () => {
  markScale($("sc-nat"));
  $("btn-next").disabled = !($("sc-nat").dataset.val && $("sc-com").dataset.val);
});
scaleButtons($("sc-com"), () => {
  markScale($("sc-com"));
  $("btn-next").disabled = !($("sc-nat").dataset.val && $("sc-com").dataset.val);
});
$("btn-next").onclick = commitRating;
</script>
</body>
</html>
"""

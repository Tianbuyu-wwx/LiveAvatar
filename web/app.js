/* LiveAvatar web demo:
   1. POST /v1/sessions → {session_id, token, url, video_ws}
   2. If the session exposes video_ws (self-developed transport), render the
      /v1/sessions/{id}/video stream onto a canvas via AvatarPlayer;
      otherwise join the LiveKit room with the token.
   3. Stream the selected wav over WS /v1/sessions/{id}/audio in ~100ms chunks
      paced at real time; "打断" sends {"type":"cancel"} with a bumped epoch. */

import { AvatarPlayer } from "/player.js";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const vstatsEl = $("vstats");

let ws = null;
let room = null;
let player = null;
let actx = null;
let sessionId = null;
let epoch = 0;
let playing = false;
let statsTimer = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function showCanvas(show) {
  $("placeholder").style.display = show ? "none" : "";
  $("canvas").style.display = show ? "block" : "none";
}

function attachPlayer(sess) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}${sess.video_ws}`;
  player = new AvatarPlayer($("canvas"), {
    onStatus: (t) => setStatus(t),
  });
  // Throttled stats line (1 Hz).
  let lastStats = null;
  player.onStats = (s) => {
    lastStats = s;
  };
  statsTimer = setInterval(() => {
    if (!lastStats) return;
    vstatsEl.textContent =
      `${lastStats.fps.toFixed(1)} fps · ` +
      `${(lastStats.kbps / 1000).toFixed(2)} Mbps · ` +
      `丢帧 ${lastStats.droppedSeqGaps} · ` +
      `关键帧 ${lastStats.keyframes}`;
  }, 1000);
  return player.connect(url).then(() => {
    showCanvas(true);
  });
}

async function start() {
  const file = $("wavFile").files[0];
  if (!file) {
    setStatus("请先选择 wav 文件");
    return;
  }
  $("startBtn").disabled = true;
  try {
    // 1. Create session.
    setStatus("创建会话…");
    const resp = await fetch("/v1/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!resp.ok) throw new Error("create session failed: " + (await resp.text()));
    const sess = await resp.json();
    sessionId = sess.session_id;

    // 2. Video: self-developed transport when available, else LiveKit.
    if (sess.video_ws) {
      setStatus("连接视频流（自研传输）…");
      await attachPlayer(sess);
    } else {
      setStatus("连接 LiveKit…");
      room = new LivekitClient.Room({
        adaptiveStream: true,
        dynacast: true,
        videoCaptureDefaults: { resolution: { width: 512, height: 512 } },
      });
      room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === "video") {
          showCanvas(false);
          $("video").style.display = "block";
          track.attach($("video"));
        }
      });
      await room.connect(sess.url, sess.token);
    }

    // 3. Stream the wav over WS in real time.
    setStatus("推流中…");
    const arrayBuf = await file.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    actx = new AudioCtx({ sampleRate: 16000 });
    const decoded = await actx.decodeAudioData(arrayBuf.slice(0));
    // Convert to 16kHz mono int16 PCM.
    const src = decoded.getChannelData(0);
    const offline = new OfflineAudioContext(1, Math.ceil(src.length * 16000 / decoded.sampleRate), 16000);
    const buf = offline.createBuffer(1, src.length, decoded.sampleRate);
    buf.copyToChannel(src, 0);
    const srcNode = offline.createBufferSource();
    srcNode.buffer = buf;
    srcNode.connect(offline.destination);
    const rendered = await offline.startRendering();
    const f32 = rendered.getChannelData(0);
    const pcm = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      pcm[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)));
    }

    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/v1/sessions/${sess.session_id}/audio`);
    ws.binaryType = "arraybuffer";
    await new Promise((res, rej) => {
      ws.onopen = res;
      ws.onerror = rej;
    });

    // Local playback so the user hears the audio while the avatar lip-syncs.
    const playSrc = actx.createBufferSource();
    playSrc.buffer = rendered;
    playSrc.connect(actx.destination);
    playSrc.start();

    playing = true;
    epoch += 1;
    ws.send(JSON.stringify({ type: "epoch", epoch }));

    const chunkSamples = 1600; // 100ms
    const chunkBytes = chunkSamples * 2;
    const bytes = pcm.buffer;
    let offset = 0;
    const pump = () => {
      if (!playing) return;
      if (offset >= bytes.byteLength) {
        setStatus("播放完成");
        stopStream(false);
        return;
      }
      ws.send(bytes.slice(offset, offset + chunkBytes));
      offset += chunkBytes;
      setTimeout(pump, 100);
    };
    ws.onopen = null;
    pump();
    $("stopBtn").disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("出错: " + err.message);
    $("startBtn").disabled = false;
  }
}

function stopStream(interrupt) {
  playing = false;
  if (ws && ws.readyState === WebSocket.OPEN && interrupt) {
    epoch += 1;
    ws.send(JSON.stringify({ type: "cancel", epoch }));
  }
  $("stopBtn").disabled = true;
  $("startBtn").disabled = false;
  if (interrupt) setStatus("已打断（epoch " + epoch + "）");
}

async function teardown() {
  stopStream(false);
  if (statsTimer) {
    clearInterval(statsTimer);
    statsTimer = null;
  }
  if (player) {
    player.close();
    player = null;
    showCanvas(false);
    vstatsEl.textContent = "";
  }
  if (room) {
    await room.disconnect();
    room = null;
  }
  // Release the server-side session immediately (avoids waiting for TTL reap).
  if (sessionId) {
    try {
      await fetch(`/v1/sessions/${sessionId}`, { method: "DELETE" });
    } catch (err) {
      console.warn("session close failed", err);
    }
    sessionId = null;
  }
  if (actx) {
    actx.close().catch(() => {});
    actx = null;
  }
}

$("startBtn").onclick = start;
$("stopBtn").onclick = () => stopStream(true);
window.addEventListener("beforeunload", teardown);

// Enable the start button once a file is picked.
$("wavFile").addEventListener("change", () => {
  $("startBtn").disabled = !$("wavFile").files.length;
});

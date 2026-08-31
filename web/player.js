/* AvatarPlayer — browser client for the self-developed video transport
   (docs/PROTOCOL.md v1).

   Server → client: JSON "ready", binary wire frames, JSON none.
   Client → server: JSON control ("keyframe_request" / "feedback").

   Wire frame layout (little-endian, 26-byte header):
     u8 msg_type(1) | u8 flags | u8 codec | u8 quality
     u16 seq | u32 epoch | i64 pts_us | u16 width | u16 height | u32 payload_len

   Flags: bit0 keyframe, bit1 epoch_boundary, bit2 eof.
   Codecs: 0 = mjpeg_full, 1 = region_delta (M4+).

   Frames are rendered as they arrive (TTS audio is the master clock); on
   epoch_boundary the canvas state is simply replaced by the next keyframe. */

const HEADER_SIZE = 26;
const REGION_PREFIX_SIZE = 2;
const PATCH_HEAD_SIZE = 12; // u16 x, u16 y, u16 w, u16 h, u32 len
const FLAG_KEYFRAME = 0x01;
const FLAG_EPOCH_BOUNDARY = 0x02;
const FLAG_EOF = 0x04;
const CODEC_MJPEG_FULL = 0;
const CODEC_REGION_DELTA = 1;

export class AvatarPlayer {
  constructor(canvas, { onStats, onStatus, feather = 4 } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.feather = feather;
    // Offscreen base image: the static background shared by region patches.
    this.base = document.createElement("canvas");
    this.baseCtx = this.base.getContext("2d");
    this._patchTmp = document.createElement("canvas");
    this._patchTmpCtx = this._patchTmp.getContext("2d", { willReadFrequently: true });
    this.onStats = onStats || null;
    this.onStatus = onStatus || null;
    this.ws = null;
    this.closed = false;
    this.lastSeq = null;
    this.stats = {
      frames: 0,
      droppedSeqGaps: 0,
      bytes: 0,
      keyframes: 0,
      startedAt: 0,
    };
    this._decodePending = 0;
    // M5 feedback loop: report congestion signals every 500 ms so the
    // server's quality controller can degrade fast and recover within
    // its 2 s budget (3 consecutive healthy reports → full quality).
    this._lastReport = null;
    this._feedbackTimer = setInterval(() => this._sendFeedback(), 500);
  }

  /** Report incremental congestion signals to the server. */
  _sendFeedback() {
    const s = this.stats;
    if (!this._lastReport) {
      this._lastReport = {
        frames: s.frames,
        droppedSeqGaps: s.droppedSeqGaps,
        bytes: s.bytes,
        at: performance.now(),
      };
      return;
    }
    const prev = this._lastReport;
    const dt = (performance.now() - prev.at) / 1000;
    const dFrames = s.frames - prev.frames;
    if (dt <= 0 || dFrames <= 0) return;
    this.sendFeedback({
      seq_gaps: s.droppedSeqGaps - prev.droppedSeqGaps,
      frames: dFrames,
      kbps: ((s.bytes - prev.bytes) * 8) / dt / 1000,
      fps: dFrames / dt,
    });
    this._lastReport = {
      frames: s.frames,
      droppedSeqGaps: s.droppedSeqGaps,
      bytes: s.bytes,
      at: performance.now(),
    };
  }

  /** Connect and start rendering. Resolves once the server sends "ready". */
  connect(url) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      let settled = false;
      ws.onopen = () => {
        /* wait for "ready" */
      };
      ws.onerror = () => {
        if (!settled) {
          settled = true;
          reject(new Error("video ws error"));
        }
      };
      ws.onclose = () => {
        if (!settled) {
          settled = true;
          reject(new Error("video ws closed before ready"));
        } else if (!this.closed) {
          this._status("video 连接断开");
        }
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          const msg = JSON.parse(ev.data);
          if (msg.type === "ready") {
            settled = true;
            this.stats.startedAt = performance.now();
            this.base.width = msg.width;
            this.base.height = msg.height;
            this._status(`视频已连接 (${msg.codec} ${msg.width}x${msg.height})`);
            resolve(msg);
          }
          return;
        }
        this._handleWire(new DataView(ev.data));
      };
    });
  }

  close() {
    this.closed = true;
    if (this._feedbackTimer) {
      clearInterval(this._feedbackTimer);
      this._feedbackTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /** Ask the server for a fresh keyframe (e.g. after tab visibility loss). */
  requestKeyframe() {
    this._sendControl({ type: "keyframe_request" });
  }

  /** Send quality feedback (M5 adaptive quality; server may ignore it). */
  sendFeedback(payload) {
    this._sendControl({ type: "feedback", ...payload });
  }

  // ------------------------------------------------------------- internals

  _handleWire(dv) {
    if (dv.byteLength < HEADER_SIZE) return;
    const flags = dv.getUint8(1);
    const codec = dv.getUint8(2);
    const seq = dv.getUint16(4, true);
    const pts = dv.getBigInt64(10, true);
    const width = dv.getUint16(18, true);
    const height = dv.getUint16(20, true);
    const payloadLen = dv.getUint32(22, true);
    if (dv.byteLength < HEADER_SIZE + payloadLen) return;

    if (flags & FLAG_EOF) {
      this._status("视频流结束");
      this.close();
      return;
    }

    // Sequence-gap detection (u16 wraparound).
    if (this.lastSeq !== null) {
      const delta = (seq - this.lastSeq) & 0xffff;
      if (delta > 1) this.stats.droppedSeqGaps += delta - 1;
    }
    this.lastSeq = seq;

    if (flags & FLAG_KEYFRAME) this.stats.keyframes += 1;

    const payload = new Uint8Array(dv.buffer, dv.byteOffset + HEADER_SIZE, payloadLen);
    if (codec === CODEC_MJPEG_FULL) {
      this._renderMjpeg(payload, flags);
    } else if (codec === CODEC_REGION_DELTA) {
      this._renderRegion(payload, flags, width, height);
    } else {
      this._status(`未知编码 ${codec}`);
    }
    this.stats.bytes += HEADER_SIZE + payloadLen;
    void pts;
  }

  _renderMjpeg(payload, flags) {
    const blob = new Blob([payload], { type: "image/jpeg" });
    this._decodePending += 1;
    createImageBitmap(blob)
      .then((bmp) => {
        this._decodePending -= 1;
        if (this.closed) return;
        // Epoch boundary: stale decodes in flight are skipped — the next
        // keyframe repaints the full canvas anyway.
        if (flags & FLAG_EPOCH_BOUNDARY) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.canvas.width !== bmp.width || this.canvas.height !== bmp.height) {
          this.canvas.width = bmp.width;
          this.canvas.height = bmp.height;
        }
        this.ctx.drawImage(bmp, 0, 0);
        bmp.close();
        this.stats.frames += 1;
        this._emitStats();
      })
      .catch(() => {
        this._decodePending -= 1;
      });
  }

  /** region_delta payload: u16 count + patches {u16 x,y,w,h, u32 len, jpeg}. */
  _renderRegion(payload, flags, frameW, frameH) {
    if (this.canvas.width !== frameW || this.canvas.height !== frameH) {
      this.canvas.width = frameW;
      this.canvas.height = frameH;
      this.base.width = frameW;
      this.base.height = frameH;
    }
    const dv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    let off = 0;
    const count = dv.getUint16(off, true);
    off += REGION_PREFIX_SIZE;
    for (let i = 0; i < count; i++) {
      const x = dv.getUint16(off, true);
      const y = dv.getUint16(off + 2, true);
      const w = dv.getUint16(off + 4, true);
      const h = dv.getUint16(off + 6, true);
      const len = dv.getUint32(off + 8, true);
      off += PATCH_HEAD_SIZE;
      const jpeg = payload.subarray(off, off + len);
      off += len;
      this._decodePending += 1;
      createImageBitmap(new Blob([jpeg], { type: "image/jpeg" }))
        .then((bmp) => {
          this._decodePending -= 1;
          if (this.closed) return;
          this._composePatch(bmp, x, y, w, h, flags);
          bmp.close();
          this.stats.frames += 1;
          this._emitStats();
        })
        .catch(() => {
          this._decodePending -= 1;
        });
    }
  }

  /** Compose one decoded patch: full-canvas patches become the new base;
      region patches blit base + feathered patch onto the display canvas. */
  _composePatch(bmp, x, y, w, h, flags) {
    const fullCanvas = x === 0 && y === 0 && w >= this.canvas.width && h >= this.canvas.height;
    if (fullCanvas) {
      // New base frame (covers the static background AND the mouth).
      this.baseCtx.drawImage(bmp, 0, 0, this.canvas.width, this.canvas.height);
      if (flags & FLAG_EPOCH_BOUNDARY) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.drawImage(this.base, 0, 0);
      return;
    }
    // Region patch: repaint base, then the feathered mouth patch on top.
    this.ctx.drawImage(this.base, 0, 0);
    const tmp = this._featherToTemp(bmp, w, h, this.feather);
    if (tmp) {
      this.ctx.drawImage(tmp, x, y);
    } else {
      this.ctx.drawImage(bmp, x, y);
    }
  }

  /** Draw a patch with a linear-alpha feathered border (4 px default) so
      JPEG block edges melt into the static background. Returns a canvas
      (the player's reusable temp surface) or null when no feathering. */
  _featherToTemp(bmp, w, h, f) {
    if (!f || f <= 0 || w <= 2 * f || h <= 2 * f) return null;
    const t = this._patchTmp;
    if (t.width !== w || t.height !== h) {
      t.width = w;
      t.height = h;
    }
    const tctx = this._patchTmpCtx;
    tctx.clearRect(0, 0, w, h);
    tctx.drawImage(bmp, 0, 0, w, h);
    const id = tctx.getImageData(0, 0, w, h);
    const d = id.data;
    for (let yy = 0; yy < h; yy++) {
      const dy = Math.min(yy, h - 1 - yy);
      if (dy >= f) continue;
      for (let xx = 0; xx < w; xx++) {
        const dx = Math.min(xx, w - 1 - xx);
        if (dx >= f) continue;
        const dist = Math.min(dx, dy); // 0..f-1
        d[(yy * w + xx) * 4 + 3] = Math.round((255 * (dist + 1)) / (f + 1));
      }
    }
    tctx.putImageData(id, 0, 0);
    return t;
  }

  _sendControl(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  _emitStats() {
    if (!this.onStats) return;
    const elapsed = (performance.now() - this.stats.startedAt) / 1000;
    this.onStats({
      fps: elapsed > 0 ? this.stats.frames / elapsed : 0,
      kbps: elapsed > 0 ? (this.stats.bytes * 8) / elapsed / 1000 : 0,
      frames: this.stats.frames,
      droppedSeqGaps: this.stats.droppedSeqGaps,
      keyframes: this.stats.keyframes,
    });
  }

  _status(text) {
    if (this.onStatus) this.onStatus(text);
  }
}

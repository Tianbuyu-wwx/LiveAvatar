# LiveAvatar 视频传输协议（Video Transport Protocol v1）

> 状态：**v1.0 冻结**（R2-M0，2026-08-30）　|　规范实现：[src/liveavatar/video_protocol.py](file:///e:/项目/LiveAvatar/src/liveavatar/video_protocol.py)
> 设计依据：[R2 开发计划](file:///e:/项目/LiveAvatar/docs/R2传输层自研开发计划_2026-08-30.md) §3–§5
> 版本策略：v1 冻结后仅允许追加（新增 msg_type / flag / codec 值），不允许修改既有字段语义；收到未知值时**必须**关闭连接（见 §5 错误处理）。

---

## 1. 通道与生命周期

- 视频下行走独立 WebSocket 端点：`GET /v1/sessions/{session_id}/video`（M1 落地）。
- 鉴权与音频 WS 一致：`api_key` 查询参数或 `X-API-Key` 头（服务端开启 `LIVEAVATAR_API_KEY` 时必填）。
- 生命周期：连接 → 服务端发 `ready`（JSON 文本）→ 二进制视频帧下行 → 会话删除/服务端停止时发 `eof_stream` 标记帧并关闭。
- 客户端可随时上行 JSON 控制消息（§4）；二进制上行**不合法**，服务端收到二进制直接关闭。

## 2. 二进制视频帧（服务端 → 客户端，小端序）

帧 = 26 字节定长头 + 变长 payload。

| 偏移 | 类型 | 字段 | 说明 |
|---|---|---|---|
| 0 | u8 | `msg_type` | `1` = video_frame（其余值非法） |
| 1 | u8 | `flags` | bit0 `keyframe`；bit1 `epoch_boundary`；bit2 `eof_stream`；bit3–7 保留（必须为 0） |
| 2 | u8 | `codec` | `0` = mjpeg_full；`1` = region_delta；其余非法 |
| 3 | u8 | `quality` | 本帧 JPEG 编码质量（1–100） |
| 4 | u16 | `seq` | 帧序号，自然回绕（0xFFFF → 0x0000） |
| 6 | u32 | `epoch` | 帧 所属打断纪元 |
| 10 | i64 | `pts_us` | 服务端 PTS（微秒，单调） |
| 18 | u16 | `width` | 画布宽（像素） |
| 20 | u16 | `height` | 画布高（像素） |
| 22 | u32 | `payload_len` | payload 字节数 |
| 26 | … | `payload` | 由 `codec` 决定（§3） |

字段语义：

- **epoch**：与音频/推理侧 epoch 同源。客户端必须丢弃 `epoch < 已见最大 epoch` 的帧（与发布侧丢弃规则一致），收到更大 epoch 时**清空抖动缓冲**并以该帧为新基准。
- **keyframe**：可独立完整解码的帧。客户端在连接建立 / 重连 / 主动请求后，必须等到第一个 keyframe 才开始合成画面。
- **epoch_boundary**：epoch 切换后的第一帧，必然同时置 `keyframe`。
- **eof_stream**：流结束标记，payload 为空；收到后服务端将关闭连接。
- **pts_us**：播放时钟的唯一真源（TTS 音频时长驱动）。客户端据此做漂移校正与播放排序，禁止使用本地到达时间排序。

## 3. Payload 格式

### 3.1 codec=0 mjpeg_full

payload 为单幅完整画面的 JPEG（BGR 源编码）。任意帧独立可解码。

### 3.2 codec=1 region_delta（自研区域独立帧）

payload 布局：

```text
u16 patch_count
patch_count × {
    u16 x          # 块左上角 X（像素）
    u16 y          # 块左上角 Y
    u16 w          # 块宽（>0，偶数对齐）
    u16 h          # 块高（>0，偶数对齐）
    u32 jpg_len    # JPEG 字节数（>0）
    bytes jpeg[jpg_len]
}
```

**核心约束（与残差编码的本质区别）**：每个 patch 是该区域的**完整替换内容**，不是差分残差。因此：

1. 任意一帧（含只含 patch 的 P 帧）都可独立解码 —— 丢帧不产生错误传播；
2. 发送端可**任意丢弃任意帧**（背压、打断、拥塞）而无需考虑参考链；
3. `patch_count = 0` 合法，表示"本帧无变化"的空帧（客户端跳过合成，仅推进时钟）。

### 3.3 通用规则

- `payload_len` 必须与实际剩余字节数一致，不一致视为协议错误。
- region_delta 的帧必须能通过 keyframe 获得底图；keyframe 频率策略由服务端自适应（基准：≥1 次/秒）。

## 4. JSON 控制消息（上行客户端 → 服务端为主）

```jsonc
// 客户端 → 服务端
{"type": "hello", "caps": {"codec": ["region_delta", "mjpeg_full"], "max_fps": 25}}
{"type": "feedback", "rtt_ms": 12.3, "received": 120, "dropped": 2}   // 每 500ms 一次
{"type": "keyframe_request"}

// 服务端 → 客户端（连接建立后第一帧文本消息）
{"type": "ready", "codec": "region_delta", "target_fps": 25, "width": 512, "height": 512}
```

规则：

- 未知的 `type` 忽略（向前兼容）；格式非法的 JSON 忽略。
- 服务端选择 codec 后在 `ready` 中声明，本次连接内不变。
- `keyframe_request` 服务端应尽快响应一个 `keyframe` 帧（限频：≥200ms 间隔）。

## 5. 错误处理

以下任一情况视为**致命协议错误**，接收方必须立即关闭连接（不做容错降级，避免静默坏帧）：

1. `msg_type` / `codec` 为未知值；
2. `flags` 使用了保留位；
3. `payload_len` 与实际不符、帧被截断；
4. region patch 几何非法（`w==0` / `h==0` / `jpg_len==0` / 边界越界 / payload 截断）。

## 6. 设计依据（为什么这样定）

- **帧独立可解码**：与 epoch 打断"丢过期帧"语义同构，拥塞时丢帧安全（借鉴 QUIC RFC 9002"宁丢不等"与 MoQ 草案的独立对象模型）；
- **PTS 单一真源**：延续"TTS 是主时钟"原则（RFC 3550 时钟模型）；
- **26 字节紧凑头**：单会话 25fps 下头开销 ~5.2 KB/s，可忽略；
- **JSON 控制面 + 二进制数据面分离**：控制低频、数据高频，互不阻塞。

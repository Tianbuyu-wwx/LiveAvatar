# LiveAvatar

[![CI](https://github.com/Tianbuyu-wwx/LiveAvatar/actions/workflows/ci.yml/badge.svg)](https://github.com/Tianbuyu-wwx/LiveAvatar/actions/workflows/ci.yml)

**实时流式数字人口型视频生成**：PCM 音频进，MuseTalk 口型视频出，经自研 WebSocket 传输（默认）或 LiveKit 推到浏览器。

LiveAvatar 从一个生产级全双工数字人系统中拆出视频生成全链路，做成独立、可嵌入的开源库 + 服务：

```text
TTS / 麦克风 / wav 文件（16kHz mono PCM S16LE）
    →  AvatarStreamingAdapter（有界队列 + epoch 打断 + 降级链）
        →  AvatarPool（租约池，每角色独占 worker，跨会话隔离）
            →  MuseTalkAvatarWorker（Whisper 特征 → UNet → VAE → 贴回原图）
            →  WebSocketSink（自研二进制协议编解码 + 多客户端扇出）
                →  浏览器 player.js（抖动缓冲 + canvas 合成，实时播放）
```

## 特性

- **毫秒级打断（epoch 机制）**：每个 PCM chunk 与视频帧都携带 epoch；打断时新旧两条通路同时生效——推理侧 CancelToken 即刻熔断，发布侧丢弃所有过期帧。打断后视频在 **1 帧（≤40ms @25fps）内**停止。
- **有界队列反压**：TTS 永远是主时钟，avatar 推理跟不上时丢 chunk 而不是阻塞音频。
- **降级链**：`MuseTalk 推理 → 连续 N 次失败自动切换 StaticAvatarWorker（静帧）→ 音频独播`，新 epoch 自动恢复主 worker。
- **租约式角色池**：每个 avatar 的 face coords / latents / mask 只加载一次，租约 + TTL 回收 + FIFO 等待队列，多会话跨角色零串音。
- **可插拔发布端**：自研 WS 传输（默认，零外部基础设施）或 LiveKit WebRTC（可选过渡）；另有本地 OpenCV 预览 / mp4 落盘（模式 B，零依赖）。
- **自研视频传输（R2）**：26 字节帧头二进制协议（seq/epoch/pts/codec/flags），MJPEG 全帧与**区域独立帧**双编码（口型区域先验，带宽 ↓55–80%），帧独立可解码、任意丢帧不花屏。
- **自适应质量**：客户端周期上报拥塞信号（丢帧率/码率），服务端 EWMA 聚合 + 5 档质量状态机，弱网"降画质不冻结"，恢复带迟滞不抖动。
- **音频来源无关**：接任意 TTS、麦克风、wav 文件——本库只负责"音频进、视频出"。

## 环境要求

- Python ≥ 3.10
- NVIDIA GPU（CUDA）用于 MuseTalk 推理；纯测试/编排/传输逻辑可在 CPU 上跑
- 可选：LiveKit Server（`LIVEAVATAR_TRANSPORT=livekit` 过渡模式）、Docker + nvidia-container-toolkit

## 快速开始

### 1. 安装

```bash
git clone https://github.com/<you>/LiveAvatar.git
cd LiveAvatar
pip install -e ".[vision,server]"   # 推理另需: pip install torch torchvision diffusers transformers einops
```

### 2. 下载模型权重与 demo 数据

```bash
python scripts/download_models.py
```

下载 `models/musetalkV15`（UNet）、`models/sd-vae-ft-mse`（VAE）、`models/whisper`（音频特征）、YuNet 人脸检测（过渡期默认检测后端，单个 onnx 资源文件），以及 yongen demo 视频与音频。

MediaPipe 关键点已不是运行依赖（R1 M5）：仅训练期教师标注自备素材时需要，用 `python scripts/download_models.py --teacher` 按需下载，并安装 `pip install -e ".[teacher]"`。

### 3. 准备一个 avatar

```bash
python scripts/prepare_avatar.py \
    --input data/video/yongen.mp4 \
    --avatar-id yongen \
    --avatar-data-root data/avatars \
    --max-frames 8
```

产出 `data/avatars/yongen/`（full_imgs / coords.pkl / latents.pt / mask/ / mask_coords.pkl）。默认带五点人脸对齐（`LANDMARK_BACKEND=mediapipe|self`，显著提升口型质量；自研 `self` 后端权重训练完成后将成为默认）。

### 4. 模式 B：本地预览（无需 LiveKit）

```bash
python -m liveavatar.preview --audio data/audio/yongen.wav --avatar yongen            # OpenCV 窗口实时预览
python -m liveavatar.preview --audio data/audio/yongen.wav --avatar yongen --save out.mp4
```

实测（单卡 CUDA，batch=4，25fps）：8s 音频 → 200 帧 / 8.00s 视频，0 丢帧 0 错误，推理队列高水位 3，全程实时。

### 5. 服务模式：浏览器 demo

```bash
uvicorn liveavatar.publish:app --host 0.0.0.0 --port 8000
# 打开 http://localhost:8000 ，选一个 wav，点「开始」
```

无需任何外部服务——视频经自研 WS 传输直推浏览器（`web/player.js`）。
CPU 体验（无需 GPU 与模型，合成画面走完整服务代码路径）：

```bash
python scripts/demo_local.py --port 8000            # MJPEG 全帧
python scripts/demo_local.py --codec region --port 8000   # 区域增量编码
```

### LiveKit 过渡模式（可选，deprecated）

```bash
export LIVEAVATAR_TRANSPORT=livekit LIVEKIT_URL=ws://localhost:7880 \
       LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=xxx
uvicorn liveavatar.publish:app --host 0.0.0.0 --port 8000
```

## 配置

全部通过 `LIVEAVATAR_*` 环境变量或 `.env` / `.env.local`：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LIVEAVATAR_AVATAR_DATA_ROOT` | `./avatars` | avatar 数据根目录 |
| `LIVEAVATAR_DEVICE` | `cuda` | 推理设备 |
| `LIVEAVATAR_IS_HALF` | `true` | fp16 推理 |
| `LIVEAVATAR_MAX_WORKERS` | `1` | 同时驻留 GPU 的 avatar worker 数 |
| `LIVEAVATAR_LEASE_TTL` | `300` | 租约时长（秒） |
| `LIVEAVATAR_TARGET_FPS` | `25` | 视频帧率 |
| `LIVEAVATAR_WIDTH` / `LIVEAVATAR_HEIGHT` | `512` / `512` | 输出分辨率 |
| `LIVEAVATAR_BATCH_SIZE` | `4` | 每次推理的音频批长（帧） |
| `LIVEAVATAR_WHISPER_MODEL_PATH` | `models/whisper` | Whisper-tiny 权重 |
| `LIVEAVATAR_MUSETALK_MODEL_DIR` | `models/musetalkV15` | UNet 权重目录 |
| `LIVEAVATAR_VAE_MODEL_DIR` | `models/sd-vae-ft-mse` | VAE 目录 |
| `LIVEAVATAR_MAX_LOADED_WORKERS` | `0` | 同时驻留的 worker 上限（超出按 LRU 卸载；0 = 不限制） |
| `LIVEAVATAR_API_KEY` | *(空)* | 服务端共享密钥；非空时 REST/WS 需携带（`X-API-Key` 头或 WS `api_key` 参数） |
| `LIVEAVATAR_MAX_SESSIONS` | `16` | 并发会话上限（超出返回 429） |
| `LIVEAVATAR_MAX_WS_FRAME_BYTES` | `65536` | WS 二进制帧上限（超限丢弃，防 DoS） |
| `LIVEAVATAR_TRANSPORT` | `ws` | 视频传输：`ws`（自研，默认）或 `livekit`（过渡，deprecated） |
| `LIVEAVATAR_CODEC` | `mjpeg` | ws 传输编码：`mjpeg`（全帧）或 `region`（区域增量，需 avatar 的 `region.json`） |

服务端（`LIVEKIT_URL`、`LIVEKIT_API_KEY`、`LIVEKIT_API_SECRET`、`LIVEKIT_ROOM`、`PUBLIC_LIVEKIT_URL`）仅 livekit 过渡模式需要，见 `liveavatar/publish.py`。

## HTTP / WebSocket API

| 端点 | 说明 |
|---|---|
| `POST /v1/sessions` | 创建会话，返回 `session_id` + video WS 路径 |
| `DELETE /v1/sessions/{id}` | 关闭会话并取消发布视频轨道 |
| `GET /v1/sessions/{id}/stats` | adapter / publisher 计数器快照（含质量档位 tier） |
| `GET /v1/avatars` | 列出可用 avatar |
| `WS /v1/sessions/{id}/audio` | 二进制帧 = PCM S16LE；文本帧 = 控制消息 |
| `WS /v1/sessions/{id}/video` | 订阅视频流（自研二进制协议，见 [docs/PROTOCOL.md](docs/PROTOCOL.md)） |
| `GET /health` | 健康检查 |

音频 WS 控制消息（JSON）：

```jsonc
{"type": "epoch",  "epoch": 3}   // 开始新一段话（新 epoch）
{"type": "cancel", "epoch": 4}   // 打断：丢弃旧 epoch 的音频与视频帧
{"type": "stop"}                  // 结束
```

视频 WS 控制消息（JSON，客户端 → 服务端）：

```jsonc
{"type": "keyframe_request"}     // 请求关键帧（重连/花屏恢复）
{"type": "feedback", "seq_gaps": 3, "frames": 120,
 "kbps": 640.0, "fps": 24.0}     // 拥塞上报 → 自适应质量（M5）
```

## 作为库嵌入

```python
from liveavatar.config import AvatarPoolConfig
from liveavatar.pipeline import AvatarPipeline
from liveavatar.ws_sink import WebSocketSink

pipeline = AvatarPipeline(
    AvatarPoolConfig(avatar_data_root="data/avatars"),
    publisher_factory=lambda cfg: WebSocketSink(width=cfg.width, height=cfg.height),
)
await pipeline.start()
await pipeline.open_session("s1", "yongen")
await pipeline.push_pcm("s1", pcm_chunk)      # pts/epoch 自动管理
pipeline.cancel_epoch("s1", 1)                 # 打断
await pipeline.close_session("s1")
await pipeline.stop()
```

测试时可注入 fake `pool` / `publisher_factory`，300+ 个单测全部不依赖 torch / LiveKit / GPU。

## 目录结构

```text
src/liveavatar/
├── pipeline.py          # 编排器（池 + 会话 + PTS 时钟 + epoch）
├── publish.py           # FastAPI 服务（/audio + /video WS 端点）
├── preview.py           # 模式 B：wav → 本地窗口 / mp4
├── adapter.py           # PCM → worker → publisher，队列/打断/降级
├── pool.py  lease.py    # avatar 租约池（基于 _common 泛型池）
├── worker.py            # AvatarWorker 抽象 + AvatarFrame
├── musetalk_worker.py   # MuseTalk 推理实现
├── static_worker.py     # 降级静帧 worker
├── sinks.py             # PublishSink 协议 + RTMP 后端
├── ws_sink.py           # 自研传输：编码 + 多客户端扇出（默认）
├── video_protocol.py    # 26B 帧头二进制协议 pack/unpack
├── region_codec.py      # 区域独立帧编码器（口型先验）
├── adaptive.py          # 反馈 EWMA + 5 档自适应质量状态机
├── video_publisher.py   # BGR24 → I420 → LiveKit VideoSource（过渡）
├── config.py            # LIVEAVATAR_* 配置
├── _common/             # 跨事件循环队列 / 租约/泛型池原语
└── musetalk/            # MuseTalk 模型定义（自包含）
scripts/                 # download_models / prepare_avatar / wsperf / e2e_bench / demo_local
web/                     # 浏览器 demo（无构建，原生 JS：player.js 抖动缓冲 + canvas 合成）
tests/                   # 300+ 个单测（含协议/传输/自适应/端到端）
```

## Roadmap

- [x] 离线批量渲染（`python -m liveavatar.batch_renderer`）
- [x] ASR/TTS 插件接口草案（`liveavatar.plugins`；官方示例插件待发布）
- [x] 更多发布后端（`liveavatar.sinks.PublishSink` 协议 + RTMP 后端）
- [x] 自研 WS 视频传输 + 区域编码 + 自适应质量（R2，默认启用）
- [ ] Apple Silicon (MPS) 支持

## 许可证

本项目代码以 [MIT](LICENSE) 发布。集成的上游组件遵守各自许可：

| 组件 | 许可 | 说明 |
|---|---|---|
| [MuseTalk](https://github.com/TMElyralab/MuseTalk) | MIT | 代码（本仓库 `musetalk/` 改编自上游）；其预训练模型可自由使用（含商用） |
| [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | MIT | VAE |
| [whisper-tiny](https://huggingface.co/openai/whisper-tiny) | MIT | 音频特征 |
| [YuNet](https://github.com/opencv/opencv_zoo) | Apache-2.0 | 人脸检测（过渡期默认后端，单个 onnx 资源文件） |
| [MediaPipe](https://developers.google.com/mediapipe) | Apache-2.0 | 训练期教师标注专用（`teacher` extra），非运行依赖 |
| [LiveKit](https://livekit.io/) | Apache-2.0 | 实时传输（可选过渡依赖，deprecated） |
| MuseTalk demo 数据（yongen） | — | **仅限非商业研究用途**，来源见上游仓库 |

使用数字人生成内容时请遵守当地法律法规，不得用于伪造他人身份等用途。

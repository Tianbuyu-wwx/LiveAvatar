# LiveAvatar

[![CI](https://github.com/Tianbuyu-wwx/LiveAvatar/actions/workflows/ci.yml/badge.svg)](https://github.com/Tianbuyu-wwx/LiveAvatar/actions/workflows/ci.yml)

**实时流式数字人口型视频生成**：PCM 音频进，MuseTalk 口型视频出，经 LiveKit/WebRTC 推到浏览器。

LiveAvatar 从一个生产级全双工数字人系统中拆出视频生成全链路，做成独立、可嵌入的开源库 + 服务：

```text
TTS / 麦克风 / wav 文件（16kHz mono PCM S16LE）
    →  AvatarStreamingAdapter（有界队列 + epoch 打断 + 降级链）
        →  AvatarPool（租约池，每角色独占 worker，跨会话隔离）
            →  MuseTalkAvatarWorker（Whisper 特征 → UNet → VAE → 贴回原图）
            →  AvatarVideoPublisher（LiveKit 视频轨道）
                →  浏览器 <video> 实时播放
```

## 特性

- **毫秒级打断（epoch 机制）**：每个 PCM chunk 与视频帧都携带 epoch；打断时新旧两条通路同时生效——推理侧 CancelToken 即刻熔断，发布侧丢弃所有过期帧。打断后视频在 **1 帧（≤40ms @25fps）内**停止。
- **有界队列反压**：TTS 永远是主时钟，avatar 推理跟不上时丢 chunk 而不是阻塞音频。
- **降级链**：`MuseTalk 推理 → 连续 N 次失败自动切换 StaticAvatarWorker（静帧）→ 音频独播`，新 epoch 自动恢复主 worker。
- **租约式角色池**：每个 avatar 的 face coords / latents / mask 只加载一次，租约 + TTL 回收 + FIFO 等待队列，多会话跨角色零串音。
- **可插拔发布端**：LiveKit WebRTC（模式 A）或本地 OpenCV 预览 / mp4 落盘（模式 B，零 LiveKit 依赖）。
- **音频来源无关**：接任意 TTS、麦克风、wav 文件——本库只负责"音频进、视频出"。

## 环境要求

- Python ≥ 3.10
- NVIDIA GPU（CUDA）用于 MuseTalk 推理；纯测试/编排逻辑可在 CPU 上跑（96 个单测不需要 torch）
- 可选：LiveKit Server（模式 A）、Docker + nvidia-container-toolkit

## 快速开始

### 1. 安装

```bash
git clone https://github.com/<you>/LiveAvatar.git
cd LiveAvatar
pip install -e ".[livekit,vision,server]"   # 推理另需: pip install torch torchvision diffusers transformers einops
```

### 2. 下载模型权重与 demo 数据

```bash
python scripts/download_models.py
```

下载 `models/musetalkV15`（UNet）、`models/sd-vae-ft-mse`（VAE）、`models/whisper`（音频特征）、YuNet 人脸检测、MediaPipe 关键点，以及 yongen demo 视频与音频。

### 3. 准备一个 avatar

```bash
python scripts/prepare_avatar.py \
    --input data/video/yongen.mp4 \
    --avatar-id yongen \
    --avatar-data-root data/avatars \
    --max-frames 8
```

产出 `data/avatars/yongen/`（full_imgs / coords.pkl / latents.pt / mask/ / mask_coords.pkl）。默认带 MediaPipe 五点人脸对齐，显著提升口型质量。

### 4. 模式 B：本地预览（无需 LiveKit）

```bash
python -m liveavatar.preview --audio data/audio/yongen.wav --avatar yongen            # OpenCV 窗口实时预览
python -m liveavatar.preview --audio data/audio/yongen.wav --avatar yongen --save out.mp4
```

实测（单卡 CUDA，batch=4，25fps）：8s 音频 → 200 帧 / 8.00s 视频，0 丢帧 0 错误，推理队列高水位 3，全程实时。

### 5. 模式 A：LiveKit 推流 + 浏览器 demo

```bash
docker compose up -d          # livekit + liveavatar（GPU）
# 打开 http://localhost:8000 ，选一个 wav，点「开始」
```

或手动运行服务：

```bash
export LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=xxx
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

服务端（`LIVEKIT_URL`、`LIVEKIT_API_KEY`、`LIVEKIT_API_SECRET`、`LIVEKIT_ROOM`、`PUBLIC_LIVEKIT_URL`）见 `liveavatar/publish.py`。

## HTTP / WebSocket API

| 端点 | 说明 |
|---|---|
| `POST /v1/sessions` | 创建会话，返回 `session_id` + LiveKit token |
| `DELETE /v1/sessions/{id}` | 关闭会话并取消发布视频轨道 |
| `GET /v1/sessions/{id}/stats` | adapter / publisher 计数器快照 |
| `GET /v1/avatars` | 列出可用 avatar |
| `WS /v1/sessions/{id}/audio` | 二进制帧 = PCM S16LE；文本帧 = 控制消息 |
| `GET /health` | 健康检查 |

WebSocket 控制消息（JSON）：

```jsonc
{"type": "epoch",  "epoch": 3}   // 开始新一段话（新 epoch）
{"type": "cancel", "epoch": 4}   // 打断：丢弃旧 epoch 的音频与视频帧
{"type": "stop"}                  // 结束
```

## 作为库嵌入

```python
from liveavatar.config import AvatarPoolConfig
from liveavatar.pipeline import AvatarPipeline

pipeline = AvatarPipeline(AvatarPoolConfig(avatar_data_root="data/avatars"))
await pipeline.start()
await pipeline.open_session("s1", "yongen", local_participant=room.local_participant)
await pipeline.push_pcm("s1", pcm_chunk)      # pts/epoch 自动管理
pipeline.cancel_epoch("s1", 1)                 # 打断
await pipeline.close_session("s1")
await pipeline.stop()
```

测试时可注入 fake `pool` / `publisher_factory`，96 个单测全部不依赖 torch / LiveKit / GPU。

## 目录结构

```text
src/liveavatar/
├── pipeline.py          # 编排器（池 + 会话 + PTS 时钟 + epoch）
├── publish.py           # FastAPI/WS 服务 + LiveKit token 签发
├── preview.py           # 模式 B：wav → 本地窗口 / mp4
├── adapter.py           # PCM → worker → publisher，队列/打断/降级
├── pool.py  lease.py    # avatar 租约池（基于 _common 泛型池）
├── worker.py            # AvatarWorker 抽象 + AvatarFrame
├── musetalk_worker.py   # MuseTalk 推理实现
├── static_worker.py     # 降级静帧 worker
├── video_publisher.py   # BGR24 → I420 → LiveKit VideoSource
├── config.py            # LIVEAVATAR_* 配置
├── _common/             # vendor 的租约/泛型池原语
└── musetalk/            # MuseTalk 模型定义（自包含）
scripts/                 # download_models / prepare_avatar / face_align
web/                     # 浏览器 demo（无构建，原生 JS + livekit-client CDN）
tests/                   # 171 个单测
```

## Roadmap

- [x] 离线批量渲染（`python -m liveavatar.batch_renderer`）
- [x] ASR/TTS 插件接口草案（`liveavatar.plugins`；官方示例插件待发布）
- [x] 更多发布后端（`liveavatar.sinks.PublishSink` 协议 + RTMP 后端；本地 WS-MJPEG 待实现）
- [ ] Apple Silicon (MPS) 支持

## 许可证

本项目代码以 [MIT](LICENSE) 发布。集成的上游组件遵守各自许可：

| 组件 | 许可 | 说明 |
|---|---|---|
| [MuseTalk](https://github.com/TMElyralab/MuseTalk) | MIT | 代码（本仓库 `musetalk/` 改编自上游）；其预训练模型可自由使用（含商用） |
| [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | MIT | VAE |
| [whisper-tiny](https://huggingface.co/openai/whisper-tiny) | MIT | 音频特征 |
| [YuNet](https://github.com/opencv/opencv_zoo) / [MediaPipe](https://developers.google.com/mediapipe) | Apache-2.0 | 人脸检测 / 五点对齐 |
| [LiveKit](https://livekit.io/) | Apache-2.0 | 实时传输（可选依赖） |
| MuseTalk demo 数据（yongen） | — | **仅限非商业研究用途**，来源见上游仓库 |

使用数字人生成内容时请遵守当地法律法规，不得用于伪造他人身份等用途。

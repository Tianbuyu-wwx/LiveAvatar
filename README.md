# LiveAvatar

[![CI](https://github.com/Tianbuyu-wwx/LiveAvatar/actions/workflows/ci.yml/badge.svg)](https://github.com/Tianbuyu-wwx/LiveAvatar/actions/workflows/ci.yml)

**实时流式数字人口型视频生成 + 全双工对话星型架构**：PCM 音频进，MuseTalk 口型视频出，经自研 WebSocket 传输推到浏览器；duplex 模式下麦克风进、完整对话（ASR → LLM → TTS → 数字人视频）出。

LiveAvatar 从一个生产级全双工数字人系统中拆出视频生成全链路，做成独立、可嵌入的开源库 + 服务：

```text
TTS / 麦克风 / wav 文件（16kHz mono PCM S16LE）
    →  AvatarStreamingAdapter（有界队列 + epoch 打断 + 降级链）
        →  AvatarPool（租约池，每角色独占 worker，跨会话隔离）
            →  MuseTalkAvatarWorker（Whisper 特征 → UNet → VAE → 贴回原图）
            →  WebSocketSink（自研二进制协议编解码 + 多客户端扇出）
                →  浏览器 player.js（抖动缓冲 + canvas 合成，实时播放）
```

## 全双工星型架构（duplex 模式）

`POST /v1/sessions {"mode": "duplex"}` 会启动一个以 RealtimeWorker 为中心的星型会话，所有辐条均可插拔、均可选：

```text
浏览器麦克风 ─▶ WS 二进制帧 ─▶ DuplexSession.push_pcm ─▶ worker.input_queue
                                                            │
                                                   worker 主循环
                                            VAD/EOU/ASR → LLM → TTS
                                                            │
浏览器 ◀─ WS 二进制 PCM + JSON 事件 ◀─ out_queue ◀─ drain ◀─ worker.output_queue
                                                            │
        /v1/sessions/{sid}/video ◀─ WebSocketSink ◀─ AvatarStreamingAdapter（可选）
```

辐条配置（全部可选，默认回退实现可在纯 CPU 上跑通）：

| 辐条 | 启用方式 | 缺省回退 |
|---|---|---|
| ASR | `LIVEAVATAR_ASR_URL`（RealtimeAsr 兼容微服务） | 进程内参考 ScriptedAsr |
| LLM | `LIVEAVATAR_LLM_BASE_URL` + `LIVEAVATAR_LLM_MODEL`（OpenAI 兼容：DeepSeek / Qwen / vLLM / Ollama） | ASR 文本直通 TTS（echo） |
| TTS | `LIVEAVATAR_VOICE_CHAR`（GPT-SoVITS VoicePool 进程内推理） | FakeTts（确定性测试音） |
| AEC | `LIVEAVATAR_AEC=1`（纯 numpy NLMS 回声消除） | 关闭 |
| Avatar | `LIVEAVATAR_DUPLEX_AVATAR=1` | 纯音频 |

epoch 打断权威仍在 worker：`cancel` 控制消息触发 `advance_epoch`，同时熔断在途 LLM/TTS 任务、清空队列、经 `AvatarStreamingAdapter.cancel_epoch` 一帧内停住过期视频。TTS 推理设备由 `LIVEAVATAR_VOICE_DEVICE` 控制（设为 `cpu` 即可完全不用 GPU）。

## 特性

- **毫秒级打断（epoch 机制）**：每个 PCM chunk 与视频帧都携带 epoch；打断时新旧两条通路同时生效——推理侧 CancelToken 即刻熔断，发布侧丢弃所有过期帧。打断后视频在 **1 帧（≤40ms @25fps）内**停止。
- **有界队列反压**：TTS 永远是主时钟，avatar 推理跟不上时丢 chunk 而不是阻塞音频。
- **降级链**：`MuseTalk 推理 → 连续 N 次失败自动切换 StaticAvatarWorker（静帧）→ 音频独播`，新 epoch 自动恢复主 worker。
- **租约式角色池**：每个 avatar 的 face coords / latents / mask 只加载一次，租约 + TTL 回收 + FIFO 等待队列，多会话跨角色零串音。
- **可插拔发布端**：自研 WS 传输（零外部基础设施）；另有本地 OpenCV 预览 / mp4 落盘（模式 B，零依赖）。
- **全双工星型架构（duplex）**：麦克风音频进 → VAD/EOU/ASR → LLM（OpenAI 兼容流式，分句降延迟）→ TTS（GPT-SoVITS VoicePool）→ 数字人视频，辐条全部可插拔；epoch 打断同时熔断音频与视频。
- **自研视频传输（R2）**：26 字节帧头二进制协议（seq/epoch/pts/codec/flags），MJPEG 全帧与**区域独立帧**双编码（口型区域先验，带宽 ↓55–80%），帧独立可解码、任意丢帧不花屏。
- **自适应质量**：客户端周期上报拥塞信号（丢帧率/码率），服务端 EWMA 聚合 + 5 档质量状态机，弱网"降画质不冻结"，恢复带迟滞不抖动。
- **音频来源无关**：接任意 TTS、麦克风、wav 文件——本库只负责"音频进、视频出"。

## 环境要求

- Python ≥ 3.10
- NVIDIA GPU（CUDA）用于 MuseTalk 推理；纯测试/编排/传输逻辑可在 CPU 上跑
- 可选：Docker + nvidia-container-toolkit

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

#### GPT-SoVITS（duplex TTS 辐条，可选）

引擎代码已 vendored 到 `third_party/GPT_SoVITS/`（MIT）。预训练权重（≈1.4 GB）不入库，用脚本下载：

```bash
python scripts/download_gptsovits.py     # 支持 HF_ENDPOINT=https://hf-mirror.com 加速
```

少数超大数据文件（`text/G2PWModel/g2pW.onnx`、`text/ja_userdic/`）同样不入库——需要多音字纠正 / 日语 TTS 时从 [上游仓库](https://github.com/RVC-Boss/GPT-SoVITS) 补齐。

### 3. 准备一个 avatar

```bash
python scripts/prepare_avatar.py \
    --input data/video/yongen.mp4 \
    --avatar-id yongen \
    --avatar-data-root data/avatars \
    --max-frames 8
```

产出 `data/avatars/yongen/`（full_imgs / coords.pkl / latents.pt / mask/ / mask_coords.pkl）。默认带五点人脸对齐（`LANDMARK_BACKEND=mediapipe|self`，显著提升口型质量；自研 `self` 后端权重训练完成后将成为默认）。

### 4. 模式 B：本地预览

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

### 全双工对话（duplex 模式）

```bash
# 最小配置：参考 ASR + echo + FakeTts，纯 CPU 即可跑通全链路
uvicorn liveavatar.publish:app --port 8000

# 完整配置：远程 ASR + LLM + GPT-SoVITS TTS（CPU 推理）
export LIVEAVATAR_ASR_URL=ws://localhost:8100
export LIVEAVATAR_LLM_BASE_URL=https://api.deepseek.com/v1
export LIVEAVATAR_LLM_API_KEY=sk-... LIVEAVATAR_LLM_MODEL=deepseek-chat
export LIVEAVATAR_VOICE_CHAR=yongen LIVEAVATAR_VOICE_DEVICE=cpu
export LIVEAVATAR_DUPLEX_AVATAR=1
uvicorn liveavatar.publish:app --port 8000
```

创建会话与对话：

```bash
curl -X POST localhost:8000/v1/sessions -d '{"mode": "duplex"}'
# → {"session_id": "...", "mode": "duplex", "spokes": {...}, ...}
# 连 WS /v1/sessions/{sid}/audio：二进制帧发麦克风 PCM，
# 收到的二进制帧即合成语音，JSON 文本帧为 asr/vad/control 事件
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
| `LIVEAVATAR_API_SECRET` | *(空)* | 会话令牌 HS256 密钥；设置后创建会话响应附带短 TTL `session_token`，WS 可用 `?token=` / `X-Session-Token` / `Authorization: Bearer` 代替静态密钥 |
| `LIVEAVATAR_TOKEN_TTL_S` | `300` | 会话令牌有效期（秒） |
| `LIVEAVATAR_MAX_SESSIONS` | `16` | 并发会话上限（超出返回 429） |
| `LIVEAVATAR_MAX_WS_FRAME_BYTES` | `65536` | WS 二进制帧上限（超限丢弃，防 DoS） |
| `LIVEAVATAR_CODEC` | `mjpeg` | ws 传输编码：`mjpeg`（全帧）或 `region`（区域增量，需 avatar 的 `region.json`） |
| `LIVEAVATAR_METRICS` | `off` | 置 `on` 启用 `GET /metrics`（Prometheus 文本格式：会话数/视频客户端/丢帧/编码错误/uptime） |
| `LIVEAVATAR_ASR_URL` | *(空)* | duplex：RealtimeAsr 兼容 ASR 微服务 WS 地址（缺省用参考 ScriptedAsr） |
| `LIVEAVATAR_LLM_BASE_URL` / `LIVEAVATAR_LLM_MODEL` | *(空)* | duplex：OpenAI 兼容 chat API（DeepSeek/Qwen/vLLM/Ollama），缺省 echo |
| `LIVEAVATAR_LLM_API_KEY` / `LIVEAVATAR_LLM_SYSTEM_PROMPT` | *(空)* | duplex：LLM 鉴权与系统提示词 |
| `LIVEAVATAR_VOICE_CHAR` | *(空)* | duplex：TTS 角色ID，启用 GPT-SoVITS VoicePool（缺省 FakeTts） |
| `LIVEAVATAR_VOICE_DEVICE` / `LIVEAVATAR_VOICE_IS_HALF` | `cuda` / `true` | VoicePool 推理设备与精度（`cpu` / `false` 即纯 CPU） |
| `LIVEAVATAR_AEC` | `0` | duplex：启用纯 numpy NLMS 回声消除 |
| `LIVEAVATAR_DUPLEX_AVATAR` | `0` | duplex：启用视频辐条（MuseTalk 生成口型视频） |

## HTTP / WebSocket API

| 端点 | 说明 |
|---|---|
| `POST /v1/sessions` | 创建会话；body `{"mode": "push"(默认) \| "duplex", "avatar_id": ...}` |
| `DELETE /v1/sessions/{id}` | 关闭会话并取消发布视频轨道 |
| `GET /v1/sessions/{id}/stats` | adapter / publisher 计数器快照（含质量档位 tier） |
| `GET /v1/avatars` | 列出可用 avatar |
| `WS /v1/sessions/{id}/audio` | 二进制帧 = PCM S16LE；文本帧 = 控制消息 |
| `WS /v1/sessions/{id}/video` | 订阅视频流（自研二进制协议，见 [docs/PROTOCOL.md](docs/PROTOCOL.md)） |
| `GET /metrics` | Prometheus 文本格式指标（需 `LIVEAVATAR_METRICS=on`，鉴权同 REST） |
| `GET /health` | 健康检查 |

音频 WS 控制消息（JSON）：

```jsonc
{"type": "epoch",  "epoch": 3}   // 开始新一段话（新 epoch）
{"type": "cancel", "epoch": 4}   // 打断：丢弃旧 epoch 的音频与视频帧
{"type": "stop"}                  // 结束
```

duplex 模式下同一 audio WS 为全双工：上行二进制帧是麦克风 PCM（16kHz mono S16LE），下行二进制帧是合成语音 PCM，下行 JSON 文本帧是管道事件（`asr` / `vad` / `eou` / `control` / `error`）。`cancel` / `epoch` 消息触发 barge-in，worker 作为 epoch 权威立即熔断在途 LLM/TTS/视频。

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

测试时可注入 fake `pool` / `publisher_factory`，470+ 个单测全部不依赖 torch / GPU。

## 目录结构

```text
src/liveavatar/
├── pipeline.py          # 编排器（池 + 会话 + PTS 时钟 + epoch）
├── publish/             # FastAPI 服务包（push/duplex 双模式）
│   ├── routes.py            # REST 端点 + app 组装
│   ├── ws_routes.py         # /audio + /video WS 端点
│   ├── session_manager.py   # pipeline/voice-pool 生命周期 + duplex 会话
│   ├── encoders.py          # 每会话发布器/编码器工厂 + avatar_id 校验
│   ├── settings.py state.py tokens.py   # 配置 / 单例状态 / JWT 签发
├── duplex.py            # 全双工星型会话（WS 音频 ⇄ RealtimeWorker 辐条）
├── spokes.py            # 可选辐条统一组装（ASR/LLM/TTS/AEC/Avatar，duplex 与 runtime 共用）
├── observability.py     # /metrics Prometheus 导出 + TraceID 日志（自研零依赖）
├── preview.py           # 模式 B：wav → 本地窗口 / mp4
├── batch_renderer.py    # 离线批量渲染（CLI：python -m liveavatar.batch_renderer）
├── plugins.py           # ASR/TTS 插件接口
├── text_source.py       # LLM 辐条协议 + OpenAI 兼容流式客户端 + 分句器
├── tts.py               # NvcStreamingTtsAdapter（VoicePool → 流式 TTS）
├── runtime/             # 全双工星型架构核心
│   ├── worker.py            # RealtimeWorker：VAD/EOU/ASR → LLM → TTS 中枢（epoch 权威）
│   ├── contracts.py         # 事件信封（Envelope）wire 协议
│   ├── queues.py fake_tts.py metrics.py
├── voice/               # GPT-SoVITS VoicePool（NvcWorker 进程内推理 + 租约池）
├── audio_in/            # 输入辐条：VAD/EOU/ASR 参考实现 + 远程客户端 + NLMS AEC
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
├── face_backend.py      # 人脸检测/关键点双后端切换（self | legacy）
├── face_self.py         # 自研 TinyFaceDetector + LandmarkNet5Self（torch）
├── face_landmarks.py    # 5 点关键点对齐（训练/推理共享）
├── face_accept.py       # M4 验收指标库（SSIM / 偏移 / IoU / 速度比）
├── config.py            # LIVEAVATAR_* 配置
├── _common/             # 跨事件循环队列 / 租约/泛型池原语
└── musetalk/            # MuseTalk 模型定义（自包含）
scripts/                 # download_models / download_gptsovits / prepare_avatar /
                         # demo_local / wsperf / e2e_bench / capacity_report /
                         # make_face_dataset / face_align / train_face_det /
                         # train_face_landmarks / accept_face_backend
web/                     # 浏览器 demo（无构建，原生 JS：player.js 抖动缓冲 + canvas 合成）
tests/                   # 49 文件、470+ 用例（协议/传输/自适应/星型架构/人脸/并发/端到端，CPU-only）
third_party/GPT_SoVITS   # GPT-SoVITS 引擎代码（MIT，vendored；预训练权重不入库）
```

## 安全与信任边界

安全漏洞请勿公开发 issue：通过仓库 **Security → Advisories** 私密披露，流程见 [SECURITY.md](SECURITY.md)。

- **API 鉴权**：公网部署必须设置 `LIVEAVATAR_API_KEY`（REST 走 `X-API-Key` 头，WS 走同名查询参数或头）。留空 = 仅限本机开发。配合 `LIVEAVATAR_API_SECRET` 时服务改为签发短 TTL 会话令牌（浏览器只接触随时过期的 token，静态密钥不出服务端），见 `liveavatar/publish/tokens.py`。
- **avatar 资产信任边界**：`coords.pkl` / `mask_coords.pkl`（pickle）与 `latents.pt` / 人脸 checkpoint（torch）反序列化可执行任意代码。请**只使用本机 `scripts/prepare_avatar.py` 与 R1 训练脚本产出的资产**，切勿加载来路不明的模型文件。代码侧已启用 `torch.load(weights_only=True)`（仅允许张量与原始类型）作为纵深防御。
- **路径安全（S4）**：`avatar_id` 会拼进文件系统路径，服务端在会话创建与区域编码器处强制白名单 `^[A-Za-z0-9_-]+$`，路径穿越/分隔符/NUL 等载荷一律 400/422 拒绝。
- **资源限制**：`LIVEAVATAR_MAX_WS_FRAME_BYTES`（默认 64 KB）限制单帧大小，`LIVEAVATAR_MAX_SESSIONS` 限制并发会话数。

## Roadmap

- [x] 离线批量渲染（`python -m liveavatar.batch_renderer`）
- [x] 全双工星型架构（duplex 模式：VAD/EOU/ASR → LLM → TTS → Avatar，epoch 打断）
- [x] ASR/TTS 插件接口草案（`liveavatar.plugins`；官方示例插件待发布）
- [x] 更多发布后端（`liveavatar.sinks.PublishSink` 协议 + RTMP 后端）
- [x] 自研 WS 视频传输 + 区域编码 + 自适应质量（R2，默认启用）
- [x] LiveKit 退役（M-C）：仅自研传输 + HS256 会话令牌 + 三路并发验证（`scripts/capacity_report.py`）
- [ ] Apple Silicon (MPS) 支持

## 许可证

本项目代码以 [MIT](LICENSE) 发布。集成的上游组件遵守各自许可：

| 组件 | 许可 | 说明 |
|---|---|---|
| [MuseTalk](https://github.com/TMElyralab/MuseTalk) | MIT | 代码（本仓库 `musetalk/` 改编自上游）；其预训练模型可自由使用（含商用） |
| [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | MIT | VAE |
| [whisper-tiny](https://huggingface.co/openai/whisper-tiny) | MIT | 音频特征 |
| [YuNet](https://github.com/opencv/opencv_zoo) | Apache-2.0 | 人脸检测（过渡期默认后端，单个 onnx 资源文件） |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | MIT | TTS 辐条（duplex 模式，`third_party/GPT_SoVITS`，可选依赖） |
| [MediaPipe](https://developers.google.com/mediapipe) | Apache-2.0 | 训练期教师标注专用（`teacher` extra），非运行依赖 |
| MuseTalk demo 数据（yongen） | — | **仅限非商业研究用途**，来源见上游仓库 |

使用数字人生成内容时请遵守当地法律法规，不得用于伪造他人身份等用途。

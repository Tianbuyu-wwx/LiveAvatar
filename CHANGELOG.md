# Changelog

所有显著变更记录于此。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [0.4.0] - 2026-08-31

### Added — R2 传输层自研（默认启用）
- **自研二进制视频协议 v1**（[docs/PROTOCOL.md](docs/PROTOCOL.md)，已冻结）：26 字节帧头（seq/epoch/pts/codec/flags/quality），`video_protocol` 纯函数编解码。
- **自研 WS 传输**：`ws_sink.WebSocketSink`（实现 `PublishSink` 协议，多客户端扇出 + 丢帧/关键帧管理）+ `WS /v1/sessions/{sid}/video` 端点 + 浏览器 `web/player.js`（抖动缓冲、时钟同步、canvas 合成、`keyframe_request` 恢复）。
- **区域独立帧编码**（`region_codec`）：首帧全图 + 口型区域独立 JPEG 块，帧间底图哈希检测；512² 合成流带宽 1670→764 kbps（↓55%），真实素材预计 ↓70–80%。
- **自适应质量**（`adaptive`）：客户端每 2s 上报拥塞信号（丢帧率/码率/fps），服务端 EWMA 聚合 + 5 档质量状态机（降快升慢 + 迟滞），弱网"降画质不冻结"。
- **跨事件循环队列**（`_common.loopqueue.LoopFreeQueue`）：修复 `asyncio.Queue` 在多事件循环（TestClient / WS portal）下绑定循环崩溃导致的静默丢帧。
- 测量工具：`scripts/wsperf.py`（合成/URL 双模式）、`scripts/e2e_bench.py`（真实 socket 端到端基准：启动/打断/重连延迟 + 带宽）。
- 传输开关 `LIVEAVATAR_TRANSPORT`（`ws` 默认 / `livekit` 过渡）与 `LIVEAVATAR_CODEC`（`mjpeg` / `region`）。

### Changed
- **默认传输切换为自研 ws**：无外部基础设施即可运行浏览器 demo；LiveKit 相关环境变量仅在过渡模式需要。
- docker-compose 移除 livekit 服务；README/DEPLOYMENT 按 ws 传输重写。
- `livekit` extra 标记 **deprecated**（计划两个小版本后移除）。

### Fixed
- `/video` WS 客户端断开后 `send_bytes` 与 ASGI close 竞态抛 RuntimeError（现按正常断开处理）。
- demo 区域编码：`region_spec_from_masks` 对缺失 mask 目录返回 None 而非抛异常；region.json 写入默认 avatar 目录。

### Performance（实测，CPU）
- 编码耗时 p50：region 0.24 ms / mjpeg 0.62 ms（512² 低熵合成，预算 2–5 ms）。
- 端到端（真实 socket，128² 合成 worker）：启动延迟 3.5–3.9 ms，打断延迟 2.8–5.2 ms，断网 0.5s 重连恢复 7.7–45.1 ms，0 丢帧（详见 `docs/R2对比报告.md`）。

### Tests
- 新增 100+ 测试：协议 roundtrip/容错、sink 丢帧/背压/epoch、区域编解码、自适应状态机、/video WS 集成、传输开关、多会话隔离、端到端反馈回路（TestClient）。
- CI 新增 Node 语法门禁（`node --check` 校验 web/ 下原生 JS，无构建链故不引入 vitest）。

## [0.3.0] - 2026-08-30

### Added
- 离线批量渲染器 `python -m liveavatar.batch_renderer`（wav → mp4，走 worker 离线长音频路径）。
- 发布后端抽象 `liveavatar.sinks.PublishSink` 协议 + `RtmpSink`（ffmpeg 子进程输出 RTMP/FLV，无新增 Python 依赖）。
- 插件接口草案 `liveavatar.plugins`（ASR/TTS 结构化协议 + 进程内注册表 + importlib entry-points 自动发现）。
- 14 个新测试（sinks/plugins/batch_renderer）。

## [0.2.0] - 2026-08-30

### Security
- REST 与 WS 端点新增可选 API-Key 鉴权（`LIVEAVATAR_API_KEY`；REST 用 `X-API-Key` 头，WS 用 `api_key` 查询参数或头）。未配置时保持开放（本地开发）。
- 浏览器 LiveKit token 降权为仅订阅（`canPublish=False`、`canPublishData=False`）；发布者 bot 保留独立可发布 token。
- WS 二进制帧大小上限（`LIVEAVATAR_MAX_WS_FRAME_BYTES`，默认 64KB），超限丢弃并告警。
- 并发会话上限（`LIVEAVATAR_MAX_SESSIONS`，默认 16），超出返回 429。
- 会话创建请求体改为 pydantic 校验（`session_id` 白名单字符 + 长度限制），非法请求 422。
- 静态文件服务由手写 catch-all 路由替换为 Starlette `StaticFiles` 挂载（目录穿越防护交给框架）。

### Performance
- `video_publisher._bgr24_to_i420` 增加 cv2 `COLOR_BGR2YUV_I420` 快路径；numpy 兜底改为 BT.601 limited-range，与 cv2 输出一致（±1）。
- 池新增 worker 驱逐：`evict_worker(resource_id)` 手动卸载；`LIVEAVATAR_MAX_LOADED_WORKERS` 超限按 LRU 自动卸载，卸载时释放 per-avatar 内存（帧/latents），共享模型保留。

### Fixed
- Web demo 关闭页面时调用 `DELETE /v1/sessions/{id}` 即时释放服务端会话（此前需等 300s TTL），并关闭 AudioContext。
- `preview.py --no-half` 死参数移除；重采样后 dtype 统一 float32。
- FastAPI `@app.on_event("shutdown")` 迁移到 lifespan（消除 deprecation 警告）。
- docker-compose 冗余环境变量清理。
- `.gitignore` 覆盖 `*.mp4` 渲染产物。

### Added
- GitHub Actions CI（Python 3.10/3.11/3.12 矩阵：ruff + mypy + pytest-cov）。
- ruff / mypy / coverage 工具链配置；`dev` extras。
- 新测试：安全（鉴权/限流/校验）、I420 数值一致性、static worker、preview wav 加载、池驱逐、MuseTalk worker（fake models，torch-gated）。

## [0.1.0] - 初始开源版本

- 流式管线：`AvatarStreamingAdapter`（epoch 打断 + 降级链）、`AvatarPool`（租约池）、`MuseTalkAvatarWorker` / `StaticAvatarWorker`、`AvatarVideoPublisher`。
- FastAPI 服务（模式 A：LiveKit 推流）与本地预览（模式 B）。
- 96 个无 GPU 依赖单元测试。

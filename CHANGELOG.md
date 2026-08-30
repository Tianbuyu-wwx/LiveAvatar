# Changelog

所有显著变更记录于此。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

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

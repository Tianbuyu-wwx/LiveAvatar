# 生产部署安全清单

LiveAvatar 服务暴露 GPU 推理与 WebSocket 入口，公网部署前逐项确认。

> 默认视频传输为自研 WS 传输：只需暴露本服务的 443（反向代理），
> 无需任何额外传输基础设施。

## 1. 鉴权（必做）
- 设置 `LIVEAVATAR_API_KEY=<强随机密钥>`，所有 REST/WS 请求需携带：
  - REST：`X-API-Key: <key>` 请求头
  - WS：`?api_key=<key>` 查询参数或同名请求头
- 未设置该变量时服务**完全开放**，仅限本机/内网调试。
- **推荐**：同时设置 `LIVEAVATAR_API_SECRET=<另一个强随机密钥>` 启用短 TTL
  会话令牌（默认 300s，`LIVEAVATAR_TOKEN_TTL_S` 可调）。此时 `POST /v1/sessions`
  响应附带 `session_token`（HS256），浏览器 WS 改用 `?token=` / `X-Session-Token` /
  `Authorization: Bearer` 携带，静态 API_KEY 不再下发到前端——泄漏凭据的影响
  被限制在 TTL 窗口内。

## 2. 传输加密
- 在反向代理（nginx/caddy）终止 TLS，对外仅暴露 `https://` 与 `wss://`。
- 自研 ws 传输的 `/v1/sessions/{id}/video` 与 `/audio` 同样走 wss；
  反向代理需关闭请求缓冲（如 nginx `proxy_buffering off`）并调大
  WS 超时（`proxy_read_timeout` ≥ 300s）。

## 3. 资源限额
- `LIVEAVATAR_MAX_SESSIONS`：按 GPU 显存设置（默认 16 偏宽松，单卡建议 2-4）。
  定容方法：GPU 空闲期运行 `python scripts/capacity_report.py --sessions N
  --seconds 600`（真实 worker）取最大满足门禁（fps ≥ 20、打断 ≤ 90ms、零饥饿）
  的 N，参考 docs/容量报告_2026-08-31.md（CPU 合成基线）。
- `LIVEAVATAR_MAX_WS_FRAME_BYTES`：保持默认 64KB（PCM 100ms 块仅 ~3.2KB）。
- `LIVEAVATAR_MAX_LOADED_WORKERS`：多 avatar 场景设置上限（如 2），防显存溢出。
- 反向代理层再加 IP 级限流（如 nginx `limit_req`）。

## 4. 网络隔离
- 仅反向代理的 443 对外；uvicorn(8000) 不直接暴露公网。
- 视频带宽预算：region 编码约 0.8–2 Mbps/会话，mjpeg 全帧约 5–8 Mbps
  （512²@25fps），按 `LIVEAVATAR_CODEC` 选择并预留出口带宽。

## 5. 日志与监控
- 结构化日志已内置（`pool_reap_completed`、`ws_sink_tier_changed` 等事件），接入采集即可。
- 关注 `/health` 与 `GET /v1/sessions/{id}/stats` 的 `queue_high_water`、
  `frames_dropped_*`、`tier`、`smoothed_gap_rate`。

## 6. 内容合规
- demo avatar（yongen）数据仅限研究用途，商用前替换为自备素材（自采指引见 README「准备一个 avatar」小节）。
- 遵守所在地深度合成（deepfake）法规；对外产品建议加可见/隐式水印。

## 7. 可选组件：RTMP 输出（外部二进制）
- **主链路零外部二进制依赖**：默认视频分发为自研 `WebSocketSink`（帧编码经现有 WS 下发，浏览器 canvas/WebCodecs 播放）。
- 仅当存在**硬性 RTMP 分发**需求时，`RtmpSink`（`src/liveavatar/sinks.py`）通过 stdin 子进程调用 `ffmpeg` 二进制（需在 PATH 上，无任何 Python 包依赖）。
- RTMP 为可选旁路：`RtmpSink` 故障不影响主 sink；服务可不安装 ffmpeg 运行。
- GPT-SoVITS TTS 引擎为 vendored 代码（来源与升级策略见 `third_party/GPT_SoVITS/README_SELF.md`），其基础权重经 `scripts/download_gptsovits.py` 拉取。

# 生产部署安全清单

LiveAvatar 服务暴露 GPU 推理与 WebSocket 入口，公网部署前逐项确认。

## 1. 鉴权（必做）
- 设置 `LIVEAVATAR_API_KEY=<强随机密钥>`，所有 REST/WS 请求需携带：
  - REST：`X-API-Key: <key>` 请求头
  - WS：`?api_key=<key>` 查询参数或同名请求头
- 未设置该变量时服务**完全开放**，仅限本机/内网调试。

## 2. 传输加密
- 在反向代理（nginx/caddy）终止 TLS，对外仅暴露 `https://` 与 `wss://`。
- LiveKit 服务同样启用 wss（生产用正式密钥，勿用 `livekit.yaml` 中的 dev 密钥）。

## 3. 资源限额
- `LIVEAVATAR_MAX_SESSIONS`：按 GPU 显存设置（默认 16 偏宽松，单卡建议 2-4）。
- `LIVEAVATAR_MAX_WS_FRAME_BYTES`：保持默认 64KB（PCM 100ms 块仅 ~3.2KB）。
- `LIVEAVATAR_MAX_LOADED_WORKERS`：多 avatar 场景设置上限（如 2），防显存溢出。
- 反向代理层再加 IP 级限流（如 nginx `limit_req`）。

## 4. 网络隔离
- LiveKit(7880)、uvicorn(8000) 端口不直接暴露公网；仅反向代理的 443 对外。
- GPU 服务器与 LiveKit 之间走内网或 VPN。

## 5. 日志与监控
- 结构化日志已内置（`pool_reap_completed`、`ws_frame_too_large` 等事件），接入采集即可。
- 关注 `/health` 与 `GET /v1/sessions/{id}/stats` 的 `queue_high_water`、`frames_dropped_*`。

## 6. 内容合规
- demo avatar（yongen）数据仅限研究用途，商用前替换为自备素材。
- 遵守所在地深度合成（deepfake）法规；对外产品建议加可见/隐式水印。

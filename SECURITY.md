# 安全策略

## 支持的版本

| 版本 | 支持状态 |
|------|----------|
| 0.4.x | ✅ 支持安全修复 |
| < 0.4 | ❌ 请升级 |

## 报告漏洞

**请勿通过公开 issue 报告安全漏洞。**

- 渠道：仓库 **Security → Advisories → Report a vulnerability**（GitHub 私密披露），或在 Issue 中申请联系渠道。
- 响应目标：48 小时内确认收到，7 天内给出初步评估。

## 信任边界与关键安全设计

LiveAvatar 是一个需要加载模型资产的实时数字人推理服务，请了解以下信任边界：

### avatar 资产（最关键）

`coords.pkl` / `mask_coords.pkl` / `latents.pt` / 人脸 checkpoint 反序列化可执行**任意代码**：

- 只使用本机 `scripts/prepare_avatar.py` 与 R1 训练脚本产出的资产；
- 切勿加载来路不明的模型文件或他人打包的 avatar 目录；
- 代码侧已启用 `torch.load(weights_only=True)`（仅允许张量与原始类型）作为纵深防御，但 pickle 资产（MuseTalk 格式要求）不受此保护。

### 服务暴露

- 公网部署**必须**设置 `LIVEAVATAR_API_KEY`（REST 走 `X-API-Key` 头，WS 走同名查询参数或头）；留空即无鉴权，仅限本机开发。
- 建议置于反向代理之后并启用 TLS；服务默认无内建 TLS。

### 资源限制

- `LIVEAVATAR_MAX_WS_FRAME_BYTES`（默认 64 KB）限制 WS 单帧大小，防止内存耗尽；
- `LIVEAVATAR_MAX_SESSIONS`（默认 16）限制并发会话数。

## 供应链

- CI 中所有 GitHub Actions 均固定到 commit SHA（见 `.github/workflows/ci.yml`），升级需显式修改；
- 每周定时运行 `pip-audit --strict`（见 `.github/workflows/audit.yml`），漏洞需通过升级 pyproject 依赖钉住版本处置。

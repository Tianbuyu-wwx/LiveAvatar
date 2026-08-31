# GPT-SoVITS vendored 声明（本仓库自研边界说明）

## 1. 来源

上游工程：[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（MIT License）。

本仓库以 **vendored 快照** 方式引入其推理引擎子集（`TTS_infer_pack/`、`module/`、`AR/`、`BigVGAN/`、`text/`、`feature_extractor/`、`eres2net/`、`f5_tts/`、`configs/` 等），首次入库于 LiveAvatar 提交 `2f4559e`（2026-08-31）。

## 2. 许可

GPT-SoVITS 上游采用 MIT License，允许自由复制、修改与再分发。本仓库遵循其许可条款；上游 LICENSE 文本以 [上游仓库](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE) 为准。

## 3. 修改清单与数据资产策略

- **代码**：当前为**零修改**纯 vendored（仅排除上游的 webui/训练入口，本仓库只保留推理所需子集）。
- **大资产排除**（按仓库 .gitignore 策略，需要时经脚本拉取）：
  - `pretrained_models/`（~1.4GB 基础权重）→ `scripts/download_gptsovits.py`
  - `text/G2PWModel/g2pW.onnx`（606MB，超 GitHub 单文件硬限制）
  - `text/ja_userdic/`（日文用户词典）

## 4. 接口冻结承诺

LiveAvatar 对本引擎的调用收敛为**唯一窄入口**：

```python
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
```

（见 `src/liveavatar/voice/pool.py`。）角色权重不可变、并发串行化、流式取消等全部由本仓库自有的 `NvcWorker` / `VoicePool` / `NvcStreamingTtsAdapter`（`src/liveavatar/voice/`、`src/liveavatar/tts.py`）封装，与引擎代码完全解耦。引擎内部结构变化不应影响 LiveAvatar 其余部分。

## 5. 升级策略

- **手动同步上游快照**（复制新版本子集 → 回归测试），禁止运行时从上游动态拉取；
- 每次同步在下方"快照记录"补一行（上游 commit/日期 + 修改内容）；
- 同步后必须跑：全量 pytest、ruff、mypy，以及一次真实 TTS 流式冒烟（`LIVEAVATAR_VOICE_*` 配置的 duplex 会话）。

## 快照记录

| 日期 | 上游版本 | 本仓库提交 | 修改 |
|---|---|---|---|
| 2026-08-31 | vendored 快照（上游 main） | `2f4559e` | 首次引入，零修改；2026-08-31 C1 补齐 `AR/models/` 4 个源文件（曾被 .gitignore 误排除） |

# Third-Party Notices

本仓库包含以下第三方组件。它们的源代码与素材**继续遵守各自上游许可条款**，
本仓库自研代码的 AGPL-3.0-or-later + 商业禁止附加限制不替代、不修改、不覆盖这些上游条款。

如上游条款与本仓库自研代码许可发生冲突，在第三方组件适用范围内**以上游条款为准**。
任何下游使用者再分发本仓库时，**必须**同时随附本文件与对应第三方上游许可证全文。

## vendored 组件（随仓库分发）

| 组件 | 上游许可 | 上游来源 | 本仓库路径 |
|---|---|---|---|
| GPT-SoVITS 引擎代码 | MIT | https://github.com/RVC-Boss/GPT-SoVITS | `third_party/GPT_SoVITS/` |
|  ├─ 3D-Speaker (ERes2Net) | All Rights Reserved (research-only, see upstream) | https://github.com/alibaba-damo-academy/3D-Speaker | `third_party/GPT_SoVITS/eres2net/` |
|  ├─ BigVGAN | MIT (NVIDIA) | https://github.com/NVIDIA/BigVGAN | `third_party/GPT_SoVITS/BigVGAN/` |
|  ├─ adefossez julius snippets | MIT | https://github.com/adefossez/julius | `third_party/GPT_SoVITS/BigVGAN/alias_free_activation/torch/filter.py` |
|  ├─ Phil Wang / lucidrains VQ-VAE | MIT | https://github.com/lucidrains/vector-quantize-pytorch | `third_party/GPT_SoVITS/module/core_vq.py`, `quantize.py` |
|  ├─ Xiaomi scaling.py | Apache-2.0 | https://github.com/k2-fsa/icefall | `third_party/GPT_SoVITS/AR/modules/scaling.py` |
|  ├─ PaddlePaddle zh_normalization | Apache-2.0 | https://github.com/PaddlePaddle/PaddleSpeech | `third_party/GPT_SoVITS/text/zh_normalization/` |
|  ├─ PaddlePaddle g2pw | Apache-2.0 | https://github.com/PaddlePaddle/PaddleNLP | `third_party/GPT_SoVITS/text/g2pw/` |
|  ├─ CMU dict (cmudict.rep) | research-only, see Carnegie Mellon Univ. notice | https://github.com/cmusphinx/cmudict | `third_party/GPT_SoVITS/text/cmudict.rep` |
| MuseTalk 自适应段 | MIT（双许可：MIT + 本仓库 AGPL-3.0-or-later） | https://github.com/TMElyralab/MuseTalk | `src/liveavatar/musetalk/`（仅派生段） |

vendored 代码均完整保留各自上游版权与许可证头；如本仓库未来收到合规删除请求，
将依据上游许可与适用法律在不影响本项目自研代码的前提下处理。

## 运行时下载的模型/素材（不入库）

| 组件 | 上游许可 | 下载脚本 | 上游来源 |
|---|---|---|---|
| [MuseTalk 预训练模型](https://github.com/TMElyralab/MuseTalk) | MIT（含商用） | `scripts/download_models.py` | 上游仓库 releases |
| [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | MIT | `scripts/download_models.py` | HuggingFace |
| [whisper-tiny](https://huggingface.co/openai/whisper-tiny) | MIT | `scripts/download_models.py` | HuggingFace |
| [YuNet](https://github.com/opencv/opencv_zoo) | Apache-2.0 | `scripts/download_models.py` | OpenCV Zoo |
| [GPT-SoVITS 预训练权重](https://github.com/RVC-Boss/GPT-SoVITS) | MIT | `scripts/download_gptsovits.py` | 上游仓库 |
| [MediaPipe 模型](https://developers.google.com/mediapipe) | Apache-2.0（仅训练期教师标注使用，运行时无依赖） | `scripts/download_models.py --teacher` | Google |

## Demo 数据

| 组件 | 许可 | 说明 |
|---|---|---|
| MuseTalk demo 数据（yongen） | **仅限非商业研究用途** | 来源见上游仓库；商用前必须替换为自采素材 |

## 与本仓库自研许可的关系

本仓库自研代码（含 `src/liveavatar/` 中除 `musetalk/` 外的全部模块、
`scripts/` 中非 vendored 脚本、`tests/`、`web/`、`docs/` 中原创内容）适用根目录 [LICENSE](LICENSE)：

- SPDX-License-Identifier: **AGPL-3.0-or-later**
- 附加限制：**严禁商业使用**（无书面商业授权不得用于任何商业用途）；
  AGPL §13 网络服务源代码公开义务继续生效；
  对历史版本一并收紧（见 LICENSE §二）。
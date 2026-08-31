# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_lightning_module.py
# reference: https://github.com/lifeiteng/vall-e
# 推理专用版本：移除 pytorch_lightning 依赖，改为纯 nn.Module
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

import torch
import torch.nn as nn

from AR.models.t2s_model import Text2SemanticDecoder


class Text2SemanticLightningModule(nn.Module):
    def __init__(self, config, output_dir, is_train=True):
        super().__init__()
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)
        pretrained_s1 = config.get("pretrained_s1")
        if pretrained_s1 and is_train:
            print(
                self.load_state_dict(
                    torch.load(
                        pretrained_s1,
                        map_location="cpu",
                        weights_only=False,
                    )["weight"],
                )
            )

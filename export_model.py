"""
模型导出脚本 (export_model.py)
基于MobileNetV2的垃圾分类 - 导出 AIR（开发板）和 MindIR（手机端）模型
"""

import numpy as np
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, export

from config import config
from mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2

# 使用训练阶段选出的最优 Checkpoint
CKPT = "mobilenetv2-27_40.ckpt"


def export_models():
    """
    加载最优 Checkpoint，分别导出：
      - mobilenetv2.air   用于 Atlas 200DK 开发板推理
      - mobilenetv2.mindir 用于手机端（Android / HarmonyOS）推理
    """
    backbone = MobileNetV2Backbone(last_channel=config.backbone_out_channels)
    head     = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes)
    network  = mobilenet_v2(backbone, head)

    load_checkpoint(f"{config.save_ckpt_path}/{CKPT}", network)

    # 构造随机 dummy 输入用于静态图导出
    dummy_input = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)

    # 导出 AIR 格式（Atlas 200DK 开发板，仅支持 MindSpore + Ascend 环境）
    export(network, Tensor(dummy_input), file_name='mobilenetv2.air', file_format='AIR')
    print("Exported: mobilenetv2.air")

    # 导出 MindIR 格式（手机端，支持 Android / HarmonyOS）
    export(network, Tensor(dummy_input), file_name='mobilenetv2', file_format='MINDIR')
    print("Exported: mobilenetv2.mindir")


if __name__ == "__main__":
    export_models()

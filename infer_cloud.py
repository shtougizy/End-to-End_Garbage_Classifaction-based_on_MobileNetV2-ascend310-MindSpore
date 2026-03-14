"""
云端推理脚本 (infer_cloud.py)
基于MobileNetV2的垃圾分类 - 在 ModelArts 上加载 Checkpoint 进行推理
"""

import numpy as np
from PIL import Image
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint

from config import config, class_en
from mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2

# 使用训练阶段选出的最优 Checkpoint（epoch=27）
CKPT = "mobilenetv2-27_40.ckpt"


def image_process(image):
    """
    单张图像预处理：归一化 + HWC -> CHW -> Tensor。

    Args:
        image: PIL.Image 对象，shape (H, W, C)。

    Returns:
        Tensor: shape (1, C, H, W) 的 float32 张量。
    """
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std  = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    image = (np.array(image) - mean) / std
    image = image.transpose((2, 0, 1))          # HWC -> CHW
    return Tensor(np.array([image], np.float32))


def infer_one(network, image_path):
    """
    对单张图像执行推理，打印预测类别。

    Args:
        network: 已加载 Checkpoint 的 MobileNetV2 网络。
        image_path (str): 图像文件路径。
    """
    image  = Image.open(image_path).resize((config.image_height, config.image_width))
    logits = network(image_process(image))
    pred   = np.argmax(logits.asnumpy(), axis=1)[0]
    print(image_path, class_en[pred])


def infer():
    """加载最优 Checkpoint，对测试集中部分图像执行推理。"""
    backbone = MobileNetV2Backbone(last_channel=config.backbone_out_channels)
    head     = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes)
    network  = mobilenet_v2(backbone, head)

    load_checkpoint(f"{config.save_ckpt_path}/{CKPT}", network)

    # 示例：推理 Cardboard 类别的第 91~99 张测试图
    for i in range(91, 100):
        infer_one(network, f'data_en/test/Cardboard/000{i}.jpg')


if __name__ == "__main__":
    infer()

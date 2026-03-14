"""
实验配置文件
基于MobileNetV2的垃圾分类 - 超参数与标签配置
"""

import math
import numpy as np
import os
import random
import shutil
import time

from matplotlib import pyplot as plt
from easydict import EasyDict
from PIL import Image

# ============================================================
# 垃圾分类数据集标签（中文分类 -> 物品列表）
# ============================================================
garbage_classes = {
    '可回收物': ['塑料瓶', '帽子', '报纸', '易拉罐', '玻璃制品', '玻璃瓶',
                '硬纸板', '篮球', '纸张', '金属制品'],
    '干垃圾':   ['一次性筷子', '打火机', '扫把', '旧镜子', '牙刷',
                '脏污衣服', '贝壳', '陶瓷碗'],
    '有害垃圾': ['油漆桶', '电池', '荧光灯', '药片胶囊'],
    '湿垃圾':   ['橙皮', '菜叶', '蛋壳', '香蕉皮'],
}

# 中文标签列表（按 index 排列）
class_cn = [
    '塑料瓶', '帽子', '报纸', '易拉罐', '玻璃制品', '玻璃瓶',
    '硬纸板', '篮球', '纸张', '金属制品',
    '一次性筷子', '打火机', '扫把', '旧镜子', '牙刷', '脏污衣服', '贝壳', '陶瓷碗',
    '油漆桶', '电池', '荧光灯', '药片胶囊',
    '橙皮', '菜叶', '蛋壳', '香蕉皮',
]

# 英文标签列表（与 class_cn 一一对应）
class_en = [
    'Plastic Bottle', 'Hats', 'Newspaper', 'Cans', 'Glassware',
    'Glass Bottle', 'Cardboard', 'Basketball', 'Paper', 'Metalware',
    'Disposable Chopsticks', 'Lighter', 'Broom', 'Old Mirror', 'Toothbrush',
    'Dirty Cloth', 'Seashell', 'Ceramic Bowl', 'Paint bucket', 'Battery',
    'Fluorescent lamp', 'Tablet capsules', 'Orange Peel', 'Vegetable Leaf',
    'Eggshell', 'Banana Peel',
]

# 英文标签 -> 类别索引 映射字典
index_en = {
    'Plastic Bottle': 0, 'Hats': 1, 'Newspaper': 2, 'Cans': 3,
    'Glassware': 4, 'Glass Bottle': 5, 'Cardboard': 6, 'Basketball': 7,
    'Paper': 8, 'Metalware': 9, 'Disposable Chopsticks': 10, 'Lighter': 11,
    'Broom': 12, 'Old Mirror': 13, 'Toothbrush': 14, 'Dirty Cloth': 15,
    'Seashell': 16, 'Ceramic Bowl': 17, 'Paint bucket': 18, 'Battery': 19,
    'Fluorescent lamp': 20, 'Tablet capsules': 21, 'Orange Peel': 22,
    'Vegetable Leaf': 23, 'Eggshell': 24, 'Banana Peel': 25,
}

# ============================================================
# 训练超参数
# ============================================================
config = EasyDict({
    "num_classes": 26,
    "image_height": 224,
    "image_width": 224,
    "backbone_out_channels": 1280,
    "batch_size": 64,
    "eval_batch_size": 8,
    "epochs": 30,
    "lr_max": 0.05,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "save_ckpt_epochs": 1,
    "save_ckpt_path": "./ckpt",
    "dataset_path": "./data_en",
    "class_index": index_en,
    "pretrained_ckpt": "./mobilenetV2-200_1067.ckpt",  # 预训练模型路径
})

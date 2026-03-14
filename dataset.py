"""
数据集处理脚本 (dataset.py)
基于MobileNetV2的垃圾分类 - 数据加载与数据增强
"""

import os

from mindspore import dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


def create_dataset(dataset_path, config, training=True, buffer_size=1000):
    """
    创建训练或验证数据集。

    Args:
        dataset_path (str): 数据集根目录路径（包含 train/ 和 test/ 子目录）。
        config (EasyDict): 训练配置，包含 image_height、image_width、
                           batch_size、eval_batch_size、class_index 等字段。
        training (bool): True 表示创建训练集，False 表示创建验证集。
        buffer_size (int): shuffle 缓冲区大小，默认 1000。

    Returns:
        Dataset: MindSpore 数据集对象。
    """
    data_path = os.path.join(dataset_path, 'train' if training else 'test')

    # 以 ImageFolder 格式加载数据集，按 class_index 映射标签
    ds = de.ImageFolderDataset(
        data_path,
        num_parallel_workers=4,
        class_indexing=config.class_index,
    )

    resize_height = config.image_height
    resize_width  = config.image_width

    # 公共变换：归一化 + HWC -> CHW + 标签类型转换
    normalize_op   = C.Normalize(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std =[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    change_swap_op = C.HWC2CHW()                          # (H,W,C) -> (C,H,W)
    type_cast_op   = C2.TypeCast(mstype.int32)            # 标签转 int32

    if training:
        # ---- 训练集：随机裁剪 + 水平翻转 + 颜色抖动 ----
        crop_decode_resize = C.RandomCropDecodeResize(
            resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333)
        )
        horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)
        color_adjust = C.RandomColorAdjust(
            brightness=0.4, contrast=0.4, saturation=0.4
        )
        train_trans = [
            crop_decode_resize,
            horizontal_flip_op,
            color_adjust,
            normalize_op,
            change_swap_op,
        ]
        train_ds = ds.map(
            input_columns="image", operations=train_trans, num_parallel_workers=8
        )
        train_ds = train_ds.map(
            input_columns="label", operations=type_cast_op, num_parallel_workers=8
        )
        train_ds = train_ds.shuffle(buffer_size=buffer_size)
        ds = train_ds.batch(config.batch_size, drop_remainder=True)

    else:
        # ---- 验证集：解码 + Resize + 中心裁剪 ----
        decode_op    = C.Decode()
        resize_op    = C.Resize((int(resize_width / 0.875), int(resize_width / 0.875)))
        center_crop  = C.CenterCrop(resize_width)
        eval_trans   = [decode_op, resize_op, center_crop, normalize_op, change_swap_op]

        eval_ds = ds.map(
            input_columns="image", operations=eval_trans, num_parallel_workers=8
        )
        eval_ds = eval_ds.map(
            input_columns="label", operations=type_cast_op, num_parallel_workers=8
        )
        ds = eval_ds.batch(config.eval_batch_size, drop_remainder=True)

    return ds

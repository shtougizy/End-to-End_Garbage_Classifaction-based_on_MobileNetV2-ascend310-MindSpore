"""
模型训练脚本 (train.py)
基于MobileNetV2的垃圾分类 - 云端训练（MindSpore + Ascend 910）
"""

import math
import os
import shutil
import time

import numpy as np
from matplotlib import pyplot as plt

import mindspore as ms
from mindspore import context, nn, Tensor, Model
from mindspore.train.serialization import load_checkpoint, export
from mindspore.train.callback import (
    Callback, ModelCheckpoint, CheckpointConfig
)
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from config import config, class_en, index_en
from dataset import create_dataset
from mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2

# ---- 运行环境配置 ----
os.environ['GLOG_v'] = '2'  # 日志级别：只显示 Error
context.set_context(
    mode=context.GRAPH_MODE,
    device_target="Ascend",
    device_id=0,
)

LOSS_SCALE = 1024


# ============================================================
# 学习率策略：Cosine Decay
# ============================================================
def cosine_decay(total_steps, lr_init=0.0, lr_end=0.0, lr_max=0.1, warmup_steps=0):
    """
    生成 Cosine Decay 学习率数组。

    Args:
        total_steps (int): 训练总步数。
        lr_init (float): 初始学习率（warmup 起点）。
        lr_end (float): 训练结束时的学习率。
        lr_max (float): 峰值学习率。
        warmup_steps (int): warmup 阶段的步数。

    Returns:
        list: 每个 step 对应的学习率列表。
    """
    lr_init, lr_end, lr_max = float(lr_init), float(lr_end), float(lr_max)
    decay_steps = total_steps - warmup_steps
    lr_all_steps = []
    inc_per_step = (lr_max - lr_init) / warmup_steps if warmup_steps else 0

    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + inc_per_step * (i + 1)
        else:
            cosine = 0.5 * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
            lr = (lr_max - lr_end) * cosine + lr_end
        lr_all_steps.append(lr)

    return lr_all_steps


# ============================================================
# 自定义回调：边训练边验证 + 早停
# ============================================================
class EvalCallback(Callback):
    """
    训练过程中每个 epoch 结束时对验证集进行评估，
    并在准确率连续 count_max 次不提升时触发早停。
    """

    def __init__(self, model, eval_dataset, history, eval_epochs=1):
        self.model        = model
        self.eval_dataset = eval_dataset
        self.eval_epochs  = eval_epochs
        self.history      = history
        self.acc_max      = 0
        self.count_max    = 5   # 早停阈值：连续 5 次不提升则停止
        self.count        = 0

    def epoch_begin(self, run_context):
        self.losses    = []
        self.startime  = time.time()

    def step_end(self, run_context):
        cb_param = run_context.original_args()
        loss = cb_param.net_outputs
        self.losses.append(loss.asnumpy())

    def epoch_end(self, run_context):
        cb_param   = run_context.original_args()
        cur_epoch  = cb_param.cur_epoch_num
        train_loss = np.mean(self.losses)
        time_cost  = time.time() - self.startime

        if cur_epoch % self.eval_epochs == 0:
            metric = self.model.eval(self.eval_dataset, dataset_sink_mode=False)

            self.history["epoch"].append(cur_epoch)
            self.history["eval_acc"].append(metric["acc"])
            self.history["eval_loss"].append(metric["loss"])
            self.history["train_loss"].append(train_loss)
            self.history["time_cost"].append(time_cost)

            print(
                "epoch: %d, train_loss: %f, eval_loss: %f, eval_acc: %f, time_cost: %f"
                % (cur_epoch, train_loss, metric["loss"], metric["acc"], time_cost)
            )

            if self.acc_max < metric["acc"]:
                self.count   = 0
                self.acc_max = metric["acc"]
            else:
                self.count += 1
                if self.count == self.count_max:
                    run_context.request_stop()  # 触发早停


# ============================================================
# 训练主函数
# ============================================================
def train():
    """构建网络、定义损失/优化器，执行训练并返回训练历史。"""

    train_dataset = create_dataset(dataset_path=config.dataset_path, config=config, training=True)
    eval_dataset  = create_dataset(dataset_path=config.dataset_path, config=config, training=False)
    step_size     = train_dataset.get_dataset_size()

    # ---- 构建模型（Backbone 冻结 + 加载预训练权重）----
    backbone = MobileNetV2Backbone()
    for param in backbone.get_parameters():
        param.requires_grad = False
    load_checkpoint(config.pretrained_ckpt, backbone)

    head    = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes)
    network = mobilenet_v2(backbone, head)

    # ---- 损失函数 / 损失缩放 / 优化器 ----
    loss       = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(LOSS_SCALE, drop_overflow_update=False)
    lrs        = cosine_decay(config.epochs * step_size, lr_max=config.lr_max)
    opt        = nn.Momentum(
        network.trainable_params(), lrs,
        config.momentum, config.weight_decay,
        loss_scale=LOSS_SCALE,
    )
    model = Model(network, loss, opt, loss_scale_manager=loss_scale, metrics={'acc', 'loss'})

    # ---- 回调 ----
    history = {'epoch': [], 'train_loss': [], 'eval_loss': [], 'eval_acc': [], 'time_cost': []}
    eval_cb = EvalCallback(model, eval_dataset, history)

    ckpt_cfg = CheckpointConfig(
        save_checkpoint_steps=config.save_ckpt_epochs * step_size,
        keep_checkpoint_max=config.epochs,
    )
    ckpt_cb = ModelCheckpoint(
        prefix="mobilenetv2",
        directory=config.save_ckpt_path,
        config=ckpt_cfg,
    )

    # ---- 开始训练 ----
    model.train(50, train_dataset, callbacks=[eval_cb, ckpt_cb], dataset_sink_mode=False)
    return history


# ============================================================
# 绘制训练曲线，选取最优 Checkpoint
# ============================================================
if __name__ == "__main__":
    if os.path.exists(config.save_ckpt_path):
        shutil.rmtree(config.save_ckpt_path)

    history = train()

    # Loss 曲线
    plt.plot(history['epoch'], history['train_loss'], label='train_loss')
    plt.plot(history['epoch'], history['eval_loss'], 'r', label='val_loss')
    plt.legend()
    plt.show()

    # Accuracy 曲线
    plt.plot(history['epoch'], history['eval_acc'], 'r', label='val_acc')
    plt.legend()
    plt.show()

    # 选取 eval_acc 最高的 Checkpoint
    CKPT = 'mobilenetv2-%d_40.ckpt' % (np.argmax(history['eval_acc']) + 1)
    print("Chosen checkpoint is", CKPT)
    # 实验结果：epoch=27 时精度最优，即 CKPT = "mobilenetv2-27_40.ckpt"

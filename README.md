# 基于 MobileNetV2 的垃圾分类端到端实验

> 华为昇腾创新实践 · Atlas 200DK 系列

---

## 实验概述

本实验完整覆盖了从**云端模型训练**到**边缘设备推理**的 AI 应用全流程，包含三个主要阶段：

1. **Atlas 200DK 合设环境搭建** — 为推理实验准备硬件运行环境
2. **MobileNetV2 垃圾分类模型训练（云端）** — 在华为云 ModelArts 上完成模型训练与导出
3. **开发板推理实验** — 将训练好的模型部署至 Atlas 200DK 上执行垃圾分类推理

---

## 实验环境

| 层级 | 组件 | 版本 / 规格 |
|------|------|------------|
| 硬件（云端） | Ascend 910 + ARM | 华为云 ModelArts |
| 硬件（边缘） | Atlas 200DK（昇腾310） | IT21DMDA 或 IT21VDMB 主板 |
| AI框架 | MindSpore | 1.5 |
| CANN | 固件驱动 + CANN 软件 | 固件驱动 1.0.10 / CANN 5.0.2alpha003 |
| 编程语言 | Python | 3.7 |
| 存储介质 | SD 卡（Samsung/Kingston 64G Class10） | — |

---

## 阶段一：Atlas 200DK 合设环境搭建



Atlas 200DK 同时充当**开发环境**和**运行环境**，无需额外配置 Ubuntu x86 PC 作为开发机，部署流程简单。

### 主要步骤

1. **准备配件**：Atlas 200DK 开发板、SD 卡、读卡器、USB Type-C 连接线
2. **下载合设镜像**：从昇腾论坛下载预装固件驱动和 CANN 软件的 dd 镜像
3. **烧录镜像**：使用 [Etcher](https://www.balena.io/etcher/) 工具将镜像写入 SD 卡
4. **配置 USB 网卡**：
   - Windows 端安装 USB 虚拟网卡驱动
   - 将 PC 侧 USB 虚拟网卡 IP 修改为 `192.168.1.x` 网段
   - 通过 `ssh HwHiAiUser@192.168.1.2` 访问开发板
5. **开发板联网**：通过直接插入网线实现联网

---

## 阶段二：MobileNetV2 垃圾分类模型训练（云端）

### MobileNetV2 简介

MobileNetV2 是 2018 年提出的轻量级网络，在 MobileNetV1 基础上引入了两项关键改进：

- **线性瓶颈（Linear Bottleneck）**：防止 ReLU 破坏低维特征信息
- **倒残差（Inverted Residual）**：通道先扩张后压缩，在瓶颈层间使用 shortcut 连接

网络输入为 `3×224×224`，经过多组 InvertedResidual 模块后输出 `k` 类分类结果。

### 训练流程

1. **登录 ModelArts 控制台**，创建 Notebook（Ascend+ARM 基础镜像，华北-北京四区域）
2. **下载垃圾分类数据集**，进行数据预处理（归一化、resize 至 224×224）
3. **配置超参数**：
   - 优化器：Momentum
   - 损失函数：SoftmaxCrossEntropyWithLogits
   - 学习率策略：Cosine Decay
   - 训练轮次：50 epochs
4. **模型训练与评估**：训练过程中实时记录 `train_loss`、`eval_loss`、`eval_acc`，绘制折线图
5. **挑选最优 Checkpoint**：根据 `eval_acc` 最大值选取，实验中 `epoch=27` 时效果最优
6. **导出模型文件**：
   - `mobilenetv2.air` — 用于 Atlas 200DK 推理
   - `mobilenetv2.mindir` — 用于手机端推理

```python
# 导出 AIR 模型
export(network, Tensor(input), file_name='mobilenetv2.air', file_format='AIR')
# 导出 MindIR 模型
export(network, Tensor(input), file_name='mobilenetv2', file_format='MINDIR')
```

---

## 阶段三：开发板推理实验

### 推理流程

```
接入开发板 → 下载项目文件 → 模型转换（ATC）→ 模型推理（AscendCL）→ 查看结果
```

### 模型转换（ATC 工具）

ATC（Ascend Tensor Compiler）是 CANN 提供的离线模型转换工具，将 `.air` 文件转换为昇腾硬件可执行的 `.om` 离线模型文件。在开发板上执行：

```bash
atc --model=mobilenetv2.air \
    --framework=1 \
    --output=mobilenetv2 \
    --soc_version=Ascend310
```

### 推理代码核心流程（AscendCL / Python）

```
初始化 ACL 资源
  └─ 加载 .om 模型文件，构建输出内存
      └─ 读取本地图像，DVPP 解码（JPEGD）+ resize
          └─ 执行模型推理
              └─ 解析输出：获取分类类别 + 置信度，标注结果图并保存
```

关键技术点：
- 使用 **DVPP**（数字视觉预处理）对图像进行硬件加速解码和缩放
- 输出格式为 `YUV420SP`，缩放至模型要求的输入分辨率
- 分类结果以置信度标注形式叠加在原图上保存

### 推理结果示例

```
data_en/test/Cardboard/00091.jpg  →  Cardboard ✓
data_en/test/Cardboard/00092.jpg  →  Cardboard ✓
...
```

---

## 项目结构

```
garbage_classification/
├── atlas_utils/          # AscendCL 工具库（模型、图像、DVPP 封装）
│   ├── acl_model.py
│   ├── acl_dvpp.py
│   ├── acl_image.py
│   └── ...
├── model/
│   └── mobilenetv2.air   # 待转换模型文件
├── data/                 # 测试图像
└── src/                  # 推理主逻辑
```

---

## 实验总结

| 实验阶段 | 核心收获 |
|---------|---------|
| 环境搭建 | 掌握 Atlas 200DK 合设环境的 SD 卡烧录与 USB 网卡配置方法 |
| 云端训练 | 了解 MindSpore 图像分类完整流程，熟悉 MobileNetV2 网络结构 |
| 开发板推理 | 掌握 ATC 模型转换和 AscendCL 推理应用开发的基本操作 |

通过本次实验，完成了 AI 模型从**云端训练**到**边缘部署**的端到端开发流程，对华为昇腾 AI 全栈（MindSpore + CANN + Atlas 硬件）有了系统性认识。

---

## 参考资源

- [Atlas 200DK 合设环境搭建（镜像恢复方式）](https://gitee.com/ascend/samples/wikis/Atlas200dk合设环境搭建--用镜像恢复的方式)
- [垃圾分类项目文件下载](https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com/CANN/garbage_classification.zip)
- [Etcher 烧录工具](https://www.balena.io/etcher/)
- [华为云 ModelArts 控制台](https://console.huaweicloud.com/modelarts/)

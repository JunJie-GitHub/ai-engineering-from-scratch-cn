# 视觉 Transformer (Vision Transformers, ViT)

> 图像是图像块 (patches) 的网格，句子是词元 (tokens) 的网格。同一个 Transformer 模型能够同时处理这两者。

**类型：** 实战构建
**语言：** Python
**前置要求：** 第 7 阶段 · 05（完整 Transformer），第 4 阶段 · 03（卷积神经网络 (CNNs)），第 4 阶段 · 14（视觉 Transformer 简介）
**预计耗时：** 约 45 分钟

## 问题背景

在 2020 年之前，计算机视觉 (Computer Vision) 几乎等同于卷积操作。ImageNet、COCO 以及各类目标检测基准测试中的每一个 SOTA (State-of-the-Art) 模型都采用 CNN (Convolutional Neural Network) 作为骨干网络 (backbone)。而 Transformer 架构则专用于自然语言处理领域。

Dosovitskiy 等人（2020）在论文《An Image is Worth 16x16 Words》中证明，你可以完全摒弃卷积操作。将图像切割为固定大小的图像块，将每个图像块线性投影为嵌入向量 (embedding)，然后将该序列输入标准的 Transformer 编码器 (vanilla Transformer encoder) 中。在足够大的数据规模下（例如使用 ImageNet-21k 进行预训练或更大规模的数据），ViT 的性能能够匹敌甚至超越基于 ResNet 的模型。

ViT 开启了 2026 年更广泛的一种趋势：单一架构，多模态 (modalities) 通用。Whisper 将音频转化为词元，ViT 将图像转化为词元，机器人控制使用动作词元，视频处理使用像素词元。Transformer 并不关心输入数据的模态——只要输入序列，它就能从中学习。

截至 2026 年，ViT 及其衍生模型（DeiT、Swin、DINOv2、ViT-22B、SAM 3）已占据计算机视觉领域的大部分版图。CNN 在边缘设备 (edge devices) 和延迟敏感型任务 (latency-sensitive tasks) 中依然保持优势。而在其他所有场景中，技术栈的某个环节几乎都少不了 ViT 的身影。

## 核心概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### 步骤 1 — 图像分块 (Patchify)

将 `H × W × C` 的图像分割为 `N × (P·P·C)` 的展平图像块 (Flat Patches) 序列。典型配置：`224 × 224` 图像，`16 × 16` 的块大小 → 196 个块，每个块包含 768 个值。

image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768

块大小 (Patch Size) 是关键调节参数。块越小 = 词元 (Tokens) 越多，分辨率越高，但注意力机制 (Attention) 的计算成本呈二次方增长。块越大 = 特征越粗糙，计算成本越低。

### 步骤 2 — 线性嵌入 (Linear Embedding)

通过一个可学习的矩阵将每个展平的图像块投影到 `d_model` 维度。这等价于一个卷积核大小为 `P`、步长为 `P` 的卷积操作。在 PyTorch 中，这本质上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` —— 仅需两行代码即可实现。

### 步骤 3 — 添加 `[CLS]` 词元与位置嵌入 (Positional Embeddings)

- 在序列前端添加一个可学习的 `[CLS]` 词元 (Token)。其最终的隐藏状态 (Hidden State) 将作为用于分类的图像表征。
- 添加可学习的位置嵌入（ViT 原版）或二维正弦位置编码（后续变体）。
- 2024 年及以后，旋转位置编码 (RoPE) 被扩展至二维以处理位置信息，有时甚至不再使用显式的位置嵌入。

### 步骤 4 — 标准 Transformer 编码器 (Transformer Encoder)

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 模块。其结构与 BERT 完全相同，不包含任何针对视觉设计的特定层。这正是该论文最精妙的教学启示。

### 步骤 5 — 任务头 (Head)

用于分类任务时：提取 `[CLS]` 的隐藏状态 → 线性层 → Softmax。对于 DINOv2 或 SAM 等模型，则舍弃 `[CLS]`，直接使用图像块的嵌入表示。

### 具有重要影响的变体模型

| 模型 | 年份 | 核心改动 |
|-------|------|--------|
| ViT | 2020 | 原始版本。固定块大小，使用全局注意力机制。 |
| DeiT | 2021 | 引入知识蒸馏 (Knowledge Distillation)；仅需在 ImageNet-1k 上即可训练。 |
| Swin | 2021 | 采用层级结构与滑动窗口机制。将计算复杂度降至次二次方。 |
| DINOv2 | 2023 | 自监督学习 (Self-supervised Learning)（无需标签）。提供最佳的通用视觉特征。 |
| ViT-22B | 2023 | 220 亿参数；验证了缩放定律 (Scaling Laws) 的适用性。 |
| SigLIP | 2023 | ViT 与语言模型配对，采用 Sigmoid 对比损失 (Sigmoid Contrastive Loss)。 |
| SAM 3 | 2025 | 分割一切模型；采用 ViT-Large 架构与可提示掩码解码器 (Promptable Mask Decoder)。 |

### 为何其发展经历了一段沉淀期

ViT 需要海量数据才能达到与卷积神经网络 (CNN) 相当的性能，因为它缺乏 CNN 固有的归纳偏置 (Inductive Biases)（如平移不变性 (Translation Invariance) 与局部性 (Locality)）。在缺乏超过 1 亿张标注图像或强自监督预训练 (Self-supervised Pretraining) 的情况下，在同等算力下 CNN 依然占据优势。DeiT 在 2021 年通过知识蒸馏技巧解决了这一问题；而 DINOv2 则在 2023 年凭借自监督学习彻底攻克了这一瓶颈。

## 动手实现

参见 `code/main.py`。仅使用标准库实现图像分块（patchify）+ 线性嵌入（linear embedding）+ 健全性检查（sanity checks）。不包含训练过程——任何实际规模的视觉 Transformer（Vision Transformer, ViT）都需要 PyTorch 和数小时的 GPU 计算时间。

### 步骤 1：构造模拟图像
一个 24 × 24 的 RGB 图像，表示为包含 `(R, G, B)` 元组的行列表。我们使用 6×6 的图像块（patch）→ 共 16 个图像块，每个对应一个 108 维的嵌入向量（embedding vector）。

### 步骤 2：图像分块（patchify）
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
光栅扫描顺序（raster order）：按行优先遍历网格。所有 ViT 均采用此顺序。

### 步骤 3：线性嵌入（linear embed）
将每个展平的图像块与一个随机的 `(patch_flat_size, d_model)` 矩阵相乘。验证在添加 `[CLS]` 标记后，输出形状是否为 `(N_patches + 1, d_model)`。

### 步骤 4：统计实际 ViT 的参数量
打印 ViT-Base 的参数量：12 层、12 个注意力头（attention heads）、d=768、patch=16。与 ResNet-50（约 2500 万）进行对比。ViT-Base 约为 8600 万。ViT-Large 约为 3.07 亿。ViT-Huge 约为 6.32 亿。

## 实际应用
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation

**DINOv2 嵌入（embeddings）已成为 2026 年图像特征的默认选择。** 冻结主干网络（backbone），仅训练一个轻量级分类头（head）。适用于图像分类、检索、目标检测和图像描述生成。Meta 的 DINOv2 预训练权重（checkpoints）在所有非文本视觉任务上均优于 CLIP。

**图像块尺寸选择。** 小型模型通常使用 16×16（如 ViT-B/16）。密集预测（dense prediction）任务（如分割）使用 8×8 或 14×14（如 SAM、DINOv2）。超大型模型则采用 14×14。

## 部署与交付
参见 `outputs/skill-vit-configurator.md`。该技能模块会根据数据集规模、图像分辨率和计算预算，为新的视觉任务自动选择合适的 ViT 变体与图像块尺寸。

## 练习
1. **简单。** 运行 `code/main.py`。验证图像块数量是否等于 `(H/P) * (W/P)`，且展平后的图像块维度是否等于 `P*P*C`。
2. **中等。** 实现二维正弦位置编码（2D sinusoidal positional embeddings）——为每个图像块的 `row`（行）和 `col`（列）生成两个独立的正弦编码，并将其拼接。将其输入到一个小型的 PyTorch ViT 中，并在 CIFAR-10 数据集上对比其与可学习位置编码（learnable positional embeddings）的准确率差异。
3. **困难。** 构建一个 3 层的 ViT（使用 PyTorch），使用 4×4 的图像块在 1,000 张 MNIST 图像上进行训练。测量测试集准确率。随后，在相同的 1,000 张图像上加入 DINOv2 预训练（简化版：仅训练编码器，使其能够根据被掩码的图像块（masked patches）预测原始图像块嵌入）。准确率是否有所提升？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 图像块 (Patch) | “视觉 Transformer 的 token” | 图像中 `P × P × C` 区域的像素值展平后的一维向量。 |
| 图像分块 (Patchify) | “切割 + 展平” | 将图像切分为不重叠的图像块，并将每个块展平为向量。 |
| `` `[CLS]` `` 标记 (Token) | “图像摘要” | 前置的可学习标记；其最终的嵌入表示 (Embedding) 即代表整张图像。 |
| 归纳偏置 (Inductive bias) | “模型的先验假设” | ViT 的先验假设比卷积神经网络 (CNN) 更少；需要更多数据来弥补这一差距。 |
| DINOv2 | “自监督 ViT” | 无需标签，通过图像增强与动量教师网络 (Momentum teacher) 进行训练。2026 年表现最佳的通用图像特征提取模型。 |
| SigLIP | “CLIP 的继任者” | 结合 ViT 与文本编码器，使用 Sigmoid 对比损失函数 (Sigmoid contrastive loss) 训练；在同等算力下表现优于 CLIP。 |
| Swin Transformer | “窗口化 ViT” | 采用局部注意力机制与滑动窗口 (Shifted windows) 的分层 ViT；计算复杂度为次二次方 (Sub-quadratic)。 |
| 寄存器标记 (Register tokens) | “2023 年的技巧” | 少量额外的可学习标记，用于吸收注意力汇聚点 (Attention sinks)；可提升 DINOv2 的特征质量。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 的原始论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT 论文。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin Transformer 论文。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2 论文。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — 针对 DINOv2 的寄存器标记 (Register tokens) 改进方案。
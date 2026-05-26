# 视觉 Transformer (Vision Transformer, ViT)

> 将图像切割为图块（patches），将每个图块视为一个词，输入标准 Transformer 进行处理。无需回头。

**类型：** 构建
**语言：** Python
**前置知识：** 第 7 阶段第 02 课（自注意力机制 Self-Attention），第 4 阶段第 04 课（图像分类 Image Classification）
**时长：** 约 45 分钟

## 学习目标

- 从零实现图块嵌入（patch embedding）、可学习位置嵌入（learned positional embedding）、类别令牌（class token）以及 Transformer 编码器块（transformer encoder blocks），以构建一个最小化的 ViT
- 解释为何 ViT 曾被认为需要海量预训练数据，以及 DeiT 和 MAE 如何推翻这一观点
- 对比 ViT、Swin 和 ConvNeXt 在架构先验（architectural priors）上的差异（无先验、局部窗口注意力、卷积骨干网络）
- 使用 `timm` 库及标准的线性探测（linear-probe）/ 微调（fine-tune）流程，在小数据集上对预训练 ViT 进行微调

## 问题背景

过去十年间，卷积（convolution）几乎成了计算机视觉的代名词。卷积神经网络（CNN）具备强大的归纳偏置（inductive biases）——局部性（locality）与平移等变性（translation equivariance）——人们曾认为这些特性无法被替代。直到 Dosovitskiy 等人（2020）证明，将纯 Transformer 直接应用于展平的图像图块，完全不依赖任何卷积结构，在大规模数据下依然能够匹敌甚至超越顶尖的 CNN。

但前提是“大规模”。在 ImageNet-1k 上直接训练的 ViT 败给了 ResNet；但若先在 ImageNet-21k 或 JFT-300M 上预训练，再在 ImageNet-1k 上微调，ViT 便能实现反超。当时的结论是：Transformer 缺乏有用的先验知识，但只要有足够的数据就能自行学习。后续研究（如 DeiT、MAE、DINO）表明，只要采用正确的训练策略（training recipes）——包括强数据增强、自监督预训练和知识蒸馏——ViT 同样能在小规模数据上取得优异表现。

截至 2026 年，纯 CNN 在边缘设备上仍具竞争力（其中 ConvNeXt 表现最强），但 Transformer 已主导其他所有领域：图像分割（Mask2Former、SegFormer）、目标检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）以及视频理解（VideoMAE、VJEPA）。ViT 的模块结构是必须掌握的核心。

## 核心概念

### 处理流程 (Pipeline)

flowchart LR
    IMG["Image<br/>(3, 224, 224)"] --> PATCH["Patch embedding<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["Flatten to<br/>(196, 768) tokens"]
    FLAT --> CAT["Prepend<br/>[CLS] token"]
    CAT --> POS["Add learned<br/>positional embed"]
    POS --> ENC["N transformer<br/>encoder blocks"]
    ENC --> CLS["Take [CLS]<br/>token output"]
    CLS --> HEAD["MLP classifier"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a

共七个步骤：图像块 (Patches) -> 词元 (Tokens) -> 注意力机制 (Attention) -> 分类器 (Classifier)。每个变体（如 DeiT、Swin、ConvNeXt、MAE 预训练）仅修改其中一到两个步骤，其余保持不变。

### 图像块嵌入 (Patch Embedding)

关键在于第一个卷积层 (Convolution)。卷积核大小为 16，步长为 16，因此 224x224 的图像会被划分为 14x14 的网格，每个网格包含一个 16x16 的图像块，并被投影为 768 维的嵌入向量。这单个卷积层同时完成了图像块划分与线性投影。

Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)

196 个图像块对应 196 个词元。每个词元的特征维度为 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### 类别词元 (Class Token)

在序列前端添加一个可学习的向量：

tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)

经过 N 个 Transformer 编码器块后，`[CLS]` 的输出即为全局图像表示。分类头仅读取这一个向量。

### 位置嵌入 (Positional Embedding)

Transformer 本身不具备空间位置的概念。需要为每个词元添加一个可学习的向量：

tokens = tokens + learned_pos_embedding   (also shape (197, 768))

该嵌入是模型的参数；基于梯度的训练会使其自适应二维图像结构。虽然存在正弦二维位置编码等替代方案，但在实际中极少使用。

### Transformer 编码器块 (Transformer Encoder Block)

标准结构。包含多头自注意力机制 (Multi-Head Self-Attention)、多层感知机 (MLP)、残差连接 (Residual Connections) 以及前置层归一化 (Pre-LayerNorm)。

x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)

ViT-B/16 堆叠了 12 个这样的块，每个块包含 12 个注意力头，总计 8600 万参数。

### 为何使用前置层归一化 (Pre-LN)

早期的 Transformer 使用后置层归一化 (Post-LN)（`x = LN(x + sublayer(x))`），若不进行预热 (Warmup)，很难训练超过 6-8 层。而前置层归一化（`x = x + sublayer(LN(x))`）无需预热即可稳定训练更深的网络。如今所有的 ViT 和现代大语言模型 (LLM) 均采用 Pre-LN。

### 图像块尺寸的权衡 (Patch Size Trade-off)

- 16x16 图像块 -> 196 个词元，为标准配置。
- 32x32 图像块 -> 49 个词元，速度更快但分辨率较低。
- 8x8 图像块 -> 784 个词元，细节更丰富，但注意力机制的计算成本呈 O(n^2) 增长，扩展性较差。

图像块越大 = 词元越少 = 速度越快但空间细节越少。SwinV2 在分层窗口中使用 4x4 的图像块。

### DeiT 在 ImageNet-1k 上训练 ViT 的方案 (DeiT's Recipe)

原始 ViT 需要 JFT-300M 数据集才能超越卷积神经网络 (CNN)。DeiT（Touvron 等人，2020）仅通过四项改进，就在 ImageNet-1k 上将 ViT-B 的 Top-1 准确率训练至 81.8%：

1. 强数据增强 (Data Augmentation)：RandAugment、Mixup、CutMix、随机擦除 (Random Erasing)。
2. 随机深度 (Stochastic Depth)：在训练期间随机丢弃整个网络块。
3. 重复增强 (Repeated Augmentation)：每个批次中对同一图像采样 3 次。
4. 从 CNN 教师模型进行知识蒸馏 (Knowledge Distillation)（可选，可进一步提升准确率）。

所有现代 ViT 的训练方案均源于 DeiT。

### Swin 与 ConvNeXt 对比

- **Swin**（Liu 等人，2021）—— 基于窗口的注意力机制。每个块仅在局部窗口内进行注意力计算；交替的块会移动窗口以跨窗口混合信息。在保留注意力算子的同时，重新引入了类似 CNN 的局部性先验 (Locality Prior)。
- **ConvNeXt**（Liu 等人，2022）—— 重新设计的 CNN，其架构选择与 Swin 保持一致（深度可分离卷积、LayerNorm、GELU、倒残差瓶颈结构）。研究表明，性能差距并非源于“注意力机制与卷积的对抗”，而是“现代训练方案与架构设计”的结合。

截至 2026 年，ConvNeXt-V2 和 Swin-V2 均已达到生产级标准；具体选择取决于你的推理技术栈（ConvNeXt 在边缘设备上的编译优化更好）以及预训练数据集。

### MAE 预训练 (MAE Pretraining)

掩码自编码器 (Masked Autoencoder, He 等人，2022)：随机掩码 (Mask) 75% 的图像块，训练编码器仅处理可见的 25% 部分，并训练一个小型解码器根据编码器的输出重建被掩码的图像块。预训练完成后，丢弃解码器并对编码器进行微调 (Fine-tuning)。

MAE 使得 ViT 仅凭 ImageNet-1k 即可进行有效训练，并达到当前最优水平 (SOTA)，现已成为默认的自监督学习 (Self-Supervised Learning) 方案。

## 构建

### 步骤 1：图像块嵌入（Patch Embedding）

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

一次卷积、一次展平、一次转置。这就是将图像转换为词元（Token）的完整步骤。

### 步骤 2：Transformer 模块（Transformer Block）

前置层归一化（Pre-Layer Normalization）、多头自注意力机制（Multi-Head Self-Attention）、采用 GELU 激活函数的多层感知机（Multi-Layer Perceptron, MLP），以及残差连接（Residual Connections）。

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

`nn.MultiheadAttention` 负责处理注意力头划分、缩放点积（Scaled Dot-Product）计算以及输出投影（Output Projection）。设置 `batch_first=True` 可使张量形状变为 `(N, seq, dim)`。

### 步骤 3：构建 ViT（Vision Transformer）

class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")

该模型约包含 280 万（2.8M）个参数，属于可在 CPU 上轻松运行的小型 ViT。标准的 ViT-B 模型参数量约为 8600 万（86M），只需将类定义中的参数改为 `dim=768, depth=12, num_heads=12` 即可实现。

### 步骤 4：合理性检查（Sanity Check）—— 单张图像推理（Inference）

logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")

代码应能无报错运行。输出的概率值之和应为 1。

## 使用方法

`timm` 为每个视觉 Transformer (Vision Transformer) 变体都提供了在 ImageNet 上预训练的权重 (Pretrained Weights)。只需一行代码：

import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)

到 2026 年，`timm` 已成为视觉 Transformer 在生产环境中的默认选择。它在统一的 API 下支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 以及数十种其他模型。

对于多模态 (Multi-modal) 任务（图像 + 文本），`transformers` 库提供了 CLIP、SigLIP、BLIP-2 和 LLaVA。这些模型中的图像编码器 (Image Encoder) 均为 ViT 的变体。

## 产出物

本课程的产出物包括：

- `outputs/prompt-vit-vs-cnn-picker.md` — 一个提示词 (Prompt)，可根据数据集规模、算力 (Compute) 和推理栈 (Inference Stack)，在 ViT、ConvNeXt 或 Swin 之间进行选择。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — 一项技能模块 (Skill)，用于验证 ViT 的图像块嵌入 (Patch Embedding) 和位置嵌入 (Positional Embedding) 的形状是否与模型预期的序列长度匹配，从而捕获最常见的模型移植错误。

## 练习

1. **(简单)** 打印上述微型 ViT 前向传播 (Forward Pass) 过程中每个中间张量 (Tensor) 的形状。验证以下流程：输入 `(N, 3, 64, 64)` -> 图像块 `(N, 16, 192)` -> 加入 CLS Token 后 `(N, 17, 192)` -> 分类器输入 `(N, 192)` -> 输出 `(N, num_classes)`。
2. **(中等)** 在第 4 课的合成 CIFAR 数据集上，对预训练的 `timm` ViT-S/16 进行微调 (Fine-tuning)。与在相同数据上微调 ResNet-18 的结果进行对比。报告训练时间和最终准确率。
3. **(困难)** 为微型 ViT 实现掩码自编码器 (Masked Autoencoder, MAE) 预训练：遮蔽 75% 的图像块，训练编码器与一个小型解码器以重建被遮蔽的图像块。在预训练前后，分别评估该模型在合成数据上的线性探测 (Linear-probe) 准确率。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 图像块嵌入 (Patch Embedding) | “第一个卷积层” | 卷积核大小（kernel size）= 步长（stride）= 图像块大小（patch size）的卷积操作；将图像转换为词元嵌入 (Token Embedding) 的网格 |
| 类别词元 (Class Token) | “[CLS]” | 一个可学习的向量，被添加到词元序列的开头；其最终输出代表全局图像特征 |
| 位置嵌入 (Positional Embedding) | “可学习的位置编码” | 一个可学习的向量，添加到每个词元上，使 Transformer 能够感知每个图像块的原始位置 |
| 前置层归一化 (Pre-LN) | “子层前的 LayerNorm” | 更稳定的 Transformer 变体结构：采用 `x + sublayer(LN(x))` 而非 `LN(x + sublayer(x))` |
| 多头注意力 (Multi-head Attention) | “并行注意力” | 标准的 Transformer 注意力机制被拆分为 `num_heads` 个独立的子空间进行计算，随后将结果拼接 |
| ViT-B/16 | “Base 模型，Patch 大小为 16” | 标准配置：隐藏层维度（dim）=768，层数（depth）=12，注意力头数（heads）=12，patch_size=16，输入图像尺寸=224；参数量约 8600 万 |
| DeiT | “数据高效型 ViT” | 仅使用 ImageNet-1k 数据集并配合强数据增强训练的 ViT；证明了大规模预训练数据集并非绝对必要 |
| MAE | “掩码自编码器” | 自监督预训练方法：遮蔽 75% 的图像块并进行重建；目前主流的 ViT 预训练方案 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT（Vision Transformer）的原始论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — 如何仅使用 ImageNet-1k 数据集训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE（Masked Autoencoder）预训练
- [timm documentation](https://huggingface.co/docs/timm) — 生产环境中使用各类视觉 Transformer（Vision Transformer）的权威参考文档
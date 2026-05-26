# 视觉 Transformer (Vision Transformer) 与 Patch-Token 原语

> 在处理任何多模态任务之前，图像必须先转换为 Transformer 能够处理的令牌 (token) 序列。2020 年的 ViT 论文通过 16x16 像素的图像块 (patch)、线性投影 (linear projection) 和位置嵌入 (positional embedding) 解决了这一问题。五年后的今天，所有 2026 年的前沿模型（原生分辨率达 2576px 的 Claude Opus 4.7、Gemini 3.1 Pro、Qwen3.5-Omni）依然沿用这一基础架构——尽管编码器已从 ViT 演进至 DINOv2 再到 SigLIP 2，引入了寄存器令牌 (register token)，位置编码方案也升级为二维旋转位置编码 (2D-RoPE)，但这一核心原语始终未变。本教程将端到端地解析图像块-令牌 (patch-token) 处理流水线，并使用 Python 标准库 (stdlib) 实现它，从而为第 12 阶段后续内容建立关于“视觉令牌”的具体心智模型。

**Type:** 学习
**Languages:** Python（标准库，patch 分词器 + 几何计算器）
**Prerequisites:** 第 7 阶段（Transformer）、第 4 阶段（计算机视觉）
**Time:** 约 120 分钟

## 学习目标

- 将 HxWx3 格式的图像转换为带有正确位置编码的 patch token 序列。
- 针对给定参数（patch 尺寸、分辨率、隐藏层维度、网络深度）的 ViT，计算序列长度、参数量和浮点运算次数 (FLOPs)。
- 列举推动 ViT 从 2020 年学术研究走向 2026 年工业落地的三项关键升级：自监督预训练 (self-supervised pretraining, DINO / MAE)、寄存器令牌 (register tokens) 以及原生分辨率打包 (native-resolution packing)。
- 针对下游任务，在 CLS 池化 (CLS pooling)、平均池化 (mean pooling) 和寄存器令牌之间做出选择。

## 问题背景

Transformer 处理的是向量序列。文本本身已经是序列形式（字节或 token）。而图像是包含三个颜色通道的二维像素网格——并非序列。如果将每个像素直接展平，一张 224x224 的 RGB 图像将变成 150,528 个 token，在此长度下计算自注意力机制 (self-attention) 是完全不可行的（其计算复杂度与序列长度呈二次方关系）。

2020 年之前的方法通常在网络前端拼接一个卷积神经网络 (CNN) 特征提取器：例如 ResNet 会生成一个 7x7 的特征图，包含 2048 维向量，再将这 49 个 token 输入 Transformer。这种方法虽然有效，但继承了 CNN 的归纳偏置 (inductive bias)（如平移等变性、局部感受野），同时也丧失了 Transformer 对数据规模的强大扩展能力。

Dosovitskiy 等人（2020）提出了一个直击本质的问题：如果我们直接跳过 CNN 会怎样？他们将图像分割为固定大小的图像块（例如 16x16 像素），将每个图像块线性投影为向量，添加位置嵌入，然后将该序列输入标准 Transformer。在当时，这无异于异端邪说——不用卷积做视觉任务。但在海量数据（JFT-300M，随后是 LAION）的加持下，它在 ImageNet 上击败了 ResNet，并持续展现出性能提升。

到了 2026 年，ViT 原语已成为毋庸置疑的基石。所有开源权重的视觉语言模型 (VLM) 的视觉塔 (vision tower) 都是其衍生架构（如 DINOv2、SigLIP 2、CLIP、EVA、InternViT）。如今的问题不再是“我们是否应该使用图像块？”，而是“该选择多大的 patch 尺寸、怎样的分辨率调度策略、何种预训练目标，以及哪种位置编码方案。”

## 核心概念

### 图像块 (Patch) 作为词元 (Token)

给定形状为 `(H, W, 3)` 的图像 `x` 和图像块大小 (Patch Size) `P`，你可以将图像划分为 `(H/P) x (W/P)` 个不重叠的网格图像块。每个图像块是一个 `P x P x 3` 的像素立方体。将每个立方体展平为 `3 P^2` 维的向量。应用一个形状为 `(3 P^2, D)` 的共享线性投影 (Linear Projection) `W_E`，将每个图像块映射到模型的隐藏维度 (Hidden Dimension) `D`。

对于 ViT-B/16 的标准配置 (Canonical Config)：
- 分辨率 224，图像块大小 16 → 网格 14x14 → 196 个图像块词元 (Patch Token)。
- 每个图像块包含 `16 x 16 x 3 = 768` 个像素值，投影至 `D = 768`。
- 添加一个可学习的 `[CLS]` 词元 → 序列长度变为 197。

图像块投影在数学上等同于一个卷积核大小 (Kernel Size) 为 `P`、步长 (Stride) 为 `P`、输出通道数为 `D` 的二维卷积 (2D Convolution)。这也是生产代码中的实际实现方式——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。“线性投影”是一种概念上的表述，而“卷积核”的表述则更注重计算效率。

### 位置嵌入 (Positional Embedding)

图像块本身没有固有的顺序——Transformer 将它们视为一个无序集合。早期的 ViT 添加了一个可学习的一维位置嵌入 (1D Positional Embedding)（每个位置对应一个 768 维向量，共 197 个）。这种方法可行，但会将模型绑定到训练时的分辨率：在推理阶段，如果改变网格大小，就必须对位置表进行插值。

现代视觉骨干网络 (Vision Backbone) 采用 2D-RoPE（如 Qwen2-VL 的 M-RoPE、SigLIP 2 的默认配置）或分解式二维位置编码。2D-RoPE 根据图像块的（行，列）索引对查询向量 (Query Vector) 和键向量 (Key Vector) 进行旋转，使模型能够从旋转角度中推断出相对的二维位置。无需位置表。模型在推理时能够处理任意大小的网格。

### CLS 词元、池化输出与寄存器词元 (Register Token)

什么是图像级表示 (Image-level Representation)？目前有三种主流方案并存：

1. `[CLS]` 词元。在图像块序列前添加一个可学习向量。经过所有 Transformer 块后，CLS 词元的隐藏状态即为图像表示。该设计继承自 BERT，被原始 ViT 和 CLIP 采用。
2. 均值池化 (Mean Pool)。对图像块词元的输出隐藏状态求平均。被 SigLIP、DINOv2 以及大多数现代视觉语言模型 (VLM) 采用。
3. 寄存器词元。Darcet 等人 (2023) 观察到，在没有显式汇聚词元 (Sink Token) 的情况下训练的 ViT 会产生高范数的“伪影”图像块，从而劫持自注意力机制 (Self-Attention)。添加 4–16 个可学习的寄存器词元可以吸收这部分负载，并提升密集预测任务（如分割、深度估计）的质量。DINOv2 和 SigLIP 2 均内置了寄存器词元。

该选择对下游任务至关重要。CLS 词元适用于分类任务。对于将图像块词元输入大语言模型 (LLM) 的 VLM，则完全跳过池化步骤——每个图像块都会直接作为 LLM 的输入词元。寄存器词元在交接给 LLM 前会被丢弃（它们仅起支撑作用，而非实际内容）。

### 预训练 (Pretraining)：监督、对比、掩码与自蒸馏

2020 年的 ViT 是在 JFT-300M 数据集上通过监督分类 (Supervised Classification) 进行预训练的。但很快被以下方法取代：

- CLIP (2021)：基于 4 亿图文对的对比学习 (Contrastive Learning)。参见第 12.02 节。
- MAE (2021, He 等人)：掩码 75% 的图像块并重建像素。属于自监督学习 (Self-Supervised Learning)，仅使用纯图像即可工作。
- DINO (2021) / DINOv2 (2023)：采用师生架构的自蒸馏 (Self-Distillation)，无需标签或图像描述。2023 年的 DINOv2 ViT-g/14 是目前最强的纯视觉骨干网络，也是“密集特征 (Dense Features)”应用场景的默认选择。
- SigLIP / SigLIP 2 (2023, 2025)：采用 Sigmoid 损失函数 (Sigmoid Loss) 和 NaFlex 以支持原生宽高比的 CLIP 变体。它是 2026 年开源 VLM（如 Qwen、Idefics2、LLaVA-OneVision）中主流的视觉编码器 (Vision Tower)。

预训练方法的选择决定了骨干网络的擅长领域：CLIP/SigLIP 适用于与文本的语义匹配，DINOv2 适用于提取密集视觉特征，而 MAE 则常作为下游微调 (Finetuning) 的起点。

### 缩放定律 (Scaling Laws)

ViT 的缩放研究 (Zhai 等人, 2022) 表明，ViT 的质量遵循与模型规模、数据规模和计算量相关的可预测缩放定律。在固定计算量下：
- 更大的模型 + 更多的数据 → 更优的质量。
- 图像块大小是调节序列长度与保真度 (Fidelity) 的杠杆。图像块大小 14（DINOv2/SigLIP SO400m 的典型配置）比大小 16 每图像生成更多词元；更利于 OCR 和密集任务，但速度较慢。
- 分辨率是另一个关键杠杆。从 224 提升到 384 再到 512 几乎总能带来性能提升，但浮点运算次数 (FLOPs) 的成本呈平方级增长。

ViT-g/14（10 亿参数，图像块大小 14，分辨率 224 → 256 个词元）和 SigLIP SO400m/14（4 亿参数，图像块大小 14）是 2026 年开源 VLM 的两大主力编码器。

### ViT 的参数量计算

完整的计算代码位于 `code/main.py`。以 224 分辨率下的 ViT-B/16 为例：

patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M

在加载检查点 (Checkpoint) 之前，建议用这种方式对每个 ViT 进行粗略估算。骨干网络的规模直接决定了你在任何下游 VLM 中所需的显存 (VRAM) 下限。

### 2026 年生产环境配置

2026 年大多数开源 VLM 搭载的编码器是支持原生分辨率 (NaFlex) 的 SigLIP 2 SO400m/14。其配置如下：
- 4 亿参数。
- 图像块大小 14，默认分辨率 384 → 每张图像生成 729 个图像块词元。
- 图像级任务采用均值池化；在视觉问答 (VQA) 中，全部 729 个图像块词元都会输入 LLM。
- 4 个寄存器词元，在交接给 LLM 前会被丢弃。
- 采用 2D-RoPE 并结合图像级缩放以支持原生宽高比。

该配置中的每一项决策，都能追溯到可供查阅的学术论文。

## 使用它

`code/main.py` 是一个图像块标记器（patch tokenizer）与几何计算器。它接收输入参数（图像高度 H、宽度 W、图像块大小 P、隐藏维度 D、网络深度 L），并报告以下内容：

- 分块处理后的网格形状与序列长度。
- 合成 8x8 像素测试图像的标记序列（逐步演示展平与投影路径）。
- 按图像块嵌入（patch embed）、位置嵌入（position embed）、Transformer 块（transformer blocks）和输出头（head）分类的参数数量。
- 在目标分辨率下，单次前向传播（forward pass）的浮点运算次数（FLOPs）。
- 针对 ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224 和 SigLIP SO400m/14 @ 384 的对比表格。

运行该脚本。将计算出的参数数量与官方公布的数值进行核对。尝试调整分块大小和分辨率，直观感受标记数量带来的计算成本。

## 交付成果

本课时将生成 `outputs/skill-patch-geometry-reader.md` 文件。给定视觉 Transformer（ViT）的配置（分块大小、分辨率、隐藏维度、网络深度），该文件会输出标记数量、参数数量以及显存（VRAM）估算值，并附带推导依据。在为视觉语言模型（VLM）选择视觉骨干网络（vision backbone）时，请务必运用此技能——它能有效避免“标记数量暴增导致大语言模型（LLM）上下文窗口溢出”的意外情况。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、分块大小为 14 时的图像块标记序列长度。这与仅使用分类标记（CLS-only representation）的表示方式相比有何差异？

2. 在分块大小为 14 的情况下，一帧 1080p 图像（1920x1080）会产生多少个标记？对于一段 5 分钟、30 FPS 的视频，总共会产生多少视觉标记？在降低计算成本方面，哪种策略最有效：池化（pooling）、帧采样（frame sampling）还是标记合并（token merging）？

3. 使用纯 Python 实现针对图像块标记的平均池化（mean pooling）。验证对 DINOv2 输出的 196 个标记进行平均池化的结果，是否与模型在请求池化嵌入时 `forward` 方法返回的结果一致。

4. 阅读论文《Vision Transformers Need Registers》（arXiv:2309.16588）的第 3 节。用两句话说明寄存器（registers）吸收了何种伪影（artifact），以及这对下游密集预测（dense prediction）任务为何重要。

5. 修改 `code/main.py` 以支持“分块与打包”（patch-n'-pack）功能：给定一组不同分辨率的图像，生成一个单一的打包序列及块对角注意力掩码（block-diagonal attention mask）。在学习到第 12.06 课时，请对照该课内容进行验证。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 图像块 (Patch) | “16x16 像素的方块” | 输入图像中固定大小且不重叠的区域；在模型中会被转换为一个 token (词元) |
| 图像块嵌入 (Patch embedding) | “线性投影” | 一个共享的可学习矩阵（或步长为 P 的 Conv2d 层），用于将展平的图像块像素映射为 D 维向量 |
| 分类词元 (CLS token) | “分类 token” | 添加在序列开头的可学习向量，其最终隐藏状态用于表征整张图像；在 2026 年的架构中已变为可选 |
| 寄存器词元 (Register token) | “汇聚 token” | 额外的可学习 token，用于吸收视觉 Transformer (ViT) 在预训练过程中产生的高范数注意力伪影 |
| 位置嵌入 (Position embedding) | “位置信息” | 为每个位置分配的向量或旋转操作，使模型具备序列顺序感知能力；2D-RoPE（二维旋转位置编码）是现代架构的默认选择 |
| 网格 (Grid) | “图像块网格” | 针对给定分辨率和图像块尺寸，由 (H/P) x (W/P) 个图像块组成的二维数组 |
| 原生灵活分辨率 (NaFlex) | “原生灵活分辨率” | SigLIP 2 的特性：单个模型无需重新训练即可支持多种宽高比和分辨率 |
| 主干网络 (Backbone) | “视觉塔” | 预训练的图像编码器，其输出的图像块 token 会作为输入馈送至视觉语言模型 (VLM) 中的大语言模型 (LLM) |
| 池化 (Pooling) | “图像级摘要” | 将多个图像块 token 聚合为单一向量的策略：包括使用 CLS token、均值池化、注意力池化或基于寄存器的方法 |
| 14x14 与 16x16 图像块 (Patch 14 vs 16) | “更细 vs 更粗的网格” | Patch 14 每张图像生成的 token 更多，OCR（光学字符识别）保真度更高但速度较慢；Patch 16 是经典的默认设置 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT (Vision Transformer) 论文。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE (掩码自编码器)，自监督预训练方法。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏技术，无需人工标注。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — 寄存器词元 (Register token) 与注意力伪影分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认的视觉主干 (Vision tower)。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 视觉 Transformer 的经验性缩放定律。
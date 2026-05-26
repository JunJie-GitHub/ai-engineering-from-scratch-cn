# Flamingo 与门控交叉注意力机制（Gated Cross-Attention）用于少样本视觉语言模型（Few-Shot VLMs）

> DeepMind 的 Flamingo（2022）率先实现了两项突破。它证明了单个模型能够处理任意交错排列的图像、视频和文本序列。同时，它展示了视觉语言模型（VLMs）具备上下文学习（in-context learning）能力——只需提供一个包含三个示例（图像、描述）对的少样本提示（few-shot prompt），模型就能在无需任何梯度更新的情况下为新图像生成描述。其核心机制在于：在冻结的大语言模型（frozen LLM）现有层之间插入门控交叉注意力层（gated cross-attention layers），并配备一个从零开始学习的 tanh 门控函数，从而在初始化阶段保留 LLM 原有的文本处理能力。本课程将深入讲解 Flamingo 的 Perceiver 重采样器（Perceiver resampler）与门控交叉注意力架构——这也是 Gemini 交错输入（interleaved inputs）和 Idefics2 视觉令牌（visual tokens）的前身。

**Type:** 学习
**Languages:** Python（标准库，门控交叉注意力机制 + Perceiver 重采样器演示）
**Prerequisites:** 第 12 阶段 · 03（BLIP-2 Q-Former）
**Time:** 约 120 分钟

## 学习目标

- 解释门控交叉注意力机制如何通过 `tanh(gate) = 0` 在初始化阶段保留冻结大语言模型的文本处理能力。
- 逐步解析 Perceiver 重采样器：通过交叉注意力机制将 N 个图像块（image patches）转换为 K 个固定的“潜在”查询向量（latent queries）。
- 描述 Flamingo 如何利用尊重图像位置的因果掩码（causal masking）来处理交错的图像-文本序列。
- 复现少样本多模态提示结构（3 个图像-描述示例后接一个查询图像）。

## 问题背景

BLIP-2 将 32 个视觉令牌（visual tokens）输入到冻结大语言模型的输入层中。这种方法适用于每个提示仅包含一张图像的场景。但如果你希望输入*多张*与文本交错的图像，例如“这是图像 A，请描述它；这是图像 B，请描述它；现在这是图像 C，请描述它”，该怎么办？此时，LLM 的自注意力机制（self-attention）必须在单一数据流中同时处理图像令牌和文本令牌，而“哪些位置可以关注哪些图像”的问题也会变得相当棘手。

Flamingo 的解决方案是：完全不改变 LLM 的输入数据流。而是在现有的 LLM 模块之间插入额外的交叉注意力层。文本令牌依然照常流经 LLM 的因果自注意力机制。在每隔几个 LLM 模块之后，文本令牌还会通过一个新的门控层与图像特征进行交叉注意力计算。该门控参数（初始化为零）意味着在训练初始阶段，这些新增层相当于空操作（no-ops）——模型的行为与预训练的 LLM 完全一致。随着训练的推进，门控逐渐打开，视觉信息开始流入模型。

Flamingo 解决的第二个问题是：如何处理每个提示中数量不定的图像（0 张、1 张或多张）？答案是引入 Perceiver 重采样器——这是一个小型的交叉注意力模块，它能够接收任意数量的图像块，并输出固定数量的视觉潜在令牌（visual latent tokens）。无论提示中包含多少张图像，LLM 的交叉注意力层所接收到的张量形状始终保持一致。

## 核心概念

### 冻结的大语言模型 (Frozen Large Language Model)

Flamingo 以一个冻结的 Chinchilla 70B 大语言模型 (Large Language Model) 为基础。全部 700 亿参数权重保持原样。模型原有的文本自注意力机制 (Self-Attention) 和前馈神经网络 (Feed-Forward Network) 照常运行。

### Perceiver 重采样器 (Perceiver Resampler)

对于提示词 (Prompt) 中的每张图像，视觉 Transformer (Vision Transformer) 会生成 N 个图像块标记 (Patch Tokens)。Perceiver 重采样器包含 K 个固定的可学习潜在向量 (Latents)（Flamingo 使用 K=64）。每个重采样器模块包含两个子步骤：

1. 交叉注意力 (Cross-Attention)：K 个潜在向量对 N 个图像块标记进行注意力计算（查询向量 Q 来自潜在向量，键值对 K/V 来自图像块）。
2. 潜在向量内部的自注意力 (Self-Attention) 与前馈神经网络 (FFN)。

经过 6 个重采样器模块后，无论 ViT 生成了多少个图像块，输出均为维度为 1024 的 K=64 个视觉标记 (Visual Tokens)。一张 224x224 的图像（196 个图像块）和一张 480x480 的图像（900 个图像块）最终都会输出为 64 个重采样标记。

对于视频，重采样器在时间维度上应用：每一帧的图像块生成 64 个潜在向量，并通过时间位置编码 (Temporal Positional Encoding) 使模型能够区分 t=0 与 t=N。整个视频最终转化为 T * 64 个视觉标记。

### 门控交叉注意力 (Gated Cross-Attention)

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控交叉注意力模块：

x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated

- `alpha` 是一个可学习的标量，初始化为零。
- `tanh(0) = 0`，因此在初始化时，门控分支的贡献为零。
- 随着 `alpha` 逐渐偏离零，交叉注意力的贡献会平滑增长。
- 残差连接 (Residual Connection) 意味着即使门控完全打开，也不会覆盖 LLM 的文本表征；它只是在原有基础上叠加视觉信息。

这是 Flamingo 中最重要的单一设计选择：视觉条件化 (Visual Conditioning) 是叠加式的、受门控控制的，且在初始化时为零。处于第 0 步的 Flamingo 在处理纯文本输入时，就是一个完美的 Chinchilla 70B。

### 用于交错输入 (Interleaved Inputs) 的掩码交叉注意力 (Masked Cross-Attention)

在类似“<图像 A> 描述 A <图像 B> 描述 B <图像 C> ？”的提示词中，每个文本标记 (Text Token) 只能“看到”序列中位于它之前的图像。交叉注意力掩码强制执行此规则：位于位置 `t` 的文本标记仅对图像索引 `i < i_t` 的图像重采样标记进行注意力计算，其中 `i_t` 是位置 `t` 之前最近的图像索引。“仅看到上一个图像”或“看到所有之前的图像”都是可行的选择；Flamingo 选择了前者。

### 上下文内少样本学习 (In-Context Few-Shot Learning)

Flamingo 的提示词格式如下：

<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a

模型识别出补全模式后，会输出“bird”（或 image3 实际显示的内容）。整个过程无需梯度更新。冻结 LLM 的上下文内学习能力通过门控交叉注意力得以延续——这正是该论文的核心亮点及其重要意义所在。

### 训练数据 (Training Data)

Flamingo 在三个数据集上进行了训练：

1. MultiModal MassiveWeb (M3W)：包含 4300 万张网页，具有交错的图像与文本，并重建了阅读顺序。
2. 图像-文本对 (Image-Text Pairs)（ALIGN + LTIP）：44 亿对。
3. 视频-文本对 (Video-Text Pairs, VTP)：2700 万个短视频片段。

OBELICS (2023) 是该交错网页语料库的开源复现版本，Idefics、Idefics2 以及大多数开源的“类 Flamingo”模型均基于此进行训练。

### OpenFlamingo 与 Otter

OpenFlamingo (2023) 是其开源复现版本。架构完全相同（在冻结的 LLaMA 或 MPT 上应用 Perceiver 重采样器与门控交叉注意力）。提供 3B、4B、9B 规模的模型检查点 (Checkpoints)。由于基础 LLM 规模较小且训练数据较少，其性能略逊于原版 Flamingo。

Otter (2023) 基于 OpenFlamingo 构建，并在 MIMIC-IT（一个多模态指令数据集）上进行了指令微调 (Instruction Tuning)，证明了门控交叉注意力同样适用于指令遵循 (Instruction Following) 任务。

### 衍生模型 (Descendants)

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力技术路线，架构逐渐简化（Idefics2 放弃了重采样器，转而采用带有自适应池化 (Adaptive Pooling) 的直接图像块标记）。
- 从 Flamingo 到 Chameleon 的演进：到 2024 年，许多团队已转向早期融合 (Early-Fusion) 架构（参见第 12.11 课）；但在需要冻结主干网络 (Backbone Freezing) 的生产环境中，Flamingo 风格的门控交叉注意力仍被广泛使用。
- Gemini 的交错输入：在概念上继承了 Flamingo 对交错格式的灵活支持，尽管其具体实现机制是专有的。

### 与 BLIP-2 的对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥接 (Visual Bridge) | 输入端单次 Q-Former | 每 M 层一次门控交叉注意力 |
| 视觉标记 (Visual Tokens) | 每张图像 32 个 | 每个交叉注意力层每张图像 64 个 |
| 冻结大语言模型 | 是 | 是 |
| 上下文内少样本学习 | 较弱 | 强大——论文的核心亮点 |
| 交错输入 | 无原生支持 | 支持，且为设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错网页 |
| 训练参数量 | 1.88 亿 | 约 100 亿（交叉注意力层） |
| 计算资源 | 8 张 A100 训练数天 | 数千张 TPUv4 训练数周 |

若预算有限且仅需处理单图像视觉问答 (Visual Question Answering)，请选择 BLIP-2。若需处理交错输入、少样本学习或多图像推理任务，请选择 Flamingo/Idefics2。

## 使用方法

`code/main.py` 演示了以下内容：

1. 一个 Perceiver 重采样器（Perceiver resampler），作用于 36 个模拟的图像块标记（patch tokens），包含 8 个可学习的潜在向量（learnable latents）（采用纯 Python 实现的交叉注意力（cross-attention））。
2. 一个门控交叉注意力（gated cross-attention）步骤：当 `alpha = 0` 时，输出等于输入（大语言模型（LLM）保持不变）；随后 `alpha = 2.0` 时，视觉信息被混合进来。
3. 一个交错掩码构建器（interleaved-mask builder），用于为 `"(image 1) (text 1) (image 2) (text 2)"` 序列生成二维注意力掩码（2D attention mask）。

## 交付成果

本节将生成 `outputs/skill-gated-bridge-diagnostic.md` 文件。给定一个开源视觉语言模型（VLM）的配置（是否使用重采样器、交叉注意力频率、门控方案），该文件会识别其中的 Flamingo 架构谱系元素，并解释权重冻结策略。这对于调试为何微调（fine-tune）会导致文本性能下降非常有用（答案通常是：门控开启得太快、幅度过大）。

## 练习题

1. 计算 Flamingo-9B 的视觉参数量：9B 参数的大语言模型（LLM） + 1.4B 参数的门控交叉注意力层 + 64M 参数的重采样器。请问可训练参数占总参数的比例是多少？

2. 在 PyTorch 中实现门控残差连接 `y = tanh(alpha) * cross + x`。通过实验证明，在初始化时当 `alpha=0` 时，严格满足 `y==x`。

3. 阅读 OpenFlamingo 论文第 3.2 节（arXiv:2308.01390），了解当批次中每个提示词（prompt）包含不同数量的图像时，他们是如何处理多图像的。请描述其填充（padding）策略。

4. 为什么 Flamingo 的交叉注意力掩码允许文本标记（text token）仅关注*最近的一个*前置图像，而不是所有前置图像？请阅读 Flamingo 论文第 2.4 节并解释其中的权衡（tradeoff）。

5. 上下文少样本学习（In-context few-shot）：为一个新的 Flamingo 变体构建一个包含 4 个“图像 → 主要物体颜色”示例的提示词。描述当示例数量从 0 变化到 8 时，预期的准确率变化趋势。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Perceiver 重采样器（Perceiver resampler） | “固定潜在向量交叉注意力” | 从可变数量的输入图像块中生成 K 个固定标记的模块 |
| 门控交叉注意力（Gated cross-attention） | “Tanh 门控桥接” | 残差层 `y = tanh(alpha)*cross + x`，alpha 可学习，初始化为 0 |
| 交错输入（Interleaved input） | “混合序列” | 图像和文本按阅读顺序自由混合的提示词格式 |
| 冻结大语言模型（Frozen LLM） | “无 LLM 梯度” | 文本大语言模型的权重不更新；仅训练重采样器与交叉注意力层 |
| 少样本学习（Few-shot） | “上下文示例” | 在提示词中提供少量（图像，答案）对；模型无需微调即可泛化 |
| OBELICS | “交错网页语料库” | 包含 1.41 亿个网页的开放数据集，图像与文本按阅读顺序排列 |
| Chinchilla | “700 亿参数冻结基座” | Flamingo 所使用的冻结文本大语言模型，源自 DeepMind 的 Chinchilla 论文 |
| 门控调度（Gate schedule） | “alpha 的变化方式” | 训练过程中交叉注意力门控开启的速率 |
| 交叉注意力频率（Cross-attn frequency） | “每 M 层一次” | 门控交叉注意力模块的插入间隔；Flamingo 采用 M=4 |
| OpenFlamingo | “开源复现版本” | MosaicML/LAION 开源的 3B-9B 参数检查点（checkpoint）；架构与 Flamingo 完全一致 |

## 扩展阅读

- [Alayrac 等人 — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla 等人 — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现版本。
- [Laurençon 等人 — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错式网络语料库 (interleaved web corpus)。
- [Jaegle 等人 — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构 (Perceiver architecture)。
- [Li 等人 — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 经过指令微调 (instruction-tuned) 的 Flamingo 衍生模型。
- [Laurençon 等人 — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — 对 Flamingo 方法的现代化简化版本。
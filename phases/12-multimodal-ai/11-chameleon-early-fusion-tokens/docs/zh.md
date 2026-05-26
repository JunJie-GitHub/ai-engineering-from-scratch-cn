# Chameleon 与早期融合（Early-Fusion）纯词元（Token）多模态模型

> 迄今为止我们见过的所有视觉语言模型（Vision-Language Model, VLM）都将图像和文本分开处理。视觉词元（Token）来自视觉编码器（Vision Encoder），流入投影器（Projector），然后在大型语言模型（Large Language Model, LLM）内部与文本汇合。视觉词表和文本词表从不重叠。Chameleon（Meta，2024 年 5 月）提出了一个问题：如果它们重叠会怎样？训练一个向量量化变分自编码器（Vector Quantized-Variational AutoEncoder, VQ-VAE），将图像转换为来自共享词表的离散词元序列。现在，每个多模态文档都变成了一个序列——文本词元和图像词元交错排列，仅使用单一的自回归损失（Autoregressive Loss）。带来的额外好处是：模型能够生成混合模态输出——在单次推理调用中交替生成文本和图像词元。本课程将研读早期融合（Early-Fusion）的相关论文，并端到端构建一个玩具版本。

**类型：** 构建
**语言：** Python（标准库，VQ-VAE 分词器 + 交错解码器）
**前置条件：** 第 12 阶段 · 05，第 8 阶段（生成式 AI）
**时长：** 约 180 分钟

## 学习目标

- 解释为什么共享词表 + 单一损失会改变模型的能力边界。
- 描述 VQ-VAE 如何将图像分词为离散序列，以适配 Transformer 的下一词元预测目标。
- 列举 Chameleon 用于提升训练稳定性的技巧：QK 归一化（QK-Norm）、Dropout 放置位置、层归一化（LayerNorm）顺序。
- 对比 Chameleon 与 BLIP-2 的 Q-Former 方法，并说明各自适用的场景。

## 核心问题

基于适配器（Adapter）的视觉语言模型（如 LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种截然不同的事物。文本词元经过 `embed(text_token)` 处理；图像则经过 `visual_encoder(image) → projector → ... pseudo_tokens` 处理。该模型拥有两条输入路径，并在中途进行融合。

这带来了三个后果：

1. LLM 只能接收图像输入，无法生成图像。输出仅限于文本。
2. 处理混合模态文档（如文章中段落与图片交替出现）十分笨拙——你要么在模型外部解析多模态输入，要么采用链式生成。
3. 分布不匹配。视觉词元和文本词元位于隐藏空间的不同区域，从而引发微妙的对齐问题。

Chameleon 摒弃了这一前提：图像本质上只是来自共享词表的离散词元序列。在交错文档上训练模型，使用单一损失和单一自回归解码器，即可免费解锁混合模态生成能力。

## 核心概念

### VQ-VAE 作为图像分词器 (Image Tokenizer)

该分词器（Tokenizer）是一个矢量量化变分自编码器（Vector-Quantized Variational Autoencoder, VQ-VAE）。其架构如下：

- 编码器（Encoder）：结合卷积神经网络（CNN）与视觉 Transformer（ViT），将图像映射为空间特征图，例如维度为 256 的 32x32 特征。
- 码本（Codebook）：一个包含 K 个向量的可学习词表（Chameleon 使用 8192 个），维度同样为 256。
- 量化（Quantization）：针对每个空间特征，通过 L2 距离查找最近的码本条目。用整数索引替换连续特征。
- 解码器（Decoder）：一个卷积神经网络，将量化后的特征还原为像素。

训练目标：变分自编码器（VAE）重建损失（Reconstruction Loss） + 承诺损失（Commitment Loss） + 码本损失。码本索引共同构成了图像的离散字母表。

对于 Chameleon 模型：单张图像被转换为 32*32 = 1024 个标记（Token），这些标记来自大小为 8192 的词表。随后与文本标记（来自大语言模型（LLM）的字节对编码（Byte Pair Encoding, BPE）词表，例如 32000 个）进行拼接。最终词表大小为 40192。Transformer 仅处理单一序列并计算单一损失。

### 共享词表 (Shared Vocabulary)

Chameleon 的词表融合了文本标记、图像标记以及模态分隔符（Modality Separators）。每个标记对应唯一的 ID。输入嵌入层（Input Embedding Layer）将每个 ID 映射为 D 维隐藏向量。输出投影层（Output Projection）将隐藏状态映射回词表对数几率（Logits）。Softmax 函数负责选择下一个标记，无论其属于何种模态。

分隔符至关重要：`<image>` 和 `</image>` 标签用于包裹图像标记序列。在生成阶段，若模型输出 `<image>`，下游软件便会知晓接下来的 1024 个标记为 VQ 索引，需将其送入解码器以渲染像素。

### 混合模态生成 (Mixed-Modality Generation)

推理过程即在共享词表中进行下一个标记的预测。示例提示词：“画一只猫并描述它。”Chameleon 的输出如下：

<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...

模型会自主决定输出顺序——它可以先生成图像再生成文本，或先文本后图像，亦或交替进行。整个过程使用相同的解码器与相同的损失函数。

相比之下，基于适配器（Adapter）的视觉语言模型（Vision-Language Model, VLM）仅能生成文本。Chameleon 重新引发了关于模型输出模态的探讨。

### 训练稳定性——QK-Norm、Dropout 与 LayerNorm 顺序

早期融合（Early-Fusion）训练在大规模下往往不稳定。Chameleon 的论文记录了三项关键技巧：

- QK-Norm：在注意力机制（Attention）内部，于点积计算之前对查询（Query）和键（Key）投影应用层归一化（LayerNorm）。这能防止深层网络中对数几率（Logit）幅值爆炸。该技术已被多款 2024 年后的大型模型采用。
- Dropout 放置位置：在每次残差相加（Residual-Add）后应用 Dropout，而不仅仅是在注意力层和多层感知机（MLP）之后。当图像标记的梯度可能占据主导时，需要更强的正则化。
- LayerNorm 顺序：在残差分支上采用前置 LayerNorm（Pre-LN，标准做法），并在最后一个模块的跳跃连接（Skip Connection）上额外增加一个 LN。这有助于稳定最后一层的梯度流。

若缺少这些技巧，340 亿参数的 Chameleon 训练会在多个检查点处发散。引入后，模型得以收敛。该训练方案（Training Recipe）的贡献与架构设计同等重要。

### 分词器的重建上限

VQ-VAE 属于有损压缩。在 8192 个码本条目且每张 512x512 图像使用 1024 个标记的配置下，重建峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）上限约为 26-28 dB。这足以生成可识别的图像，但明显逊于连续空间扩散模型（Diffusion Model，Stable Diffusion 3 可达 32+ dB）。

分词器是性能瓶颈。更优秀的分词器（如 MAGVIT-v2、IBQ、SBER-MoVQGAN）能够突破这一上限。Emu3（第 12.12 课）仅凭更优的分词器便实现了媲美 SDXL 的生成质量。

### Chameleon 与 BLIP-2 / LLaVA 的对比

Chameleon（早期融合，共享词表）：
- 单一损失函数，单一解码器。
- 支持混合模态输出。
- 分词器决定了质量上限。
- 计算成本高：推理路径中每生成一张图像都需调用 VQ-VAE 解码器。

BLIP-2 / LLaVA（晚期融合（Late Fusion），独立塔式结构）：
- 仅支持视觉输入、文本输出。
- 复用预训练的大语言模型。
- 在理解任务上不存在分词器瓶颈。
- 成本低廉：仅需单次前向传播。

根据任务需求进行选择。若需图像生成，选择 Chameleon 系列；若仅需理解能力，基于适配器的 VLM 更为简单，且能复用更多预训练算力。

### Fuyu 与 AnyGPT

Fuyu（Adept, 2023）采用了一种相关思路：完全摒弃独立的视觉编码器，将原始图像块（Image Patches）直接通过 LLM 的输入投影层，如同处理标记一般，无需分词器。该方案比 Chameleon 更简单，但牺牲了基于共享词表的输出生成能力。

AnyGPT（Zhan 等人, 2024）将 Chameleon 扩展至四种模态：文本、图像、语音与音乐。每种模态均采用相同的 VQ-VAE 技巧，并共享 Transformer 架构。支持任意模态到任意模态的生成。相关内容将在第 12.16 课中详细探讨。

## 使用它

`` `code/main.py` `` 构建了一个玩具级的端到端 (end-to-end) 早期融合 (early-fusion) 模型：

- 一个微型 VQ-VAE 风格量化器，将 8x8 图像块映射到码本索引 (codebook indices)（K=16）。
- 一个共享词表 (shared vocabulary)，包含（文本 ID 0..31）+（图像 ID 32..47）+（分隔符 48, 49）。
- 一个玩具级自回归解码器 (autoregressive decoder，基于二元语法表 bigram table)，在合成字幕与图像 token 序列上进行训练。
- 采样循环 (sampling loop)，在给定提示词 (prompt) 的情况下，交替输出文本与图像 token。

代码有意将 Transformer 保持得极小（仅使用二元语法），以便你能端到端地追踪信号流。

## 交付成果

本课时将生成 `` `outputs/skill-tokenizer-vs-adapter-picker.md` ``。根据产品规格（仅理解 vs 理解+生成、所需图像质量、成本预算），该脚本会在 Chameleon 系列（早期融合 early fusion）与 LLaVA 系列（晚期融合 late fusion）之间做出选择，并基于定量的经验法则给出论证依据。

## 练习

1. Chameleon 使用 K=8192 个码本条目，每张 512x512 图像对应 1024 个 token。估算其相对于 24 位 RGB 图像的压缩率。它是有损压缩吗？有损程度如何？
2. 在相同的 VQ-VAE 密度下，一张 4K 图像（3840x2160）会产生多少个图像 token？Chameleon 风格的模型能否在一次推理调用中生成 4K 图像？最先遇到瓶颈的是什么——上下文窗口 (context)、分词器质量 (tokenizer quality)，还是 KV 缓存 (KV cache)？
3. 使用纯 Python 实现 QK 归一化 (QK-Norm)。给定一个 64 维的查询向量 (query) 和键向量 (key)，展示应用层归一化 (LayerNorm) 前后的点积结果。为什么在深层网络中控制向量幅度 (magnitude control) 至关重要？
4. 阅读 Chameleon 论文第 2.3 节关于训练稳定性的内容。描述该论文在 34B 参数规模下未使用 QK-Norm 时观察到的具体失败模式。“范数爆炸 (norm explosion)”的典型特征是什么？
5. 扩展该玩具级解码器，使其在仅文本提示词下能够输出混合模态 (mixed-modality) 响应。在训练数据分布为 60% 文本优先 / 40% 图像优先的情况下，测量模型选择图像优先与文本优先的频率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 早期融合 (Early fusion) | “统一 token” | 图像从第一步起就被转换为离散 token，并与 Transformer 共享同一词表 |
| VQ-VAE | “图像分词器” | 结合 CNN、ViT 与码本的结构，将图像映射为 Transformer 可预测的整数索引 |
| 共享词表 (Shared vocabulary) | “一本字典” | 覆盖文本、图像及模态分隔符的单一 token ID 空间 |
| QK 归一化 (QK-Norm) | “注意力稳定器” | 在查询向量与键向量进行点积前对其应用层归一化，防止范数爆炸 |
| 混合模态生成 (Mixed-modality generation) | “文本+图像输出” | 单次推理过程中自主生成交错排列的文本与图像 token |
| 码本大小 (Codebook size) | “K 个条目” | VQ-VAE 可量化到的离散向量数量；在压缩率与保真度之间进行权衡 |
| 分词器上限 (Tokenizer ceiling) | “重建极限” | 解码 VQ token 所能达到的最佳峰值信噪比 (PSNR)；决定了模型图像质量的上限 |

## 延伸阅读

- [Chameleon 团队 — Chameleon：混合模态早期融合基础模型 (Mixed-Modal Early-Fusion Foundation Models) (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan 等人 — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu 等人 — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan 等人 — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B 博客 (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
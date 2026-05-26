# 视觉自回归建模（Visual Autoregressive Modeling, VAR）：下一尺度预测

> 扩散模型（Diffusion Models）在时间维度上进行迭代采样（去噪步骤）。VAR 则在尺度维度上进行迭代采样——它先预测 1x1 的词元（token），然后是 2x2，接着是 4x4，直至最终分辨率，每个尺度都以前一个尺度为条件。2024 年的论文表明，VAR 在图像生成中符合类似 GPT 的缩放定律（Scaling Laws），并且在相同计算预算下优于 DiT。本课程将构建其核心机制。

**类型：** 构建
**语言：** Python（使用 PyTorch）
**前置知识：** 第 7 阶段第 03 课（多头注意力机制（Multi-Head Attention）），第 8 阶段第 06 课（去噪扩散概率模型（DDPM））
**时长：** 约 90 分钟

## 问题所在

自回归生成（Autoregressive Generation）之所以在语言建模中占据主导地位，是因为其性能缩放具有可预测性：计算量越大、参数越多，困惑度（Perplexity）越低，输出质量越好。在 2024 年之前，图像生成领域主要有两种自回归（Autoregressive, AR）尝试：PixelRNN/PixelCNN（逐像素生成）以及 DALL-E 1 / Parti / MuseGAN（基于 VQ-VAE 编码逐词元生成）。

这两种方法都受限于生成顺序问题。像素和词元虽然排列在二维网格中，但自回归模型必须按照一维光栅顺序（Raster Order）依次访问它们。位于起始角落的像素根本无法预知图像最终会呈现什么样子。其生成质量的缩放效果远不及基于文本的 GPT，且在同等计算量下始终无法达到扩散模型的质量。

VAR 通过改变生成对象来解决这一顺序问题。VAR 不再在空间上逐个预测图像词元，而是以递增的分辨率预测整幅图像。步骤 1：预测 1x1 的词元（图像的宏观“摘要”）。步骤 2：预测 2x2 的词元网格（较粗糙的特征）。步骤 3：预测 4x4 的网格。步骤 K：预测最终的 (H/8)x(W/8) 网格。

每个尺度都会关注所有先前的尺度（按“尺度顺序”进行因果注意力计算），并在自身尺度内并行处理。顺序问题随之消失：尺度 k 的整幅图像只需一次 Transformer 前向传播即可生成。

## 核心概念

### VQ-VAE 多尺度标记器 (Multi-Scale Tokenizer)

VAR 需要一个**多尺度离散标记器 (multi-scale discrete tokenizer)**。对于图像 x，它会生成一系列分辨率逐渐提高的标记网格 (token grids)：

x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)

每个 z_k 使用相同的码本 (codebook)（典型大小为 4096-16384）。每个尺度的标记化过程并非相互独立——其训练目标是使各尺度的残差 (residuals) 之和能够重建 f：

f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)

这是一种**残差向量量化 (residual VQ)** 变体。尺度 k 负责捕捉尺度 1..k-1 所遗漏的信息。解码器 (Decoder) 将所有尺度的嵌入 (embeddings) 求和，并生成图像。

多尺度 VQ 标记器只需训练一次（类似于 VQGAN），随后即被冻结。所有的生成工作均由其上方的自回归模型 (autoregressive model) 完成。

### 下一尺度预测 (Next-Scale Prediction)

生成模型是一个 Transformer，它接收所有先前尺度的标记，并预测下一个尺度的标记。

输入序列结构：
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]

位置嵌入 (Position embeddings) 同时编码尺度索引以及尺度内的空间位置。注意力机制在尺度顺序上是因果的 (causal)：位于尺度 k、位置 (i, j) 的标记可以关注尺度 1..k 的所有标记，以及尺度 k 内部按所用顺序排在其前面的标记（VAR 采用固定的位置注意力，且尺度内无因果性——尺度内的所有位置均并行预测）。

训练损失 (Training loss)：在每个尺度 k，基于所有先前尺度的标记预测标记 z_k。对离散 VQ 码使用交叉熵损失 (Cross-entropy loss)。其结构与 GPT 相同，区别在于“序列”现在具有尺度结构。

### 生成 (Generation)

在推理阶段：
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)

对于 K = 10 个尺度，生成过程需要 10 次 Transformer 前向传播 (forward passes)。每次前向传播并行生成整个尺度的标记——尺度内不存在逐标记的自回归。对于 256x256 的图像，这大约需要 10 次前向传播，而 DiT 需要 28-50 次。

### 为什么下一尺度预测优于下一标记预测

三大结构性优势：
1. **由粗到细符合自然图像统计特性。** 人类视觉感知和图像数据集均表现出依赖于尺度的规律性：低频结构稳定且可预测；高频细节则以低频内容为条件。下一尺度预测充分利用了这一特性。
2. **尺度内并行生成。** 与 GPT 风格的逐标记自回归 (token AR) 不同，VAR 在单步内即可生成某一尺度的所有标记。有效生成长度呈对数级而非线性增长。
3. **无生成顺序偏差。** 尺度 k 的标记能够看到尺度 k-1 的全部内容；不存在“左侧”或“上方”的偏差，从而避免了早期标记在缺乏后期上下文的情况下被迫做出确定性预测 (commit)。

### 缩放定律 (Scaling Law)

Tian 等人证明，VAR 在 ImageNet 上的 FID 指标遵循幂律缩放曲线 (power-law scaling curve)——正如 GPT 在困惑度 (perplexity) 上的表现一样。将参数量或计算量翻倍，误差即可稳定减半。这是首个在图像生成模型中展现出与语言模型同样清晰缩放行为的模型。其结果是，VAR 的规模预测变得可通过计算量进行推算，而不再依赖于针对特定架构的经验猜测。

### 与扩散模型的关系

VAR 与扩散模型 (diffusion) 共享相同的数据压缩逻辑：两者都将生成问题分解为一系列更简单的子问题。

- 扩散模型：逐步添加噪声，学习撤销单步操作。
- VAR：逐步提升分辨率，学习预测下一个尺度。

它们是从不同维度切入该问题的方法。两者均能导出易于处理的条件分布 (tractable conditional distributions)。经验表明，VAR 在推理阶段速度更快（前向传播次数更少，且尺度内完全并行），在类别条件 ImageNet 任务上表现与 DiT 相当或更优。文本条件 VAR（如 VARclip、HART）是当前活跃的研究方向。

## 动手构建

在 `code/main.py` 中，你将：
1. 在合成的“图像”数据（二维高斯环）上构建一个小型的**多尺度 VQ 标记器（multi-scale VQ tokenizer）**。
2. 训练一个 **VAR 风格 Transformer（VAR-style transformer）**，以进行下一尺度的标记预测。
3. 通过调用 Transformer 4 次（对应 4 个尺度）并进行解码来生成样本。
4. 验证按尺度顺序训练是否能使同一尺度内的生成过程实现并行。

这是一个玩具级实现。其核心目的是直观展示尺度结构化注意力掩码（scale-structured attention mask）以及尺度内并行生成（parallel-within-scale generation）的实际运行机制。

## 交付

本课时将生成 `outputs/skill-var-tokenizer-designer.md` 文件——这是一份关于设计多尺度标记器的技能指南，涵盖尺度数量、尺度比例、码本大小（codebook size）、残差共享以及解码器架构等内容。

## 练习

1. **尺度数量消融实验（Scale count ablation）。** 分别使用 4、6、8、10 个尺度训练 VAR 模型。对比重建质量与自回归（autoregressive）推理步数之间的关系。尺度越多 = 残差越精细 = 质量越好，但推理步数也越多。

2. **码本大小（Codebook size）。** 使用 512、4096、16384 的码本大小训练标记器。更大的码本能带来更好的重建效果，但预测难度也会增加。请寻找性能拐点（knee）。

3. **尺度内并行性验证（Parallel-within-scale check）。** 针对训练好的 VAR 模型，显式测量其注意力模式。在尺度 k 内，模型是否只关注跨尺度位置而不关注尺度内位置？请验证掩码（mask）的实现是否正确。

4. **VAR 与 DiT 的扩展性对比（VAR vs DiT scaling）。** 在相同的 ImageNet 类别条件生成任务中，使用匹配的参数量预算（例如 33M、130M、458M）分别训练 VAR 和 DiT（Diffusion Transformer）。绘制 FID（Fréchet Inception Distance）与计算量的关系曲线。VAR 应在每个参数量级上均优于 DiT——请在小规模下复现论文中的结果。

5. **文本条件控制（Text conditioning）。** 扩展 VAR 模型，使其通过 adaLN（自适应层归一化）接收文本嵌入（CLIP pooled）作为额外的条件输入。这是 HART 方案的核心做法。请评估在文本对齐采样下，FID 指标能提升多少。

## 关键术语

| 术语 | 常见表述 | 实际含义 |
|------|----------------|----------------------|
| 视觉自回归 (Visual AutoRegressive) | “视觉自回归” | 基于 VQ 词元网格金字塔的下一尺度预测进行图像生成 |
| 下一尺度预测 (Next-scale prediction) | “先预测粗粒度，再预测细粒度” | 模型在分辨率递增的尺度上预测词元，并以所有先前尺度为条件 |
| 多尺度 VQ 分词器 (Multi-scale VQ tokenizer) | “残差 VQ” | 一种 VQ-VAE，可生成 K 个分辨率递增的词元网格，其解码器会对所有尺度的特征进行求和 |
| 尺度 k (Scale k) | “金字塔第 k 层” | K 个分辨率层级之一，范围从 k=1 时的 1x1 到 k=K 时的 (H/p)x(W/p) |
| 尺度内并行 (Parallel-within-scale) | “每个尺度一次前向传播” | 尺度 k 的所有词元在一次 Transformer 前向传播中完成预测，而非自回归方式 |
| 跨尺度因果 (Causal-across-scales) | “按尺度排序的注意力机制” | 尺度 k 的词元可以关注尺度 1..k 的所有内容，但无法关注尺度 k+1..K |
| 残差 VQ (Residual VQ) | “加性分词” | 每个尺度的词元编码了较低尺度留下的残差；解码器对所有尺度的嵌入向量进行求和 |
| VAR 缩放定律 (VAR scaling law) | “图像 GPT 缩放” | FID 指标随计算量呈现可预测的幂律变化，类似于语言模型的困惑度 |
| HART (Hybrid Autoregressive Transformer) | “混合 VAR + 文本” | 一种文本条件 VAR 变体，结合了 MaskGIT 风格的迭代解码与 VAR 的尺度结构 |
| 尺度位置嵌入 (Scale position embedding) | “(尺度, 行, 列) 三元组” | 位置编码同时包含尺度索引以及该尺度内的空间坐标 |

## 进一步阅读

- [Tian 等人, 2024 — 《视觉自回归建模：通过下一尺度预测实现可扩展的图像生成》](https://arxiv.org/abs/2404.02905) — VAR 原始论文，权威参考文献
- [Peebles 和 Xie, 2022 — 《基于 Transformer 的可扩展扩散模型》](https://arxiv.org/abs/2212.09748) — DiT，扩散模型的对比基线
- [Esser 等人, 2021 — 《驾驭 Transformer 实现高分辨率图像合成》](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度分词器所扩展的分词器系列
- [van den Oord 等人, 2017 — 《神经离散表示学习》](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像分词技术的基础
- [Tang 等人, 2024 — 《HART：基于混合自回归 Transformer 的高效视觉生成》](https://arxiv.org/abs/2410.10812) — 文本条件 VAR
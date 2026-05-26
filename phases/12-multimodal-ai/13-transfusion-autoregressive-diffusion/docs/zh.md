# Transfusion：单一 Transformer 中的自回归文本与扩散图像

> Chameleon 和 Emu3 将全部筹码押在离散词元（discrete tokens）上。它们确实有效，但量化瓶颈（quantization bottleneck）显而易见——图像质量在连续空间扩散模型（continuous-space diffusion models）之下便触及天花板。Transfusion（Meta，Zhou 等人，2024 年 8 月）则反其道而行之：保持图像的连续性，彻底抛弃 VQ-VAE，并使用两个损失函数训练同一个 Transformer。文本词元采用下一词元预测（next-token prediction）。图像块（image patches）采用流匹配（flow matching）/ 扩散损失。两个目标共同优化同一组权重。Stable Diffusion 3 底层架构（MMDiT）与其可谓近亲。本教程将解读 Transfusion 论文，构建一个玩具级的双损失训练器，并剖析让单一 Transformer 兼顾两项任务的注意力掩码（attention mask）。

**Type:** 构建
**Languages:** Python（标准库，基于 MNIST 规模的玩具级双损失训练器）
**Prerequisites:** 第 12 阶段 · 第 11 阶段（Chameleon），第 8 阶段（生成式 AI）
**Time:** 约 180 分钟

## 学习目标

- 在单一主干网络（backbone）上搭建一个同时运行两个损失函数（文本词元的下一词元预测 NTP，图像块的扩散均方误差 MSE）的 Transformer。
- 解释为何在图像块上使用双向注意力（bidirectional attention），而在文本词元上使用因果注意力（causal attention）是正确的掩码选择。
- 从算力消耗、生成质量和代码复杂度三个维度，对比 Transfusion 风格（连续图像、扩散损失）与 Chameleon 风格（离散图像、NTP）。
- 阐明 MMDiT 的核心贡献：每个模块具备模态专属权重（modality-specific weights），并在残差流（residual stream）中进行联合注意力计算。

## 问题背景

关于图像词元应采用离散还是连续表示的争论，其历史早于大语言模型（LLMs）。连续表示（原始像素、VAE 潜变量）能够保留更多细节。离散词元（VQ 索引）虽然契合 Transformer 的原生词表，但会在量化步骤中丢失细节。

Chameleon / Emu3 选择了离散路线：单一损失函数、单一架构，但图像保真度受限于分词器（tokenizer）的质量。

扩散模型（diffusion models）则走向连续：图像质量卓越，但需要独立于 LLM 的单独模型，涉及复杂的噪声调度（noise schedule）工程，且无法与文本生成无缝集成。

Transfusion 提出了一个问题：能否两者兼得？保持图像的连续性，依然只训练一个模型，并将两个损失函数缝合到同一次梯度更新步骤（gradient step）中。

## 核心概念

### 双损失架构 (Two-Loss Architecture)

一个仅解码器 Transformer (decoder-only Transformer) 处理包含以下内容的序列：

- 文本词元 (text tokens)（离散型，来自 BPE 词表）。
- 图像块 (image patches)（连续型，16x16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器 (Vision Transformer encoder) 的输入方式相同）。
- `<image>` 和 `</image>` 标签，用于标记连续图像块的位置。

前向传播 (forward pass) 仅执行一次。损失函数根据每个词元选择两个输出头之一：

- 对于文本词元：在词表 logits 输出头上使用标准的交叉熵 (cross-entropy)。
- 对于图像块：在连续图像块上使用扩散损失 (diffusion loss)——预测添加到每个块中的噪声。

梯度流经共享的 Transformer 主干网络。两种损失同时优化共享权重。

### 注意力掩码：因果文本 + 双向图像

文本词元必须是因果的 (causal)——不能让文本词元关注未来的文本，否则教师强制 (teacher forcing) 机制会失效。然而，图像块代表单一快照；它们应在同一图像块内相互进行双向 (bidirectional) 注意力。

掩码 (mask) 定义如下：

M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text

在训练和推理阶段，该掩码以块三角掩码 (block-triangular mask) 的形式实现。

### Transformer 内部的扩散损失

扩散损失是标准的：向图像块添加噪声，要求模型预测该噪声（或等效地预测干净的图像块）。Transfusion 的版本使用流匹配 (flow matching)——预测从含噪状态到干净状态的速度场 (velocity field)。

训练期间：
1. 对于每个图像块 x0，采样一个随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（用于流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列中的文本下一词元预测 (Next Token Prediction, NTP) 损失一起进行反向传播。

在推理阶段，生成过程为：
- 文本词元：标准的自回归采样 (autoregressive sampling)。
- 图像块：扩散采样循环（通常为 10-30 步），以先前的文本词元为条件。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等人，2024 年 3 月）与 Transfusion 大致同期发布了 MMDiT（多模态扩散 Transformer，Multimodal Diffusion Transformer）。这两种架构属于同源分支。

MMDiT 的主要区别在于：

- 每个模块使用模态特定权重 (modality-specific weights)。每个 Transformer 模块为文本词元和图像块分别设置独立的 Q、K、V 和 MLP 权重。注意力机制是联合的（跨模态）；其余部分均为模态特定。
- 整流流训练 (rectified flow training)。一种特定的流匹配变体，具有已知的采样路径，且数学计算比 DDPM (Denoising Diffusion Probabilistic Models) 更简单。
- 规模。MMDiT 是 SD3 的主干网络（包含 20 亿和 80 亿参数变体）。Transfusion 的论文扩展至 70 亿参数。

两者最终收敛于同一核心理念：由一个 Transformer 对文本执行 NTP，并对连续图像表示执行扩散过程。

### 为何优于 Chameleon 风格架构

在图像生成任务中，连续扩散与离散 NTP 之间的质量差距是可量化的。Transfusion 论文报告称：

- 在 70 亿参数规模下，其 FID (Fréchet Inception Distance) 得分比同等规模的 Chameleon 风格模型高出 3-5 分。
- 无需训练图像分词器 (tokenizer)——图像编码器更简单（线性投影至隐藏层，与 ViT 的输入层相同）。
- 推理时可并行化图像块去噪，这与自回归图像词元不同。

缺点：Transfusion 是双损失模型，导致训练动态 (training dynamics) 更复杂。需要调整损失权重。NTP 与扩散之间的调度不匹配可能导致其中一个输出头占据主导。

### 下游演进

Janus-Pro（第 12.15 课）通过解耦用于理解和生成的视觉编码器来优化 Transfusion 的理念——一个使用 SigLIP，另一个使用 VQ (Vector Quantization)，同时共享 Transformer 主干。Show-o（第 12.14 课）将扩散替换为离散扩散 (discrete-diffusion，即掩码预测)。在 Transfusion 之后，统一生成 (unified-generation) 家族迅速分化出多个分支。

2026 年投入生产的具备图像生成能力的视觉语言模型 (Vision-Language Models, VLMs)——如 Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定采用了该家族的某种衍生架构。具体细节属于商业机密。

## 实际应用

`code/main.py` 在一个微型类 MNIST 问题上构建了一个玩具版 Transfusion (跨模态融合模型)：

- 文本描述是用于描述数字（0-9）的短整数序列。
- 图像是 4x4 的字节网格。
- 一对共享权重的线性投影充当 Transformer 的替代组件；文本部分使用下一词元预测 (Next Token Prediction, NTP) 损失，噪声图像块部分使用均方误差 (Mean Squared Error, MSE) 损失。
- 训练循环交替优化这两种损失，注意力掩码 (Attention Mask) 是显式构建的。
- 生成过程通过一次前向传播 (Forward Pass) 同时输出文本描述和 4x4 图像。

这里的 Transformer 仅是玩具模型。真正的核心产出在于双损失调度管线、注意力掩码构建以及推理循环。

## 交付上线

本课时将生成 `outputs/skill-two-loss-trainer-designer.md`。针对新的多模态训练任务（文本+图像、文本+音频、文本+视频），它会设计双损失调度方案（损失权重、掩码形状、共享模块与模态专属模块的划分），并标记实现过程中的潜在风险。

## 练习

1. 一个采用 Transfusion 风格的模型训练 70% 的文本词元 (Token) 和 30% 的图像块 (Patch)。图像扩散损失 (Diffusion Loss) 的数值量级约为文本 NTP 损失的 10 倍。应设置怎样的损失权重来平衡两者？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块三角掩码 (Block-Triangular Mask)。将每个位置标记为 0 或 1。

3. MMDiT (Multimodal Diffusion Transformer, 多模态扩散 Transformer) 拥有模态专属的 QKV (Query, Key, Value) 权重。与 Transfusion 的完全共享 Transformer 相比，这会增加多少参数量开销？在 70 亿 (7B) 参数规模下，这种设计是否值得？

4. 生成过程：给定文本提示词 (Prompt)，模型先运行 NTP 生成 50 个词元，随后遇到 `<image>` 标记，接着在 20 个去噪步骤中对 256 个图像块运行扩散过程。总共需要多少次前向传播？

5. 阅读 SD3 (Stable Diffusion 3) 论文的第 3 节。描述整流流 (Rectified Flow) 的原理，并解释为何它比 DDPM (Denoising Diffusion Probabilistic Models, 去噪扩散概率模型) 需要更少的推理步骤即可收敛。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 双损失训练 (Two-loss training) | “NTP + 扩散” | 单个 Transformer 在同一梯度更新步骤中，同时优化文本词元的交叉熵损失与连续图像块的均方误差损失 |
| 流匹配 (Flow matching) | “整流流” | 扩散模型的一种变体，用于预测从噪声到干净数据的速度场；其数学推导比 DDPM 更简洁 |
| MMDiT | “多模态 DiT” | Stable Diffusion 3 的架构：采用联合注意力机制，以及模态专属的多层感知机 (MLP) 和归一化层 |
| 块三角掩码 (Block-triangular mask) | “因果文本 + 双向图像” | 一种注意力掩码，在文本部分保持因果性（单向），而在图像区域内允许双向交互 |
| 连续图像表示 (Continuous image representation) | “无向量量化 (VQ)” | 将图像块表示为实数值向量，而非离散的整数码本索引 |
| 速度预测 (Velocity prediction) | “v-参数化” | 网络输出的是噪声与数据之间的速度场，而非噪声本身 |

## 延伸阅读

- [Zhou 等人 — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser 等人 — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles 与 Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao 等人 — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
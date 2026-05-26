# Janus-Pro：用于统一多模态模型的解耦编码器

> 统一多模态模型（Unified Multimodal Models）面临着一个不可避免的内在张力。理解任务需要语义特征（Semantic Features）——即 SigLIP 或 DINOv2 输出的富含概念级信息的向量。生成任务则偏好利于重建的编码（Reconstruction-friendly Codes）——即能够重新组合成清晰像素的 VQ 令牌（VQ Tokens）。单一编码器无法同时兼容这两个目标。Janus（DeepSeek，2024 年 10 月）与 Janus-Pro（DeepSeek，2025 年 1 月）提出了解决方案：停止强行统一，将两个编码器解耦（Decouple）。在任务间共享 Transformer 主体（Transformer Body），但让理解任务通过 SigLIP 路由，生成任务通过 VQ 分词器（VQ Tokenizer）路由。在 7B 参数规模下，Janus-Pro 在 GenEval 基准上超越了 DALL-E 3，同时在 MMMU 基准上与 LLaVA 持平。本节将深入解析为何双编码器架构能在单一编码器失效的地方取得成功。

**Type:** 构建
**Languages:** Python（标准库，双编码器路由 + 共享主体信号）
**Prerequisites:** 第 12 阶段 · 13（Transfusion），第 12 阶段 · 14（Show-o）
**Time:** 约 120 分钟

## 学习目标

- 解释为何单一共享编码器会损害理解或生成任务的质量。
- 描述 Janus-Pro 的路由机制：输入端使用 SigLIP 特征进行理解，输入与输出端均使用 VQ 令牌进行生成。
- 梳理数据混合扩展（Data-mix Scaling）策略，阐明 Janus-Pro 为何能取得 Janus 未能实现的成功。
- 对比解耦架构（Janus-Pro）、连续耦合架构（Coupled-Continuous，如 Transfusion）与离散耦合架构（Coupled-Discrete，如 Show-o）。

## 核心问题

统一模型在理解与生成任务间共享同一个 Transformer 主体。此前的尝试（如 Chameleon、Show-o、Transfusion）均在双向任务中使用同一个视觉分词器（Visual Tokenizer）。这种设计本质上是一种妥协：

- 偏向重建优化（生成任务）：VQ-VAE 能够捕捉细粒度的像素细节，但生成的令牌语义连贯性较弱。
- 偏向语义优化（理解任务）：SigLIP 嵌入（SigLIP Embeddings）能将“猫”的图像与“猫”的文本标记聚集在相近的向量空间，但无法支持高质量的图像重建。

Show-o 和 Transfusion 为此付出了代价，在某一方向上出现了肉眼可见的质量损失。Janus-Pro 则提出质疑：既然任务需求不同，为何非要强求使用同一个分词器？

## 核心概念

### 解耦视觉编码 (Decoupled Visual Encoding)

Janus-Pro 的架构将两个编码器分离开来：

- 理解路径 (Understanding Path)。输入图像 → SigLIP-SO400m → 2层 MLP → Transformer 主体 (Transformer Body)。
- 生成路径 (Generation Path)。输入图像（若以现有图像为条件） → VQ 分词器 (VQ Tokenizer) → Token ID → Transformer 主体。
- 输出生成。Transformer 预测的图像 Token → VQ 解码器 (VQ Decoder) → 像素。

Transformer 主体是共享的。主体上游和下游的所有组件均针对特定任务设计。

输入通过提示词格式 (Prompt Format) 进行消歧：`<understand>` 标签将路由至 SigLIP；`<generate>` 标签路由至 VQ。或者，路由逻辑由任务隐式决定。

### 为何有效

理解损失 (Understanding Loss) 获取的是 SigLIP 特征，这些特征经过 CLIP 风格预训练 (CLIP-style Pretraining)，已针对语义相似性进行了优化。由于输入特征更契合任务需求，该模型在感知基准测试上的表现优于 Show-o / Transfusion。

生成损失 (Generation Loss) 获取的是 VQ Token，分词器已针对图像重建对其进行了优化。由于 VQ 编码能够干净地重组回像素，其图像生成质量优于 Show-o。

共享的 Transformer 主体会接触两种输入分布 (Input Distributions)（SigLIP 和 VQ），并学习同时处理它们。核心观点是：只要数据量足够大、参数量足够多，主体就能自然吸收并适应这种切换。

### 数据扩展 —— Janus 与 Janus-Pro 对比

Janus（原始版本，arXiv 2410.13848）首次引入了这种解耦设计，但规模较小（13 亿参数，数据有限）。Janus-Pro（arXiv 2501.17811）则进行了大规模扩展：

- 70 亿参数（对比 13 亿）。
- 第一阶段（对齐阶段）使用 9000 万图文对，较之前的 7200 万有所增加。
- 第二阶段（统一阶段）使用 7200 万数据，较之前的 2600 万大幅提升。
- 第三阶段新增了 20 万图像生成指令样本。

最终成果：Janus-Pro-7B 在 MMMU 基准上与 LLaVA 持平（60.3 对比约 58），并在 GenEval 上超越了 DALL-E 3（0.80 对比 0.67）。作为一个开源模型，它在统一架构的两大能力维度上均具备强劲竞争力。

### JanusFlow —— 整流流 (Rectified Flow) 变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为整流流生成路径（连续型）。架构分工变为：SigLIP 负责理解 + 整流流负责生成。这进一步提升了质量上限。整体架构依然保持“编码器解耦、主体共享”的设计。

### 共享主体的职责

Transformer 主体处理统一的序列，但需应对两种不同的输入分布。其核心职责包括：

- 理解任务：接收 SigLIP 特征 + 文本 Token → 自回归 (Autoregressively) 生成文本。
- 生成任务：接收文本 Token +（可选的图像 VQ Token）→ 自回归生成图像 VQ Token。

主体内部的每个模块均不包含模态特定权重 (Modality-specific Weights)。它本质上就是你在 Qwen 或 Llama 中常见的那种纯文本 Transformer，外加两个输入适配器 (Input Adapters)。

有趣的是，这意味着 Janus-Pro 的主体可以直接从预训练大语言模型 (Pretrained LLM) 初始化。Janus-Pro 确实采用了 DeepSeek-MoE-7B 进行初始化。这一选择至关重要：大语言模型赋予了模型强大的推理能力，这是纯从头训练的统一模型难以企及的。

### 与 InternVL-U 对比

InternVL-U（Lesson 12.10）是 2026 年的后续演进版本。它融合了以下特性：

- 原生多模态预训练 (Native Multimodal Pretraining)（基于 InternVL3 骨干网络）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头 (Diffusion Heads) 输出）。
- 统一的理解、生成与编辑能力。

InternVL-U 将 Janus-Pro 的架构选择纳入了一个更庞大的框架中。如今，“解耦编码器”理念已成为大规模统一模型的默认标准。

### 局限性

解耦编码器增加了架构的复杂性。需要训练两个分词器、维护两条输入路径，并应对两套不同的故障模式 (Fail Modes)。对于不需要生成能力的产品，Janus-Pro 显得过度设计——直接选择 LLaVA 系列的理解模型即可。

对于不需要理解能力的产品，Janus-Pro 则显得大材小用——选择 Stable Diffusion 3 或 Flux 模型更为合适。

而对于同时需要理解与生成能力的产品，Janus-Pro 现已成为参考级的开源架构。

## 使用它

`code/main.py` 模拟了 Janus-Pro 的路由机制：

- 两个模拟编码器（mock encoders）：类 SigLIP 编码器（生成 256 维语义向量）和类 VQ（向量量化，Vector Quantization）编码器（生成整数编码）。
- 一个提示词路由器（prompt router），根据任务标签选择对应的编码器。
- 一个共享主干网络（shared body，占位实现），无论 token 序列由哪个编码器生成，均对其进行统一处理。
- 从阶段 1（对齐，alignment）切换到阶段 3（指令微调，instruction tuning）的加权采样调度策略（weighted-sample schedule）。

打印 3 个示例的路由路径：图像问答（image QA）、文本生成图像（Text-to-Image, T2I）、图像编辑。

## 交付它

本节将生成 `outputs/skill-decoupled-encoder-picker.md`。针对一款追求前沿级质量、且需要统一生成与理解能力的产品，该脚本会推荐 Janus-Pro、JanusFlow 或 InternVL-U，并提供具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 基准上超越了 DALL-E 3。请解释为何一个 7B 参数的开源模型能在生成任务上匹敌前沿闭源模型，却在理解任务上有所不及。

2. 实现一个路由函数：输入提示词文本，将其分类为 `understand` 或 `generate`。你将如何处理诸如“先描述再绘制”这类模糊提示词？

3. JanusFlow 使用整流流（rectified flow）替换了 VQ 路径。此时 Transformer 主干网络输出的是什么？损失函数（loss）发生了哪些变化？

4. 提出 Janus-Pro 架构可通过增加一个解耦编码器（decoupled encoder）来处理的第四项任务。示例：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 论文第 4.2 节关于数据缩放（data scaling）的内容。相较于 Janus，哪个数据阶段对 T2I 质量提升的贡献最大？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 解耦编码 (Decoupled encoding) | “两个视觉编码器” | 为不同任务方向配备独立的分词器或编码器：理解方向使用语义特征，生成方向使用重建特征 |
| 共享主干网络 (Shared body) | “一个 Transformer” | 单个 Transformer 统一处理来自任一编码器的输出；不包含特定模态的专属权重 |
| 用于理解的 SigLIP | “语义特征” | CLIP 家族的视觉塔（vision tower），提供丰富的概念特征，但图像重建能力较弱 |
| 用于生成的 VQ | “重建编码” | 向量量化（Vector-Quantized）的 token，能够清晰、准确地解码回像素 |
| JanusFlow | “整流流变体” | 采用连续流匹配（flow-matching）生成头替代 VQ 的 Janus-Pro 架构 |
| 路由标签 (Routing tag) | “任务标签” | 提示词标记（`<understand>` / `<generate>`），用于动态选择输入编码器 |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
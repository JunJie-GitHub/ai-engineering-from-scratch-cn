# 从 CLIP 到 BLIP-2 —— Q-Former 作为模态桥梁 (Modality Bridge)

> CLIP 能够对齐图像与文本，但无法生成图像描述、回答问题或进行对话。BLIP-2（Salesforce，2023）通过一个小型可训练桥梁解决了这一问题：32 个可学习查询向量 (learnable query vectors) 通过交叉注意力机制 (cross-attention) 关注冻结 (frozen) 的视觉变换器 (ViT) 特征，随后直接接入冻结的大语言模型 (LLM) 的输入流中。该桥梁仅含 1.88 亿参数，便将一个 110 亿参数的 LLM 与 ViT-g/14 连接起来。截至 2026 年，所有基于适配器的视觉语言模型 (VLM) ——如 MiniGPT-4、InstructBLIP 以及 LLaVA 的同类模型——皆是其后继者。本课程将解析 Q-Former 的架构，阐述其两阶段训练过程，并构建一个简易版本，将视觉词元 (visual tokens) 输入至冻结的文本解码器中。

**类型：** 构建实践
**语言：** Python（标准库，交叉注意力与可学习查询演示）
**前置知识：** 第 12 阶段 · 02（CLIP），第 7 阶段（Transformers）
**时长：** 约 180 分钟

## 学习目标

- 解释为何在冻结的视觉编码器与冻结的 LLM 之间使用可训练瓶颈 (trainable bottleneck)，在成本与稳定性上优于端到端微调 (end-to-end finetuning)。
- 实现一个交叉注意力模块 (cross-attention block)，使一组固定的可学习查询向量关注外部图像特征。
- 逐步解析 BLIP-2 的两阶段预训练：表征学习阶段 (representation)（ITC + ITM + ITG）与生成阶段（使用冻结解码器的语言模型损失 (LM loss)）。
- 将 Q-Former 与 LLaVA 中使用的更简单的多层感知机投影器 (MLP projector) 进行对比，并论证各自适用的场景。

## 核心问题

假设你拥有一个冻结的 ViT，每张图像会输出 256 个维度为 1408 的图像块词元 (patch tokens)。同时你还有一个冻结的 70 亿参数 LLM，其期望的词元嵌入 (token embeddings) 维度为 4096。最直接的桥梁方案——一个从 1408 映射到 4096 的线性层 (linear layer)——确实可行，但将全部 256 个图像块词元输入 LLM 的上下文 (context)，意味着每张图像会额外消耗 256 个词元。在一个包含 32 张图像的批次中，仅视觉模态 (visual modality) 就会占用 8192 个词元。

BLIP-2 提出的核心问题是：能否将 256 个词元的图像表征 (image representation) 压缩为少得多的词元（例如 32 个），同时保留足够的信息，使 LLM 能够生成描述、回答问题并对图像进行推理？此外，能否在不改动冻结骨干网络 (backbones) 的前提下训练该桥梁，从而将训练成本严格限制在桥梁自身的参数范围内？

答案是：Q-Former。它使用 32 个可学习的“查询”向量，通过交叉注意力机制关注 ViT 的图像块词元，生成一个包含 32 个词元的视觉摘要 (visual summary) 供 LLM 消费。总参数量仅为 1.88 亿。在接入 LLM 之前，该桥梁已通过对比、匹配与生成目标 (contrastive, matching, and generative objectives) 进行预训练。

## 核心概念

### 可学习查询向量 (Learnable Queries)

Q-Former 的核心技巧在于：不再让大语言模型（Large Language Model, LLM）的文本词元直接关注图像块（image patches），而是引入一组全新的 32 个可学习查询向量 `Q`，并让*它们*去关注图像块。这些查询向量是模型的参数——它们在训练过程中被学习得到，并且每张图像都使用相同的 32 个查询向量。

经过交叉注意力（cross-attention）机制后，每个查询向量都保存了图像的压缩摘要——例如“描述主要物体”、“描述背景”、“统计物体数量”等。这些查询向量并不会真正针对特定的语义标签进行特化；它们会学习任何能够降低下游任务损失的编码方式。

### 架构 (Architecture)

Q-Former 是一个小型 Transformer（12 层，约 1 亿参数），包含两条路径：

1. 查询路径：32 个查询向量首先经过自注意力（self-attention，在它们自身之间进行），然后对冻结的视觉 Transformer（Vision Transformer, ViT）的图像块词元进行交叉注意力计算，最后通过前馈神经网络（Feed-Forward Network, FFN）。
2. 文本路径：一个类似 BERT 的文本编码器与查询路径共享自注意力和 FFN 的权重。文本路径中禁用了交叉注意力。

在训练阶段，两条路径同时运行。查询向量与文本通过共享的自注意力机制进行交互，这意味着对于需要文本条件的任务（如图像-文本匹配 ITM、图像引导文本生成 ITG），查询向量可以以文本为条件。在视觉语言模型（Vision-Language Model, VLM）交接的推理阶段，只有查询向量流经网络，最终输出 32 个视觉词元。

### 两阶段训练 (Two-Stage Training)

BLIP-2 的预训练分为两个阶段：

第一阶段：表征学习（不涉及 LLM）。包含三种损失函数：
- ITC（图像-文本对比学习，Image-Text Contrastive）：在池化后的查询词元与文本 CLS 词元（CLS token）之间进行类似 CLIP 的对比学习。
- ITM（图像-文本匹配，Image-Text Matching）：二分类器——判断该图像-文本对是否匹配？采用困难负样本挖掘（hard-negative mining）。
- ITG（图像引导文本生成，Image-Grounded Text Generation）：在文本上使用因果语言模型头（causal LM head），并以查询向量为条件。这迫使查询向量编码出可生成文本的内容。

此阶段仅训练 Q-Former。ViT 保持冻结。不涉及 LLM。

第二阶段：生成式学习。接入一个冻结的 LLM（如 OPT-2.7B 或 Flan-T5-XL 等）。通过一个小型线性层将 32 个查询输出投影到 LLM 的嵌入维度。将它们拼接到文本提示词（prompt）的前面。仅在线性投影层和 Q-Former 上，针对拼接后的“提示词 + 图像 + 描述”序列计算语言模型损失并进行训练。

第二阶段结束后，Q-Former 加上投影层就构成了完整的视觉适配器（visual adapter）。推理流程为：图像 → ViT → Q-Former → 线性投影 → 拼接到文本前 → 冻结的 LLM 生成输出。

### 参数量与成本效益 (Parameter Economics)

采用 ViT-g/14（11 亿参数，冻结）+ OPT-6.7B（67 亿参数，冻结）+ Q-Former（1.88 亿参数，可训练）的 BLIP-2，总参数量为 80 亿，其中仅 1.88 亿参与训练。Q-Former 本身仅占全栈参数量的约 2.4%。训练成本也反映了这一点：只需在少量 A100 显卡上训练数天，而端到端训练则需要数周。

质量表现：BLIP-2 在零样本视觉问答（Zero-Shot Visual Question Answering, VQA）任务上达到或超越了 Flamingo-80B，而模型体积仅为后者的 1/50。这座“桥梁”确实行之有效。

### InstructBLIP 与指令感知型 Q-Former

InstructBLIP（2023）为 Q-Former 增加了一个额外输入：指令文本本身。在进行交叉注意力计算时，查询向量现在可以同时访问图像块和指令文本。查询向量能够针对每条指令进行特化（例如“统计汽车数量”、“描述画面情绪”），而不是学习单一的固定摘要。这在保留测试集（held-out tasks）的基准测试中带来了性能提升。

### MiniGPT-4 与仅投影器方案 (Projector-Only Approach)

MiniGPT-4 保留了 Q-Former，但仅训练输出端的线性投影层，其余部分全部冻结。这种方法成本较低，但代价是质量受限——查询向量直接沿用了 BLIP-2 的预训练结果，而非针对自身任务优化。它适合快速迭代，但并非最优架构。

### LLaVA 为何选择更简单的方案

LLaVA（2023，第 12.05 课）用一个简单的 2 层多层感知机（Multi-Layer Perceptron, MLP）取代了 Q-Former，将 ViT 的每个图像块词元直接投影到 LLM 空间——对于 24x24 的网格，每张图像产生 576 个词元，全部输入给 LLM。这种方案的压缩率较低，但允许 LLM 直接关注原始图像块。当时这一做法颇具争议；但到了 2023 年底，它已成为主流，因为视觉指令数据（LLaVA-Instruct-150k）证明 MLP 经过训练后足以保留足够的信号。其权衡在于：LLaVA 的上下文窗口消耗更快，但它能自然地扩展到多图像和视频任务。

到 2026 年，该领域出现分化：在词元预算（token budget）受限的场景（如长视频、多图像）中，Q-Former 依然存活；而在追求单词元原始质量的场景中，MLP 投影器占据主导地位。

### 门控交叉注意力：先驱 Flamingo

Flamingo（第 12.04 课）早于 BLIP-2 提出，它采用了相同的交叉注意力思想，但将其应用于冻结 LLM 的每一层，而非仅作为单一的桥梁。BLIP-2 证明了仅压缩到输入层依然有效。Gemini 和 Idefics 则结合了两者的优点：交错输入的词元，加上可选的门控交叉注意力（gated cross-attention），用于上下文少样本学习（in-context few-shot）。

### 2026 年的衍生架构

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4 以及大多数视频-语言模型（出于词元预算的考虑）。
- Perceiver 重采样器（Perceiver resampler）：Flamingo 的变体（第 12.04 课）；Idefics 系列、Eagle、OmniMAE。
- MLP 投影器：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- 注意力池化（Attention pool）：VILA、PaliGemma。

这四种方案均行之有效。决定性因素在于：你的瓶颈是词元预算，还是单词元质量。

## 使用它

`code/main.py` 构建了一个标准库（stdlib）风格的 Q-Former 式交叉注意力（cross-attention）模块：

1. 模拟 256 个图像块 token（维度为 128）。
2. 实例化 32 个可学习查询向量（queries，维度为 128）。
3. 执行缩放点积交叉注意力（scaled-dot-product cross-attention）计算（Q 来自查询向量，K/V 来自图像块）。
4. 通过线性层（linear layer）将其投影至大语言模型（LLM）维度（512）。
5. 输出 32 个可供 LLM 直接使用的视觉 token。

所有数学运算均使用纯 Python 实现（基于向量的嵌套循环）。虽为示例代码，但张量形状（shape）完全正确。程序会打印注意力权重矩阵（attention-weight matrix），以便你观察每个查询向量具体关注了哪些图像块。

## 交付成果

本课时将生成 `outputs/skill-modality-bridge-picker.md` 文件。给定目标视觉语言模型（VLM）配置（视觉编码器 token 数量、LLM 上下文预算、部署约束条件、质量目标），该文件会推荐采用 Q-Former、多层感知机（MLP）还是 Perceiver 重采样器（Perceiver resampler），并为每种桥接模块（bridge）提供简短的理由及参数量估算。

## 练习

1. 在 PyTorch 中实现该交叉注意力（cross-attention）模块。验证当使用 32 个查询向量和 256 个键/值（keys/values）时，注意力权重矩阵的形状为 32 x 256，且经过 softmax 操作后每一行的和均为 1。

2. 在 BLIP-2 的第一阶段，Q-Former 会同时计算三种损失函数：图像-文本对比损失（ITC）、图像-文本匹配损失（ITM）和图像-文本生成损失（ITG）。请用伪代码写出各自的 `forward` 函数签名。其中哪一种需要激活文本编码器（text encoder）路径？

3. 对比参数量：Q-Former（12 层，隐藏层维度 768）与 2 层 MLP 投影器（1408 → 4096，共两层）。当 LLM 规模达到多大时，Q-Former 高达 1.88 亿（188M）的参数量成本能在训练效率上得到回报？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 初始化的内容。解释为何使用 BERT-base 进行初始化（而非随机初始化）能够加速模型收敛。

5. 对于一段 10 分钟、1 FPS 采样率（共 60 帧）的视频，分别计算使用 Q-Former（每帧 32 个 token）与 MLP 投影器（每帧 576 个 token）时的单帧 token 开销。哪种方案能够适配 128k token 的 LLM 上下文窗口（context window）？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Q-Former | “查询 Transformer” | 包含 32 个可学习查询向量的小型 Transformer，通过交叉注意力（cross-attention）机制与冻结的视觉 Transformer（ViT）特征进行交互 |
| 可学习查询向量（Learnable queries） | “视觉软提示（soft prompt）” | 一组固定参数，充当交叉注意力中的查询端；针对每个模型独立学习，并在所有输入间共享 |
| 交叉注意力（Cross-attention） | “Q 来自此处，K/V 来自彼处” | 查询（Query）、键（Key）和值（Value）源自不同数据源的注意力机制；查询向量借此从 ViT 图像块（patches）中提取信息 |
| ITC | “图像-文本对比学习” | 应用于 Q-Former 池化查询向量与文本 CLS 标记之间的 CLIP 风格损失函数 |
| ITM | “图像-文本匹配” | 基于难负样本挖掘（hard-negative mining）的二分类器；迫使查询向量能够区分细粒度的不匹配情况 |
| ITG | “图像引导的文本生成” | 因果语言模型（Causal LM）损失，文本在查询向量的条件下生成；迫使查询向量编码可被文本解码器理解的内容 |
| 两阶段预训练（Two-stage pretraining） | “先表征，后生成” | 第一阶段仅训练 Q-Former（使用 ITC/ITM/ITG）；第二阶段接入冻结的大语言模型（LLM），仅训练投影层与 Q-Former |
| 冻结主干网络（Frozen backbone） | “不进行微调” | 视觉编码器（vision encoder）与大语言模型权重保持固定；仅训练中间的桥接模块 |
| 投影头（Projection head） | “线性映射至 LLM 维度” | 最终的线性层，负责将 Q-Former 的输出映射到大语言模型的嵌入维度（embedding dimension） |
| Perceiver 重采样器（Perceiver resampler） | “Flamingo 的版本” | 类似的可学习查询交叉注意力机制，被 Flamingo 模型应用于每一层，而非作为单一的桥接模块 |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 前作，引入了 ITC/ITM/ITG 三重损失机制。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — “先对齐，后融合（align before fuse）”——第一阶段训练的概念先驱。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — 具备指令感知能力的 Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — 仅使用投影器的方案。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 可学习查询交叉注意力的通用架构。
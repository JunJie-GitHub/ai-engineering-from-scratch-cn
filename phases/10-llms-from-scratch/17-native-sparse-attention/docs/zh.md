# 原生稀疏注意力（Native Sparse Attention, DeepSeek NSA）

> 在 64k token 长度下，注意力机制（Attention）占据了 70-80% 的解码延迟（Decode Latency）。每个开源模型实验室都有相应的优化方案。DeepSeek 的 NSA（ACL 2025 最佳论文）是最终落地并经受住考验的方案：它包含三个并行的注意力分支——压缩的粗粒度 token、选择性保留的细粒度 token，以及用于局部上下文的滑动窗口（Sliding Window）——并通过一个可学习的门控（Learned Gate）进行融合。该架构与硬件对齐（Hardware-aligned，对底层算子友好）、支持原生可训练（Natively Trainable，在预训练（Pre-training）阶段即可生效，而非仅在推理（Inference）阶段临时挂载），并且在 64k 解码任务中，其运行速度优于 FlashAttention，同时生成质量达到或超越全注意力（Full Attention）。本课程将端到端（End-to-End）构建这三个分支，并阐明其稀疏性为何具备端到端可微性（End-to-End Differentiable）。

**Type:** 构建实践
**Languages:** Python（标准库）
**Prerequisites:** 第 7 阶段 · 12（KV 缓存（KV Cache）、FlashAttention），第 7 阶段 · 15（注意力变体），第 10 阶段 · 16（可微注意力）
**Time:** 约 60 分钟

## 学习目标

- 阐述 NSA 的三个注意力分支及其各自捕获的信息特征。
- 解释为何 NSA 支持“原生可训练”，而以往的稀疏注意力（Sparse Attention）方法仅能用于推理阶段。
- 根据压缩块大小（Compression Block Size）和 Top-K 选择策略，计算在 64k 上下文长度下，NSA 相较于全注意力的计算量节省比例。
- 使用 Python 标准库在短合成序列上实现三分支融合逻辑，并验证门控权重（Gating Weights）的行为是否符合预期。

## 问题背景

在序列长度为 N 时，全注意力（Full Attention）的时间复杂度为 `O(N^2)`，且每层需要 `O(N)` 的 KV 缓存（KV Cache）。在 64k token 长度下，其计算量与内存带宽需求是灾难性的。NSA 论文中的实测理论估算表明：在 64k 长度下，注意力机制占据了总解码延迟的 70-80%。所有下游指标——首字延迟（Time To First Token, TTFT）、每秒生成 token 数（Tokens/sec）、每百万 token 成本——均被注意力计算成本所主导。

稀疏注意力（Sparse Attention）是显而易见的解决方案。以往的尝试主要分为两类。固定模式稀疏（Fixed-pattern Sparsity，如滑动窗口、跨步采样、局部块）会直接丢弃信息，在长程召回任务中表现不佳。推理期稀疏（Inference-time Sparsity，如 KV 缓存剪枝（KV Cache Pruning）、H2O、StreamingLLM）则应用于基于稠密注意力预训练的模型，由于模型从未学习过如何通过稀疏模式路由信息，因此只能挽回一小部分潜在的速度提升。

原生稀疏注意力（Native Sparse Attention, NSA）（Yuan 等人，DeepSeek + 北京大学 + 华盛顿大学，ACL 2025 最佳论文，arXiv:2502.11089）兼顾了两者：它让模型在预训练阶段学习稀疏模式，并以与底层算子对齐（Kernel-aligned）的算法实现，从而在推理阶段真正兑现计算量的节省。两年后，NSA 或其直接演进版本将成为所有前沿长上下文模型（Frontier Long-context Model）的默认注意力机制。

## 核心概念

### 三个并行分支

对于每个查询（query），NSA 会执行三次注意力计算（attention），分别针对键值缓存（KV cache）的三种不同视图：

1. **压缩分支（Compressed branch）。** 词元（tokens）被分组为大小为 `l`（通常为 32 或 64）的块。每个块通过一个小型可学习多层感知机（MLP）压缩为单个摘要词元。查询在这些压缩词元上进行注意力计算，从而获得整个序列的粗粒度视图。

2. **选择分支（Selected branch）。** 利用压缩分支的注意力分数，识别出与当前查询最相关的 top-k 个块。读取这些块中的细粒度（未压缩）词元，查询将对它们全部进行注意力计算。可以将压缩分支的注意力视为选择操作的路由信号。

3. **滑动窗口分支（Sliding-window branch）。** 查询对最近的 `W` 个词元（通常为 512）进行注意力计算，以获取局部上下文。该分支捕捉其他两个分支可能遗漏的、结构密集的短程模式（如句法、局部共指）。

三个分支的输出通过一个按位置学习的门控机制进行融合：

out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win

`g_cmp, g_sel, g_win` 是来自查询端小型 MLP 的门控权重。它们无需总和为 1，可以独立地为各个分支分配权重。

### 为何它是“原生可训练的”

选择步骤（top-k 块）是离散的。离散操作会中断梯度流。以往的稀疏注意力（sparse-attention）研究要么在选择步骤中跳过反向传播（backpropagation）（限制了训练效果），要么使用连续松弛方法，但这在推理时无法实现真正的稀疏性。

NSA 巧妙地规避了这一问题：压缩分支的注意力机制本身就是针对整个序列的可微（differentiable）粗粒度注意力计算。top-k 操作仅仅是复用压缩分支中较高的注意力分数，来决定加载哪些细粒度块。梯度可以通过压缩分支的分数进行传播（这些分数同时影响压缩输出和选择逻辑），且所选块对最终输出的贡献也是可微的。不可微的 `top_k` 操作在前向计算图中相当于空操作（no-op）——它仅控制从内存中加载哪些块。

这正是 NSA 能够用于端到端预训练的原因。模型会联合学习如何通过这三个分支路由信息，从而在推理时产生一种稀疏模式，真正实现预期的加速效果。

### 硬件对齐的内核

NSA 的内核（kernel）专为现代 GPU 内存层次结构设计。该内核按分组查询注意力（Grouped-Query Attention, GQA）组加载查询（外层循环），按组获取对应的稀疏 KV 块（内层循环），并在静态随机存取存储器（SRAM）上执行注意力计算。由于每个查询组看到的所选块是相同的（选择是基于查询组而非查询头进行的），KV 加载的开销在组内被分摊。算术强度（arithmetic intensity）得以保持高位。

论文报告称，在 64k 解码长度下，Triton 内核的运行速度比 FlashAttention 快 9 倍，且加速比随序列长度增加而提升。同时提供了前向和反向内核。

### 计算开销预算

设 `N` 为序列长度，`l` 为压缩块大小，`k` 为 top-k 选择数量，`w` 为滑动窗口大小，`b` 为所选块大小（通常等于 `l`）。

- 压缩分支：每个查询 `O(N/l)` 个键（keys），总计 `O(N * N / l)`。
- 选择分支：每个查询 `O(k * b)` 个键，总计 `O(N * k * b)`。
- 滑动窗口分支：每个查询 `O(w)` 个键，总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

当 `N = 64k, l = 64, k = 16, b = 64, w = 512` 时：每个查询的开销为 `1000 + 1024 + 512 = 2536` 个键。全注意力（full attention）为 `64000` 个键。计算量减少 25 倍。

当 `N = 128k, l = 64, k = 16, b = 64, w = 512` 时：每个查询的开销为 `2000 + 1024 + 512 = 3536` 个键。全注意力为 `128000` 个键。计算量减少 36 倍。序列越长，收益越大，这正是该设计的核心所在。

### 与其他方法的对比

| 方法 | 可微 | 实际推理加速 | 长程召回 |
|--------|---------------|----------------------|-------------------|
| 仅滑动窗口 | 是 | 是 | 失效 |
| 跨步/块稀疏 | 是 | 是 | 部分 |
| KV 剪枝（H2O, StreamingLLM） | 不适用（推理阶段） | 是 | 部分 |
| MoBA（Moonshot） | 部分 | 是 | 良好 |
| NSA | 是（原生） | 是（64k 长度下 9 倍） | 与全注意力相当 |

MoBA（Moonshot，arXiv:2502.13189）同期发表，采用了类似的“三管齐下优于单一”思路，将混合专家（Mixture of Experts, MoE）原则应用于注意力块。NSA 与 MoBA 是面向 2026 年长上下文预训练必须了解的两种架构。

## 实现它

`code/main.py` 在一个简短的合成序列上实现了这三个分支，并展示了以下内容：

- 压缩多层感知机 (Compression MLP)（为便于教学演示，此处使用了简单的均值池化 (Mean-Pool) 基线；实际的 NSA 使用的是经过训练的多层感知机）。
- 由压缩分支得分驱动的 Top-k 块选择 (Top-k Block Selection)。
- 针对最后 `w` 个词元 (Token) 的滑动窗口注意力 (Sliding-Window Attention)。
- 门控融合 (Gated Combination)。
- 与全注意力 (Full Attention) 对比的计算量统计输出。

### 步骤 1：将词元压缩为块

def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out

### 步骤 2：压缩分支注意力

对查询向量 (Query) 与压缩后的键向量 (Keys) 执行 Softmax 注意力 (Softmax Attention) 计算。压缩分支的得分同时作为 Top-k 选择的信号。

### 步骤 3：Top-k 块选择

选取得分最高的 `k` 个压缩块的索引。从这些块中加载原始的未压缩词元，并对其执行注意力计算。

### 步骤 4：滑动窗口注意力

提取最后 `w` 个词元，并对它们执行标准注意力计算。

### 步骤 5：门控与融合

基于查询向量输入一个小型多层感知机 (MLP)，生成三个门控权重。最终输出为三个分支输出的加权和。

### 步骤 6：计算量统计

打印每个分支及总计中每个查询向量所关注的键向量数量。与 `N`（全注意力）进行对比。在长度为 1024 个词元的合成序列上（设置 `l = 32, k = 4, w = 128`），NSA 每个查询向量仅需关注 `32 + 128 + 128 = 288` 个键向量，而全注意力需要关注 1024 个——计算量减少了 3.5 倍。

## 实际应用

NSA 已集成至 DeepSeek 自身的长上下文预训练流水线 (Pre-training Pipeline) 中。截至 2026 年 4 月，其在公开推理框架 (Inference Stacks) 中的集成状态如下：

- **DeepSeek 内部**：原生支持，已发布的权重均使用 NSA 或其后续版本 DSA (Deepseek Sparse Attention)。
- **vLLM**：正在开发针对 DeepSeek-V3.x 权重的实验性 NSA 支持。
- **SGLang**：已发布 NSA 基准测试；生产环境部署路径将跟随 vLLM。
- **llama.cpp / CPU**：不支持；在 CPU 吞吐量下，内核分解 (Kernel Decomposition) 带来的开销得不偿失。

何时选择使用 NSA：

- 面向 64k 以上上下文长度，且具备充足算力预算的预训练或继续训练任务。
- 推理 DeepSeek 自身的长上下文检查点 (Checkpoints)。这些权重原生适配 NSA。

何时不建议使用：

- 部署现有的密集注意力 (Dense Attention) 预训练模型。若不进行继续训练，无法直接为模型改造适配 NSA。
- 上下文长度低于 16k。此时三分支架构的开销将抵消其带来的性能收益。
- Batch-1 交互式对话。虽然延迟敏感的解码 (Decode) 阶段能从中受益，但仅限于长上下文场景。

## 部署上线

本课时将生成 `outputs/skill-nsa-integrator.md`。给定一份长上下文预训练运行配置，它将输出一份原生稀疏注意力 (Native Sparse Attention) 集成方案，涵盖：压缩块大小、Top-k 选择、滑动窗口 (Sliding Window)、门控多层感知机 (Gate MLP) 宽度、内核选择，以及能够证明该架构变更合理性的具体长上下文评估指标。

## 练习

1. 在 1024 token 的合成数据上运行 `code/main.py`。在三个预设配置中遍历 `(l, k, w)` 参数并打印计算量。找出在“大海捞针” (Needle-in-Haystack) 测试中，相对于全注意力 (Full Attention) 能保持 95% 召回率，且每个查询所需键数量最少的预设配置。

2. 将均值池化压缩器 (Mean-Pool Compressor) 替换为一个小型可学习多层感知机（2 层，隐藏层维度 32）。在一个合成任务上训练该模型，其中信号定义为块内 token 的平均值。在保留数据集上，测量其相对于均值池化基线的困惑度 (Perplexity) 差距。

3. 实现门控多层感知机 (Gate MLP)。该模块以查询向量作为输入，输出三个标量。验证门控机制的行为是否符合预期：对随机查询赋予近乎均匀的权重；当查询命中序列靠后的块时，对选定的分支赋予显著更高的权重。

4. 计算支持 NSA 的 70B 模型在 128k 上下文长度下的 KV 缓存 (KV Cache) 内存预算。已知 KV 头数为 8，头维度为 128，数据格式为 BF16。将其与全注意力机制以及多头潜在注意力 (Multi-Head Latent Attention)（Phase 10 · 14 中已展示 MLA 的相关数据）进行对比。找出 NSA 细粒度分支的 KV 缓存大小与全注意力机制持平的序列长度。

5. 阅读 NSA 论文（arXiv:2502.11089）的第 4 节，用三句话解释为何直接复用压缩分支的注意力分数进行 Top-k 选择，而非单独计算路由分数。请将答案与梯度流 (Gradient Flow) 机制联系起来。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 压缩分支 (Compressed branch) | “粗粒度视图” | 对块平均键进行注意力计算，以每个查询 O(N/l) 个键的复杂度提供全局上下文 |
| 选定分支 (Selected branch) | “Top-k 块” | 对压缩分支得分最高的 `k` 个块进行细粒度注意力计算 |
| 滑动窗口 (Sliding window) | “局部上下文” | 对最后 `W` 个 token 进行注意力计算，用于捕捉短程模式 |
| 原生可训练性 (Native trainability) | “预训练时即开启稀疏性” | 稀疏模式在预训练阶段学习得到，而非在推理阶段临时添加 |
| 压缩块大小 l (Compression block size l) | “粗粒度视图的分组大小” | 多少个 token 被合并为一个摘要；通常为 32-64 |
| Top-k | “保留的块数” | 读取其未压缩 token 的压缩块数量；通常为 16 |
| 滑动窗口 W (Sliding window W) | “局部注意力半径” | 通常为 512；过小会损害局部连贯性，过大会浪费计算资源 |
| 分支门控 (Branch gate) | “如何混合三者” | 每个位置的 MLP 输出，用于对三个分支的贡献进行加权 |
| 硬件对齐 (Hardware alignment) | “对内核友好的稀疏性” | 选择的稀疏模式需确保实际 GPU 内核能够实现理论上的加速效果 |
| DSA | “NSA 的继任者” | DeepSeek 稀疏注意力 (Deepseek Sparse Attention)，是 DeepSeek 技术路线中继 NSA 之后的架构 |

## 延伸阅读

- [Yuan 等人 — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 最佳论文)](https://arxiv.org/abs/2502.11089) — 该论文
- [DeepSeek-V3 技术报告 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 原生稀疏注意力（Native Sparse Attention）所针对的架构系列
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期研究，基于分块的混合专家（Mixture of Experts）式注意力机制
- [Beltagy 等人 — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — 滑动窗口（Sliding Window）机制的起源
- [Xiao 等人 — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — 原生稀疏注意力所改进的推理时稀疏性（Inference-time Sparsity）基线
- [Dao 等人 — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — 原生稀疏注意力算子（Kernels）在 64k 上下文长度下超越的全注意力（Full Attention）基线
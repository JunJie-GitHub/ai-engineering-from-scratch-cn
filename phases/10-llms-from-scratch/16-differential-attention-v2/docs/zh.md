# 差分注意力（Differential Attention）V2

> Softmax 注意力（Softmax Attention）会将少量概率分散到每一个非匹配词元（token）上。在超过 10 万个词元的序列中，这种噪声会不断累积并淹没有效信号。差分 Transformer（Differential Transformer, Ye et al., ICLR 2025）通过将注意力计算为两个 Softmax 的差值，减去共享的噪声基底（noise floor），从而解决了这一问题。DIFF V2（Microsoft, 2026 年 1 月）是对生产级技术栈的重写：其解码延迟（decode latency）与基线 Transformer 持平，无需自定义内核（custom kernels），且兼容 FlashAttention。本课程将完整梳理从 V1 到 V2 的端到端演进，并提供一个可在 Python 标准库中运行的差分操作简易实现。

**类型：** 构建实践
**语言：** Python（标准库）
**前置知识：** 第 7 阶段 · 02（自注意力 self-attention）、第 7 阶段 · 15（注意力变体 attention variants）、第 10 阶段 · 14（架构走读 architecture walkthrough）
**时长：** 约 60 分钟

## 学习目标

- 准确阐述 Softmax 注意力为何存在噪声基底，以及该基底为何会随上下文长度（context length）增加而增长。
- 推导差分注意力公式，并解释减法操作如何在保留有效信号的同时抵消共享噪声分量。
- 梳理从 V1 到 V2 的差异：哪些部分变得更快、更简单、更稳定，以及为何每项改动对于生产级预训练都必不可少。
- 使用纯 Python 从零实现差分注意力，并在合成的“信号加噪声”查询（query）上实证验证其噪声抵消特性。

## 问题背景

标准的 Softmax 注意力机制具有一个数学特性，在大规模应用中会演变为工程难题。对于查询向量 `q`，其注意力权重为 `softmax(qK^T / sqrt(d))`。Softmax 函数永远无法输出精确的零值——每个非匹配词元都会分配到一定的正概率质量（probability mass）。这种残留质量即为噪声，且其总量随上下文长度线性增长。在 12.8 万个词元的场景下，即使每个非匹配词元仅分得 0.001% 的概率，127,999 个词元叠加后也会贡献约 12% 的总概率。模型必须学会在这种随上下文增长的噪声基底中寻找有效路径。

在实际应用中，这表现为注意力头干扰（attention-head interference）：长上下文检索增强生成（RAG）中出现虚构引用、在 10 万词元检索任务中发生“中间迷失（lost-in-the-middle）”现象，以及在超过 3.2 万词元的“大海捞针（needle-in-a-haystack）”基准测试中出现细微的准确率下降。《差分 Transformer》论文（arXiv:2410.05258, ICLR 2025）量化了这一差距：与同等规模的基线模型相比，DIFF Transformer 实现了更低的困惑度（perplexity）、更高的长上下文准确率以及更少的幻觉现象。

DIFF V1 存在三个缺陷，导致其无法进入前沿预训练流水线：每次解码步骤需加载两次值缓存（value cache）；依赖破坏 FlashAttention 兼容性的自定义 CUDA 内核（custom CUDA kernels）；其逐头 RMS 归一化（per-head RMSNorm）在 700 亿参数以上规模的长期训练中会导致不稳定。DIFF V2（Microsoft UniLM 博客，2026 年 1 月 20 日）彻底修复了这三点。本课程将对比讲解两个版本，构建差分算子（difference operator），并在简易查询上对噪声抵消效果进行基准测试。

## 核心概念

### Softmax 的噪声基底 (Noise Floor)

对于查询向量 `q` 和键向量 `K = [k_1, ..., k_N]`，注意力权重 (Attention Weights) 的计算公式为：

w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))

没有任何一个 `w_i` 会严格为零。如果 `k_i` 与 `q` 完全无关，其得分 `q . k_i` 并非 0，而是以 `||q||^2 / d` 为方差在零附近波动。经过 Softmax 归一化后，每个无关的 Token 仍会对加权和贡献 `O(1/N)`。所有无关 Token 的总贡献量为 `O((N-1)/N) = O(1)`，这并非一个可以忽略的小数值。

模型真正期望的是一种类似硬 Top-K (Hard Top-K) 的机制：在匹配的 Token 上赋予高权重，而在其他位置赋予接近零的权重。Softmax 函数过于平滑，无法直接实现这一目标。

### 差分机制 (Differential Mechanism) 的核心思想

将每个注意力头 (Attention Head) 的 Q 和 K 投影拆分为两部分：`Q = (Q_1, Q_2)` 和 `K = (K_1, K_2)`。分别计算两个注意力图 (Attention Maps)：

A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))

输出为：

DiffAttn = (A_1 - lambda * A_2) V

相减操作会抵消两个注意力图共有的噪声分布。如果两个图在 127k 个无关 Token 上具有大致均匀的权重（在随机初始化时确实如此），这些权重就会相互抵消。而信号——即集中在少数真正相关 Token 上的峰值权重——只有在两个图中以相同幅度出现时才会被抵消，但模型一旦开始训练，这种情况就不会发生。

`lambda` 是每个注意力头可学习的标量，其参数化形式为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。该值可以为负数。`lambda_init` 默认初始化为一个较小的正数（例如 0.8）。

### 为何这与逐头降噪 (Headed Noise-Canceling) 原理一致

想象两个带有噪声的麦克风同时录制同一个人的声音。两者都会拾取说话人的声音以及相关的背景噪声。将其中一个信号减去另一个，共享的噪声就会被消除。人声得以保留，是因为两个信号在相位或幅度上存在足够差异，从而避免了完全抵消。每个注意力头的 `lambda` 正是通过学习来掌握这种平衡。

### V1 与 V2 的差异

V1 版本保持了与基线 Transformer 相同的参数量。为了在每个头中获得两个查询向量，它将头维度 (Head Dimension) 减半。这牺牲了头的表达能力，且更严重的是，每个头的值缓存 (Value Cache) 也被减半。在解码 (Decode) 阶段，每一步都需要加载两次值缓存（每个 Softmax 分支各一次）。结果是：尽管参数量相同，但解码速度反而慢于基线模型。

V2 版本将查询头 (Query Heads) 的数量翻倍，同时保持 KV 头数量不变（参数从升维投影中借用）。头维度保持与基线一致。在相减操作之后，多余的维度会被重新投影降维，以匹配基线 Transformer 的 `O_W` 投影。这一改动同时带来了三个效果：

1. 解码速度与基线持平（KV 缓存仅需加载一次）。
2. FlashAttention 可直接运行，无需修改（无需自定义内核）。
3. 解码阶段的算术强度 (Arithmetic Intensity) 提升（从 HBM 加载每字节数据对应的计算量增加）。

V2 还移除了 V1 中用于稳定相减操作的逐头 RMSNorm (Root Mean Square Normalization)。在 70B 级别的预训练规模下，该 RMSNorm 会导致训练后期不稳定。V2 采用了一种更简单的初始化方案来替代它，无需额外模块即可保持训练稳定。

### 适用场景

| 工作负载 | 收益 |
|----------|---------|
| 长上下文 RAG (64k+) | 注意力图更清晰，减少幻觉引用 |
| 大海捞针 (Needle-in-a-Haystack) 基准测试 | 超过 32k 上下文时准确率显著提升 |
| 多文档问答 (Multi-document QA) | 减少跨文档干扰 |
| 8k 上下文代码补全 | 收益甚微，不值得为此修改架构 |
| 短对话 (< 4k) | 与基线模型基本无异 |

该机制的价值随上下文长度增加而提升。在 4k Token 时，噪声基底足够小，标准注意力机制即可胜任。但在 128k 时，噪声基底已成为性能瓶颈。

### 与 2026 年其他架构调优方案 (Knobs) 的兼容性

| 特性 | 是否与 DIFF V2 兼容？ |
|---------|------------------------|
| GQA (Grouped-Query Attention) | 是（V2 增加的是 Q 头数量，而非 KV 头） |
| MLA (DeepSeek) | 理论上兼容，暂无结合两者的已发表论文 |
| MoE (Mixture of Experts) | 是（注意力机制与 MLP 模块相互独立） |
| RoPE (Rotary Position Embedding) | 是（保持不变） |
| YaRN / 长上下文缩放 | 是（这正是 DIFF 发挥最大作用的场景） |
| FlashAttention | 是（V2 支持，V1 不支持） |
| 投机解码 (Speculative Decoding) | 是（注意力机制的改动对投机解码循环透明） |

## 构建实现

`code/main.py` 使用纯 Python 实现了差分注意力（Differential Attention）。通过一个具有已知“信号加噪声”结构的示例查询，您可以直接测量噪声消除比（Noise-Cancellation Ratio）。

### 步骤 1：标准 Softmax 注意力

标准库矩阵运算：使用嵌套列表、手动实现矩阵乘法（Matrix Multiplication），以及通过减去最大值来保证数值稳定性的 Softmax 计算。

def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]

### 步骤 2：将 Q 和 K 拆分为两半

V1 风格：将注意力头维度（Head Dimension）减半。V2 风格：保持注意力头维度不变，将注意力头数量翻倍。本示例实现采用 V1 风格以便于理解——两者的数学原理完全一致，仅在维度管理与数据记录上有所区别。

### 步骤 3：双 Softmax 分支与相减

A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]

注意：输出权重可能为负值。这没有问题——值缓存（Value Cache）依然能够处理带符号的贡献值，后续的 V 投影（Projection）操作会自然吸收该符号。

### 步骤 4：噪声消除测量

构建一个长度为 1024 的合成序列。将信号 Token 放置在已知位置，其余部分填充噪声。分别计算：(a) 信号位置处的标准 Softmax 注意力权重，以及 (b) 差分注意力权重。测量两者的信噪比（Signal-to-Noise Ratio）。差分注意力能稳定地将信噪比提升 3 到 10 倍，具体倍数取决于两个分支在训练过程中被优化出的差异程度。

### 步骤 5：V1 与 V2 的参数量核算

给定配置（hidden=4096, heads=32, d_head=128），输出如下：

- 基线 Transformer（Baseline Transformer）：Q、K、V 的维度均为 `hidden * hidden`，多层感知机（MLP）维度为 4 * hidden。
- DIFF V1：Q、K 维度均为 `hidden * hidden`，V 维度保持 `hidden * hidden` 不变，内部将头维度减半。额外增加每个头的 `lambda` 参数（复杂度为 O(heads * d_head)）。
- DIFF V2：Q 维度为 `2 * hidden * hidden`，K 维度为 `hidden * hidden`，V 维度为 `hidden * hidden`。在 O_W 投影前将额外维度投影回原尺寸。同样增加 `lambda` 参数。

该示例实现会测量 V2 版本的额外参数开销（每个注意力块大约增加 `hidden * hidden` 个参数）并将其打印输出。

## 使用它

截至 2026 年 4 月，DIFF V2 尚未在所有生产级推理服务器（Production Inference Server）中全面部署，但 vLLM 和 SGLang 的集成工作正在进行中。与此同时，该模式已出现在以下场景中：

- 微软内部的长上下文（Long-Context）生产模型。
- 针对 256k 以上上下文长度的多个开源模型训练实验中的研究复现。
- 在交替层中将 DIFF 注意力机制（DIFF Attention）与滑动窗口注意力机制（Sliding-Window Attention）相结合的混合架构。

在 2026 年，建议在以下情况下采用该技术：

- 从零开始训练目标有效上下文（Effective Context）长度超过 64k 的新模型。应从一开始就引入差分注意力机制（Differential Attention）；后期重新训练的成本极高。
- 微调长上下文模型时，若“中间丢失”（Lost-in-the-Middle）现象在评估中占据主导。可在 Q 投影（Q Projections）上应用 LoRA 来近似 DIFF 结构。

不建议在以下情况下使用：

- 正在部署具有稳定长上下文性能的预训练稠密模型（Pre-trained Dense Model）。对现有权重进行重新训练的成本通常难以收回。
- 上下文长度始终低于 16k。此时噪声基底（Noise Floor）可忽略不计。

## 交付成果

本课时将生成 `outputs/skill-diff-attention-integrator.md` 文件。给定模型架构、目标上下文长度、幻觉特征（Hallucination Profile）和训练预算，它将输出一份集成方案，用于在新的预训练（Pre-training）流程或 LoRA 微调（LoRA Fine-tune）中添加差分注意力机制。

## 练习

1. 运行 `code/main.py`。验证在合成查询（Synthetic Query）上，差分注意力机制报告的信噪比（Signal-to-Noise Ratio）是否高于标准 Softmax 注意力机制（Standard Softmax Attention）。调整噪声幅度，并展示标准注意力机制变得不可用的临界点。

2. 针对 7B 级别模型（hidden=4096, heads=32, d_head=128, 32 layers），计算从基线模型到 DIFF V1 以及从基线模型到 DIFF V2 的参数量差值（Parameter-Count Delta）。说明哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文（arXiv:2410.05258）的第 3 节以及 DIFF V2 Hugging Face 博客的第 2 节。用两句话解释：为什么 V1 版本必须使用逐头 RMSNorm（Per-Head RMSNorm），以及为什么 V2 版本可以在不导致训练发散（Training Divergence）的情况下将其移除。

4. 实现消融实验（Ablation Study）：分别计算 `lambda = 0`（仅保留首个 Softmax）和 `lambda = 1`（完全相减）时的差分注意力。在合成查询上，测量在整个参数遍历范围内信噪比的变化情况。找出使信噪比最大化的 `lambda` 值。

5. 将该示例扩展至分组查询注意力机制（Grouped-Query Attention, GQA）+ DIFF V2。选择 8 个 KV 头（KV Heads）和 32 个 Q 头（Q Heads）。证明其 KV 缓存（KV Cache）大小与具有相同 (8, 32) 配置的基线 GQA 模型一致。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 差分注意力 (Differential Attention) | “两个 softmax 相减” | 将 Q、K 分为两半，分别计算两个 softmax 映射，从第一个中减去第二个（按 lambda 缩放），再与 V 相乘 |
| 噪声基底 (Noise Floor) | “softmax 的非零尾部” | softmax 分配给每个无关 token 的 O(1/N) 权重，在长上下文中累加后达到 O(1) 量级 |
| lambda | “减法缩放系数” | 每个注意力头可学习的标量参数，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可为负值 |
| DIFF V1 | “ICLR 2025 版本” | 原始差分 Transformer；将头维度减半以保持参数量不变，需自定义算子，解码速度较慢 |
| DIFF V2 | “2026 年 1 月的修复版” | 将 Q 头数量翻倍，保持 KV 头数量不变；解码速度与基线一致，且兼容 FlashAttention |
| 逐头 RMSNorm (Per-head RMSNorm) | “V1 稳定器” | V1 在差分操作后应用的额外归一化层；V2 将其移除以防止训练后期的不稳定性 |
| 信噪比 (Signal-to-Noise Ratio) | “注意力被浪费的程度” | 真实信号位置上的权重与无关位置平均权重的比值 |
| 中间迷失现象 (Lost in the Middle) | “长上下文失效模式” | 经验现象：长上下文中间部分的文档检索准确率会下降——DIFF 注意力机制可缓解此问题 |
| 算术强度 (Arithmetic Intensity) | “每加载一字节数据对应的 FLOPs” | V2 通过每次加载 KV 时处理双倍的查询（queries）提升了解码阶段的该指标；对内存受限的解码至关重要 |

## 延伸阅读

- [Ye 等人 — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，包含噪声抵消理论与长上下文消融实验
- [Microsoft unilm — Differential Transformer V2 (Hugging Face 博客，2026 年 1 月)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 面向生产环境技术栈的重写版本，解码速度与基线一致，且兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 理论分析，阐释减法操作如何恢复预训练自注意力结构
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani 等人 — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 进行减法操作所基于的基线 Transformer
- [Liu 等人 — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF 注意力机制旨在优化的长上下文基准测试
# 注意力机制变体 —— 滑动窗口、稀疏与差分

> 全注意力机制（Full Attention）如同一个完整的圆。每个词元（Token）都与所有其他词元进行交互，内存开销随之剧增。四种变体重塑了这一结构，成功将成本降低了一半。

**类型：** 构建
**语言：** Python
**前置知识：** 第 7 阶段 · 02（自注意力机制 Self-Attention）、第 7 阶段 · 03（多头注意力 Multi-Head Attention）、第 7 阶段 · 12（键值缓存 KV Cache / Flash Attention）
**预计耗时：** 约 60 分钟

## 核心问题

全注意力机制在序列长度（Sequence Length）上的内存与计算复杂度均为 `O(N²)`。对于上下文长度为 128K 的 Llama 3 70B 模型而言，这意味着每层需要处理 160 亿个注意力条目，再乘以 80 层。Flash Attention（第 12 课）虽然隐藏了 `O(N²)` 的激活内存（Activation Memory），但并未改变算术计算成本——每个词元依然需要关注所有其他词元。

三类变体直接改变了注意力矩阵（Attention Matrix）本身的拓扑结构：

1. **滑动窗口注意力（Sliding Window Attention, SWA）**。每个词元仅关注固定窗口内的相邻词元，而非完整的前缀序列。内存与计算复杂度降至 `O(N · W)`，其中 `W` 为窗口大小。应用于 Gemma 2/3、Mistral 7B 的前几层以及 Phi-3-Long。
2. **稀疏/块注意力（Sparse / Block Attention）**。仅对选定的 `(i, j)` 词元对进行打分，其余位置的权重强制为零。应用于 Longformer、BigBird 和 OpenAI 稀疏 Transformer。
3. **差分注意力（Differential Attention）**。使用独立的 Q/K 投影计算两张注意力图，并将两者相减。此举消除了将权重“泄漏”到前几个词元的“注意力汇（Attention Sink）”现象。应用于微软的 DIFF Transformer（2024）。

这些机制通常共存。2026 年的前沿模型往往会将它们混合使用：大多数层采用 SWA-1024，每五层插入一次全局全注意力机制，另有少量差分注意力头用于优化检索结果。Gemma 3 采用的 5:1 滑动窗口与全局注意力比例，已成为当前的标准配置。

## 核心概念

### 滑动窗口注意力（Sliding Window Attention, SWA）

位于位置 `i` 的每个查询（query）仅关注 `[i - W, i]`（因果滑动窗口注意力，causal SWA）或 `[i - W/2, i + W/2]`（双向，bidirectional）范围内的位置。窗口外的词元（token）在得分矩阵（score matrix）中会被赋值为 `-inf`。

full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x

当 `N = 8192` 且 `W = 1024` 时，得分矩阵在期望上仅有 1024 × 8192 个非零行——计算量减少了 8 倍。

**KV 缓存（KV cache）随 SWA 缩小。** 每层只需保留 K 和 V 的最后 `W` 个词元。对于类似 Gemma-3 的配置（窗口大小为 1024，上下文长度为 128K），KV 缓存将减少 128 倍。

**质量代价。** 仅使用 SWA 的 Transformer 在长程信息检索方面表现不佳。解决方法：将 SWA 层与全注意力（full-attention）层交错排列。Gemma 3 采用了 5:1 的 SWA 与全局注意力比例。Mistral 7B 则使用了一种因果 SWA 堆叠结构，信息通过重叠的窗口“向前流动”——每一层将有效感受野（receptive field）扩展 `W`，经过 `L` 层后，模型能够回溯关注 `L × W` 个词元。

### 稀疏/块注意力（Sparse / Block Attention）

预先选定一个 `N × N` 的稀疏模式（sparsity pattern）。三种典型结构如下：

- **局部+跨步（Local + strided，OpenAI 稀疏 Transformer）**。关注最后 `W` 个词元，以及此前每隔 `stride` 个的词元。以 `O(N · sqrt(N))` 的计算复杂度同时捕获局部与长程依赖。
- **Longformer / BigBird**。局部窗口 + 少量全局词元（global tokens，例如 `[CLS]`），这些词元与所有词元相互关注，并辅以随机稀疏连接。在质量相当的情况下，经验上可将上下文长度扩展 2 倍。
- **原生稀疏注意力（Native Sparse Attention，DeepSeek, 2025）**。学习 `(Q, K)` 中哪些块（blocks）是重要的；在内核（kernel）级别跳过零块。兼容 FlashAttention。

稀疏注意力本质上是一项内核工程（kernel-engineering）工作。其数学原理很简单（对得分矩阵进行掩码处理）；真正的优势在于永远不将零值条目加载到 SRAM 中。FlashAttention-3 和 2026 年的 FlexAttention API 使得自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力（Differential Attention，DIFF Transformer, 2024）

常规注意力存在“注意力汇聚（attention sink）”问题：softmax 强制每一行的和为 1，因此那些不需要特别关注任何内容的词元会将权重倾泻到第一个词元（或前几个）上。这窃取了本应分配给实际内容的模型容量。

差分注意力通过计算**两个**注意力图（attention maps）并相减来解决此问题：

A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V

其中 `λ` 是一个可学习的标量（scalar，通常为 0.5–0.8）。A1 捕获实际内容的权重；A2 捕获汇聚效应。相减操作抵消了汇聚效应，将权重重新分配给相关的词元。

报告结果（Microsoft 2024）：困惑度（perplexity）降低 5–10%，在相同训练长度下有效上下文长度延长 1.5–2 倍，大海捞针（needle-in-haystack）检索能力更精准。

### 变体对比

| 变体 | 计算复杂度 | KV 缓存 | 质量对比（全注意力） | 生产应用 |
|---------|---------|----------|-----------------|----------------|
| 全注意力（Full attention） | O(N²) | 每层 O(N) | 基准 | 所有模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | 困惑度低 0.1，配合全局层效果佳 | Gemma 2/3, Phi-3-Long |
| 局部+跨步稀疏 | O(N·√N) | 混合 | 与 SWA 相似 | OpenAI 稀疏 Transformer, Longformer |
| BigBird（局部+全局+随机） | 约 O(N) | 混合 | 在 2 倍上下文下匹配全注意力 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃比例) | O(N) | 困惑度差异在 0.05 以内 | DeepSeek-V3.2, 2025 |
| 差分注意力 | O(2·N²) | O(2N) | 困惑度降低 5% 至 10% | DIFF Transformer, 2026 年初期模型 |

## 动手构建

参见 `code/main.py`。我们实现了一个因果掩码（causal mask）比较器，可在一个示例序列（toy sequence）上并排对比全注意力（full attention）、滑动窗口注意力（sliding window attention, SWA）、局部+跨步（local+strided）以及差分注意力（differential attention）。

### 步骤 1：全因果掩码（基线）

def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]

源自第 07 课的基线（baseline）实现。下三角矩阵；对角线上方权重为零。

### 步骤 2：滑动窗口因果掩码

def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M

仅包含一个参数 `window`。当 `window >= n` 时，可恢复为全因果注意力。当 `window = 1` 时，每个词元（token）仅关注自身。

### 步骤 3：局部 + 跨步稀疏掩码

def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M

密集局部窗口加上从序列开头起每隔 `stride` 个词元的采样。随着网络层数的增加，感受野（receptive field）呈对数级增长。

### 步骤 4：差分注意力

def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V

进行两次注意力计算，并通过学习到的混合系数进行相减。在代码中，我们对比了单次注意力与差分注意力的注意力汇聚（attention-sink）热力图，并观察汇聚效应的消除。

### 步骤 5：KV 缓存大小

打印各变体在 `N = 131072` 时每层的缓存大小。SWA 与稀疏变体的缓存大小下降 10 至 100 倍。差分注意力则翻倍。请明确权衡相应的内存开销。

## 使用它

2026 年生产环境模式：

from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)

PyTorch 2.5+ 中的灵活注意力机制 (FlexAttention) 接受一个掩码函数 (mask function)：

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)

该代码将编译为自定义的 Triton 内核 (Triton kernel)。在常见模式下，其性能与 FlashAttention-3 的差距在 10% 以内，且掩码函数支持直接使用 Python 可调用对象 (Python callable)。

**何时选择各方案：**

- **纯全注意力机制 (Pure Full Attention)** — 适用于上下文长度约 16K 以内的每一层，或在检索质量至关重要的场景。
- **滑动窗口与全局注意力混合 (SWA + Global Mix)** — 适用于长上下文（>32K），且训练与推理受内存限制的场景。这是 2026 年 32K 以上上下文的默认配置。
- **稀疏块注意力 (Sparse Block Attention)** — 需自定义内核与模式。专用于特定工作负载（如检索、音频处理）。
- **差分注意力 (Differential Attention)** — 适用于任何受注意力汇聚点污染 (attention-sink contamination) 影响的工作负载（如长上下文检索增强生成 (RAG)、大海捞针任务 (needle-in-haystack)）。

## 发布上线 (Ship It)

请参阅 `outputs/skill-attention-variant-picker.md`。该技能模块 (skill) 会根据目标上下文长度、检索需求以及训练/推理的计算配置，为新模型选择合适的注意力拓扑结构 (attention topology)。

## 练习

1. **简单。** 运行 `code/main.py`。验证当 `window=4` 时，滑动窗口注意力 (SWA) 是否将每行最后 4 个词元 (token) 之外的所有值置零。验证当 `window=n` 时，是否能逐位一致地 (bit-identically) 复现完整的因果注意力 (causal attention)。
2. **中等。** 在第 07 课的综合项目基础上，实现 `window=1024` 的因果滑动窗口注意力。在 `tinyshakespeare` 数据集上训练 1,000 步。与全注意力机制相比，验证集损失 (validation loss) 会退化多少？峰值内存 (peak memory) 会下降多少？
3. **困难。** 在综合项目模型中实现类似 Gemma-3 的 5:1 层混合结构（5 层 SWA，1 层全局）。在参数量匹配的条件下，与纯 SWA 和纯全局基线模型 (baselines) 对比损失、内存占用及生成质量。
4. **困难。** 实现差分注意力机制，为每个注意力头 (attention head) 学习一个独立的 `λ` 参数。在合成检索任务（1 个目标信息，2,000 个干扰项）上进行训练。在参数量匹配的条件下，测量其检索准确率并与单一注意力基线模型进行对比。

## 关键术语

| 术语 | 业界俗称 | 技术实质 |
|------|-----------------|-----------------------|
| 滑动窗口注意力 (Sliding Window Attention, SWA) | “局部注意力” | 每个查询（query）仅关注其最近的 `W` 个 token；键值缓存（KV cache）大小缩减至 `O(W)`。 |
| 有效感受野 (Effective Receptive Field) | “模型能回溯多远” | 在窗口大小为 `W` 的 `L` 层 SWA 堆叠中，最大可达 `L × W` 个 token。 |
| Longformer / BigBird | “局部 + 全局 + 随机” | 采用稀疏模式，包含少量始终参与注意力的全局 token；属于早期的长上下文处理方案。 |
| 原生稀疏注意力 (Native Sparse Attention) | “DeepSeek 的内核优化技巧” | 学习块级稀疏性；在内核层面直接跳过零块，同时保持模型质量。 |
| 差分注意力 (Differential Attention) | “两张注意力图，相减抵消” | DIFF Transformer：从第一张注意力图中减去学习到的 `λ` 倍第二张注意力图，以抵消注意力汇聚（attention sinks）。 |
| 注意力汇聚 (Attention Sink) | “权重泄漏至第 0 个 token” | Softmax 归一化强制每行权重之和为 1；信息量低的查询会将权重倾泻到位置 0。 |
| FlexAttention | “掩码即 Python 代码” | PyTorch 2.5+ 提供的 API，可将任意掩码函数编译为类似 FlashAttention 的内核。 |
| 层类型混合 (Layer Type Mix) | “SWA 与全局注意力比例为 5:1” | 在堆叠中交替使用稀疏注意力层和全注意力层，以在更低内存占用下保持模型质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 滑动窗口与全局 token 结合的经典论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部 + 全局 + 随机注意力模式。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 提出的局部+跨步（strided）模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 采用 1:1 的 SWA 与全局注意力混合比例。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 采用 5:1 混合比例且窗口大小为 1024，现已成为教科书级的默认配置。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 原始论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 采用的学习型稀疏注意力机制。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — “使用指南（Use It）”部分中“掩码即可调用对象”模式的 API 参考文档。
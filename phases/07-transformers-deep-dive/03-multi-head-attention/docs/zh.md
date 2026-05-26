# 多头注意力（Multi-Head Attention）

> 单个注意力头（attention head）一次仅能学习一种关系。八个头便能学习八种。注意力头成本极低，尽管多用。

**类型：** 构建
**语言：** Python
**前置要求：** 第 7 阶段 · 02（从零实现自注意力（Self-Attention））
**耗时：** 约 75 分钟

## 问题背景

单个自注意力头（self-attention head）仅计算一个注意力矩阵。该矩阵只能捕捉一种关系——通常是在当前训练信号下使损失最小化的那一种。如果你的数据中主谓一致、指代消解、长程语篇和句法分块等特征交织在一起，单个头会将它们模糊地混合进同一个 softmax 分布中，从而丢失大量信号。

2017 年 Vaswani 等人提出的解决方案是：并行运行多个注意力函数，每个函数拥有独立的 Q、K、V 投影（projections），最后将输出拼接（concatenate）起来。每个头在维度为 `d_model / n_heads` 的更小子空间中运行。总参数量保持不变，但模型的表达能力显著提升。

多头注意力已成为 2026 年所有 Transformer 架构的默认配置。唯一的争议在于使用*多少*个头，以及键（keys）和值（values）是否共享投影（例如分组查询注意力（Grouped-Query Attention）、多查询注意力（Multi-Query Attention）、多头潜在注意力（Multi-head Latent Attention））。

## 核心概念

![多头注意力：拆分、计算注意力、拼接](../assets/multi-head-attention.svg)

**拆分（Split）。** 获取形状为 `(N, d_model)` 的输入 `X`。将其投影为形状均为 `(N, d_model)` 的 Q、K、V。重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行计算注意力（Attend in parallel）。** 在每个头内部执行缩放点积注意力（scaled dot-product attention）。每个头输出形状为 `(N, d_head)` 的结果。各个头在嵌入空间的不同子空间中独立运作，在注意力计算过程中彼此不交互。

**拼接与投影（Concatenate and project）。** 将各个头的输出重新堆叠回 `(N, d_model)` 形状，并与形状为 `(d_model, d_model)` 的可学习输出矩阵 `W_o` 相乘。`W_o` 正是各个头进行信息融合的地方。

**为何有效。** 每个头都可以专注于特定任务，而无需与其他头争夺表征容量（representational budget）。2019 至 2024 年的探针研究（probing studies）揭示了不同头的明确分工：位置头、关注前一个 token 的头、复制头、命名实体头以及归纳头（induction heads，后者是上下文学习（in-context learning）的基础）。

**2026 年的主流变体谱系：**

| 变体 | Q 头数量 | K/V 头数量 | 代表模型 |
|---------|---------|-----------|---------|
| 多头注意力（MHA） | N | N | GPT-2, BERT, T5 |
| 多查询注意力（MQA） | N | 1 | PaLM, Falcon |
| 分组查询注意力（GQA） | N | G（例如 N/8） | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| 多头潜在注意力（MLA） | N | 压缩至低秩 | DeepSeek-V2, V3 |

GQA 已成为现代默认选择，因为它能在几乎不损失质量的前提下，将 KV 缓存（KV-cache）内存占用降低 `N/G` 倍。MLA 则更进一步，将 K/V 压缩至潜在空间（latent space），在计算时再投影回原空间——这会消耗更多浮点运算次数（FLOPs），但能大幅节省内存。

## 动手实现

### 步骤 1：从已有的单头注意力（Single-Head Attention）中拆分多头

使用第 02 课中的 `SelfAttention`，并用拆分/拼接（Split/Concat）操作将其包裹。NumPy 实现请参考 `code/main.py`，其核心逻辑如下：

def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)

仅需一次重塑（reshape）和一次转置（transpose），无需循环。这正是 PyTorch 在 `nn.MultiheadAttention` 底层所采用的实现方式。

### 步骤 2：在每个头上运行缩放点积注意力（Scaled-Dot-Product Attention）

每个注意力头（Attention Head）将独立获取 Q、K、V 的切片。此时，注意力计算将转化为批量矩阵乘法（Batched Matrix Multiplication）：

def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights

在实际硬件上，`Qh @ Kh.transpose(...)` 对应一次 `bmm` 操作。GPU 将其视为形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的单一批量矩阵乘法。增加注意力头数几乎不带来额外的计算开销。

### 步骤 3：分组查询注意力（Grouped-Query Attention）变体

仅键（Key）和值（Value）的投影层发生变化。Q 被划分为 `n_heads` 个组；而 K 和 V 仅划分为 `n_kv_heads`（`n_kv_heads < n_heads`）个组，并通过重复操作以匹配 Q 的维度：

def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)

在推理阶段，这能显著节省内存，因为 KV 缓存（KV Cache）中只需存储 `n_kv_heads` 份副本，而非 `n_heads` 份。例如，Llama 3 70B 模型采用了 64 个查询头搭配 8 个 KV 头，使缓存体积缩小了 8 倍。

### 步骤 4：探查每个头所学到的特征

使用 4 个注意力头对一句简短文本运行多头注意力（MHA）。针对每个头，打印其 `(N, N)` 维度的注意力矩阵（Attention Matrix）。你会发现，即便在随机初始化下，不同的头也会捕捉到不同的结构特征——这既部分源于输入信号本身，也部分归因于子空间中的旋转对称性（Rotational Symmetry）。

## 上手使用

在 PyTorch 中，单行版本如下：

import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

PyTorch 2.5+ 中的 GQA（分组查询注意力，Grouped-Query Attention）：

from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

**设置多少个注意力头（Attention Head）？** 基于 2026 年生产环境模型的经验法则：

| 模型规模 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小型 (~1.25亿) | 768 | 12 | 64 |
| 基础 (~3.5亿) | 1024 | 16 | 64 |
| 大型 (~10亿) | 2048 | 16 | 128 |
| 前沿 (~700亿) | 8192 | 64 | 128 |

`d_head` 几乎总是落在 64 或 128。它决定了单个注意力头能够“感知”的信息量。如果低于 32，注意力头就会开始与缩放因子（Scaling Factor）`sqrt(d_head)` 产生冲突；如果高于 256，则会丧失“众多小型专家协同工作”的优势。

## 部署上线

请参阅 `outputs/skill-mha-configurator.md`。该技能模块会根据参数预算、序列长度和部署目标，为新建的 Transformer（Transformer 架构）推荐注意力头数量、KV 头数量以及投影策略。

## 练习

1. **简单。** 提取 `code/main.py` 中的 MHA（多头注意力，Multi-Head Attention），在固定 `d_model=64` 的情况下，将 `n_heads` 从 1 改为 16。绘制一个微型单层模型在合成复制任务上的损失曲线。增加注意力头数量是有帮助、趋于平稳，还是会产生负面影响？
2. **中等。** 实现 MQA（多查询注意力，Multi-Query Attention，即所有查询头共享一个 KV 头）。测量与完整 MHA 相比参数量下降了多少。计算在推理阶段且 `N=2048` 时，KV 缓存（KV-Cache）大小缩减了多少。
3. **困难。** 实现一个微型的多头潜在注意力（Multi-Head Latent Attention）版本：将 K、V 压缩为秩为 `r` 的潜在表示（Latent Representation），将该潜在表示存入 KV 缓存中，并在计算注意力时进行解压。当 `r` 为何值时，缓存内存会降至完整 MHA 的 1/8 以下，同时验证集困惑度（Perplexity, PPL）的质量损失保持在 1 bit 以内？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 注意力头（Head） | “单个注意力电路” | 维度为 `d_head = d_model / n_heads` 的一组 Q/K/V 投影，各自拥有独立的注意力矩阵。 |
| 头维度（d_head） | “头维度” | 每个头的隐藏层宽度；在实际生产环境中几乎固定为 64 或 128。 |
| 拆分/合并（Split / combine） | “张量形状变换技巧” | 在注意力计算前后执行 `(N, d_model) ↔ (n_heads, N, d_head)` 的 reshape（重塑）与 transpose（转置）操作。 |
| 输出投影（W_o） | “输出投影” | 拼接所有注意力头后应用的 `(d_model, d_model)` 矩阵；各头在此处进行信息混合。 |
| 多查询注意力（MQA） | “单一 KV 头” | Multi-Query Attention（多查询注意力）：共享单个 K/V 投影。KV 缓存体积最小，但会伴随一定的质量损失。 |
| 分组查询注意力（GQA） | “Llama 2 以来的默认配置” | Grouped-Query Attention（分组查询注意力），满足 `n_kv_heads < n_heads`；通过重复 KV 投影来匹配 Q 的数量。 |
| 多头潜在注意力（MLA） | “DeepSeek 的独门技巧” | Multi-head Latent Attention（多头潜在注意力）：将 K、V 压缩为低秩潜在表示，在计算注意力时再进行解压。 |
| 归纳头（Induction head） | “上下文学习背后的电路” | 一对注意力头，负责检测先前出现过的 token 序列，并复制紧随其后的内容。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 多头注意力（Multi-Head Attention）机制的原始规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — 提出多查询注意力（Multi-Query Attention, MQA）的论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 介绍如何在训练后将多头注意力（Multi-Head Attention, MHA）转换为分组查询注意力（Grouped-Query Attention, GQA）。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — 详解多头潜在注意力（Multi-head Latent Attention, MLA）及其在缓存内存占用上优于 MHA/GQA 的原因。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 从机制可解释性角度剖析注意力头的实际作用。
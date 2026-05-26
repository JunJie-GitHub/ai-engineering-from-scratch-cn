# 推理优化（Inference Optimization）

> 大语言模型（Large Language Model, LLM）推理（Inference）由两个阶段构成。预填充（Prefill）阶段并行处理提示词（Prompt）——属于计算受限（Compute-bound）。解码（Decode）阶段逐个生成词元（Token）——属于内存受限（Memory-bound）。所有的优化手段都旨在解决其中一个或两个阶段的瓶颈。

**类型：** 实践构建
**语言：** Python
**前置条件：** 第10阶段，课程01-08（Transformer架构、注意力机制）
**时长：** 约120分钟

## 学习目标

- 实现键值缓存（KV-cache），以消除自回归（Autoregressive）词元生成过程中的冗余计算
- 解释大语言模型推理中的预填充与解码阶段，以及为何各自存在不同的瓶颈（计算受限与内存受限）
- 实现连续批处理（Continuous Batching）与分页注意力（PagedAttention）概念，以在并发请求下最大化GPU利用率
- 对比各项推理优化技术（KV缓存、投机解码（Speculative Decoding）、Flash Attention）及其在吞吐量（Throughput）与延迟（Latency）之间的权衡

## 问题背景

你在4块A100 GPU上部署了Llama 3 70B模型。单用户使用时，生成速度约为每秒50个词元。感觉很快。但当100个用户同时访问该端点时，吞吐量骤降至每人每秒3个词元。你每月25,000美元的GPU账单，换来的响应速度甚至比人类打字还慢。

模型本身在1个用户和100个用户之间没有任何变化。相同的权重、相同的架构、相同的数学运算。真正改变的是工作调度方式。朴素的推理（Naive Inference）会浪费90%以上的可用GPU算力。当用户等待第47个词元时，会占用整个批处理槽位，而GPU内存总线在矩阵乘法（MatMul）运算间隙处于空闲状态。与此同时，一个新用户的2000词元提示词本可以利用这段空闲时间进行有效计算。

这不是一个扩展性问题，而是一个调度问题。本课程介绍的技术——KV缓存、连续批处理、PagedAttention、投机解码、前缀缓存（Prefix Caching）——正是将每月2.5万美元的推理账单降至5000美元（在相同流量下）的关键所在。

在4块A100-80GB GPU上使用vLLM服务Llama 3 70B模型时，低并发下可实现约每人每秒50个词元的速度；通过连续批处理与PagedAttention，在100个并发请求下仍能维持每人每秒15-25个词元（TPS）的吞吐量。若没有这些优化，相同硬件在该并发量下仅能提供每人每秒5个词元的吞吐量。相同的GPU，相同的模型，吞吐量却提升了4倍。

## 核心概念

### 预填充（Prefill）与解码（Decode）

每个大语言模型（LLM）推理请求都包含两个截然不同的阶段。

**预填充（Prefill）**负责处理整个输入提示词（prompt）。由于所有 token 都是已知的，因此可以在整个序列上并行计算注意力机制（attention）。这是一个大规模的矩阵乘法运算——GPU 核心会保持满载。此阶段的瓶颈在于计算能力：即硬件每秒能执行多少次浮点运算（FLOPS）。一块 A100 可提供 312 TFLOPS（BF16 精度）。在单块 A100 上，对 70B 模型进行 4,096 token 提示词的预填充大约需要 400 毫秒。

**解码（Decode）**则逐个生成输出 token。每个新 token 都会关注所有先前 token 的键向量（key）和值向量（value），但每次前向传播（forward pass）仅生成一个 token。权重矩阵的大小与预填充阶段相同，但此时你是将它们与单个向量相乘，而非矩阵。GPU 核心在微秒级即可完成计算，随后便等待下一批权重从内存中加载。此阶段的瓶颈在于内存带宽：即你能以多快的速度将模型权重从高带宽内存（HBM）传输到计算单元。A100 的带宽为 2 TB/s。一个 FP16 精度的 70B 模型大小为 140 GB。完整读取一次模型需要 70 毫秒——这就是单次解码步骤的理论下限。

graph LR
    subgraph "Prefill (compute-bound)"
        P1["All prompt tokens"] --> P2["Parallel attention"]
        P2 --> P3["Full matmul utilization"]
    end

    subgraph "Decode (memory-bound)"
        D1["One token at a time"] --> D2["Sequential generation"]
        D2 --> D3["Waiting on memory reads"]
    end

    P3 --> D1

**ops:byte 比率**（也称为算术强度，arithmetic intensity）刻画了这种权衡关系。它衡量的是每从内存加载一个字节所执行的运算次数。

ops:byte ratio = FLOPs per token / bytes read from memory

在批量大小为 4,096 token 的预填充阶段，每加载一个权重，你会执行约 4,096 次乘加运算。此时比率很高——你处于计算受限（compute-bound）状态。而在批量大小为 1 的解码阶段，每加载一个权重仅执行约 1 次运算。此时比率很低——你处于内存带宽受限（memory-bound）状态。

核心洞察在于：*解码之所以受限于内存带宽，是因为你需要读取整个模型才能生成单个 token*。下文提到的每一项优化技术，要么减少读取的数据量，要么增加每次读取所处理的 token 批量大小，要么完全避免读取。

### KV 缓存（KV Cache）

在注意力计算过程中，每个 token 的查询向量（query）都会关注所有先前 token 的键向量（key）和值向量（value）。如果不使用缓存，生成第 N 个 token 就需要重新计算前 N-1 个 token 的键值投影。在生成 token 2 时会对 token 1 进行投影，生成 token 3 时再次投影，生成 token 4 时又一次投影。到了生成第 1,000 个 token 时，token 1 已经被投影了整整 999 次。

KV 缓存（KV Cache）会保存所有先前 token 的键值投影。在生成第 N 个 token 时，你只需计算 token N 的键和值，然后将它们与缓存中 token 1 到 N-1 的 K/V 进行拼接。

graph TD
    subgraph "Without KV Cache"
        A1["Token 5: recompute K,V for tokens 1-4"]
        A2["Token 6: recompute K,V for tokens 1-5"]
        A3["Token 7: recompute K,V for tokens 1-6"]
    end

    subgraph "With KV Cache"
        B1["Token 5: compute K5,V5, read K1-4,V1-4 from cache"]
        B2["Token 6: compute K6,V6, read K1-5,V1-5 from cache"]
        B3["Token 7: compute K7,V7, read K1-6,V1-6 from cache"]
    end

**KV 缓存的内存计算公式：**

KV cache size = 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_param

以 Llama 3 70B 为例（80 层，采用分组查询注意力 GQA 的 8 个 KV 头，head_dim=128，BF16 精度）：

per token: 2 * 80 * 8 * 128 * 2 bytes = 327,680 bytes = 320 KB
at 4,096 tokens: 320 KB * 4,096 = 1.28 GB
at 128K tokens: 320 KB * 131,072 = 40 GB

对于 Llama 3 70B 模型，单次 128K 上下文长度的对话将消耗 40 GB 的 KV 缓存——这相当于一块 A100 显存的一半。如果有 100 个并发用户，每人 4K token，仅 KV 缓存就需要 128 GB。这正是为什么 KV 缓存管理成为推理优化核心挑战的原因。

### 连续批处理（Continuous Batching）

静态批处理（Static batching）会等待凑齐 N 个请求后一起处理，并且必须等到*所有*请求都完成后才接受新请求。如果一个请求需要生成 500 个 token，而另一个只需要 10 个，那么短请求在完成后，将白白闲置 490 个解码步骤。

连续批处理（Continuous batching，也称为迭代级批处理）会在任意请求完成时，立即将新请求插入当前批次。系统会在每个解码步骤重新评估批次状态。一个在 10 个 token 后完成的请求，会立刻被等待中的新请求替换。

sequenceDiagram
    participant GPU
    participant R1 as Request 1 (50 tokens)
    participant R2 as Request 2 (10 tokens)
    participant R3 as Request 3 (30 tokens)
    participant R4 as Request 4 (waiting)

    Note over GPU: Static batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    Note over R2: Wasting 40 steps...
    Note over R3: R3 done at step 30
    Note over R3: Wasting 20 steps...
    GPU->>R4: Finally start R4 at step 50

    Note over GPU: Continuous batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    GPU->>R4: Insert R4 at step 11
    Note over R3: R3 done at step 30

吞吐量的提升幅度取决于输出长度的差异程度。如果长度一致，连续批处理的效果与静态批处理相当。但在长度可变的情况下（这是常见场景），由于 GPU 计算槽位永远不会空闲，连续批处理可带来 2 到 5 倍的吞吐量提升。

### 分页注意力（PagedAttention）

每个请求的 KV 缓存都是一块连续的内存区域。随着请求的不断到达和结束，内存会产生碎片——这与操作系统中的 RAM 碎片化现象完全一致。一个 4K token 的请求需要 1.28 GB 的连续内存。即使你总共有 2 GB 空闲内存，也可能无法提供 1.28 GB 的*连续空间*。你要么浪费内存，要么拒绝该请求。

PagedAttention（源自 vLLM）将操作系统风格的虚拟内存机制应用于 KV 缓存。它不再为每个请求分配一块连续内存，而是分配固定大小的“页”（page，通常每页包含 16 个 token）。这些页可以分散在物理 GPU 显存的任意位置。页表（page table）负责将每个请求的逻辑序列位置映射到物理页的位置。

graph TD
    subgraph "Contiguous allocation"
        C1["Request A: 2GB block"]
        C2["[free: 0.5GB]"]
        C3["Request B: 1GB block"]
        C4["[free: 1.5GB -- but fragmented]"]
    end

    subgraph "PagedAttention"
        P1["Page pool: 256 pages of 16 tokens each"]
        P2["Request A: pages 3,7,12,45,88..."]
        P3["Request B: pages 1,4,9,22,67..."]
        P4["No fragmentation, no waste"]
    end

PagedAttention 还支持共享前缀的**写时复制（copy-on-write）**机制。如果 50 个请求共享相同的系统提示词（system prompt），该提示词的 KV 缓存页只需存储一次，并由所有 50 个请求共同引用。只有当某个请求产生分支（例如用户输入不同的消息）时，才会为其分配独立的页。对于具有共享系统提示词的应用，这能大幅降低内存占用。

借助 PagedAttention，vLLM 报告其内存浪费率接近于零（约 4%，而朴素分配方式约为 60-80%）。

### 投机解码（Speculative Decoding）

解码速度慢是因为它是串行的——你生成一个 token，将其反馈回去，再生成下一个。但如果能以极低的成本先“猜”出接下来的 5 个 token，然后一次性验证它们呢？

投机解码（Speculative decoding）使用一个小型、快速的**草稿模型（draft model）**来生成 K 个候选 token。随后，大型**目标模型（target model）**在一次前向传播中处理所有 K 个候选 token（这看起来像预填充——并行、计算受限、高效）。如果目标模型同意草稿模型的预测，你就可以在相当于一次目标模型前向传播的时间内接受全部 K 个 token。如果目标模型在第 j 个位置产生分歧，则接受第 1 到 j-1 个 token，并丢弃剩余部分。

graph LR
    D["Draft model (1B)"] -->|"Generate 5 tokens<br/>~5ms"| C["Candidates: the cat sat on the"]
    C --> T["Target model (70B)"]
    T -->|"Verify all 5 in one pass<br/>~70ms"| V{"Match?"}
    V -->|"4 of 5 match"| A["Accept 4 tokens in 75ms<br/>vs 280ms sequential"]
    V -->|"Mismatch at pos 5"| R["Reject token 5<br/>Resample from target"]

加速效果取决于**接受率（acceptance rate）**——即草稿模型的预测与目标模型匹配的频率。在使用 Llama 3 8B 为 Llama 3 70B 生成草稿时，自然语言场景下的接受率通常在 70-85% 之间。这相当于解码速度提升 2 到 3 倍。

投机解码的三种主要方法：

| 方法 | 草稿来源 | 接受率 | 开销 |
|--------|-------------|-----------------|----------|
| 草稿-目标模型法（Leviathan 等） | 独立的小型模型 | 70-85% | 草稿模型内存 |
| EAGLE（Li 等） | 目标模型上的轻量级头部 | 75-90% | 约 1% 额外参数 |
| N-gram 查找 | Token N-gram 表 | 40-60% | 可忽略不计 |

**EAGLE** 在目标模型的隐藏状态之上训练了一个小型自回归头部（autoregressive head）。它利用目标模型倒数第二层的特征来预测下一个 token 的嵌入向量（embedding）。由于它直接基于目标模型自身的表征进行运算（而非独立模型），因此能以极少的额外内存实现更高的接受率。EAGLE-2 进一步引入了动态草稿树，可根据上下文动态调整候选 token 的数量。

**N-gram 投机解码**会维护一个基于当前上下文或预构建语料库的 N-gram 续写表。如果草稿匹配了同一对话中先前出现过的内容（如重复模式、代码、结构化输出），它将以零神经网络开销直接触发。虽然平均接受率较低，但每次推测的成本几乎为零。

投机解码在*数学上是精确的*——其输出分布与目标模型的分布完全一致。它并非近似方法。验证步骤确保了每一个被接受的 token，其概率都与目标模型原本赋予的概率完全相同。

### 前缀缓存（Prefix Caching）

许多请求共享相同的前缀。例如聊天机器人的系统提示词、检索增强生成（RAG）的上下文块，或少样本（few-shot）示例集。如果没有前缀缓存，每个请求都必须从头重新计算这些共享 token 的 KV 缓存。

前缀缓存（Prefix caching）会存储常见前缀的 KV 缓存，并在不同请求间复用。当带有已知前缀的新请求到达时，系统会直接复制（或引用）已缓存的 KV 条目，仅针对独有的后缀部分计算 KV。

对于一个所有请求共享的 2,000 token 系统提示词，前缀缓存可为每个请求节省约 400 毫秒的预填充时间。在每秒 100 个请求的负载下，这相当于每秒节省 40 秒的 GPU 计算时间——超过了一块 GPU 的算力。

SGLang 的 RadixAttention 使用基数树（radix tree，即字典树 trie）按 token 内容对前缀进行索引，从而实现前缀缓存。任何匹配已存储前缀的请求都能免费获得其 KV 缓存。该树结构支持部分前缀匹配——如果你与某个缓存条目共享了 2,000 个前缀 token 中的 1,500 个，系统会复用这 1,500 个，仅重新计算剩余的 500 个。

### 推理引擎（Inference Engines）

在生产环境的大语言模型服务中，三大推理引擎占据主导地位：

| 引擎 | 核心创新 | 适用场景 |
|--------|---------------|----------|
| vLLM | PagedAttention、连续批处理 | 通用服务，兼容性最高 |
| SGLang | RadixAttention（前缀缓存）、结构化生成 | 多轮对话机器人、受限解码 |
| TensorRT-LLM | NVIDIA 内核融合（kernel fusion）、FP8 量化 | NVIDIA 硬件上的单卡最大吞吐量 |

**vLLM** 是默认的起点。它支持最广泛的模型，可在任何 GPU 厂商（NVIDIA、AMD、Intel）的硬件上运行，并通过 PagedAttention + 连续批处理实现强大的吞吐量。其兼容 OpenAI 的 API 意味着你可以直接将其作为任何 OpenAI API 调用的替代品无缝接入。

**SGLang** 建立在 vLLM 相同的基础之上，但增加了用于前缀缓存的 RadixAttention 以及用于结构化 LLM 程序的领域特定语言（DSL）。如果你的工作负载涉及多轮对话、工具调用或受限解码（如 JSON 输出、正则表达式引导生成），SGLang 通常能通过前缀复用实现比 vLLM 高 2 到 5 倍的性能。

**TensorRT-LLM** 将模型编译为针对 NVIDIA GPU 优化的内核。它融合了多种算子（将注意力、线性层和激活函数合并到一个内核中），在 H100 GPU 上使用 FP8 精度，并与 NVIDIA Triton Inference Server 集成以支持生产部署。它在 NVIDIA 硬件上能实现最高的单卡吞吐量，但配置更为复杂，且仅支持 NVIDIA GPU。

Llama 3 70B 的实际性能数据（4x A100-80GB，BF16 精度）：

| 指标 | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|---------------|
| 吞吐量（1 用户） | ~50 TPS | ~55 TPS | ~65 TPS |
| 吞吐量（100 用户） | ~2,500 总 TPS | ~3,200 总 TPS | ~3,000 总 TPS |
| 首 token 延迟（Time to first token） | ~400ms | ~300ms（前缀命中） | ~350ms |
| 最大上下文 | 128K | 128K | 128K |

### Ops:Byte 框架

无法衡量，就无法优化。ops:byte 比率能告诉你当前是计算受限还是内存带宽受限，从而决定哪些优化手段真正有效。

Compute roof: peak FLOPS of the GPU
Memory roof:  peak bandwidth * ops:byte ratio

当 ops:byte 较低时（解码、小批量），你会触及内存带宽上限。增加更多算力（更高主频、更多核心）无济于事。你需要减少内存读取（如量化、KV 缓存压缩），或增大批量大小，将读取开销分摊到更多有效计算上。

当 ops:byte 较高时（预填充、大批量），你会触及计算上限。优化内存带宽无济于事。你需要更快的 GPU、内核融合或降低精度，以榨取更多的 FLOPS。

| 场景 | ops:byte | 瓶颈类型 | 优化手段 |
|----------|----------|-------|---------------|
| 预填充，batch=1 | ~4,096 | 计算受限 | 内核融合、FP8 |
| 解码，batch=1 | ~1 | 内存受限 | 量化、KV 压缩 |
| 解码，batch=32 | ~32 | 内存受限 | 增大批量、连续批处理 |
| 解码，batch=256 | ~256 | 过渡阶段 | 两者皆需 |
| 解码，batch=1024 | ~1,024 | 计算受限 | 内核融合、张量并行（tensor parallelism） |

在 A100 上，交叉点大约在 ops:byte = 156（312 TFLOPS / 2 TB/s）。低于 156 时，系统受限于内存带宽；高于 156 时，系统受限于计算能力。连续批处理通过在每次迭代中打包更多 token，将解码阶段推向这一交叉点。

## 构建

### 步骤 1：从零构建 KV 缓存 (KV Cache)

我们构建一个多头 KV 缓存（Multi-head KV Cache），用于按层和按注意力头（Attention Head）存储键（Key）与值（Value）投影，并展示其内存增长模式。

import numpy as np

class KVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, dtype=np.float16):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.k_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.v_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.seq_len = 0

    def update(self, layer_idx, new_keys, new_values):
        num_new = new_keys.shape[1]
        end = self.seq_len + num_new
        self.k_cache[layer_idx, :, self.seq_len:end, :] = new_keys
        self.v_cache[layer_idx, :, self.seq_len:end, :] = new_values
        return (
            self.k_cache[layer_idx, :, :end, :],
            self.v_cache[layer_idx, :, :end, :]
        )

    def advance(self, num_tokens):
        self.seq_len += num_tokens

    def memory_bytes(self):
        return self.k_cache.nbytes + self.v_cache.nbytes

    def used_bytes(self):
        per_token = 2 * self.num_layers * self.num_heads * self.head_dim * np.dtype(self.dtype).itemsize
        return per_token * self.seq_len

### 步骤 2：结合 KV 缓存的注意力机制 (Attention)

这是一个简化的多头注意力机制（Multi-head Attention），在解码（Decode）阶段利用 KV 缓存进行计算。

def scaled_dot_product_attention(query, keys, values):
    head_dim = query.shape[-1]
    scores = np.matmul(query, keys.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    seq_len_q = scores.shape[-2]
    seq_len_k = scores.shape[-1]
    if seq_len_q > 1:
        mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=np.float32), k=seq_len_k - seq_len_q + 1)
        scores = scores + mask * (-1e9)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, values)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

    def forward(self, x, kv_cache=None, layer_idx=0):
        batch, seq_len, d_model = x.shape
        Q = np.matmul(x, self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            K_full, V_full = kv_cache.update(layer_idx, K[0], V[0])
            K = K_full[np.newaxis, :, :, :]
            V = V_full[np.newaxis, :, :, :]
            if seq_len == 1:
                kv_cache.advance(1)

        attn_out = scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, -1, d_model)
        return np.matmul(attn_out, self.W_o)

### 步骤 3：连续批处理模拟器 (Continuous Batching Simulator)

本部分模拟静态批处理（Static Batching）与连续批处理（Continuous Batching）在调度策略上的差异。

import heapq

class Request:
    def __init__(self, request_id, prompt_tokens, output_tokens, arrival_step):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.arrival_step = arrival_step
        self.tokens_generated = 0
        self.start_step = None
        self.end_step = None

    def is_done(self):
        return self.tokens_generated >= self.output_tokens


def simulate_static_batching(requests, batch_size):
    step = 0
    completed = []
    queue = list(requests)
    queue.sort(key=lambda r: r.arrival_step)

    while queue:
        batch = []
        while queue and len(batch) < batch_size:
            r = queue.pop(0)
            r.start_step = max(step, r.arrival_step)
            batch.append(r)

        if batch:
            step = max(step, max(r.start_step for r in batch))
            max_output = max(r.output_tokens for r in batch)
            for r in batch:
                r.tokens_generated = r.output_tokens
                r.end_step = step + max_output
            step += max_output
            completed.extend(batch)

    return completed


def simulate_continuous_batching(requests, batch_size):
    step = 0
    completed = []
    queue = sorted(requests, key=lambda r: r.arrival_step)
    queue_idx = 0
    active = []
    waiting = []

    while queue_idx < len(queue) or active or waiting:
        while queue_idx < len(queue) and queue[queue_idx].arrival_step <= step:
            waiting.append(queue[queue_idx])
            queue_idx += 1

        while waiting and len(active) < batch_size:
            r = waiting.pop(0)
            r.start_step = step
            active.append(r)

        if not active:
            if waiting:
                step += 1
                continue
            elif queue_idx < len(queue):
                step = queue[queue_idx].arrival_step
                continue
            else:
                break

        for r in active:
            r.tokens_generated += 1

        done = [r for r in active if r.is_done()]
        for r in done:
            r.end_step = step + 1
            completed.append(r)
        active = [r for r in active if not r.is_done()]

        step += 1

    return completed


def batching_stats(completed):
    latencies = [r.end_step - r.arrival_step for r in completed]
    total_time = max(r.end_step for r in completed) - min(r.arrival_step for r in completed)
    total_tokens = sum(r.output_tokens for r in completed)
    return {
        "avg_latency": np.mean(latencies),
        "p50_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "total_time": total_time,
        "throughput": total_tokens / total_time if total_time > 0 else 0,
    }

### 步骤 4：前缀缓存 (Prefix Cache)

这是一个基于字典树（Trie）的前缀缓存，用于存储共享前缀对应的 KV 条目。

class TrieNode:
    def __init__(self):
        self.children = {}
        self.kv_data = None
        self.hit_count = 0


class PrefixCache:
    def __init__(self, max_entries=1000):
        self.root = TrieNode()
        self.max_entries = max_entries
        self.total_entries = 0
        self.hits = 0
        self.misses = 0

    def _walk(self, token_ids):
        node = self.root
        depth = 0
        for tid in token_ids:
            if tid not in node.children:
                break
            node = node.children[tid]
            depth += 1
        return node, depth

    def lookup(self, token_ids):
        node, depth = self._walk(token_ids)
        if depth > 0:
            self.hits += 1
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                current.hit_count += 1
            kv_entries = []
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                if current.kv_data is not None:
                    kv_entries.append(current.kv_data)
            return depth, kv_entries
        self.misses += 1
        return 0, []

    def insert(self, token_ids, kv_per_token):
        node = self.root
        for i, tid in enumerate(token_ids):
            if tid not in node.children:
                if self.total_entries >= self.max_entries:
                    return i
                node.children[tid] = TrieNode()
                self.total_entries += 1
            node = node.children[tid]
            if i < len(kv_per_token):
                node.kv_data = kv_per_token[i]
        return len(token_ids)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

### 步骤 5：投机解码模拟器 (Speculative Decoding Simulator)

我们模拟草稿-目标（Draft-Target）投机解码（Speculative Decoding）流程，并支持配置接受率（Acceptance Rate）。

class DraftModel:
    def __init__(self, vocab_size, acceptance_rate=0.8):
        self.vocab_size = vocab_size
        self.acceptance_rate = acceptance_rate

    def generate(self, context, num_tokens):
        tokens = np.random.randint(0, self.vocab_size, size=num_tokens)
        return tokens

    def get_probs(self, context, token):
        probs = np.random.dirichlet(np.ones(self.vocab_size))
        return probs


class TargetModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_probs(self, context, tokens=None):
        if tokens is not None:
            return [np.random.dirichlet(np.ones(self.vocab_size)) for _ in tokens]
        return np.random.dirichlet(np.ones(self.vocab_size))


def speculative_decode(draft_model, target_model, context, num_speculative=5,
                       draft_cost=1.0, target_cost=10.0, verify_cost=12.0):
    total_tokens = 0
    total_cost = 0.0
    accepted_counts = []
    context = list(context)

    max_tokens = 100

    while total_tokens < max_tokens:
        draft_tokens = draft_model.generate(context, num_speculative)
        total_cost += draft_cost * num_speculative

        target_probs = target_model.get_probs(context, draft_tokens)
        total_cost += verify_cost

        accepted = 0
        for i, token in enumerate(draft_tokens):
            draft_p = draft_model.get_probs(context + list(draft_tokens[:i]), token)
            target_p = target_probs[i]

            r = np.random.random()
            acceptance_prob = min(1.0, target_p[token] / (draft_p[token] + 1e-10))

            if r < draft_model.acceptance_rate:
                accepted += 1
                context.append(token)
                total_tokens += 1
            else:
                new_token = np.random.choice(draft_model.vocab_size, p=target_p)
                context.append(new_token)
                total_tokens += 1
                break

        accepted_counts.append(accepted)

        if accepted == num_speculative:
            bonus_probs = target_model.get_probs(context)
            bonus_token = np.random.choice(draft_model.vocab_size, p=bonus_probs)
            context.append(bonus_token)
            total_tokens += 1

    sequential_cost = total_tokens * target_cost
    return {
        "total_tokens": total_tokens,
        "speculative_cost": total_cost,
        "sequential_cost": sequential_cost,
        "speedup": sequential_cost / total_cost if total_cost > 0 else 1.0,
        "avg_accepted": np.mean(accepted_counts),
        "acceptance_rate": np.mean(accepted_counts) / num_speculative,
    }


def compare_speculation_strategies(vocab_size=1000, num_trials=20):
    results = {}

    for name, acceptance_rate, spec_tokens in [
        ("Draft-target (8B->70B)", 0.78, 5),
        ("EAGLE", 0.85, 6),
        ("N-gram", 0.50, 4),
        ("No speculation", 0.0, 0),
    ]:
        if spec_tokens == 0:
            results[name] = {
                "speedup": 1.0,
                "acceptance_rate": 0.0,
                "avg_accepted": 0.0,
            }
            continue

        trial_results = []
        for _ in range(num_trials):
            draft = DraftModel(vocab_size, acceptance_rate=acceptance_rate)
            target = TargetModel(vocab_size)
            context = list(np.random.randint(0, vocab_size, size=10))
            result = speculative_decode(draft, target, context, num_speculative=spec_tokens)
            trial_results.append(result)

        results[name] = {
            "speedup": np.mean([r["speedup"] for r in trial_results]),
            "acceptance_rate": np.mean([r["acceptance_rate"] for r in trial_results]),
            "avg_accepted": np.mean([r["avg_accepted"] for r in trial_results]),
        }

    return results

### 步骤 6：KV 缓存内存分析器 (KV Cache Memory Profiler)

计算真实模型配置下的 KV 缓存内存需求。

MODEL_CONFIGS = {
    "Llama-3-8B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 8, "gqa": True,
    },
    "Llama-3-70B": {
        "num_layers": 80, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 70, "gqa": True,
    },
    "Llama-3-405B": {
        "num_layers": 126, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 405, "gqa": True,
    },
    "Mistral-7B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 7, "gqa": True,
    },
    "GPT-4-est": {
        "num_layers": 120, "num_kv_heads": 96, "head_dim": 128,
        "model_params_b": 1800, "gqa": False,
    },
}


def kv_cache_memory(config, seq_len, dtype_bytes=2):
    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * dtype_bytes
    total = per_token * seq_len
    return {
        "per_token_bytes": per_token,
        "per_token_kb": per_token / 1024,
        "total_bytes": total,
        "total_mb": total / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
    }


def memory_budget(config, gpu_memory_gb, model_dtype_bytes=2, kv_dtype_bytes=2):
    model_memory_gb = config["model_params_b"] * 1e9 * model_dtype_bytes / (1024 ** 3)
    overhead_gb = gpu_memory_gb * 0.1
    available_for_kv = gpu_memory_gb - model_memory_gb - overhead_gb

    if available_for_kv <= 0:
        return {"error": "Model does not fit in GPU memory", "model_memory_gb": model_memory_gb}

    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * kv_dtype_bytes
    max_tokens = int(available_for_kv * (1024 ** 3) / per_token)

    return {
        "gpu_memory_gb": gpu_memory_gb,
        "model_memory_gb": round(model_memory_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "available_for_kv_gb": round(available_for_kv, 1),
        "max_total_tokens": max_tokens,
        "max_users_at_2k": max_tokens // 2048,
        "max_users_at_4k": max_tokens // 4096,
        "max_users_at_32k": max_tokens // 32768,
    }


## 使用方法

使用 vLLM：

from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=4,
    enable_prefix_caching=True,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain inference optimization in one paragraph."], params)

使用 SGLang 实现前缀缓存（Prefix Caching）与结构化输出（Structured Output）：

import sglang as sgl

@sgl.function
def classify(s, text):
    s += sgl.system("You are a classifier. Output JSON only.")
    s += sgl.user(f"Classify this text: {text}")
    s += sgl.assistant(sgl.gen("result", regex=r'\{"label": "(positive|negative|neutral)"\}'))

runtime = sgl.Runtime(model_path="meta-llama/Llama-3-70B-Instruct", tp_size=4)
sgl.set_default_backend(runtime)

results = classify.run_batch([
    {"text": "This product is amazing!"},
    {"text": "Terrible experience."},
    {"text": "It was okay I guess."},
])

使用 TensorRT-LLM：

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./llama-70b-trt-engine/", rank=0)

outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("Explain KV caching.")],
    max_new_tokens=256,
    temperature=0.7,
)

## 交付成果

本章节将产出：
- `outputs/skill-inference-optimization.md` —— 一份用于诊断和优化大语言模型（Large Language Model, LLM）推理服务（Inference Serving）的技能指南

## 练习

1. 修改键值缓存（KV Cache）分析器，以对比 FP16、FP8 与 INT4 的 KV 缓存量化（KV Cache Quantization）效果。针对上下文长度为 4K 的 Llama 3 70B 模型，计算在 4 张 A100-80GB GPU 上每种配置的最大并发用户数。将 KV 缓存量化至 INT4 应能使用户容量提升约 4 倍。

2. 扩展连续批处理（Continuous Batching）模拟器，以追踪 GPU 利用率（每步填充的批处理槽位比例）。针对 50 个输出长度服从帕累托分布（Pareto Distribution，shape=1.5, scale=20）的请求，绘制静态批处理与连续批处理随时间变化的利用率曲线。连续批处理应能维持 80% 以上的利用率。

3. 实现分组查询注意力（Grouped-Query Attention, GQA）版本的 KV 缓存，其中 `num_kv_heads < num_query_heads`。Llama 3 70B 使用 64 个查询头（Query Heads），但仅使用 8 个 KV 头。计算其相较于完整多头注意力（Multi-Head Attention）的内存节省情况（KV 缓存大小减少 8 倍）。

4. 构建一个采用最近最少使用（Least Recently Used, LRU）淘汰策略的前缀缓存。将 `max_entries` 设置为 500，并生成 1,000 个请求，其中 60% 的请求共享 5 个常见前缀之一。测量缓存命中率（Hit Rate），并与无限容量缓存进行对比。在良好的淘汰策略下，命中率应保持在 55% 以上。

5. 扩展投机解码（Speculative Decoding）模拟器，以实现基于树的投机机制（EAGLE-2 风格）。不再使用单一的 K 个草稿令牌（Draft Tokens）链，而是生成候选令牌树（例如，3 层结构每层 2 个分支 = 8 个叶子候选）。对比每轮验证中接受的令牌总数与线性投机解码的差异。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 预填充 (Prefill) | "处理提示词" | 并行计算所有输入 Token 的注意力机制——属于计算密集型 (compute-bound)，因为完整的矩阵乘法会让 GPU 核心保持满载。 |
| 解码 (Decode) | "生成 Token" | 每次前向传播 (forward pass) 生成一个 Token，且每次都需要读取完整的模型权重——属于内存密集型 (memory-bound)，因为计算完成时下一批权重尚未加载完毕。 |
| KV 缓存 (KV Cache) | "缓存注意力状态" | 存储所有历史 Token 的键 (Key) 和值 (Value) 投影，避免在每次解码步骤中重复计算——以内存空间换取计算时间。 |
| 连续批处理 (Continuous Batching) | "动态批处理" | 只要批次中有任何请求完成，就立即将新请求插入正在运行的批次中；该策略在每次解码迭代时进行评估，而非等待整个批次完成。 |
| 分页注意力 (PagedAttention) | "KV 缓存的虚拟内存" | 以固定大小的页（而非连续内存块）分配 KV 缓存，消除内存碎片化，并为共享前缀启用写时复制 (copy-on-write) 机制。 |
| 投机解码 (Speculative Decoding) | "起草与验证" | 使用快速草稿模型 (draft model) 一次性生成多个候选 Token，随后在目标模型 (target model) 的一次前向传播中统一验证——数学上完全等价，可实现 2-3 倍加速。 |
| EAGLE | "自投机解码" | 投机解码的一种变体，直接在目标模型自身的隐藏状态 (hidden states) 上训练一个轻量级预测头 (head)，相比使用独立的草稿模型，能获得更高的 Token 接受率。 |
| 前缀缓存 (Prefix Caching) | "复用系统提示词 KV" | 存储常见前缀（如系统提示词、少样本示例）已计算好的 KV 缓存条目，并在不同请求间复用，从而跳过冗余的预填充计算。 |
| 计算访存比 (Ops:Byte Ratio) | "算术强度" | 计算操作数与读取内存字节数的比值——用于判断工作负载是计算密集型（比值高）还是内存密集型（比值低）。 |
| 首字延迟 (Time to First Token, TTFT) | "TTFT" | 从接收请求到生成第一个输出 Token 的延迟——对于长提示词，该延迟主要由预填充时间决定。 |

## 延伸阅读

- Kwon 等人，《Efficient Memory Management for Large Language Model Serving with PagedAttention》（2023）—— vLLM 的论文，引入了分页 KV 缓存（Paged KV Cache）管理技术，现已成为推理服务（Inference Serving）的行业标准。
- Leviathan 等人，《Fast Inference from Transformers via Speculative Decoding》（2023）—— 该领域的基础性论文，证明了草稿-验证（Draft-Verify）推测机制能够生成与目标模型分布（Target Model Distributions）完全一致的结果，同时实现 2-3 倍的加速。
- Li 等人，《EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty》（2024）—— 该方法不再依赖独立的草稿模型（Draft Model），而是直接在目标模型自身的特征上训练预测头（Prediction Head），从而实现了更高的令牌接受率（Acceptance Rate）。
- Zheng 等人，《SGLang: Efficient Execution of Structured Language Model Programs》（2024）—— 引入了用于前缀缓存（Prefix Caching）的基数注意力（RadixAttention）机制，以及专为多次调用大语言模型程序设计的编程模型（Programming Model）。
- Williams 等人，《Roofline: An Insightful Visual Performance Model for Multicore Architectures》（2009）—— 屋顶线模型（Roofline Model）的原始论文，正式确立了“操作数:字节”（Ops:Byte）分析框架，用于评估计算瓶颈与访存瓶颈（Compute vs Memory Bottlenecks）。
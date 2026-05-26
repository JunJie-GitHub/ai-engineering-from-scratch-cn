# 梯度检查点（Gradient Checkpointing）与激活重计算（Activation Recomputation）

> 反向传播（Backpropagation）会保留每一个中间激活值（Activation）。在 700 亿参数和 128K 上下文长度的设定下，每个进程（Rank）的激活值将高达 3 TB。检查点技术以浮点运算次数（FLOPs）换取内存：通过重新计算来替代保存。核心问题在于应该丢弃哪些片段，而答案绝不是“全部丢弃”。

**类型：** 构建
**语言：** Python（含 numpy，可选 torch）
**前置知识：** 第 10 阶段第 04 课（预训练 Mini-GPT）、第 10 阶段第 05 课（扩展与分布式）
**时长：** 约 70 分钟

## 问题所在

训练 Transformer 时，模型会为每一层保存反向传播中需要求导的每个算子（Operator）的输入：包括注意力机制的输入、Q/K/V 投影、Softmax 输出、前馈神经网络（FFN）的输入、归一化（Normalization）输出以及残差流（Residual Stream）。对于隐藏层维度为 `d`、序列长度为 `L`、批次大小为 `B` 的层，每层所需的浮点数数量级约为 `12 * B * L * d`。

当 `d=8192, L=8192, B=1` 时，在 BF16 精度下每层占用约 800 MB。一个 64 层的模型将产生 51 GB 的激活值——这还没有乘以微批次（Microbatch）大小，没有加上注意力 Softmax 的中间结果（每个注意力头为 `L^2`），也没有考虑张量并行（Tensor Parallel）的部分副本开销。

现实的双重约束在于：BF16 权重加上优化器状态（Optimizer State）或许能塞进 80 GB 显存，但激活值会直接导致溢出。梯度检查点（Gradient Checkpointing，又称激活重计算 Activation Recomputation）是标准的解决方案。其做法是丢弃大部分激活值，在反向传播时重新执行前向传播（Forward Pass）以恢复它们。代价是额外的计算量（FLOPs），收益是内存占用将按检查点分段数与总层数的比例下降。

若采用简单粗暴的实现方式，检查点技术会使每一步的前向传播计算量增加约 33%。但若优化得当——例如采用 Korthikanti 等人提出的“智能选择”策略进行选择性检查点——你只需承担不到 5% 的计算量开销，就能节省 5 倍的内存。在 FP8 矩阵乘法（FP8 Matmul）、FSDP 卸载（FSDP Offload）以及专家并行混合专家模型（Expert-Parallel MoE）的背景下，这一点至关重要：你既无法承受内存溢出，也负担不起计算资源的浪费。

## 核心概念

### 反向传播（Backward）的实际需求

`output = layer(input)`。反向传播需要计算 `grad_input` 和 `grad_params`。为此，它需要以下信息：

- `input`（用于线性层计算 `grad_params = input.T @ grad_output`）
- 部分激活函数导数的中间值（ReLU/GELU/softmax 的导数依赖于激活值本身）

前向传播（Forward Pass）会自动将这些信息存储在自动求导图（Autograd Graph）中。每次调用 `tensor.retain_grad()` 以及每个需要其输入的操作（Op）都会保留相应的引用。

### 朴素的全量检查点（Naive Full Checkpointing）

将网络划分为 `N` 个分段。在前向传播期间，仅保存每个分段的*输入*。当反向传播需要中间值时，重新运行该分段的前向传播以重新生成（Materialize）这些中间值，然后再进行微分计算。

示例：将一个 32 层的 Transformer 划分为 32 个分段，每个分段包含 1 层。

- 内存占用：32 个层输入（较小）对比 32 *（每层的激活值体积）（极大）。
- 额外计算量：每个分段增加 1 次前向传播，即总前向浮点运算次数（FLOPs）增加约 33%（由于反向传播的计算量是前向的 2 倍，完整训练步的计算单元从 1 + 2 = 3 变为 1 + 1 + 2 = 4）。

这是 Chen 等人 2016 年提出的原始方案：每隔 `sqrt(L)` 层设置一个检查点（Checkpoint），以平衡内存与计算开销。当 L=64 时，即设置 8 个检查点。

### 选择性检查点（Selective Checkpointing，Korthikanti 2022）

并非所有激活值（Activation）的存储成本都相同。注意力机制的 Softmax 输出维度为 `B*L*L*heads`，随序列长度呈*二次方*增长。而前馈神经网络（FFN）的隐藏层激活值维度为 `B*L*4d`，呈线性增长。对于长序列而言，Softmax 的内存占用占据主导地位。

选择性检查点策略会保留存储成本较低的激活值（如线性投影、残差连接），仅对成本较高的部分（如注意力机制）进行重新计算。你只需付出极少的额外 FLOPs 进行重算，即可节省 O(L^2) 级别的内存。

Megatron-Core 将其实现为“选择性”激活重计算。该策略已被广泛应用于 2024 年及以后的大多数前沿模型训练中。

### 卸载（Offload）

作为重计算的替代方案：在前向与反向传播之间，将激活值传输至 CPU 内存（RAM）。该方案依赖 PCIe 带宽；当空闲带宽高于重新生成激活值的计算成本时，收益显著。混合策略较为常见：对部分层使用检查点，对其他层使用卸载。

FSDP2 将卸载作为一等公民（First-class）选项提供。当 GPU 受限于内存瓶颈，但 CPU-GPU 数据传输仍有富余带宽时，卸载策略能发挥最大优势。

### 重计算成本模型

在 `L` 层网络中每隔 `k` 层设置一次朴素检查点时，单步的 FLOPs 计算如下：

flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%

采用选择性检查点时，你只需重计算注意力内核（Attention Kernel），而非整个层：

flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%

### 内存节省模型

每层的激活值体积为 `A`。对于 `L` 层网络，总激活值内存占用为 `L * A`。

全量检查点（分段大小为 1）：仅保存 `L * input_volume`（对于标准 Transformer 约为 `L * 1/10 A`）。可节省约 `9 * L * A * 1/10` 的内存。

每隔 `k` 层设置检查点：需保存 `L/k * A`，加上当前活跃分段内 `k-1` 层的激活值。

当 `k = sqrt(L)` 时，内存占用与重计算成本均按 `sqrt(L)` 缩放——这是针对计算成本均匀的层的最优权衡点。

### 何时不应使用检查点

- 流水线阶段（Pipeline Stage）中已在执行的最内层。无论如何它们都必须完成计算。
- 若首层和末层占据了该阶段的主要计算量（在 Transformer 中较为罕见）。
- 注意力内核已使用 FlashAttention 时——FlashAttention 本身已能快速重计算 Softmax，因此额外的层级别检查点带来的收益微乎其微。

### 实现模式

1. **函数包装器（Function Wrapper）：** 使用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装一个分段。PyTorch 仅保存 `input`，在反向传播时重新计算其余所有内容。
2. **基于装饰器（Decorator-based）：** 将层标记为可检查点；训练器在配置阶段决定哪些分段需要被包装。
3. **手动显式重计算：** 自行编写反向传播逻辑，调用自定义的 `recompute_forward` 函数，利用保存的输入重复执行前向传播。

这三种方式在功能上结果一致。使用包装器是业界的标准惯用法。

### 与张量并行（TP）/ 流水线并行（PP）/ FP8 的交互

- **张量并行（Tensor Parallel）：** 重计算时，检查点的输入必须进行 Gather 或重新 Scatter 操作；需妥善处理由此产生的通信开销。
- **流水线并行（Pipeline Parallel）：** 典型模式是对每个流水线阶段的前向传播设置检查点，以便逆序执行的微批次（Microbatch）能够复用激活值内存。
- **FP8 重计算：** 重计算过程中更新的 `amax` 历史记录必须与原始前向传播保持一致，否则会导致 FP8 缩放因子（Scale）发生漂移。大多数框架会对此缩放因子进行快照保存。

## 构建

### 步骤 1：带分段（Segments）的简易模型（Toy Model）

import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations

### 步骤 2：依赖全部激活值（Activations）的朴素反向传播（Naive Backward）

def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads

### 步骤 3：每 k 层检查点（Checkpoint-Every-k）的内存策略

def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads

### 步骤 4：成本模型（Cost Model）

def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }

### 步骤 5：内存估算器（Memory Estimator）

def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6

### 步骤 6：最优分段大小（Optimal Segment Size）

def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))

### 步骤 7：选择性检查点（Selective Checkpoint）决策

def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False


## 使用它

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch 中的标准包装器（wrapper）。它用于包装函数，仅保存输入数据，并在反向传播（backward pass）阶段重新计算激活值（activation）。
- **Megatron-Core 激活重计算（activation recomputation）**：支持 `selective`（选择性）、`full`（全量）和 `block`（分块）模式。这是 2024 年及以后前沿模型训练（frontier training）的标准配置。
- **FSDP2 卸载（offload）**：在 FSDP2 中结合 `offload_policy` 使用 `module.to_empty(device="cpu")`，可将激活值分片（shard）至 CPU，而非进行重新计算。
- **DeepSpeed ZeRO-Offload**：针对优化器状态（optimizer states）和激活值的 CPU 卸载方案，可作为检查点机制（checkpointing）的有效补充。

## 交付成果

本课时将生成 `outputs/prompt-activation-recompute-policy.md` 文件——这是一个提示词（prompt），它接收你的模型配置（层数、隐藏层维度、序列长度、批次大小）以及可用 GPU 显存，并输出逐层的重计算策略（无 / 选择性 / 全量 / 卸载）。

## 练习

1. 验证正确性。分别运行 `model_forward` + `model_backward`（保留完整激活值）与 `model_forward_checkpointed` + `model_backward_checkpointed`（分段计算）。参数梯度必须在机器精度（machine precision）范围内完全一致。

2. 将分段大小 `k` 从 1 遍历至 `L`。绘制浮点运算开销（FLOP overhead）与内存占用的关系曲线，并找出曲线的拐点（knee of the curve）。

3. 实现选择性检查点（selective checkpointing）：仅保存注意力模块（attention module）的输入，而不保存其中间计算结果。在序列长度 seq=8192 的 32 层模型上，对比测量其与全层检查点机制的浮点运算开销。

4. 添加卸载功能。将分段输入保存至模拟的“CPU 缓冲区”（一个独立的列表）。以“字节/时间”为单位测量 PCIe 带宽（PCIe bandwidth），并找出卸载与重新计算之间的盈亏平衡点（breakeven point）。

5. 对真实的 PyTorch Transformer 模型进行基准测试（benchmark），分别对比使用与不使用 `torch.utils.checkpoint` 的情况。测量显存占用（通过 `torch.cuda.max_memory_allocated`）以及单步训练时间。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 梯度检查点 (Gradient Checkpointing) | “通过重做前向传播节省显存” | 仅保存分段输入；在反向传播期间重新计算中间值，以获取计算梯度所需的张量 |
| 激活值重计算 (Activation Recomputation) | “与检查点技术相同” | 同一技术在高性能计算（HPC）领域的惯用名称 |
| 分段大小 (Segment Size, k) | “每个检查点覆盖的层数” | 中间值被丢弃并一同重新物化的层数 |
| 选择性检查点 (Selective Checkpointing) | “Korthikanti 的优化技巧” | 仅重计算存储开销大的激活值（如注意力 Softmax）；保留存储开销小的激活值 |
| 全量检查点 (Full Checkpointing) | “朴素实现版本” | 在每个分段内重新计算所有层的中间值 |
| 块检查点 (Block Checkpointing) | “粗粒度策略” | 以整个 Transformer 块为单位设置检查点；粒度最大 |
| FLOP 开销 (FLOP Overhead) | “算力税” | 单步额外 FLOP =（重计算 FLOP）/（前向 + 反向 FLOP）；朴素版约 33%，选择性版约 5% |
| 激活值卸载 (Activation Offload) | “转移至 CPU” | 在前向至反向传播阶段将激活值移至 CPU 内存；作为重计算的替代方案 |
| 平方根-L 法则 (sqrt-L Rule) | “经典最优间隔” | 对于计算成本均匀的层，最优检查点间隔为 sqrt(L) 层 |
| 注意力 Softmax 数据量 (Attention-Softmax Volume) | “O(L^2) 复杂度问题” | L^2 × 注意力头数 × 批次大小的浮点数；在长上下文场景下主导激活值内存占用 |

## 扩展阅读

- [Chen 等人, 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 正式提出梯度检查点技术的原始论文
- [Korthikanti 等人, 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 选择性激活值重计算及其形式化成本分析
- [Pudipeddi 等人, 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过反向模式重物化实现恒定内存占用的替代方案
- [Ren 等人, 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 大规模激活值卸载技术
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core 激活值重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- 涵盖选择性、全量及块模式
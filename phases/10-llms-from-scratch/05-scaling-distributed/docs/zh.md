# 规模化：分布式训练（Distributed Training）、FSDP 与 DeepSpeed

> 你的 1.24 亿参数模型可以在单张 GPU 上训练。现在试试 70 亿参数。模型无法装入内存。在单机上处理数据需要数周时间。在规模化场景下，分布式训练不再是可选项，而是唯一的前进道路。

**类型：** 实战
**语言：** Python
**前置要求：** 第 10 阶段，第 04 课（预训练迷你 GPT）
**时长：** 约 120 分钟

## 学习目标

- 解释三种并行方式：数据并行（Data Parallelism）、张量并行（Tensor Parallelism）和流水线并行（Pipeline Parallelism），并根据模型和集群规模说明各自适用的场景
- 使用 PyTorch DDP（Distributed Data Parallel）实现数据并行训练，并在多 GPU 间进行梯度（Gradients）同步
- 计算给定模型规模（权重 Weights + 优化器状态 Optimizer States + 梯度 Gradients + 激活值 Activations）的内存预算，以确定最低硬件需求
- 配置 FSDP（Fully Sharded Data Parallel）或 DeepSpeed ZeRO（Zero Redundancy Optimizer）阶段，将模型状态分片至多张 GPU，以容纳超出单卡显存的模型

## 问题背景

一个 70 亿（7B）参数的半精度浮点数（FP16）模型仅权重（Weights）就需要 14GB 显存。Adam 优化器（Adam Optimizer）会为每个参数额外保存两份副本（一阶和二阶动量估计）。这又增加了 28GB。反向传播过程中的梯度（Gradients）再增加 14GB。在存储任何激活值（Activations）之前，显存占用已达到 56GB。

NVIDIA A100 的显存为 80GB。

80GB 中已消耗 56GB，仅剩 24GB 用于存放激活值（Activations）——即前向传播过程中计算的中间值，反向传播时必须保留。对于序列长度为 2048 个词元（Token）、隐藏层维度为 4096 的模型，单层的激活值约占 64MB。32 层网络下，每个样本需要 2GB。批次大小（Batch Size）为 8 时需要 16GB。你还有 24GB 可用。但如果批次大小设为 12，显存就会溢出。

现在试试 700 亿（70B）参数。仅权重在 FP16 下就需要 140GB。单张 GPU 根本装不下。仅存放权重就至少需要 2 张 A100（2 x 80GB = 160GB）。再加上优化器状态和梯度，需求会大得多：至少需要 3 张以上 GPU，而根据分片策略的不同，实际通常需要 8 到 16 张。

Llama 3 405B 模型在 16,384 张 NVIDIA H100 GPU 上完成训练。该次训练的算力成本估计高达 1 亿美元。DeepSeek V3 通过巧妙的架构设计（混合专家模型 Mixture of Experts 意味着每个词元仅激活部分参数）和高效的训练策略，以约 560 万美元的成本训练出了同等规模的模型。

本课将介绍实现大规模训练的四种核心策略：数据并行、张量并行、流水线并行和完全分片数据并行（Fully Sharded Data Parallelism）。在接触任何分布式训练框架之前，你将使用纯 Python 模拟每种策略，以深入理解其底层机制。

## 核心概念

### 为什么需要分布式训练 (Distributed Training)

以下是真实模型的显存计算。所有数字均为精确计算，而非估算。

| 模型 | 参数量 (Params) | 权重 (FP16) | Adam 状态 (Adam States) | 梯度 (FP16) | 总计（不含激活值） |
|-------|--------|----------------|-------------|------------------|----------------------|
| GPT-2 Small | 124M | 248 MB | 992 MB | 248 MB | 1.5 GB |
| Llama 3 8B | 8B | 16 GB | 64 GB | 16 GB | 96 GB |
| Llama 3 70B | 70B | 140 GB | 560 GB | 140 GB | 840 GB |
| Llama 3 405B | 405B | 810 GB | 3,240 GB | 810 GB | 4,860 GB |

“Adam 状态”这一列是显存占用的主要瓶颈。Adam 优化器会为每个参数存储一个运行均值（m）和一个运行方差（v），两者均使用 FP32 格式。对于 70B 模型，计算为 70B x 4 字节 x 2 = 560GB。仅优化器状态就需要占用七张 A100 显卡的显存。

单张 H100 显卡仅有 80GB 显存。要容纳 Llama 3 405B 的权重、优化器状态和梯度，至少需要 61 张 H100。如果再加上激活值（activations），所需数量还会进一步增加。Meta 使用 16,384 张 GPU 并非出于自愿，而是迫于硬件限制的必要之举。

### 数据并行 (Data Parallelism)

这是最基础的分布式训练策略。将完整模型复制到 N 张 GPU 上。将每个训练批次（batch）均分为 N 份。每张 GPU 在其分配到的数据分片上执行前向传播（Forward Pass）和反向传播（Backward Pass）。反向传播结束后，对所有 GPU 的梯度进行平均。每张 GPU 使用相同的平均梯度更新其本地的权重副本，从而保持所有副本同步。

**优势：** 吞吐量呈线性扩展。N 张 GPU 每步可处理 N 倍的数据。通信仅限于梯度平均操作，且该操作可与计算过程重叠。

**劣势：** 每张 GPU 都必须保存完整的模型、优化器状态和梯度副本。对于 70B 模型，单张 GPU 需要 840GB 显存。数据并行无法降低单卡显存占用，它只能缩短训练时间。

**计算示例：** 有效批次大小（Effective Batch Size）= 单卡批次大小 x N。当 N=64 张 GPU 且单卡批次为 16 时，有效批次为 1,024。Llama 3 在训练时每步使用的有效批次大小为 1600 万个 token。

graph TD
    subgraph DataParallel["Data Parallelism (N=4 GPUs)"]
        B["Full Batch\n(1024 samples)"] --> S["Split"]
        S --> G1["GPU 1\nFull Model Copy\n256 samples"]
        S --> G2["GPU 2\nFull Model Copy\n256 samples"]
        S --> G3["GPU 3\nFull Model Copy\n256 samples"]
        S --> G4["GPU 4\nFull Model Copy\n256 samples"]
        G1 --> AR["AllReduce\nAverage Gradients"]
        G2 --> AR
        G3 --> AR
        G4 --> AR
        AR --> U["Update\n(identical on all GPUs)"]
    end

    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style G1 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G3 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G4 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style AR fill:#1a1a2e,stroke:#51cf66,color:#fff
    style U fill:#1a1a2e,stroke:#51cf66,color:#fff

### 张量并行 (Tensor Parallelism)

将单个网络层拆分到多张 GPU 上。一次矩阵乘法（Matrix Multiplication）被分配到多张 GPU 执行，每张 GPU 计算结果的一部分。

以全连接层（Feedforward Layer）中形状为 (8192, 8192) 的权重矩阵为例。采用 4 路张量并行时，每张 GPU 仅持有 (8192, 2048) 的分片。每张 GPU 将输入与其持有的分片相乘，得到部分结果。这些部分结果随后通过集合通信操作（如全规约 all-reduce 或全收集 all-gather）进行合并，生成完整输出。

**优势：** 降低模型权重的单卡显存占用。将 70B 模型拆分到 8 张 GPU 上，意味着每张 GPU 仅需存储约 8.75B 参数对应的权重。

**劣势：** 每个网络层计算后都需要高速的 GPU 间通信。每次矩阵乘法后的 all-reduce 操作会增加延迟。该策略在配备 NVLink（同一节点内 GPU 间带宽达 900 GB/s）的环境下表现优异，但在通过 InfiniBand 互联的跨节点环境（400 Gb/s，约 50 GB/s）中表现较差。因此，张量并行几乎总是被限制在单个节点内（通常为 8 张 GPU）。

**实际应用：** Megatron-LM 率先引入了张量并行技术。Llama 3 405B 在每个节点内采用了 8 路张量并行。

### 流水线并行 (Pipeline Parallelism)

按网络层对模型进行拆分。GPU 1 负责第 1-8 层，GPU 2 负责第 9-16 层，GPU 3 负责第 17-24 层，GPU 4 负责第 25-32 层。数据沿流水线流动：GPU 1 计算完其负责的层后，将激活值发送给 GPU 2；GPU 2 计算完毕后发送给 GPU 3，依此类推。

**优势：** GPU 间通信量极小——仅需传输层边界的激活值，其数据量远小于梯度或权重。由于带宽需求较低，该策略非常适合跨节点部署。

**劣势：** 流水线气泡（Pipeline Bubbles）。当 GPU 4 正在对微批次 1 进行前向传播时，GPU 1、2 和 3 处于空闲状态（它们已完成各自部分的前向计算）。在反向传播阶段，空闲模式会反转。在朴素的流水线并行中，对于 N 个流水线阶段，GPU 利用率仅为 1/N。

**GPipe 和 PipeDream** 通过将批次拆分为微批次（Micro-batches）来解决气泡问题。GPU 1 一旦完成微批次 1 的前向传播，便立即开始处理微批次 2。这使得不同流水线阶段的计算得以重叠。当使用 M 个微批次和 N 个阶段时，气泡比例降至 (N-1)/M。例如，采用 M=16 个微批次和 N=4 个阶段时，气泡占比为 3/16 = 18.75% 的空闲时间。

### FSDP：完全分片数据并行 (Fully Sharded Data Parallel)

FSDP 结合了数据并行的可扩展性与分片（Sharding）的显存高效性。与每张 GPU 保存完整模型副本不同，FSDP 让每张 GPU 仅持有 1/N 的参数、梯度和优化器状态。

在某个层的前向传播开始前，FSDP 会执行 **all-gather** 操作，从所有 GPU 收集完整参数到当前 GPU 的显存中。前向传播结束后，每张 GPU 会丢弃非本地的参数。在反向传播阶段，再次执行 all-gather 以重建参数用于梯度计算。反向传播结束后，通过 **reduce-scatter** 操作分发梯度分片，使得每张 GPU 仅存储 1/N 的梯度。

**8 张 GPU 训练 70B 模型的显存计算：**

| 组件 | 未使用 FSDP | 使用 FSDP |
|-----------|-------------|-----------|
| 权重 (FP16) | 每张 GPU 140 GB | 每张 GPU 17.5 GB |
| Adam 状态 (FP32) | 每张 GPU 560 GB | 每张 GPU 70 GB |
| 梯度 (FP16) | 每张 GPU 140 GB | 每张 GPU 17.5 GB |
| **总计** | **每张 GPU 840 GB** | **每张 GPU 105 GB** |

不使用 FSDP 时，你根本无法将 70B 模型塞进单张 80GB 显存的 GPU。在 8 张 GPU 上使用 FSDP 时，单卡占用为 105GB——等等，这仍然放不下。你至少需要 16 张 GPU 才能将单卡占用降至 80GB 以下，或者将 FSDP 与激活检查点（Activation Checkpointing，在反向传播时重新计算激活值而非存储它们）结合使用。

由于每个层前都需要执行 all-gather，其通信开销高于原生数据并行。但显存的大幅节省使得原本无法进行的训练任务成为可能。

graph TD
    subgraph FSDP["FSDP: Fully Sharded Data Parallel (4 GPUs)"]
        direction TB
        S["Model: 4 layers, sharded"]

        subgraph GPU1["GPU 1"]
            G1S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU2["GPU 2"]
            G2S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU3["GPU 3"]
            G3S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU4["GPU 4"]
            G4S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end

        AG["All-Gather\n(reconstruct full params\nbefore each layer)"]
        FW["Forward Pass\n(full params temporarily)"]
        RS["Reduce-Scatter\n(distribute gradient shards\nafter backward)"]

        S --> GPU1
        S --> GPU2
        S --> GPU3
        S --> GPU4
        GPU1 --> AG
        GPU2 --> AG
        GPU3 --> AG
        GPU4 --> AG
        AG --> FW
        FW --> RS
    end

    style G1S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G3S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G4S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style AG fill:#1a1a2e,stroke:#e94560,color:#fff
    style FW fill:#1a1a2e,stroke:#51cf66,color:#fff
    style RS fill:#1a1a2e,stroke:#e94560,color:#fff

### DeepSpeed ZeRO

DeepSpeed 的 ZeRO（Zero Redundancy Optimizer，零冗余优化器）在概念上与 FSDP 完全一致，但由微软独立研发。它定义了三个阶段，分片策略逐级激进：

| 阶段 | 分片内容 | 显存节省 | 通信开销 |
|-------|--------|---------------|---------------|
| ZeRO-1 | 仅优化器状态 | 约降低 4 倍 | 与数据并行相同 |
| ZeRO-2 | + 梯度 | 约降低 8 倍 | 略高 |
| ZeRO-3 | + 参数 | 约降低 N 倍（N 为 GPU 数量） | 每层执行 all-gather |

ZeRO-3 与 FSDP 等效。两者命名不同，但底层机制一致。在 DeepSpeed 验证了该概念的可行性后，PyTorch 将其作为原生实现加入了 FSDP。

DeepSpeed 还推出了 ZeRO-Offload（将优化器状态卸载至容量更大且成本更低的 CPU 内存）和 ZeRO-Infinity（卸载至 NVMe SSD）。这些技术以计算速度换取显存容量——卸载操作的速度较慢，但能释放宝贵的 GPU 显存。

### 混合精度训练 (Mixed Precision Training)

现代模型训练会同时使用多种浮点数格式：

- **前向传播**：使用 FP16 或 BF16（16 位）。显存占用仅为 FP32 的一半。在 Tensor Core 上，矩阵乘法速度可提升 2 倍。
- **主权重（Master Weights）**：使用 FP32（32 位）。由优化器维护，以确保权重更新时的数值精度。
- **损失缩放（Loss Scaling）**：在反向传播前将损失值乘以一个较大的常数，防止 FP16 梯度下溢（Underflow）为零。在优化器更新步骤前，再除以相同的常数。

BF16（Brain Float 16）拥有与 FP32 相同的指数范围（8 位指数），但精度较低（7 位尾数，而 FP32 为 23 位）。由于它能表示与 FP32 相同的数值范围，因此极少需要损失缩放。FP16 具有 5 位指数和 10 位尾数——它能表示更精细的数值，但在极大或极小值时容易发生上溢或下溢。

Google 的 TPU 原生支持 BF16。NVIDIA 的 A100 和 H100 同时支持 FP16 和 BF16。业界已普遍转向 BF16，因为它彻底免去了损失缩放的繁琐配置。

**7B 模型的显存对比：**

| 精度格式 | 权重 | 优化器 | 梯度 | 总计 |
|-----------|---------|-----------|-----------|-------|
| 全 FP32 | 28 GB | 56 GB | 28 GB | 112 GB |
| 混合精度 (BF16 + FP32 主权重) | 14 GB | 56 GB | 14 GB | 84 GB |

混合精度为该模型节省了 28GB 显存。无论采用何种格式，优化器状态始终保持在 FP32——这也是显存消耗的主要来源。

### Megatron-LM 与 3D 并行 (3D Parallelism)

实际的大规模训练会结合上述三种并行策略：

- **数据并行**：跨节点组部署（扩大批次大小）
- **张量并行**：在单个节点内部署（将网络层拆分到 8 张 GPU）
- **流水线并行**：跨节点部署（将层组拆分到不同机器）

在 16,384 张 H100 上训练 Llama 3 405B：
- 每个节点内采用 8 路张量并行（每节点 8 张 GPU）
- 跨节点采用 16 路流水线并行（16 个流水线阶段）
- 剩余维度采用 128 路数据并行（16,384 / 8 / 16 = 128）

这种 3D 分解（8 x 16 x 128 = 16,384）正是将训练规模扩展至数千张 GPU 的核心方法。每张 GPU 处理不同的数据分片（数据并行），持有每个网络层的一个切片（张量并行），并计算不同的层组（流水线并行）。

DeepSeek V3 则采用了不同的路线。其混合专家（Mixture of Experts, MoE）架构在处理每个 token 时仅激活 671B 参数中的 37B。这意味着每张 GPU 只需计算（并存储激活值）被激活的参数。他们仅使用 2,048 张 H800 GPU（不到 Meta 用量的 1/8）完成了训练，成本约为 560 万美元，而 Meta 的预估成本高达 1 亿美元。

graph TD
    subgraph ThreeD["3D Parallelism (Llama 3 405B)"]
        direction TB
        subgraph DP["Data Parallel (128-way)\nSplit batch across 128 groups"]
            subgraph PP["Pipeline Parallel (16-way)\nSplit layers across 16 stages"]
                subgraph TP["Tensor Parallel (8-way)\nSplit each layer across 8 GPUs"]
                    G1["GPU 1\nSlice of layers 1-N"]
                    G2["GPU 2\nSlice of layers 1-N"]
                    G8["GPU 8\nSlice of layers 1-N"]
                end
            end
        end
    end

    N1["Total: 8 x 16 x 128 = 16,384 GPUs"]

    style G1 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G8 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style N1 fill:#1a1a2e,stroke:#e94560,color:#fff


## 构建

### 步骤 1：模拟数据并行 (Data Parallelism)

将一个批次 (batch) 的数据划分到多个模拟 GPU 上。每个 GPU 在其分配的数据分片 (shard) 上执行一次前向传播 (forward pass)。对“梯度” (gradients) 进行平均（此处我们使用损失值来模拟梯度）。

import numpy as np

def simulate_data_parallelism(data, num_gpus, model_fn):
    batch_size = len(data)
    shard_size = batch_size // num_gpus
    remainder = batch_size % num_gpus

    gpu_losses = []
    gpu_gradients = []

    offset = 0
    for gpu_id in range(num_gpus):
        extra = 1 if gpu_id < remainder else 0
        shard = data[offset:offset + shard_size + extra]
        offset += shard_size + extra

        loss, grad = model_fn(shard)
        gpu_losses.append(loss)
        gpu_gradients.append(grad)

    avg_loss = np.mean(gpu_losses)
    avg_gradient = np.mean(gpu_gradients, axis=0)

    return avg_loss, avg_gradient

全规约 (all-reduce，即对梯度求平均) 操作是数据并行中唯一的通信环节。在实际应用中，NVIDIA GPU 通常使用 NCCL 库来实现环形全规约 (ring all-reduce)：每个 GPU 将其 1/N 的梯度发送给相邻节点，同时从另一相邻节点接收 1/N 的梯度；经过 N-1 步后，每个 GPU 都获得了完整的平均值。总通信量为：2 × gradient_size × (N-1)/N，当 N 较大时，该值趋近于梯度大小的 2 倍。

### 步骤 2：模拟张量并行 (Tensor Parallelism)

将权重矩阵 (weight matrix) 拆分到多个 GPU 上。每个 GPU 计算部分矩阵乘法 (matrix multiplication)。最后将结果合并。

def simulate_tensor_parallelism(input_data, weight_matrix, num_gpus):
    d_in, d_out = weight_matrix.shape
    assert d_out % num_gpus == 0, f"d_out {d_out} not divisible by num_gpus {num_gpus}"
    shard_size = d_out // num_gpus

    partial_results = []
    for gpu_id in range(num_gpus):
        start = gpu_id * shard_size
        end = start + shard_size
        weight_shard = weight_matrix[:, start:end]

        partial = input_data @ weight_shard
        partial_results.append(partial)

    full_output = np.concatenate(partial_results, axis=-1)

    direct_output = input_data @ weight_matrix
    error = np.abs(full_output - direct_output).max()

    return full_output, error

误差应严格为零（或处于机器精度 (machine epsilon) 范围内）。张量并行在数学上是精确的——其计算结果与在单个 GPU 上执行完整矩阵乘法 (matmul) 完全一致。拆分是沿着输出维度 (output dimension) 进行的，因此每个 GPU 会生成不同的列块，通过拼接 (concatenation) 即可还原完整结果。

对于列并行 (column-parallel) 线性层（拆分输出维度），需要进行拼接操作。对于行并行 (row-parallel，拆分输入维度)，则需要进行求和操作。在 Transformer 的前馈神经网络 (FFN) 中，第一个线性层（扩展/expand）采用列并行，第二个线性层（压缩/contract）采用行并行。这种设计避免了在两个层之间执行全规约 (all-reduce) 操作。

### 步骤 3：模拟流水线并行 (Pipeline Parallelism)

将模型的层 (layers) 划分到多个虚拟 GPU 上。展示“气泡问题” (bubble problem)，即早期阶段在后期阶段计算时处于空闲状态。

def simulate_pipeline_parallelism(num_layers, num_stages, num_microbatches):
    layers_per_stage = num_layers // num_stages

    timeline = {}
    clock = 0

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            start_time = max(
                timeline.get((stage, mb - 1, "fwd"), (0, 0))[1] if mb > 0 else 0,
                timeline.get((stage - 1, mb, "fwd"), (0, 0))[1] if stage > 0 else 0,
            )
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "fwd")] = (start_time, end_time)

    last_fwd_end = max(v[1] for v in timeline.values())

    for mb in range(num_microbatches - 1, -1, -1):
        for stage in range(num_stages - 1, -1, -1):
            deps = [last_fwd_end]
            if mb < num_microbatches - 1 and (stage, mb + 1, "bwd") in timeline:
                deps.append(timeline[(stage, mb + 1, "bwd")][1])
            if stage < num_stages - 1 and (stage + 1, mb, "bwd") in timeline:
                deps.append(timeline[(stage + 1, mb, "bwd")][1])
            start_time = max(deps)
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "bwd")] = (start_time, end_time)

    total_time = max(v[1] for v in timeline.values())
    compute_time = num_microbatches * num_stages * layers_per_stage * 2
    bubble_fraction = 1.0 - compute_time / (total_time * num_stages)

    return timeline, total_time, bubble_fraction

当设置为 4 个阶段 (stages) 和 1 个微批次 (micro-batch) 时，气泡占比为 75%——即任意时刻 4 个 GPU 中有 3 个处于空闲状态。当微批次增加到 16 个时，该比例降至约 19%。消除气泡的代价是内存开销：你必须同时存储所有正在处理 (in-flight) 的微批次的激活值 (activations)。

### 步骤 4：内存计算器

精确计算训练任意规模模型所需的内存。

def memory_calculator(
    params_billions,
    precision_bytes=2,
    optimizer="adam",
    num_gpus=1,
    sharding="none",
    sequence_length=2048,
    batch_size_per_gpu=1,
    hidden_dim=None,
    num_layers=None,
):
    params = params_billions * 1e9

    weight_memory = params * precision_bytes

    if optimizer == "adam":
        optimizer_memory = params * 4 * 2
    elif optimizer == "sgd":
        optimizer_memory = params * 4
    else:
        optimizer_memory = 0

    gradient_memory = params * precision_bytes

    total_no_activation = weight_memory + optimizer_memory + gradient_memory

    if hidden_dim and num_layers:
        activation_per_layer = (
            sequence_length * batch_size_per_gpu * hidden_dim * precision_bytes * 4
        )
        activation_memory = activation_per_layer * num_layers
    else:
        activation_memory = params * precision_bytes * 0.5

    if sharding == "fsdp" or sharding == "zero3":
        weight_memory /= num_gpus
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero2":
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero1":
        optimizer_memory /= num_gpus

    per_gpu_total = weight_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        "params_billions": params_billions,
        "weights_gb": weight_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "gradients_gb": gradient_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "per_gpu_total_gb": per_gpu_total / 1e9,
        "total_across_gpus_gb": per_gpu_total * num_gpus / 1e9,
        "fits_on_80gb": per_gpu_total / 1e9 <= 80,
        "num_gpus": num_gpus,
        "sharding": sharding,
    }

该计算器回答了每位机器学习 (ML) 工程师都会问的问题：“我需要多少张 GPU？”输入模型规模，即可查看是否能够容纳。调整分片 (sharding) 策略，直到单张 GPU 的总内存占用降至 80GB 以下。

### 步骤 5：混合精度模拟

对比 FP32、FP16 与混合精度训练 (mixed precision training) 的内存使用情况。

def mixed_precision_comparison(params_billions):
    params = params_billions * 1e9

    fp32_weights = params * 4
    fp32_optimizer = params * 4 * 2
    fp32_gradients = params * 4
    fp32_total = fp32_weights + fp32_optimizer + fp32_gradients

    fp16_weights = params * 2
    fp16_master = params * 4
    fp16_optimizer = params * 4 * 2
    fp16_gradients = params * 2
    fp16_total = fp16_weights + fp16_master + fp16_optimizer + fp16_gradients

    mixed_weights = params * 2
    mixed_optimizer = params * 4 * 2
    mixed_gradients = params * 2
    mixed_total = mixed_weights + mixed_optimizer + mixed_gradients

    return {
        "fp32_total_gb": fp32_total / 1e9,
        "fp16_with_master_gb": fp16_total / 1e9,
        "mixed_bf16_gb": mixed_total / 1e9,
        "savings_vs_fp32": 1 - mixed_total / fp32_total,
    }

对大多数人来说最大的意外是：混合精度并不会将内存占用减半。无论采用何种精度，优化器状态 (optimizer states，如 Adam 的 m 和 v) 始终保持在 FP32 格式。对于 7B（70 亿参数）模型，FP32 训练需要 112GB 内存，而混合精度训练需要 84GB。这相当于减少了 25%，而非 50%。优化器状态占据了主导地位。

## 使用方法

### 运行所有模拟

def run_all_demos():
    print("=" * 70)
    print("DATA PARALLELISM SIMULATION")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.randn(64, 32)
    weight = np.random.randn(32, 16)

    def model_fn(batch):
        output = batch @ weight
        loss = np.mean(output ** 2)
        grad = 2 * batch.T @ (batch @ weight) / len(batch)
        return loss, grad

    for n_gpus in [1, 2, 4, 8]:
        loss, grad = simulate_data_parallelism(data, n_gpus, model_fn)
        print(f"  {n_gpus} GPUs: loss={loss:.4f}, grad_norm={np.linalg.norm(grad):.4f}")

    print()
    print("=" * 70)
    print("TENSOR PARALLELISM SIMULATION")
    print("=" * 70)

    x = np.random.randn(4, 8192)
    W = np.random.randn(8192, 8192)

    for n_gpus in [1, 2, 4, 8]:
        output, error = simulate_tensor_parallelism(x, W, n_gpus)
        print(f"  {n_gpus} GPUs: output_shape={output.shape}, max_error={error:.2e}")

    print()
    print("=" * 70)
    print("PIPELINE PARALLELISM SIMULATION")
    print("=" * 70)

    for n_mb in [1, 4, 8, 16, 32]:
        _, total_t, bubble = simulate_pipeline_parallelism(32, 4, n_mb)
        print(f"  {n_mb:2d} micro-batches: total_time={total_t:4d}, bubble={bubble:.1%}")

    print()
    print("=" * 70)
    print("MEMORY CALCULATOR")
    print("=" * 70)

    configs = [
        (7, "none", 1),
        (7, "fsdp", 8),
        (70, "none", 1),
        (70, "fsdp", 8),
        (70, "fsdp", 16),
        (405, "fsdp", 64),
        (405, "fsdp", 128),
    ]

    print(f"  {'Model':>8} {'Sharding':>8} {'GPUs':>5} {'Per-GPU':>10} {'Fits 80GB':>10}")
    print("  " + "-" * 50)
    for params, shard, gpus in configs:
        result = memory_calculator(params, num_gpus=gpus, sharding=shard)
        fits = "Yes" if result["fits_on_80gb"] else "No"
        print(f"  {params:>6}B {shard:>8} {gpus:>5} {result['per_gpu_total_gb']:>8.1f}GB {fits:>10}")

    print()
    print("=" * 70)
    print("MIXED PRECISION COMPARISON")
    print("=" * 70)

    for params_b in [7, 13, 70, 405]:
        result = mixed_precision_comparison(params_b)
        print(f"  {params_b}B: FP32={result['fp32_total_gb']:.0f}GB, "
              f"Mixed BF16={result['mixed_bf16_gb']:.0f}GB, "
              f"Savings={result['savings_vs_fp32']:.0%}")

## 交付上线

本课时将生成 `outputs/prompt-distributed-training-planner.md` 文件——这是一个提示词（Prompt），它接收模型规模与可用硬件配置，随后生成一份完整的分布式训练（Distributed Training）计划，涵盖并行策略（Parallelism Strategy）、内存预算（Memory Budget）、通信开销（Communication Overhead）以及预期吞吐量（Expected Throughput）。

## 练习

1. 修改显存计算器（Memory Calculator）以加入激活检查点（Activation Checkpointing）功能。启用检查点后，仅每隔 K 层保存一次激活值（通常 K=1，意味着全部重新计算）。展示显存与计算开销的权衡（Memory-Compute Tradeoff）：检查点能节省多少显存，又会使训练速度降低多少（完全检查点通常会增加约 33% 的计算量）？

2. 扩展流水线并行（Pipeline Parallelism）模拟程序，以实现 PipeDream 使用的 1F1B（一次前向，一次反向）调度（1F1B Schedule）。在 4 个阶段和 8 个微批次（Micro-batches）的配置下，将其气泡占比（Bubble Fraction）与朴素调度（Naive Schedule）进行对比。由于 1F1B 调度会更早启动反向传播，因此其峰值显存（Peak Memory）占用应更小。

3. 实现一个梯度累积（Gradient Accumulation）模拟器。不再在每个微批次后执行全归约（All-Reduce），而是先在本地累积 K 步的梯度，然后再执行全归约。展示该方法如何将通信量降低 K 倍，同时生成完全相同的最终梯度（从而保证训练结果一致）。

4. 构建一个成本估算器（Cost Estimator）。给定模型规模、目标 Token 数量、GPU 类型（A100 为 2 美元/小时，H100 为 3.50 美元/小时）以及并行策略（Parallelism Strategy），估算以美元计的总训练成本。使用已知成本进行验证：据报道，Llama 3 405B 的成本约为 1 亿美元，DeepSeek V3 的成本约为 560 万美元。

5. 在显存计算器中加入 ZeRO-Offload 功能。假设每个节点的 CPU 内存（CPU RAM）为 512GB，NVMe 存储为 2TB。展示将优化器状态（Optimizer States）卸载至 CPU 后，如何使 70B 参数的模型仅需 4 张 GPU 即可训练（而非 16 张），代价是优化器步骤（Optimizer Steps）的速度会减慢 30-50%。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 数据并行 (Data Parallelism) | “把模型复制到每张 GPU 上” | 每张 GPU 处理不同的数据分片；每个训练步后通过 all-reduce 操作对梯度进行平均 |
| 张量并行 (Tensor Parallelism) | “将单个层拆分到多张 GPU 上” | 对权重矩阵进行分区，使每张 GPU 仅计算矩阵乘法的一部分；需要高速 NVLink 互连 |
| 流水线并行 (Pipeline Parallelism) | “将不同层分配到不同 GPU 上” | 每张 GPU 负责运行一组不同的网络层；数据以微批次 (micro-batches) 形式流经流水线，以减少空闲气泡 |
| FSDP (Fully Sharded Data Parallel) | “把所有东西都切分” | 完全分片数据并行——每张 GPU 仅持有 1/N 的权重、梯度和优化器状态；在计算前通过 all-gather 操作重建完整参数 |
| ZeRO (Zero Redundancy Optimizer) | “DeepSpeed 版的 FSDP” | 零冗余优化器，包含三个阶段：分片优化器状态（阶段 1）、+ 梯度（阶段 2）、+ 参数（阶段 3） |
| All-reduce (全规约) | “在 GPU 间求平均” | 集合通信操作，使每张 GPU 最终都获得所有 GPU 输入数据的总和（或平均值）——通常以环形 all-reduce 方式实现 |
| All-gather (全收集) | “从所有 GPU 收集数据” | 集合通信操作，使每张 GPU 最终都获得所有 GPU 数据的拼接结果——在 FSDP 中用于重建完整参数 |
| Reduce-scatter (规约分散) | “求和并分发” | 集合通信操作，先对数据进行规约（求和），再将不同数据块分散到不同 GPU——在 FSDP 中用于梯度分片 |
| 混合精度 (Mixed Precision) | “用半精度训练” | 前向/反向传播使用 FP16/BF16，优化器状态使用 FP32——可节省约 25% 的显存（而非 50%），因为优化器状态占用了大部分内存 |
| 流水线气泡 (Pipeline Bubble) | “流水线中的空闲时间” | GPU 等待上一阶段数据而处于空闲状态的时间比例——通过增加微批次数量可有效降低该比例 |

## 延伸阅读

- [Rajbhandari 等人，2020 -- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054) -- DeepSpeed ZeRO 论文，正式定义了三种分片阶段
- [Shoeybi 等人，2020 -- "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"](https://arxiv.org/abs/1909.08053) -- NVIDIA 针对 Transformer 架构的张量并行实现
- [Narayanan 等人，2021 -- "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"](https://arxiv.org/abs/2104.04473) -- 结合数据并行、张量并行与流水线并行的三维并行 (3D Parallelism) 方案
- [Zhao 等人，2023 -- "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"](https://arxiv.org/abs/2304.11277) -- PyTorch 原生的 FSDP 实现
- [Llama 3 技术报告](https://arxiv.org/abs/2407.21783) -- 使用 16,384 张 GPU 进行训练的三维并行细节
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) -- 混合专家 (MoE) 架构如何将训练成本降低一个数量级
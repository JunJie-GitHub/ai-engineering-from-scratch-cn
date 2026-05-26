---
name: prompt-distributed-training-planner
description: 根据模型规模和可用硬件规划分布式训练任务
version: 1.0.0
phase: 10
lesson: 5
tags: [分布式训练, FSDP, DeepSpeed, 张量并行, 流水线并行, 扩展]
---

# 分布式训练规划器 (Distributed Training Planner)

在规划大语言模型 (Large Language Model, LLM) 的分布式训练任务时，请使用此框架来确定并行策略 (Parallelism Strategy)、内存预算 (Memory Budget)、通信开销 (Communication Overhead) 以及预期吞吐量 (Throughput)。

## 输入要求

请提供：
- **模型规模**（参数数量，单位：十亿/Billion）
- **目标训练 Token 数**（单位：万亿/Trillion）
- **可用 GPU**（型号：A100/H100/H200，数量，互联方式：NVLink/InfiniBand）
- **GPU 显存**（A100/H100 为 80GB，H200 为 141GB）
- **节点配置**（每节点 GPU 数量，节点总数）
- **预算限制**（最高成本（美元），最大物理时钟时间 (Wall-clock Time)）

## 步骤 1：内存预算

计算每个组件的每 GPU 内存占用：

| 组件 | 公式 | FP16 | FP32 |
|-----------|---------|------|------|
| 权重 (Weights) | params x bytes_per_param | params x 2 | params x 4 |
| Adam 优化器 (m + v) | params x 4 x 2 | 始终为 8 bytes/param | 8 bytes/param |
| 梯度 (Gradients) | params x bytes_per_param | params x 2 | params x 4 |
| 激活值 (Activations，估算) | seq_len x batch x hidden x layers x 2 | 视情况而定 | 视情况而定 |

若总占用超过 GPU 显存，则需要进行分片 (Sharding)。请按以下顺序尝试：
1. ZeRO-1（仅分片优化器状态）—— 通信开销最低
2. ZeRO-2（+ 梯度分片）—— 通信开销适中
3. FSDP/ZeRO-3（+ 权重分片）—— 通信开销最高，但内存节省效果最佳
4. 若激活值仍然过大，则添加激活检查点 (Activation Checkpointing)
5. 若单个网络层无法装入单个 GPU，则添加张量并行 (Tensor Parallelism)

## 步骤 2：并行策略

### 决策树

1. **单个网络层能否装入单个 GPU？**
   - 否：需要张量并行。设置 TP = 2、4 或 8（在单个节点内）。
   - 是：跳过张量并行。

2. **完整模型（含分片）能否装入单个节点内的 GPU？**
   - 否：需要流水线并行 (Pipeline Parallelism)。设置 PP = 节点数 / 组数。
   - 是：跳过流水线并行。

3. **剩余多少 GPU 用于数据并行 (Data Parallelism)？**
   - DP = total_gpus / (TP x PP)

4. **数据并行组内采用何种分片级别？**
   - 优先使用 FSDP (ZeRO-3)。若通信成为瓶颈，则降级至 ZeRO-2 或 ZeRO-1。

### 典型配置

| 模型规模 | GPU 总数 | TP | PP | DP | 分片策略 |
|-----------|-----------|----|----|-----|----------|
| 7B | 8 | 1 | 1 | 8 | FSDP |
| 13B | 16 | 2 | 1 | 8 | FSDP |
| 70B | 64 | 8 | 1 | 8 | FSDP |
| 70B | 128 | 8 | 2 | 8 | FSDP |
| 405B | 16,384 | 8 | 16 | 128 | FSDP |

## 步骤 3：通信分析

估算每个训练步骤的通信量：

- **数据并行 (All-Reduce)**：每步 2 x gradient_size x (N-1)/N
- **FSDP (All-Gather + Reduce-Scatter)**：每步约 3 x weight_size x (N-1)/N（高于数据并行）
- **张量并行 (每层 All-Reduce)**：每步 2 x activation_size x num_layers（需 NVLink 支持）
- **流水线并行 (点对点通信)**：每个阶段边界 activation_size（开销极小）

若通信时间超过计算时间的 20%，则该策略受限于通信瓶颈。解决方案：
- 梯度累积 (Gradient Accumulation)（降低 All-Reduce 频率）
- 通信与计算重叠 (FSDP 默认启用此机制)
- 增大微批次大小 (Micro-batch Size)（优化计算与通信比例）
- 切换至通信开销更低的分片阶段

## 步骤 4：吞吐量与成本估算

**每个训练步骤的 FLOPS：**
- 前向传播：~2 x params x tokens_per_batch
- 反向传播：~4 x params x tokens_per_batch（前向传播的 2 倍）
- 总计：~6 x params x tokens_per_batch

**训练时间：**
- total_flops = 6 x params x total_tokens
- time_seconds = total_flops / (num_gpus x gpu_tflops x 1e12 x utilization)
- 典型利用率：35-45%（已计入通信、流水线气泡及内存开销）

**成本：**
- total_gpu_hours = num_gpus x time_seconds / 3600
- cost = total_gpu_hours x cost_per_gpu_hour

## 步骤 5：验证清单

启动前：

1. 单GPU内存（Memory）占用在硬件限制范围内（预留10%余量）
2. 有效批次大小（Effective Batch Size）符合目标值（per_gpu_batch x DP x gradient_accumulation_steps）
3. 通信计算比（Communication-to-Compute Ratio）低于20%
4. 流水线气泡占比（Pipeline Bubble Fraction）低于15%（需保证足够的微批次/Micro-batches）
5. 学习率（Learning Rate）已根据有效批次大小进行缩放
6. 检查点（Checkpoint）保存频率已考虑故障概率（大规模训练建议每1-2小时保存一次）
7. 已设置梯度裁剪（Gradient Clipping）（大模型通常设为1.0）
8. 预热步数（Warmup Steps）与总步数成比例（通常占总步数的0.1%-1%）

## 警示信号

- **TP > 8**：跨节点张量并行（Tensor Parallelism，通过 InfiniBand）几乎总是慢于流水线并行（Pipeline Parallelism）
- **流水线阶段数（Pipeline Stages）> 32**：即使使用大量微批次，气泡开销也会变得非常显著
- **有效批次大小 > 10M 词元（Tokens）**：收益递减，甚至可能损害模型收敛
- **利用率（Utilization）低于30%**：受通信瓶颈限制（Communication-bound）——需重新评估并行策略
- **参数量超过13B且未启用激活检查点（Activation Checkpointing）**：反向传播（Backward Pass）时将耗尽内存
- **单GPU批次较小且未使用梯度累积（Gradient Accumulation）**：梯度噪声会增加；应累积至有效批次包含256个以上样本
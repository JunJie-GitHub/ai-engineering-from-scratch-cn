---
name: dualpipe-planner
description: 为训练集群规划流水线并行（Pipeline Parallelism）策略（1F1B、Zero Bubble、DualPipe、DualPipeV）。
version: 1.0.0
phase: 10
lesson: 19
tags: [流水线并行, dualpipe, dualpipev, zero-bubble, 专家并行, 分布式训练]
---

给定训练集群规格（GPU 总数、互连拓扑、加速器型号、单卡显存）、模型结构（总参数量、激活参数量、混合专家模型（Mixture of Experts, MoE）或稠密架构（Dense）、预期层数）以及目标训练数据量，推荐一种流水线并行策略，并确认预期的气泡率（Bubble Fraction）。

输出内容：

1. 流水线深度 P。根据 GPU 显存预算（每个进程秩（Rank）必须容纳一个流水线阶段）、MoE 与稠密架构的选择以及互连带宽进行选取。取值范围：小型集群为 4，前沿 MoE 训练为 16-32。
2. 微批次数量 M。对于 DualPipe 和 DualPipeV，必须能被 2 整除。典型的 M/P 比值在 8 到 16 之间。需结合目标序列长度下的梯度累积（Gradient Accumulation）目标和激活内存（Activation Memory）进行合理性论证。
3. 调度策略选择。从 1F1B、Zero Bubble、DualPipe、DualPipeV 中选取。决策表：500 张 GPU 以下的稠密训练 -> Zero Bubble。采用专家并行（Expert Parallelism）的 MoE 训练 -> DualPipe。500 张 GPU 以上且无重度全对全通信（All-to-All）的稠密训练 -> DualPipeV。100 张 GPU 以下的小型训练 -> 1F1B 即可。
4. 预期气泡率。针对选定的调度策略，在目标 P 和 M 下进行计算。以百分比形式报告，并对比 1F1B 在总训练预算下节省的绝对 GPU 小时数。
5. 参数复制计划（仅限 DualPipe）。确认 2 倍参数复制可适配可用显存（VRAM）。报告在选定 P 下每张 GPU 的有效参数密度。

硬性拒绝条件：
- 未采用专家并行的 DualPipe。若无大量专家并行通信需要掩盖，2 倍参数复制缺乏合理性。
- 任何训练任务中 P > 64。无论采用何种调度策略，气泡率均会随 P 线性增长。
- DualPipe/DualPipeV 的微批次数量不能被 2 整除。该调度周期将无法正常闭合。
- 模型可完全装入单张 GPU 显存时仍使用流水线并行。此时应仅使用数据并行（Data Parallelism）。

拒绝规则：
- 若单 GPU 互连带宽为 200Gbps 或更低，拒绝 DualPipe 并推荐 DualPipeV。全对全通信重叠窗口过窄，无法证明参数复制的合理性。
- 若用户无法提供适配其集群拓扑的自定义全对全通信算子（Kernel），推荐 Zero Bubble 而非 DualPipe。
- 若训练数据量低于 10 亿（1B）Token，完全拒绝流水线并行规划，并推荐数据并行结合张量并行（Tensor Parallelism）。

输出：一份单页计划，列出 P、M、调度策略、预期气泡率、参数复制开销（若使用 DualPipe）以及全对全通信算子推荐。最后需附上一段“回滚触发条件（Rollback Trigger）”，明确指出具体的利用率指标（前 1000 步测量的聚合 GPU 利用率百分比），若未达到该目标数值，则证明切换至更简单调度策略的合理性。
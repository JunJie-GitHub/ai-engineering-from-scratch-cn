# DualPipe 并行 (DualPipe Parallelism)

> DeepSeek-V3 在 2,048 张 H800 GPU 上进行训练，其混合专家模型（Mixture of Experts, MoE）的专家（Experts）分散在不同节点上。跨节点的专家全对全通信（all-to-all communication）开销极大，每 1 GPU 小时的计算就需要消耗 1 GPU 小时的通信时间，导致 GPU 有一半时间处于空闲状态。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线（bidirectional pipeline），它将前向（forward）与后向（backward）计算与其触发的全对全通信进行重叠。流水线气泡（pipeline bubbles）随之减少，吞吐量得以提升；而在专家并行（Expert Parallelism）本身已将专家分散至各个进程（ranks）的情况下，保留两份模型参数副本（即名称中“Dual/双”的由来）的额外开销也变得微乎其微。本教程将以“学习（Learn）”模式逐步拆解 DualPipe 的实际工作机制，并探讨为何 Sea AI Lab 的 DualPipeV 改进版能够以略微收紧流水线气泡为代价，彻底消除 2 倍的参数复制成本。

**Type:** 学习（Learn）
**Languages:** Python（标准库、调度模拟器）
**Prerequisites:** 第 10 阶段 · 05（分布式训练、FSDP、DeepSpeed），第 10 阶段 · 14（开源模型架构与 MoE）
**Time:** 约 60 分钟

## 学习目标

- 指出 DualPipe 前向-后向计算块（forward-backward chunk）的四个组成部分，并解释为何每个部分都需要独立的计算-通信重叠窗口（overlap window）。
- 解释大规模训练中的流水线气泡（pipeline bubble）问题，并阐明“无气泡（bubble-free）”在实际工程与市场营销中的含义差异。
- 手动推演 8 个流水线并行（Pipeline Parallelism, PP）进程与 16 个微批次（micro-batches）的 DualPipe 调度过程，验证前向与反向数据流如何相互填补空闲时间片。
- 说明 DualPipeV（Sea AI Lab，2025 年）所做的权衡：在专家并行未激活时，以略微收紧流水线气泡为代价，省去 2 倍的参数复制开销。

## 问题背景

在 2,000 张 H800 GPU 上训练 671B 参数的 MoE 模型时，会面临三个相互叠加的瓶颈：

1. **显存压力（Memory pressure）。** 每张 GPU 仅持有模型的一部分切片。在 128 个注意力头（heads）、61 层网络且序列长度（sequence length）为 8k 的情况下，激活值（activations）占用的显存极为庞大。
2. **流水线气泡（Pipeline bubbles）。** 传统的流水线并行（Pipeline Parallelism，如 GPipe、1F1B）在等待当前阶段输入或梯度时，会导致 GPU 处于空闲状态。即使采用 1F1B 调度策略，在 8 个阶段（stages）的配置下，仍可能有约 12% 的 GPU 时间消耗在气泡上。
3. **跨节点全对全通信（Cross-node all-to-all）。** 结合专家并行的 MoE 架构会将专家分散到不同节点。每次前向传播（forward pass）都会触发一次全对全通信以将 Token 分发至对应专家，并在计算结束后再次触发以聚合结果。在 2,000 张 GPU 的规模下，计算与通信的时间比极易达到 1:1。

上述问题各自已有独立的解决方案：针对显存压力可采用梯度检查点（gradient checkpointing），针对流水线气泡可采用 Zero Bubble 算法（Sea AI Lab，2023 年），针对全对全通信可优化专家并行通信内核（expert-parallel comm kernels）。而 DualPipe 的核心价值在于将这些技术协同整合。其调度策略在单个前向-后向计算块内实现计算与通信的重叠，同时从流水线两端注入微批次，并利用由此生成的调度表，将全对全通信完全隐藏于计算窗口之内。

实测结果：流水线气泡近乎消除，在 DeepSeek-V3 处理 14.8T Token 的训练任务中，GPU 利用率超过 95%。

## 核心概念

### 流水线并行（Pipeline Parallelism）回顾

将一个包含 N 层的模型划分到 P 个设备上。设备 `i` 负责第 `i * N/P` 层到第 `(i+1) * N/P - 1` 层。一个微批次（micro-batch）会依次从设备 0 到 P-1 进行前向传播（forward pass），然后从 P-1 到 0 进行反向传播（backward pass）。每个设备只有在前一个设备发送其输出后才能开始前向阶段，并且只有在下游设备发送上游梯度后才能开始反向阶段。

GPipe（Huang 等人，2019）每次仅调度一个微批次，这会浪费大量 GPU 时间。1F1B（Narayanan 等人，2021）对多个微批次的前向和反向传播进行交错调度。Zero Bubble（Qi 等人，2023）将反向传播拆分为两部分——针对输入的反向传播（B）和针对权重的反向传播（W）——并通过调度它们来填补流水线气泡（bubble）。在 Zero Bubble 之后，流水线几乎达到了紧凑状态。

DualPipe 是下一步的演进。它在上述基础上引入了两个核心思路：

### 思路一：块分解（Chunk Decomposition）

每个前向计算块（forward chunk）被拆分为四个组件：

- **注意力机制（Attention）。** Q/K/V 投影、注意力计算、输出投影。
- **All-to-all 分发（All-to-all dispatch）。** 跨节点通信，负责将 token 发送至对应的专家（expert）。
- **多层感知机（MLP）。** 混合专家模型（MoE）的专家计算。
- **All-to-all 合并（All-to-all combine）。** 跨节点通信，负责将专家的输出结果汇总返回。

反向计算块（backward chunk）则包含上述各组件对应的梯度计算版本。DualPipe 的调度策略使得：当前块的 all-to-all 分发与下一个块的注意力计算并行执行，而 all-to-all 合并则与再下一个块的 MLP 计算并行执行。

### 思路二：双向调度（Bidirectional Scheduling）

大多数流水线调度方案仅从阶段 0 注入微批次，并使其流向阶段 P-1。DualPipe 则从**两端**同时注入微批次。阶段 0 会看到从自身发起的前向微批次；阶段 P-1 同样会看到从自身发起的前向微批次。这两股数据流在中间汇合。

为实现这一机制，设备 `i` 必须同时持有流水线前端的第 `i` 层和流水线后端的第 `P - 1 - i` 层。这正是 DualPipe 中“双（Dual）”的含义：每个设备保留其所需服务模型层的两份副本（每个方向各一份）。在 DeepSeek-V3 的规模下，这相当于 2 倍的参数复制开销。但这种开销是完全可以承受的，因为专家并行（Expert Parallelism）已经将 MoE 专家分散得非常稀疏，将非专家层复制两份的代价相比之下微不足道。

关键在于，一个方向的前向流与另一个方向的反向流，恰好重叠在单向调度中原本会产生气泡的位置。气泡由此消失。

### 手动推演的调度示例

假设 P = 4 个设备 rank（rank），共 8 个微批次，分为 4 个前向 / 4 个反向。时间从左向右推进；每一行代表一个设备 rank。

           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...

解读“F4/F5R”标记：rank 1 在同一时间片内，同时执行微批次 4 的前向传播（在流水线中从左向右）和微批次 5 的前向传播（从右向左）。这就是“双向”在实际操作中的含义。

在 rank 2 处，交叉流会更早重叠；而在 rank 0 和 P-1 处，重叠发生得最晚。在调度的稳定中间阶段，每个 rank 都在执行 X 方向的前向计算与 Y 方向的反向计算的重叠运行。计算单元始终保持忙碌。前向传播所需的 all-to-all 分发被隐藏在反向计算中，而 all-to-all 合并则被隐藏在前向计算中。流水线气泡被彻底挤出。

### 气泡（Bubble）核算

标准 1F1B 流水线的气泡（每个 rank 浪费的时间）：

bubble_1F1B = (P - 1) * forward_chunk_time

Zero Bubble 的优化降低了气泡时间，但无法降至零。DualPipe 在稳定阶段，若微批次数量能被流水线深度的 2 倍整除，则气泡时间为零。在稳定阶段之外（预热和冷却期），虽然存在一定的空闲时间，但它不会随微批次数量的增加而增长——这是论文强调的一个关键特性。

用市场术语来说就是“无气泡（bubble-free）”。用技术术语来说：气泡不随微批次数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明，只有在专家并行不是瓶颈时，才能实现完全的空闲时间为零；在由 EP 驱动的 all-to-all 通信下，调度上总会存在一定的妥协。

### DualPipeV —— 优化版本

Sea AI Lab（2025）指出，当 EP 通信重叠并非核心目标时，2 倍的参数复制是一种浪费。他们提出的 DualPipeV 调度方案将双向注入折叠为一种“V 形”调度，仅需单份参数副本即可运行。其气泡时间略大于 DualPipe，但内存节省效果显著。DeepSeek 在其开源的 DualPipe 实现中采用了 DualPipeV，作为关闭 EP（EP-off）模式下的选项。

权衡对比：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每个设备的参数副本数 | 2 | 1 | 1 | 1 |
| 气泡 vs 微批次数量 | 恒定 | 小幅增长 | 增长 | 增长 |
| 计算与通信重叠度 | 完全重叠 | 部分重叠 | 极低 | 部分重叠 |
| 适用场景 | 重度依赖 EP 的 MoE | 稠密模型或轻量 EP | 基线方案 | 任意流水线 |

### 对 14.8T token 训练任务的意义

DeepSeek-V3 的预训练在 2,048 块 H800 GPU 上消耗了约 280 万 GPU 小时，处理了 14.8T 个 token。若采用基础的 1F1B 方案，流水线气泡将导致 12-15% 的算力损失——相当于 34 万至 42 万 GPU 小时，这足以完整训练一个 70B 参数的模型。DualPipe 挽回了其中的大部分损失。在没有内部日志的情况下很难直接量化其具体贡献，但论文声称在整个训练过程中，平均 GPU 利用率超过了 95%。

对于较小规模的训练（少于 1000 块 GPU），DualPipe 显得大材小用——流水线气泡占总成本的比例较小，且稠密模型训练很少触及 all-to-all 通信瓶颈。但在数千块 GPU 规模的前沿 MoE 训练中，它几乎是必不可少的。

### 在技术栈中的定位

- 与 **FSDP**（第 10 阶段 · 05）互补。FSDP 负责在多个 rank 间对模型参数进行分片（sharding）；DualPipe 负责在多个 rank 间调度计算。两者可结合使用。
- 兼容 **ZeRO-3** 梯度分片。双副本复制的元数据管理需要与 ZeRO 的分片梯度机制协同工作。
- 需要针对特定集群拓扑优化的**自定义 all-to-all 内核（kernels）**。DeepSeek 开源的内核可作为参考实现。

## 使用

`code/main.py` 是一个流水线调度（pipeline schedule）模拟器。它接收参数 `(P, n_micro_batches, schedule)`，并输出 1F1B、Zero Bubble、DualPipe 和 DualPipeV 在稳定阶段（stable-phase）的利用率（utilization）。这是一个教学工具——其中的数值与论文中的定性结论相符，并非针对生产环境实测加速比（speedup）的声明。

该模拟器的价值在于：你可以使用不同的 P 和微批次（micro-batch）数量运行它，观察 1F1B 的气泡比例（bubble fraction）如何增长，而 DualPipe 则不会。

实际训练运行中的集成注意事项：

- 选择一个能被微批次数量整除的流水线并行深度（pipeline-parallel depth）。
- 确保你的专家并行网格（expert-parallel mesh）支持双向全对全通信（bidirectional all-to-all）。DeepSeek 的算子内核（kernels）可作为参考实现。
- 首次集成时，预计需要花费一周时间专门调试调度逻辑本身。其中的状态记录（bookkeeping）非常繁琐。
- 监控每个进程（rank）的 GPU 利用率，而不仅仅是总体平均值。DualPipe 的优势正是来自于优化落后节点（stragglers）的利用率。

## 交付

本课时将生成 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑结构、互联方式、模型形状），它会推荐一种流水线并行（pipeline parallelism）策略、应使用的调度算法，以及在目标规模下预期的气泡比例。

## 练习

1. 使用 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 运行 `code/main.py`。计算 GPU 利用率差异，并将其换算为每训练百万 token 所挽回的 GPU 小时数。

2. 手动绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。在每个时间片（time slot）中标注微批次 ID 和方向。找出第一个不存在气泡的时间片。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）中的图 5。找出 DualPipe 前向计算块（forward chunk）内部全对全分发（all-to-all dispatch）的重叠窗口（overlap window）。解释计算调度（compute schedule）是如何将其隐藏的。

4. 计算 DualPipe 在 P=8 流水线阶段的 70B 稠密模型（dense model）和 P=16 流水线阶段的 671B 混合专家模型（MoE model）上的 2 倍参数开销。说明为何 MoE 场景下的开销比例更小（因为大部分参数属于专家网络，且已在较大的专家并行组（EP group）中进行了分片）。

5. 将 DualPipe 与 Chimera（2021 年提出的另一种双向调度器（bidirectional scheduler））进行对比。参考论文第 3.4 节，指出 DualPipe 新增而 Chimera 不具备的两个具体特性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 流水线气泡 (Pipeline Bubble) | “每个 Rank 的空闲时间” | 因流水线阶段等待输入或梯度而浪费的 GPU 计算周期 |
| 1F1B 调度 (One Forward One Backward) | “默认流水线调度” | 一次前向/一次反向交错调度；DualPipe 旨在超越的基线方案 |
| 零气泡 (Zero Bubble) | “Sea AI Lab 2023 方案” | 将反向传播拆分为 B（输入梯度）和 W（权重梯度）；几乎完全消除了流水线空闲 |
| DualPipe | “DeepSeek-V3 调度方案” | 双向流水线 + 计算与通信重叠；气泡时间不随微批次数量增加而增长 |
| DualPipeV | “对半切分 (Cut-in-half)” | V 形优化方案，以略微增大气泡为代价，省去了 2 倍的参数冗余复制 |
| 计算块 (Chunk) | “流水线工作单元” | 单个微批次在单个流水线阶段完成的一次前向或反向传播 |
| 全对全分发 (All-to-all Dispatch) | “将词元 (Token) 发送给专家” | 跨节点通信，负责将词元路由至其分配的混合专家模型 (MoE) 专家 |
| 全对全聚合 (All-to-all Combine) | “将专家输出传回” | 跨节点通信，负责在多层感知机 (MLP) 层后收集各专家的输出结果 |
| 专家并行 (Expert Parallelism, EP) | “专家分布在不同 GPU 上” | 将 MoE 专家分片至不同计算节点 (Rank)，使各 GPU 持有不同的专家模型 |
| 流水线并行 (Pipeline Parallelism, PP) | “模型层分布在不同 GPU 上” | 将模型层分片至不同计算节点；DualPipe 进行调度的维度 |
| 气泡占比 (Bubble Fraction) | “浪费的 GPU 时间” | (气泡时间 / 总时间)；DualPipe 致力于将该比例趋近于零 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 技术报告 (arXiv:2412.19437)，第 3.3.2 节与图 5](https://arxiv.org/abs/2412.19437) — DualPipe 的主要参考文献
- [DeepSeek — DualPipe GitHub 仓库](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包含 DualPipeV（对半切分）模式
- [Qi 等人 — 零气泡流水线并行 (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 的前身工作
- [Sea AI Lab — 去掉 Dual 的 DualPipe 可能更好](https://sail.sea.com/blog/articles/63) — 为 DeepSeek 的 EP-off 模式提供依据的 DualPipeV 分析
- [Narayanan 等人 — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 对比的 1F1B 调度方案原始文献
- [Huang 等人 — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 流水线并行原始论文及气泡问题探讨
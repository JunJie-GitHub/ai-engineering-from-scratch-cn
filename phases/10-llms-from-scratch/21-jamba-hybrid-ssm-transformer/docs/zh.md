# Jamba — 混合 SSM-Transformer

> 状态空间模型（State Space Models, SSMs）与 Transformer 追求的目标截然不同。Transformer 通过注意力机制（Attention）以二次方成本换取生成质量；SSM 则通过递归机制（Recurrence）实现线性时间推理和恒定内存占用，但在质量上有所欠缺。AI21 于 2024 年 3 月发布的 Jamba 和 2024 年 8 月发布的 Jamba 1.5 将两者融合于同一模型中：每 7 个 Mamba 层搭配 1 个 Transformer 层，每隔一个模块应用混合专家模型（Mixture of Experts, MoE），并支持 256k 上下文窗口（Context Window），且能完整部署在单张 80GB GPU 上。Mamba-3（ICLR 2026）通过引入复数值状态空间（Complex-valued State Spaces）和多输入多输出投影（Multiple-Input Multiple-Output, MIMO Projections）进一步收紧了 SSM 端。本课程将端到端解析这两种架构，并阐明为何在纯 SSM 和纯 Transformer 的长上下文尝试均告失败的情况下，这种混合配方能在三年的模型扩展中存活下来。

**Type:** 学习
**Languages:** Python（标准库，层混合计算器）
**Prerequisites:** 第 10 阶段 · 14（开放模型架构），第 10 阶段 · 17（原生稀疏注意力）
**Time:** 约 60 分钟

## 学习目标

- 解释 Jamba 模块中的三个基本组件——Transformer 层、Mamba 层和 MoE——以及 1:7 比例与偶数模块交错的架构设计。
- 概述 SSM 递归机制的高层工作原理，以及它为何能实现恒定内存推理。
- 计算 Jamba 模型在 256k 上下文下的键值缓存（KV Cache）占用空间，并与纯 Transformer 模型的需求进行对比。
- 列举 Mamba-3 的三项创新（指数梯形离散化、复数值状态更新、MIMO），并说明每项创新所针对的问题。

## 核心问题

注意力机制的计算复杂度随序列长度呈二次方增长，而状态空间模型仅为线性增长。这种差异会随着规模扩大而急剧放大：在 256k token 的序列中，Transformer 每个注意力头的注意力图包含 650 亿个条目；而 SSM 的递归状态大小是固定的，与序列长度无关。

纯 SSM 模型（如 Mamba、Mamba-2）在小规模下能达到与 Transformer 相当的困惑度（Perplexity），但在状态跟踪任务上表现落后，且在部分上下文内检索（In-context Retrieval）类别中会失效。其直观原因在于：SSM 将历史信息压缩为固定大小的状态，当历史过长时，信息会发生泄漏。注意力机制能够精确记住所有内容，但需付出二次方计算成本。

显而易见的解决方案是：两者兼用。在需要精确回忆的位置部署 Transformer 层，在其他位置使用 SSM 层，并动态调整两者的比例。Jamba 是首个以生产级规模落地该混合配方的模型（总参数量 52B，激活参数量 12B，支持 256k 上下文，单张 80GB GPU 即可运行）。Jamba 1.5 将该系列扩展至总参数量 398B / 激活参数量 94B。Mamba-3（ICLR 2026）则是目前最优的纯 SSM 基线模型，混合架构可围绕其进行重构。

本课程将精读上述三篇论文，并为你构建关于“如何选择最佳比例”的认知模型。

## 核心概念

### 一页纸看懂状态空间模型 (State Space Model, SSM)

状态空间模型通过固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

h_t = A h_{t-1} + B x_t
y_t = C h_t

在每一步中，状态通过线性动力学 `A` 进行演化，接收输入 `B x_t`，并输出 `C h_t`。`A, B, C` 均可学习。请注意其关键特性：计算 `y_t` 仅需 `h_{t-1}` 和 `x_t`，无需任何更早的 `x`。内存占用是恒定的。每个 token 的推理复杂度为 O(1)。

提升建模质量的关键在于 `A` 的结构设计。S4 (Gu 2021) 采用了一种高度结构化的矩阵，在训练时可作为长卷积高效计算。Mamba (Gu, Dao 2023) 将固定的 `A, B, C` 替换为依赖数据的参数（即“选择性”部分）。Mamba-2 (2024) 进一步简化了该结构。Mamba-3 (2026) 则在特定位置重新引入了复杂性。

核心特性：对于解码器大语言模型 (Large Language Model, LLM) 而言，SSM 层可直接替换注意力层 (Attention Layer)，其每层状态大小固定，无需像 KV 缓存 (KV Cache) 那样随序列增长。

### Jamba 模块结构

Jamba 模块根据两个参数交错排列网络层：

- `l`：注意力层与 Mamba 层的比例。Jamba 采用 `l = 8`，即每 7 个 Mamba 层搭配 1 个 Transformer 层（每组共 8 层：7 Mamba + 1 Attention）。
- `e`：混合专家模型 (Mixture of Experts, MoE) 的应用频率。Jamba 采用 `e = 2`，即每隔一层应用一次 MoE。

模块内的层序列如下：

M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)

每个 Jamba 模块包含 8 层。当堆叠 4 个模块（共 32 层）时，将包含 28 个 Mamba 层和 4 个注意力层。其中 16 层使用了 MoE。

### 为何采用 1:7 的比例

AI21 进行了消融实验：在长上下文评估中，何种注意力与 Mamba 的比例能在“单位参数困惑度”和“上下文内召回能力”上取得最佳平衡？

- 注意力层过多（1:1）：模型质量提升，但内存占用增加且推理速度下降。
- 注意力层过少（1:15）：内存表现优异，但上下文内检索能力失效。
- 最佳平衡点：1:7 或 1:8。

直观理解：Transformer 层负责精确召回与状态追踪，而 Mamba 层则以极低的计算成本处理大部分常规信息。

### 位置编码

Mamba 层本身具备位置感知能力（通过循环机制实现）。早期基于 Mamba 的混合架构中，注意力层并未使用旋转位置编码 (Rotary Position Embedding, RoPE)，因为 SSM 层已提供位置信息。Jamba 1.5 为注意力层引入了 RoPE，以提升长上下文泛化能力，这是基于长上下文实证评估所做的后期优化。

### 内存预算

以 Jamba-1 的架构配置为例（32 层：28 Mamba + 4 Attention，隐藏层维度 4096，32 个注意力头）：

- KV 缓存（仅注意力层）：在 256k 上下文长度、BF16 精度下，`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。仅 4 个注意力层产生此开销。
- SSM 状态：每个 token 前缀的状态大小为 `28 * hidden * state_size`，但这是每层的固定大小，不随序列长度增长。典型的 Mamba 状态为每个特征 16 维，隐藏层 4096：总计 `28 * 4096 * 16 * 2 = 3.7 MB`。

对比同等配置（32 层、相同隐藏层维度、32 头全多头注意力机制 (Multi-Head Attention, MHA)）的纯 Transformer 模型：在 256k 上下文、BF16 精度下，KV 缓存高达 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。Jamba 将 KV 缓存缩减了 8 倍。即便与 2024 年主流模型采用的分组查询注意力 (Grouped-Query Attention, GQA(8)) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`）相比，Jamba 1:7 混合架构的 16 GB 缓存仍小了一半。

这正是 AI21 宣称“单张 80GB GPU 即可支持 256k 上下文”的含义。纯 Transformer 全 MHA 的 KV 缓存根本无法装入显存；即便是 GQA 基线，剩余空间也不足以容纳模型权重与激活值；而 Jamba 则可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3 (ICLR 2026, arXiv:2603.15569) 在纯 SSM 方向引入了三项创新：

1. **指数梯形离散化 (Exponential-trapezoidal discretization)**。用表达能力更强的循环机制替代了 Mamba-2 中的欧拉法离散化。在核心循环内部对状态与输入应用类卷积操作，而非在 `x_t` 外部进行卷积。
2. **复数值状态更新 (Complex-valued state update)**。此前的 Mamba 版本将状态矩阵从复数（S4）简化为实数对角阵（Mamba），再进一步简化为缩放单位阵（Mamba-2）。Mamba-3 重新引入复数值——这等价于对状态应用依赖数据的旋转嵌入。此举恢复了此前实数简化所牺牲的状态追踪能力。
3. **多输入多输出 (Multi-Input Multi-Output, MIMO) 投影**。摒弃逐特征的标量投影，改用矩阵值投影。在不增加解码延迟的前提下，提升了建模能力与推理时的硬件利用率。

在 1.5B 参数量下，Mamba-3 的平均下游任务准确率较 Gated DeltaNet 提升 0.6 个百分点；MIMO 变体再提升 1.2 个百分点，累计增益达 1.8 个百分点。在相同状态维度下，Mamba-3 仅需一半的状态大小即可达到 Mamba-2 的性能。

Mamba-3 尚未在大规模生产级混合架构中落地——但它无疑是下一代 Jamba 级模型中 SSM 组件的明确候选方案。

### 何时选择混合架构

混合架构在以下场景胜出：

- 上下文足够长，导致纯 Transformer 的 KV 缓存成为瓶颈（64k+）。
- 任务同时包含短程结构（SSM 擅长）与长程召回（需 Transformer 支持）。
- 希望在单 GPU 显存预算内部署，而纯 Transformer 的 KV 缓存本身就已超出显存限制。

混合架构在以下场景处于劣势：

- 上下文较短（16k 以下）。SSM 的额外开销成为浪费，纯 Transformer 已足够。
- 任务需要全局注意力（如深度推理、多文档交叉引用）。混合架构中注意力层的稀疏性会损害性能。
- 正在扩展至万亿参数级的前沿模型。目前，“纯 Transformer + 多头潜在注意力 (Multi-Head Latent Attention, MLA) + MoE”（DeepSeek-V3 风格）在能力竞赛中占据领先。

### 竞争格局

| 模型 | 架构家族 | 规模 | 核心优势 |
|-------|--------|------|-------------|
| Mamba-2 | 纯 SSM | 3B | 线性时间复杂度，恒定内存占用 |
| Jamba | 混合架构 | 52B/12B | 单卡 80GB 支持 256k 上下文 |
| Jamba 1.5 Large | 混合架构 | 398B/94B | 企业级长上下文支持 |
| Mamba-3 | 纯 SSM | 1.5B (论文) | 恢复状态追踪能力 |
| DeepSeek-V3 | 纯 Transformer + MoE | 671B/37B | 前沿模型能力 |

2026 年格局：纯 Transformer MoE 主导前沿能力，但混合架构牢牢占据 256k 以上长上下文细分市场。Mamba-3 在状态追踪上的突破，可能会在下一代模型中推动混合比例进一步降低（即增加 SSM 占比，减少注意力层）。

## 使用方法

`code/main.py` 是一个用于混合架构（hybrid architectures）的显存计算器。给定状态空间模型（SSM）与 Transformer 的比例，以及隐藏维度（hidden-size）/ 层数（layer-count）配置，它将计算：

- 目标上下文长度下的 KV 缓存（KV cache）。
- SSM 状态内存。
- 不同模型形状（model shapes）在上下文长度 N 下的总内存占用。

该计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格的 1:7 混合架构。
- 纯 SSM（完全无 KV 缓存）。

对于已发布的模型形状，数据直接取自 Jamba-1 和 Jamba-1.5 论文；对于假设的变体，则通过外推法得出。

实际部署时的集成注意事项：

- 大多数生产级推理服务器（如 vLLM、SGLang）均支持 Jamba 和 Mamba。请核对具体版本。
- 在 256k 上下文长度下，Jamba 的显存优势会体现在并发请求吞吐量（concurrent-request throughput）上。在相同的显存（VRAM）下，你能容纳的 Jamba 序列数量多于 Transformer 序列。
- Mamba-3 作为独立模型尚未投入生产环境使用——目前仅为 1.5B 参数的研究预览版。

## 交付成果

本课程的产出文件为 `outputs/skill-hybrid-picker.md`。给定工作负载规范（workload specification）（包含上下文长度分布、任务混合比例、内存预算），它会在纯 Transformer、Jamba 风格混合架构和纯 SSM 之间给出推荐，并明确阐述显存与模型质量之间的权衡（tradeoffs）逻辑。

## 练习

1. 运行 `code/main.py`，计算 32 层纯 Transformer（隐藏维度 4096，32 个注意力头）在 256k 上下文长度下的 KV 缓存，以及相同形状的 Jamba-1 混合架构的对应值。验证 AI21 论文中声称的约 8 倍显存缩减。

2. 修改计算器以模拟 1:3 混合架构（4 个 Mamba 层 : 1 个注意力层）和 1:15 混合架构（14 个 Mamba 层 : 1 个注意力层）。绘制 KV 缓存随比例变化的曲线。在何种比例下，KV 缓存大小等于 SSM 状态内存？

3. 阅读 Jamba 论文（arXiv:2403.19887）的第 3 节。解释为何尽管 Mamba-2 速度更快，AI21 仍选择使用 Mamba-1。提示：混合架构消融实验（hybrid ablation）部分对此有详细记录。

4. 计算 Jamba 1.5 Large（总参数 398B，激活参数 94B）中“每隔一层使用混合专家模型（MoE）”带来的参数开销（parameter overhead）。将其激活比例（active ratio）与 DeepSeek-V3（37B/671B）进行对比，并解释为何 Jamba 的架构能将激活比例推得更高。

5. 阅读 Mamba-3 论文（arXiv:2603.15569）的第 3 节。用三句话解释为何复数值状态更新（complex-valued state update）等价于数据依赖的旋转位置编码（data-dependent rotary embedding）。请将答案与第 7 阶段 · 第 04 课中的旋转位置编码（RoPE）推导过程联系起来。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 状态空间模型 (State Space Model, SSM) | “具有固定状态的循环机制” | 包含可学习循环公式 `h_t = A h_{t-1} + B x_t` 的网络层；每个词元 (token) 的内存占用为常数 |
| 选择性状态空间模型 (Selective SSM) | “Mamba 的核心技巧” | 数据依赖的 A、B、C 参数，使模型能够在线性时间内实现类似门控的选择性 |
| 注意力与 Mamba 层比例 (Attention-to-Mamba Ratio) | “注意力层的数量” | 在 Jamba 中，`l = 8` 表示每 7 个 Mamba 层配置 1 个注意力层 |
| Jamba 模块 (Jamba Block) | “8 层结构组” | 一个注意力层 + 七个 Mamba 层，并在交替位置应用混合专家模型 (Mixture of Experts, MoE) |
| SSM 状态 (SSM State) | “隐藏缓冲区” | 固定大小的逐层状态，用于替代 Mamba 层的键值缓存 (KV Cache) |
| 256k 上下文窗口 (256k Context) | “Jamba 的标志性指标” | Jamba-1 可在单张 80GB GPU 上处理的序列长度；同等规模下纯 Transformer 架构无法实现 |
| Mamba-3 | “2026 年纯 SSM 架构” | 当前最优的纯 SSM 架构，采用复数状态与多输入多输出 (Multi-Input Multi-Output, MIMO) 机制；主流混合架构均以其为基础进行重构 |
| 多输入多输出 (Multi-Input Multi-Output, MIMO) | “多输入多输出” | Mamba-3 的创新设计，使用矩阵值投影替代逐特征的标量计算 |
| 指数梯形离散化 (Exponential-Trapezoidal Discretization) | “Mamba-3 的循环机制” | 表达能力更强的循环机制，涵盖了 Mamba-2 的欧拉法离散化 |
| 混合架构 (Hybrid Architecture) | “混合注意力与 SSM” | 任何交错堆叠 Transformer 层与 SSM 层的模型；Jamba 是其中的工业级典范 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — Jamba 原始论文，包含比例消融实验与 256k 上下文窗口的主张
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 扩展版模型家族，公开了 398B/94B 与 12B/52B 参数版本
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 所基于的选择性 SSM 原始论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化版结构化状态空间模型的后续演进
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — 引入复数状态与 MIMO 机制，代表 2026 年纯 SSM 架构的前沿进展
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 原始论文，标志着 SSM 技术在大语言模型 (Large Language Model, LLM) 领域的起源
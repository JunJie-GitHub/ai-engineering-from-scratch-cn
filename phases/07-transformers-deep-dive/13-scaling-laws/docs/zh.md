# 缩放定律 (Scaling Laws)

> 2020 年的 Kaplan 论文指出：模型越大，损失 (Loss) 越低。2022 年的 Hoffmann 论文则指出：你之前的训练量不足。算力 (Compute) 主要分配在两个“桶”里——参数 (Parameters) 和词元 (Tokens)——而这两者的分配比例并不显而易见。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 7 阶段 · 05（完整 Transformer），第 7 阶段 · 07（GPT）
**Time:** 约 45 分钟

## 问题所在

当你拥有 C FLOPs 的训练算力并希望获得最佳模型时，你需要权衡两个调节旋钮：

1. **参数 (N) 的数量是多少？** 模型越大，模型容量 (Model Capacity) 越高。
2. **训练词元 (D) 的数量是多少？** 数据越多，对容量的利用越充分。

浮点运算次数 (FLOPs) 的缩放比例大致为 `6 × N × D`。你可以提高 N 并降低 D，或者提高 D 并降低 N。哪种更好？

在 2022 年之前，答案是“大幅增加 N”。GPT-3（2020 年）拥有 175B 参数，在约 300B 词元上进行训练。比例约为每个参数对应 1.7 个词元。Kaplan 缩放定律 (Kaplan Scaling Laws) 支持了这一做法。

Hoffmann 等人（2022 年）在训练一个名为 Chinchilla 的小型模型家族时，发现了不同的结论：最优比例更接近**每个参数对应 20 个词元**。GPT-3 的训练量仅为最优值的十分之一。Chinchilla（70B 参数，1.4T 词元）在各项基准测试 (Benchmark) 中均击败了 GPT-3（175B 参数，300B 词元），且推理成本 (Inference Cost) 降低了 2.5 倍。

2026 年依然是 Chinchilla 定律的天下——但有一个重要的转折。Llama 3 8B 在 15 万亿词元上进行了训练，比例达到每个参数对应 1,875 个词元。这已是 Chinchilla 最优比例的 94 倍。对于将大规模部署的模型而言，推理成本比训练成本更重要，因此为了获得更小的部署资源占用而进行超额训练（超越 Chinchilla 比例）已成为 2026 年的默认做法。

## 核心概念

![Chinchilla 曲线：不同 N/D 比例下的损失 (loss) 与计算量 (compute) 关系](../assets/scaling-laws.svg)

### Hoffmann 定律 (Hoffmann's Law)

根据 Chinchilla 论文，损失函数遵循以下公式：

L(N, D) = A / N^α + B / D^β + E

- `N` = 参数量 (parameters)（非嵌入层参数）。
- `D` = 训练词元 (training tokens)。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，即不可约损失上限 (irreducible loss ceiling)。
- `A ≈ 406`，`B ≈ 411`。

随着模型规模扩大，这两项会相互权衡。在固定计算量（C = 6ND）下对 `N` 求导并求解：

N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20

计算最优 (Compute-optimal)：每个参数对应 20 个词元。

### 为何要进行过训练 (Over-training)

Chinchilla 最优策略旨在最小化每次训练浮点运算 (FLOP) 的训练损失。但训练成本只需支付一次，而推理成本 (inference cost) 却是持续不断的。

对于每月处理万亿级词元的聊天机器人而言，推理成本占据了总成本的主导地位。Llama 的策略是：训练更小的模型，但使用更长的训练时间。在 15T 词元上训练的 8B 模型经过了深度的推理优化：

- 可部署于消费级 GPU。
- 延迟仅为 70B Chinchilla 最优模型的几分之一。
- 对于大多数任务而言，其质量已足够接近。

DeepMind 2024 年的论文《过训练即新最优》（"Over-training is the new optimal"）对此进行了形式化论证。对于以推理为主的工作负载，根据服务规模的不同，最佳比例更接近于每个参数对应 100–500 个词元。

### 涌现 (Emergence) 与平滑性

观点认为：某些能力（如算术、多步推理、思维链遵循）会在达到特定规模时突然“涌现”。

Schaeffer 等人（2023）指出，这实际上是一种测量假象 (measurement artifact)：涌现性指标采用了不连续的评分方式（如精确匹配、阈值准确率），从而掩盖了底层逻辑值 (logits) 的平滑提升。而连续指标（如交叉熵 (cross-entropy)）则呈现出平滑的曲线。

到 2026 年，业界共识已明确：基于连续损失函数的预测是可靠的。基准测试中的性能跃升往往是评分器带来的假象。在规划预算时，应以连续指标为依据。

### 2026 年的技术图景

缩放定律 (Scaling laws) 依然适用，但：

| 因素 | 变化方式 |
|--------|-------------|
| 数据质量 | 筛选“优质”词元（Phi 风格）可使曲线偏移，等效计算量提升超过 2 倍 |
| 混合专家模型 (MoE) | 总参数量与活跃浮点运算数解耦；缩放定律按活跃 FLOP 计算 |
| 后训练 (Post-training) | 部分能力（如指令遵循、代码生成）受 SFT+RLHF 的影响大于预训练阶段 |
| 多模态 (Multimodality) | 图像与文本词元共同缩放；各模态拥有独立的曲线 |
| 合成数据 (Synthetic data) | 模型自行生成训练数据；等效计算量可产生复利效应 |

Muon 优化器（Kimi Moonlight, 2024）在数据量相同的情况下，相较于 AdamW 实现了约 2 倍的等效计算量增益。部分 2026 年的训练任务已默认采用 Muon。它改变了缩放定律中的绝对常数，但并未改变其曲线形态。

## 动手构建

参见 `code/main.py`。我们实现了 Chinchilla 损失方程（Chinchilla loss equation），并在多个计算预算（compute budgets）下求解计算最优的 `(N, D)`。

### 步骤 1：Chinchilla 损失（Chinchilla loss）

def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E

在固定 `C = 6ND` 的条件下，将 `L` 绘制为 `(N, D)` 上的等高线图。寻找最小值。

### 步骤 2：计算最优前沿（compute-optimal frontier）

对于从 `1e17` 到 `1e25` FLOPs 的计算预算，在满足 `6ND = C` 的约束下，找到使损失最小化的 `(N, D)`。验证比例 `D/N ≈ 20`。

### 步骤 3：过训练成本（over-training cost）

计算训练一个缩小 10 倍的模型（最优 N 的 1/10，最优 D 的 10 倍）所付出的额外损失。报告由此换取的推理 FLOP 节省量（与 N 成正比）。

### 步骤 4：与真实模型对比

代入 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（活跃参数）已知的 `(N, D)` 对，并对比预测损失与报告损失。

## 实际应用

你不太可能亲自训练前沿模型（frontier model）。但缩放定律（scaling laws）会告诉你：

1. **你的微调数据是否充足。** 如果你的任务特定数据低于基础模型每个参数 20 个 token，预计损失会在某个下限处达到饱和。
2. **是否选择更大的基础模型。** 如果你的预算全部用于推理，应优先选择规模更小但训练更充分的模型。
3. **收益递减的临界点。** 超过 Chinchilla 最优值的 1000 倍后，对数损失（log-loss）的变化将沦为噪声。

**2026 年的研究轨迹：**

- **数据受限阶段（Data-constrained regime）。** 互联网上的高质量 token 数量是有限的（过滤后英文约 5–10 万亿）。前沿预训练正逼近这一上限。合成数据、多语言、多模态以及基于人类反馈的强化学习（RLHF）扩展的微调将是下一个发力点。
- **计算乘数技巧（Compute-multiplier tricks）。** Muon 优化器（Muon optimizer）、混合专家模型（Mixture of Experts, MoE）、更优的数据筛选——每一项改变的只是绝对常数，而非渐近线。
- **强化学习的缩放定律（Scaling laws for RL）。** 这是一个开放性问题。早期证据表明 RL 样本中存在幂律关系，但其指数与预训练截然不同。

## 部署上线

参见 `outputs/skill-training-budget-estimator.md`。该技能会根据计算预算、部署约束和目标损失，为新的训练运行挑选 `(N, D, hours, GPU)` 配置。

## 练习

1. **简单。** 运行 `code/main.py`。打印计算预算为 `1e20`、`1e22`、`1e24` 时的 Chinchilla 最优 `(N, D)`。与真实模型表格进行对比。
2. **中等。** 实现 Hoffmann 损失-计算量函数曲线。针对计算最优前沿，绘制损失与 `log10(C)` 的关系图。确定该定律预测我们需要多少 FLOPs（`>10^28`）才能实现交叉熵（cross-entropy）的下一个 0.1 降幅。
3. **困难。** 在相同数据集上训练的 5 个微型模型（10 万到 1000 万参数）上拟合你自己的缩放定律。估算 `α` 和 `E`。你的指数与已发表的数值匹配度如何？

## 关键术语

| 术语 | 业界常见说法 | 实际技术含义 |
|------|-----------------|-----------------------|
| 参数量 (Parameters, N) | “模型规模” | 非嵌入层 (Embedding) 权重数量；决定模型容量。 |
| 词元数 (Tokens, D) | “训练数据量” | 训练过程中见过的词元 (Token) 总数；决定参数被利用的充分程度。 |
| 计算量 (Compute, C) | “消耗的浮点运算次数” | 对于标准 Transformer 架构，约为 `6 × N × D`。 |
| Chinchilla 最优配比 (Chinchilla-optimal) | “D/N ≈ 20” | 使预训练阶段每单位浮点运算次数 (FLOP) 损失最小化的比例。 |
| 过训练 (Over-training) | “超越 Chinchilla 配比” | 消耗额外的训练 FLOP 以节省推理 FLOP；D/N >> 20。 |
| 不可约损失 (Irreducible loss) | “损失下限” | 缩放定律 (Scaling Law) 中的 `E` 项；即数据本身的熵。 |
| 涌现能力 (Emergent capability) | “规模扩大时的突变” | 通常是评估指标的人为假象；连续损失实际上是平滑变化的。 |
| 有效计算量 (Effective compute) | “训练效率乘数” | 更优质的数据/优化器/架构能放大单个 FLOP 的实际效用。 |

## 延伸阅读

- [Kaplan 等人 (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 首篇提出缩放定律 (Scaling Law) 的论文；模型处于欠训练 (Undertrained) 状态。
- [Hoffmann 等人 (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — 提出 Chinchilla 模型。
- [Schaeffer 等人 (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 指出涌现能力实为测量假象。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 解释为何 Llama 的过训练策略契合其实际工作负载。
- [Jordan 等人 (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 实现 2 倍计算效率乘数。
# 多智能体辩论与协作 (Multi-Agent Debate and Collaboration)

> Du 等人（ICML 2024，《Society of Minds》）运行 N 个模型实例，各自独立提出答案，随后经过 R 轮迭代相互批判以收敛至一致结论。该方法能有效提升事实准确性 (factuality)、规则遵循能力 (rule-following) 与推理能力 (reasoning)。在 Token 成本方面，稀疏拓扑 (sparse topology) 优于全连接网格 (full mesh)。

**类型：** 学习 + 构建
**编程语言：** Python (stdlib)
**前置条件：** 第 14 阶段 · 12（工作流模式 (Workflow Patterns)），第 14 阶段 · 05（自我优化 (Self-Refine) 与 CRITIC）
**预计耗时：** 约 60 分钟

## 学习目标

- 阐述辩论协议 (debate protocol)：N 个提议者，R 轮迭代，收敛至共享答案。
- 说明辩论为何能提升事实准确性、规则遵循能力与推理能力。
- 解释稀疏拓扑：并非每个辩论者都需要感知其他所有辩论者。
- 基于脚本化大语言模型 (scripted LLM) 使用标准库实现全连接网格与稀疏变体的辩论流程；对比测量 Token 成本与准确率。

## 问题背景

自我优化 (Self-Refine)（第 05 课）是单一模型对自身进行批判——容易陷入群体思维 (groupthink)。CRITIC（第 05 课）将批判建立在外部工具之上——但这些工具并非总是可用。辩论引入了第三种模式：多实例并行、交叉批判 (cross-critique)、基于分歧的收敛 (convergence by disagreement)。

## 核心概念

### 心智社会（Society of Minds，Du 等人，ICML 2024）

- N 个模型实例独立地对同一问题提出答案。
- 经过 R 轮交互，每个模型阅读其他模型的提议并进行批判性评估（critique）。
- 模型根据批判性反馈更新自身的答案。
- R 轮结束后，返回收敛的答案。

原始实验出于成本考虑采用 N=3、R=2。在难题（如 MMLU、GSM8K、国际象棋走法有效性、传记生成）上，增加智能体（agent）数量和交互轮数可提升准确率。

跨模型组合优于单模型辩论：ChatGPT 与 Bard 协同工作的效果优于任一模型单独使用。

### 稀疏拓扑（Sparse Topology）

《通过稀疏通信拓扑改进多智能体辩论》（"Improving Multi-Agent Debate with Sparse Communication Topology"，arXiv:2406.11776，2024-2025）表明，全连接（full-mesh）辩论并非总是最优解。稀疏拓扑结构（如星型、环型、中心辐射型）能够以更低的词元（Token）消耗达到相当的准确率。每个辩论者仅能看到部分对等节点（peer）的发言。

实际影响：

- 全连接拓扑（N=5, R=3）：5 × 3 = 15 个提议，每个提议需阅读 4 个对等节点 = 60 次批判操作（critique ops）。
- 星型拓扑（N=5, R=3，1 个中心节点 + 4 个边缘节点）：15 个提议，边缘节点仅阅读中心节点 = 12 次批判操作。

### 辩论机制的适用场景

- **事实准确性（Factuality）。** N 个独立提议通过交叉验证可有效降低幻觉（hallucination）。
- **规则遵循（Rule-following）。** 例如国际象棋走法有效性验证——若一个模型遗漏了某条规则，其他模型可将其捕获。
- **开放式推理（Open-ended reasoning）。** 多种分析视角相互收敛，逐步逼近正确答案。

### 辩论机制的局限场景

- **延迟敏感型用户体验（Latency-sensitive UX）。** N × R 轮串行交互会引入可能无法接受的延迟。
- **成本敏感型规模化应用（Cost-sensitive scale）。** 每个问题需消耗 N × R 倍的词元（Token）。
- **简单事实查询。** 单次查询的成本远低于五次辩论。

### 2026 年的实际落地方案

- **Anthropic 编排器-工作节点模式（Anthropic orchestrator-workers）**（第 12 课）——一种包含综合（synthesis）步骤的辩论变体。
- **LangGraph 监督器模式（LangGraph supervisor）**（第 13 课）——通过中央路由（central router）与专家智能体（specialist agents），可将辩论实现为一个图节点。
- **OpenAI Agents SDK**（第 16 课）——智能体之间通过反复任务交接（handoff）实现迭代式批判。
- **多智能体评估（Multi-agent evals）**——将辩论与评估器-优化器（evaluator-optimizer）配对，以生成评估信号。

### 该模式的常见陷阱

- **收敛崩溃（Convergence collapse）。** 所有智能体过早收敛于首个错误答案。可通过强制设置分歧轮次（required disagreement rounds）来缓解。
- **中心节点故障（Hub failure）。** 在星型拓扑中，劣质中心节点会污染所有边缘节点。可采用轮换机制或使用多中心节点。
- **提示词同质化（Prompt homogenization）。** 所有智能体使用相同提示词会导致输出趋同。应采用多样化的提示词和/或模型。

## 动手实现

`code/main.py` 实现了标准辩论逻辑：

- `Debater` 类（具备独立观点漂移特性的脚本化大语言模型（LLM））。
- `FullMeshDebate` 与 `SparseDebate` 运行器。
- 三类问题：事实型、规则型、推理型。
- 评估指标：收敛答案、收敛所需轮数、总批判操作次数。

运行方式：

python3 code/main.py

输出结果：各协议的准确率与成本；在 2/3 的问题上，稀疏拓扑以更低的成本达到了与全连接拓扑相当的效果。

## 实际应用

- **Anthropic 编排器-工作节点模式**：适用于简单的 2-3 个工作节点辩论。
- **LangGraph**：适用于带检查点机制（checkpointing）的有状态多轮辩论。
- **自定义实现**：适用于学术研究或需要特定正确性保障的场景。

## 交付上线

`outputs/skill-debate.md` 搭建了一个多智能体辩论（multi-agent debate）框架，支持可配置的拓扑结构（topology）、N（智能体数量）、R（辩论轮数）以及收敛规则（convergence rule）。

## 练习

1. 实现“强制分歧”（forced disagreement）规则：在第一轮中，每位辩手必须提出互不相同的提案。测量其对收敛速度的影响。
2. 添加置信度加权聚合（confidence-weighted aggregation）：辩手返回（答案，置信度）；聚合器根据置信度进行加权。这是否有效？
3. 将其中一个“智能体”替换为持有不同观点的其他脚本化大语言模型（LLM）。异质性（heterogeneity）能否提升准确率？
4. 针对你的 3 个问题，分别测量全连接拓扑（full mesh）与稀疏拓扑（sparse topology）的 Token 成本。绘制成本与准确率的对比图。
5. 阅读《心智社会》（Society of Minds）论文。将你的原型代码移植到 N=5、R=3 的配置。哪些部分会失效？哪些部分会得到优化？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 辩论（Debate） | “多智能体互评” | N 个提案者，进行 R 轮交叉互评，最终收敛 |
| 全连接拓扑（Full mesh） | “所有人阅读所有人” | 每轮中每位辩手阅读所有其他辩手的内容 |
| 稀疏拓扑（Sparse topology） | “有限的同伴视角” | 辩手仅阅读部分同伴的内容 |
| 中心辐射型拓扑（Hub-and-spoke） | “星型结构” | 一个中心辩手，其余 N-1 个节点仅读取中心节点的内容 |
| 收敛（Convergence） | “达成一致” | 辩手们最终收敛于一个共同的答案 |
| 心智社会（Society of Minds） | “Du 等人的辩论论文” | ICML 2024 提出的多智能体辩论方法 |

## 扩展阅读

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — 权威的多智能体辩论（multi-agent debate）文献
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — 稀疏拓扑（sparse topology）相关实验结果
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 将编排器-工作者（orchestrator-workers）架构作为辩论模式的变体
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 单模型自我批评（self-critique）的对应方案
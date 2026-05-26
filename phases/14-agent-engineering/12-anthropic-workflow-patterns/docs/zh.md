# Anthropic 的工作流模式：简单优于复杂

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）将工作流（workflow，预定义路径）与智能体（agent，动态工具使用）进行了区分。五种工作流模式涵盖了大多数应用场景。应从直接调用 API 开始。仅在步骤无法预测时才引入智能体。

**类型：** 学习 + 实践
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（智能体循环）
**耗时：** 约 60 分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：提示词链（prompt chaining）、路由（routing）、并行化（parallelization）、编排器-工作者（orchestrator-workers）、评估器-优化器（evaluator-optimizer）。
- 解释智能体与工作流的区别，以及各自的工程成本。
- 明确何时应选择工作流而非智能体（反之亦然）。
- 使用标准库针对脚本化的大语言模型（LLM）实现全部五种模式。

## 问题

团队常常为只需单次函数调用即可解决的问题，直接套用多智能体框架（multi-agent framework）。其代价是实实在在的：框架引入的抽象层会掩盖提示词（prompt）、隐藏控制流（control flow），并引发过早的复杂性。Schluntz 和 Zhang 于 2024 年 12 月发表的文章是业界引用最多的反思之声：从简单入手，仅在复杂性带来的收益足以抵消其成本时，才增加复杂度。

## 核心概念

### 工作流（Workflow）与智能体（Agent）

- **工作流（Workflow）。** 通过预定义的代码路径编排大语言模型（LLM）与工具。工程师负责维护执行图（graph）。
- **智能体（Agent）。** 大语言模型动态调度自身工具并自主决策执行步骤。模型自身掌控执行图。

两者各有适用场景。工作流成本更低、速度更快且更易于调试。智能体能够解决开放式问题，但也使得故障模式（failure modes）更难追溯与分析。

### 增强型大语言模型（Augmented LLM）

上述五种模式的基础：一个集成了三项核心能力的大语言模型——搜索（检索）、工具（动作执行）与记忆（状态持久化）。任何 API 调用均可利用这些能力。

### 五种核心模式

1. **提示词链（Prompt chaining）。** 第一次调用的输出作为第二次调用的输入。适用于可清晰线性拆解的任务。步骤之间可设置可选的程序化门控（programmatic gates）。

2. **路由（Routing）。** 由一个分类器大语言模型决定调用下游的哪个大语言模型或工具。适用于不同类型输入需要差异化处理的场景（例如：一线客服支持、退款处理、Bug 反馈或销售咨询）。

3. **并行化（Parallelization）。** 并发执行 N 次大语言模型调用，并聚合结果。包含两种形态：分块处理（sectioning，处理不同数据块）与投票机制（voting，相同提示词运行 N 次，取多数结果或进行综合）。

4. **编排器-工作节点（Orchestrator-workers）。** 由一个编排器大语言模型动态决定运行哪些工作节点（同样为大语言模型），并综合它们的输出。类似于智能体循环，但编排器不会无限循环。

5. **评估器-优化器（Evaluator-optimizer）。** 一个大语言模型生成答案，另一个大语言模型对其进行评估。迭代进行直至评估通过。这是第 05 课中“自我优化（Self-Refine）”模式的泛化应用。

### 工作流优于智能体的场景

- **可预测任务。** 如果能够明确枚举出所有步骤，就应当优先使用工作流。
- **成本受限任务。** 工作流的步骤数量是有上限的；而智能体可能会陷入无限循环导致成本失控。
- **合规性要求高的任务。** 审计人员需要直接查看执行图，而不是从运行轨迹中反推逻辑。

### 智能体优于工作流的场景

- **开放式研究。** 当下一步操作取决于上一步返回结果时。
- **可变长度任务。** 耗时从几分钟到几小时不等，且步骤数量无法预先确定的任务。
- **全新领域。** 当你尚未确定最佳工作流时——先进行探索，后续再将其固化为代码。

### 上下文工程（Context Engineering）配套指南

《AI 智能体的有效上下文工程》（Anthropic, 2025）正式确立了这一相邻学科的核心原则：200k 的上下文窗口是一项预算（budget），而非单纯的容器。它指导我们该包含哪些内容、何时进行压缩、何时允许上下文增长。本课程的第 14 阶段“上下文压缩”课程（重编号前的第 14 阶段第 06 课）对此进行了详细讲解。

## 动手实践

`code/main.py` 基于 `ScriptedLLM` 实现了全部五种工作流模式（Workflow Patterns）：

- `prompt_chain(input, steps)` — 顺序执行。
- `route(input, classifier, handlers)` — 分类与路由分发。
- `parallel_vote(prompt, n, aggregator)` — 并行运行 N 次并聚合结果。
- `orchestrator_workers(task, workers)` — 编排器动态挑选工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 循环迭代直至通过评估。

运行方式：

python3 code/main.py

每种模式都会打印其执行轨迹（Trace）。每种模式的代码量仅约 10-15 行；而引入完整框架的代价往往高达数千行。

## 使用指南

- 对于大多数任务，直接调用 API 即可。
- 仅当模式确实需要持久化状态（Durable State，如 LangGraph）、Actor 模型并发（Actor-Model Concurrency，如 AutoGen v0.4）或角色模板（Role Templating，如 CrewAI）时，才引入框架。
- 若希望获得类似 Claude Code 的控制架构（Harness Shape）而无需从零重建，可直接选用 Claude Agent SDK。

## 部署与交付

`outputs/skill-workflow-picker.md` 可根据给定的任务描述自动匹配最佳模式，其中包含决策依据，以及在工作流无法满足需求时向智能体（Agent）重构的演进路径。

## 练习

1. 实现带置信度阈值（Confidence Threshold）的路由逻辑。低于阈值时 -> 升级至人工处理。在一线客服（Tier-1 Support）场景中，该阈值通常设定在什么水平？
2. 为 `parallel_vote` 添加超时机制。当某个调用挂起（Hang）时会发生什么？在缺少部分投票结果的情况下如何进行聚合？
3. 将 `evaluator_optimizer` 改造为多臂老虎机（Bandit）模式：在迭代过程中保留排名前 2 的输出，以避免后期出现的优质结果被更晚出现的劣质结果覆盖。
4. 将提示词链（Prompt Chaining）与路由结合：由路由器从三条链中选择其一。对比该方案与单一长提示词（Big-Prompt）方案的 Token 消耗成本。
5. 选取你生产环境中的一个功能。绘制其工作流图（Workflow Graph）。统计步骤数量。在此场景下，使用智能体（Agent）是否真的更优？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 工作流（Workflow） | “预定义流程” | 由工程师掌控的 LLM 与工具调用图 |
| 智能体（Agent） | “自主 AI” | 由模型自主掌控的图结构；动态决定工具调用方向 |
| 增强型 LLM（Augmented LLM） | “带工具的 LLM” | LLM + 搜索 + 工具 + 记忆；构成系统的基本原子单元 |
| 提示词链（Prompt Chaining） | “顺序调用” | 第 N 次调用的输出作为第 N+1 次调用的输入 |
| 路由（Routing） | “分类器分发” | 决定由哪条链或哪个模型处理输入 |
| 并行化（Parallelization） | “扇出（Fan-out）” | N 个并发调用；通过分块或投票机制聚合结果 |
| 编排器-工作者（Orchestrator-workers） | “分发器智能体” | 编排器 LLM 动态挑选专业 LLM 执行任务 |
| 评估器-优化器（Evaluator-optimizer） | “提议者 + 裁判” | 持续迭代直至评估器通过；Self-Refine 模式的泛化 |

## 延伸阅读

- [Anthropic，构建高效智能体（2024年12月）](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式（workflow patterns）
- [Anthropic，AI 智能体的高效上下文工程（context engineering）](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 互补领域（companion discipline）
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图（stateful graphs）何时能证明其成本价值
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 编排器-工作者模式（orchestrator-workers pattern）的产品化实现
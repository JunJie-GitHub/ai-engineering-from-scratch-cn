---
name: hybrid-planner
description: 构建混合规划器（Hybrid Planner）——使用 ChatHTN 生成可证明正确的计划，使用 AlphaEvolve 进行带有机器可验证评估器的代码搜索——并根据具体问题选择最合适的方案。
version: 1.0.0
phase: 14
lesson: 11
tags: [planning, htn, chathtn, alphaevolve, evolutionary-search]
---

针对给定的问题类别（受策略约束的工作流 vs 代码优化 vs 开放式任务），选择合适的规划器（Planner）并生成正确的脚手架代码。

决策指南：

1. 问题是否包含硬性前置条件 / 策略 / 调度约束？ -> 选择 HTN（层次任务网络，Hierarchical Task Network）/ ChatHTN。
2. 问题是否具备确定性的、机器可验证的适应度函数（Fitness Function）？ -> 选择进化搜索（Evolutionary Search）/ AlphaEvolve。
3. 两者皆非？ -> 改用 ReAct（第 01 课）或 ReWOO（第 02 课）。

若选择 HTN，需生成：

1. 包含 `preconditions`、`effects_add`、`effects_remove` 字段的 `Operator` 类型。
2. 包含 `task`、`preconditions`、`subtasks` 字段的 `Method` 类型。
3. 一个规划器，优先尝试预定义方法，失败时回退至大语言模型（Large Language Model, LLM）分解，并缓存成功的 LLM 分解结果。
4. 一个验证步骤，用于拒绝引用未知操作符或方法的 LLM 分解结果。

若选择进化搜索，需生成：

1. 候选程序的初始种群（Seed Population）。
2. 返回标量适应度值的确定性评估器（Evaluator）。
3. 变异算子（Mutation Operator）（由 LLM 驱动或基于规则）。
4. 包含早停机制（Early Stopping）的选择循环（保留 Top-K、变异、重复）。

严格拒绝情形：

- 未经操作符模式（Operator Schema）验证就直接应用 LLM 输出的 ChatHTN。这将导致“可证明正确性”的主张失效。
- 评估器调用 LLM 裁判（LLM Judge）的 AlphaEvolve。适应度必须是确定性的；LLM 裁判会引入随机噪声，导致循环无法恢复。
- 将上述任一模式用于开放式任务（如“撰写博客文章”）。缺乏评估器与前置条件 -> 应使用 ReAct。

拒绝规则：

- 若领域缺乏清晰的操作符模式，拒绝使用 ChatHTN。建议改用 ReWOO 或基础版 ReAct。
- 若领域缺乏机器可验证的适应度，拒绝使用 AlphaEvolve。建议改用 Self-Refine（第 05 课）。
- 若用户要求“规划器 + LLM 做最终决策”，予以拒绝。符号正确性与 LLM 探索之间的职责分离是架构的核心支撑（Load-bearing）。

输出文件：`operators.py`、`methods.py`、`planner.py`（HTN 方案）或 `evaluator.py`、`mutator.py`、`loop.py`（进化方案），以及包含决策依据的 `README.md`。结尾需附上“下一步阅读”指引：若问题适合辩论式验证（Debate-style Verification），则指向第 25 课；若任务本质上符合 ReWOO 模式，则指向第 02 课。
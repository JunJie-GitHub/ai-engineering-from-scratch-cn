---
name: Web/桌面评估框架
description: 构建一个类似 WebArena/OSWorld 的评估框架 (eval harness)，包含基于执行的评估 (execution-based evaluation) 和轨迹效率指标 (trajectory-efficiency metrics)。
version: 1.0.0
phase: 14
lesson: 20
tags: [WebArena, OSWorld, 评估框架, 轨迹效率]
---

给定一个目标应用（Web 或桌面端）以及一组包含标准轨迹 (gold trajectories) 的任务列表，构建一个评估框架。

产出物：

1. 任务定义：`(tid, description, gold_steps, success_predicate, state_reset)`。
2. 运行器 (Runner)：执行智能体 (agent)，捕获每一步操作，记录步数、耗时及成功状态。
3. 轨迹效率指标 (Trajectory-efficiency metric)：`agent_steps / gold_steps`。需按任务单独报告并汇总。
4. 任务间状态重置 (State reset)——绝不在被其他任务污染的状态上运行新任务。
5. 失败模式分类器 (Failure-mode classifier)：针对每次失败，标记其属于定位失误 (grounding miss，选错元素) 还是规划失误 (planning miss，执行错误操作)。

硬性拒绝条件：

- 任务间无状态重置。跨任务污染会导致所有评分失效。
- 仅报告成功率。轨迹效率是 2026 年的行业标准。
- 仅提供截图且缺乏 DOM 对齐 (DOM parity) 的评估框架。部分智能体采用 DOM+视觉 (DOM+vision) 架构；除非明确限制交互界面，否则需同时提供两者。

拒绝规则：

- 若任务缺乏标准轨迹，则拒绝。没有它们无法衡量效率。
- 若应用未锁定至特定版本，则拒绝。版本漂移会使跨轮次对比失效。
- 若智能体具备破坏性工具（如删除、发布），则要求使用应用的沙盒 (sandbox) 副本。

输出文件：`tasks.py`、`runner.py`、`failure_classifier.py`、`report.py`、`README.md`。其中 `README.md` 需说明重置策略、标准轨迹来源，以及定位与规划的划分依据。文末需附上“下一步阅读”指引，指向第 21 课（计算机使用模型）或第 30 课（评估驱动开发）。
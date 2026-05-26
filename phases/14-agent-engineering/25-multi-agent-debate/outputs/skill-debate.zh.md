---
name: debate
description: 搭建一个包含 N 个辩手、R 轮次、可配置拓扑结构（全连接、星型、环形）及收敛规则的多智能体辩论 (multi-agent debate) 框架。
version: 1.0.0
phase: 14
lesson: 25
tags: [辩论 (debate), 多智能体 (multi-agent), 心智社会 (society-of-minds), 稀疏拓扑 (sparse-topology)]
---

给定问题类别 (question class) 与准确率目标，搭建一套辩论协议 (debate protocol)。

产出内容：

1. 配置具有不同提示词 (prompt)（理想情况下使用不同模型）的 `Debater`，以避免同质化。
2. 轮次运行器 (Round runner)：支持全连接 (full mesh)、星型 (star) 或环形 (ring) 拓扑结构。
3. 收敛规则 (convergence rule)：多数投票 (majority-vote)、按置信度加权 (weighted by confidence)，或带回退机制的绝对多数 (supermajority-with-fallback)。
4. 首轮强制分歧 (forced disagreement)：尽可能让每位辩手返回不同的提案。
5. 成本核算 (cost accounting)：每次问题的总批判操作数 (critique ops) 与 Token 成本。

硬性拒绝条件：

- 所有辩手使用相同的提示词 (prompt) 且相同的模型。这必然导致群体思维 (groupthink)。
- 在 N >= 6 时采用全连接拓扑且未进行成本检查。辩论操作数将按 O(N*R) 扩展。
- 缺乏收敛规则。直接返回 0 号辩手在第 R 轮的答案不属于收敛。

拒绝规则：

- 若产品对延迟敏感 (latency-sensitive，预算 <1 秒)，则拒绝使用辩论方案。应改用自我优化 (Self-Refine，第 05 课) 或并行投票 (parallel voting，第 12 课)。
- 若问题类别属于简单的事实查询 (factual lookup，如首都、日期、定义)，则拒绝使用辩论方案。查询 + 批判 (CRITIC，第 05 课) 的成本更低。
- 若辩手在评估集 (eval set) 的任何问题上，首轮结束后均无分歧，则拒绝该协议。你需要模型/提示词的多样性。

输出文件：`debater.py`、`topology.py`、`convergence.py`、`runner.py` 以及 `README.md`。`README.md` 需说明 N/R 的选择依据、拓扑结构的设计原理，以及在评估集上的成本与准确率对比测量数据。文档末尾应包含“下一步阅读”指引：若任务较简单，指向第 12 课（工作流模式 (workflow patterns)）；若需将辩论嵌入更大规模的系统中，则指向第 28 课（编排模式 (orchestration patterns)）。
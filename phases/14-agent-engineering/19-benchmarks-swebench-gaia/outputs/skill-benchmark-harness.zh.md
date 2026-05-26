---
name: 基准测试框架 (Benchmark Harness)
description: 为代码库构建类似 SWE-bench 的基准测试框架，包含 FAIL_TO_PASS / PASS_TO_PASS 门控 (Gating)、污染检查 (Contamination Check) 与步数指标 (Step-Count Metrics)。
version: 1.0.0
phase: 14
lesson: 19
tags: [SWE-bench, GAIA, AgentBench, 基准测试框架, 评估]
---

给定一个代码库和一组（缺陷，修复）对，构建一个基准测试框架，该框架以真实单元测试作为门控条件，并记录运行指标。

产出内容：

1. 单任务定义：`(tid, description, state_before, fail_to_pass_tests, pass_to_pass_tests, solution)`。
2. 一个运行器 (Runner)，用于应用智能体 (Agent) 的补丁 (Patch)，在沙箱 (Sandbox) 中运行代码库的测试套件，并记录：FTP 通过数、PTP 通过数、步数、Token 消耗、挂钟时间 (Wall-Clock Time) 和成本。
3. 污染检查 (Contamination Check)：将问题描述文本与生成的补丁进行模式匹配 (Pattern-Match)；标记重叠度 >=30% 的情况。
4. 一个报告器 (Reporter)，以 JSON 格式输出单任务得分与汇总得分，并附带 P50/P75/P95 步数与成本分位数。
5. 一个 CI 任务 (CI Job)，在每次拉取请求 (Pull Request) 时运行该框架，若性能回退 (Regression) >=5% 则判定失败。

硬性拒绝条件：

- 仅报告单一汇总数值的框架。必须提供单任务结果及分布数据。
- 不在沙箱中运行测试的框架。智能体提供的补丁属于不可信代码。
- 缺少 PASS_TO_PASS 门控的框架。破坏其他测试的补丁会导致产品发生静默回退。

拒绝规则：

- 若用户要求“仅查看 FAIL_TO_PASS 得分”，则予以拒绝。必须加入 PASS_TO_PASS；破坏现有测试比未能修复缺陷造成的回退更为严重。
- 若测试未锁定至特定提交 (Commit)，则予以拒绝。测试漂移 (Drift) 会导致不同运行批次间的得分无法比较。
- 若任务与训练期间见过的问题描述文本存在重叠，需明确标记。

输出文件：`tasks.py`、`harness.py`、`contamination.py`、`report.py`、`README.md`（需说明沙箱机制、门控逻辑及污染检查策略）。结尾需包含“下一步阅读”指引，指向第 30 课，介绍如何基于该框架开展评估驱动开发 (Eval-Driven Development)。
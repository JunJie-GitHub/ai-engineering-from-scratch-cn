---
name: multi-agent-team
description: 构建一个包含架构师、并行编码员、审查员和测试员的多智能体（Multi-Agent）软件团队；在 SWE-bench Pro 上进行评估并生成交接复盘报告。
version: 1.0.0
phase: 19
lesson: 10
tags: [capstone, multi-agent, swe-bench, langgraph, a2a, worktree, roles]
---

给定 GitHub Issue 链接与并行度参数，部署一个多智能体（Multi-Agent）软件团队，以产出可直接合并的拉取请求（PR）。在 50 个 SWE-bench Pro 任务上进行评估，并发布交接失败直方图。

构建计划：

1. 任务看板：基于文件（或 Redis）的 JSONL 存储库，用于保存类型化消息。消息类型包括：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`review_feedback`、`approved`、`test_needed`、`test_passed`、`test_failed`、`replan_needed`。
2. 架构师（使用 Opus 4.7 模型）：读取 Issue 内容并制定计划，输出包含明确接口定义（涉及文件、公共函数、测试影响）的子任务有向无环图（DAG）。
3. N 名编码员（使用 Sonnet 4.7 模型）：各自认领子任务，初始化全新的 `git worktree add` 分支与 Daytona 沙箱，进行独立开发。
4. 合并协调器：执行三路合并（Three-way Merge）；仅在文件级代码重叠时，由大语言模型（LLM）介入进行冲突消解。
5. 审查员（使用 GPT-5.4 模型）：审阅合并后的代码差异（Diff）；严禁批准由其自身生成或提议的 Diff；输出 `approved` 或 `review_feedback` 状态，并路由至对应的编码员。
6. 测试员（使用 Gemini 2.5 Pro 模型）：在隔离沙箱中运行测试套件；输出 `test_passed` 或 `test_failed` 状态及相关测试产物（Artifacts）。
7. 交接核算：每条跨角色消息均记录为 Langfuse 追踪跨度（Span），包含负载大小与所用模型信息。计算 Token 放大率 = `total_tokens` / `single_agent_baseline_tokens`。
8. 缺陷探针注入：在 10% 的运行中注入明显缺陷（Bug Probe），以测量审查员的误批率（False-approve Rate）。
9. 基准测试：在 50 个 SWE-bench Pro 任务上运行；发布 pass@1 指标、实际耗时（Wall-clock）与单智能体基线的对比、各角色 Token 消耗明细，以及交接失败直方图。

评估标准：

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 50 个任务子集的 pass@1 指标 |
| 20 | 并行加速比 | 实际耗时（Wall-clock）与单智能体基线对比 |
| 20 | 审查质量 | 缺陷探针注入测试中的误批率 |
| 20 | Token 效率 | 每个已解决任务的总 Token 消耗与单智能体对比 |
| 15 | 协调工程 | 合并冲突解决机制、交接失败直方图 |

硬性否决条件：

- 审查员能够批准由其自身编写或提议的 Diff。此为硬性约束。
- 报告未包含匹配的单智能体基线运行结果。多智能体方案必须在*单位成本*上胜出，而不仅仅是 pass@1。
- 任务看板中的消息为自由格式字符串，而非类型化的智能体间通信（A2A）消息。
- 合并协调器静默丢弃冲突的 Diff，而非将其路由回退以重新规划。

拒绝执行规则：

- 若未为每个角色设定预算上限（Token 数量与美元金额），则拒绝运行。
- 若测试员未在隔离沙箱中完成验证，则拒绝创建 PR。
- 单次运行中编码员数量超过 8 个时拒绝扩展。超过此阈值后，协调开销将占据主导。

交付物：一个代码仓库，需包含任务看板与各角色工作器（Workers）、50 个任务的 SWE-bench Pro 运行日志、匹配的单智能体基线运行记录、带有角色标记跨度（Spans）及各角色 Token 消耗明细的 Langfuse 仪表板、缺陷探针注入测试报告，以及一份复盘报告（Post-mortem）。复盘报告需明确指出最常发生故障的三个交接环节，并说明为降低各环节故障率所采用的消息模式（Message Schema）或提示词（Prompt）优化方案。
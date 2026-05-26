---
name: terminal-coding-agent
description: 构建并评估一款终端原生编程智能体（Terminal Coding Agent），在成本可控、工具沙箱化（Sandbox）及完整支持 2026 版钩子（Hook）机制的前提下，与 SWE-bench Pro 进行对标测试。
version: 1.0.0
phase: 19
lesson: 01
tags: [综合项目, 编程智能体, claude-code, swe-bench, mcp, 钩子, 沙箱]
---

给定目标代码仓库与自然语言任务，构建一个测试框架（Harness），使其能够进行任务规划、在沙箱中执行操作并自动发起拉取请求（Pull Request）。在单任务预算不超过 5 美元的前提下，于包含 30 个任务的 SWE-bench Pro 子集上，达到或超越 mini-swe-agent 基线模型的性能。

构建计划：

1. 搭建基于 Bun 与 Ink 的终端用户界面（TUI）测试框架，包含规划面板、工具调用流以及实时 Token/美元预算显示。
2. 基于模型上下文协议（Model Context Protocol）的 StreamableHTTP 传输层定义六项工具（`read_file`、`edit_file`、`ripgrep`、`tree_sitter_symbols`、`run_shell`、`git`）。每次调用最多返回 4k 个 Token。
3. 在通过 `git worktree add` 创建的全新工作树（Worktree）分支上，将所有工具调用置于 E2B 或 Daytona 沙箱中运行。严禁触碰宿主机文件系统。
4. 接入全部八项 2026 版钩子事件：`SessionStart`、`SessionEnd`、`PreToolUse`、`PostToolUse`、`UserPromptSubmit`、`Notification`、`Stop`、`PreCompact`。交付至少四项自定义钩子（破坏性命令防护、Token 计量、OpenTelemetry 跨度发射器、追踪包写入器）。
5. 严格执行三项预算限制：50 轮对话、20 万 Token、5 美元。当 Token 消耗达到 15 万时触发 `PreCompact` 事件，对早期对话轮次进行摘要压缩。
6. 遵循生成式人工智能（GenAI）语义规范，向自托管的 Langfuse 实例发送 OpenTelemetry 跨度（Span）数据。
7. 任务成功后，推送该分支并创建拉取请求，在请求描述中附带任务规划与追踪包。
8. 在包含 30 个问题的 SWE-bench Pro Python 子集上与 mini-swe-agent 进行对比评估，并记录每个任务的 pass@1（首次通过率）、对话轮数、Token 消耗及美元成本。

评估标准：

| 权重 | 评估维度 | 衡量标准 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 在匹配的 30 任务子集上对比 mini-swe-agent 基线 |
| 20 | 架构清晰度 | 规划/执行/观察模块分离、钩子接口覆盖度、工具模式（Schema）可读性 |
| 20 | 安全性 | 沙箱逃逸红队测试（Red-team） + 破坏性命令防护审计 |
| 20 | 可观测性 | 100% 工具调用均生成跨度，按轮次进行 Token 计量 |
| 15 | 开发者体验 | 冷启动（Cold-start）时间低于 2 秒、崩溃恢复机制、Ctrl-C 取消语义 |

硬性否决项：

- 测试框架绕过沙箱，直接在宿主机文件系统上调用 git 命令。
- 任何能够在工作树目录外写入文件，或在未配置显式白名单钩子的情况下通过 curl 访问外部 URL 的智能体。
- 报告评估数据时，未在同一 30 个问题集上运行匹配的基线模型进行对照。
- 依赖重试期间执行 `git reset --hard` 来宣称“通过率”；SWE-bench Pro 仅认可 pass@1。

拒绝执行规则：

- 在任何配置下均拒绝直接推送至 main 分支。仅允许推送至 PR 分支。
- 拒绝禁用破坏性命令防护功能。此为评估标准的硬性要求。
- 拒绝在无预算上限的情况下运行。无限制运行会污染评估对比结果。

交付物：一个包含测试框架的代码仓库、一个针对固定 30 任务 SWE-bench Pro 的评估框架（附带匹配的 mini-swe-agent 基线运行结果）、至少 5 次完整运行的 OpenTelemetry 追踪归档文件，以及一份说明文档，明确指出该框架解决了哪些基线未能解决的任务，反之亦然。文档末尾需包含一个章节，总结观察到的三大主要故障模式，以及修复每种模式所对应的钩子变更。
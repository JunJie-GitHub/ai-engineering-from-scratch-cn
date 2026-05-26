# 综合项目 01 — 终端原生编程智能体 (Terminal-Native Coding Agent)

> 到 2026 年，编程智能体 (Coding Agent) 的形态已基本定型。它包含一个终端用户界面控制框架 (TUI Harness)、一个有状态计划 (Stateful Plan)、一个沙盒化工具面 (Sandboxed Tool Surface)，以及一个“规划-执行-观察-恢复”循环 (Plan-Act-Observe-Recover Loop)。从宏观视角来看，Claude Code、Cursor 3 和 OpenCode 的架构如出一辙。本综合项目要求你从零到一构建这样一个系统——以命令行接口 (CLI) 输入，以拉取请求 (Pull Request) 输出——并在 SWE-bench Pro 基准测试中将其与 mini-swe-agent 和 Live-SWE-agent 进行对比评估。你将深刻体会到，真正的难点并非模型调用，而是工具循环 (Tool Loop)、沙盒环境，以及在 50 轮交互中严格控制成本上限。

**Type:** 综合项目
**Languages:** TypeScript / Bun（控制框架）、Python（评估脚本）
**Prerequisites:** 第 11 阶段（大语言模型工程）、第 13 阶段（工具与协议）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 小时

## 问题背景

编程智能体在 2026 年已成为主导性的 AI 应用类别。Claude Code (Anthropic)、搭载 Composer 2 和 Agent Tabs 的 Cursor 3 (Cursor)、Amp (Sourcegraph)、OpenCode（11.2 万星标）、Factory Droids 以及 Google Jules 等产品，均交付了同一架构的不同变体：一个终端控制框架、一个带权限控制的工具面、一个沙盒环境，以及围绕前沿模型 (Frontier Model) 构建的“规划-执行-观察”循环。前沿模型的能力边界虽窄——Live-SWE-agent 在使用 Opus 4.5 时于 SWE-bench Verified 上达到了 79.2% 的得分——但工程实现的广度却极大。大多数故障模式并非源于模型本身的错误，而是工具循环不稳定、上下文污染 (Context Poisoning)、Token 成本失控 (Runaway Token Cost) 以及破坏性的文件系统操作。

你无法仅从外部视角来剖析这些智能体。你必须亲手构建一个，亲眼目睹当 ripgrep 返回 8MB 匹配结果时循环在第 47 轮崩溃，然后重新设计截断层。这正是本综合项目的核心意义所在。

## 核心概念

该控制框架包含四个核心层面。**规划 (Plan)** 维护一个类似 TodoWrite 的状态对象，模型会在每一轮交互中重写它。**执行 (Act)** 负责分发工具调用（读取、编辑、运行、搜索、Git 操作）。**观察 (Observe)** 捕获标准输出 (stdout) / 标准错误 (stderr) / 退出码，进行截断处理，并将摘要反馈给模型。**恢复 (Recover)** 负责处理工具错误，避免撑爆上下文窗口或陷入无限循环。2026 年的架构新增了一个关键要素：**钩子 (Hooks)**。包括 `PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——这些是可配置的扩展点，供开发者注入策略、遥测数据和安全护栏。

沙盒环境采用 E2B 或 Daytona。每个任务都在全新的开发容器 (Devcontainer) 中运行，并挂载一个可读写的 Git 工作树 (Git Worktree)。控制框架绝不会直接触碰宿主机的文件系统。无论任务成功或失败，工作树都会被彻底清理。成本控制通过三层机制强制执行：单轮 Token 上限、单次会话美元预算，以及硬性轮次限制（通常为 50 轮）。可观测性层 (Observability Layer) 基于 OpenTelemetry 追踪跨度 (Spans) 并遵循 GenAI 语义规范 (Semantic Conventions)，数据将发送至自托管的 Langfuse 实例。

## 架构设计

  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app

## 技术栈 (Stack)

- 调度运行时 (Harness runtime)：Bun 1.2 + Ink 5（终端内 React）
- 模型接入 (Model access)：通过 OpenRouter 统一 API 接入 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro 及 Opus 4.5（用于处理最复杂的任务）
- 工具传输 (Tool transport)：模型上下文协议 (Model Context Protocol, MCP) StreamableHTTP（2026 修订版）
- 沙箱环境 (Sandbox)：E2B 沙箱（JS SDK）或 Daytona 开发容器 (devcontainers)
- 代码搜索 (Code search)：`ripgrep` 子进程，支持 17 种语言的 `tree-sitter` 解析器（预编译）
- 隔离机制 (Isolation)：每个任务使用 `git worktree add` 创建独立工作树，任务成功或失败后自动清理
- 评估框架 (Eval harness)：SWE-bench Pro（已验证子集） + Terminal-Bench 2.0 + 您自定义的 30 个任务保留集 (holdout)
- 可观测性 (Observability)：集成 `gen_ai.*` 语义约定 (semconv) 的 OpenTelemetry SDK → 自托管 Langfuse
- PR 提交 (PR posting)：通过 GitHub App 使用细粒度令牌 (fine-grained token)，权限范围仅限目标仓库

## 开始构建 (Build It)

1. **终端用户界面（TUI）与命令循环。** 使用 Ink 初始化一个 Bun 项目。支持接收 `agent run <repo> "<task>"` 命令。输出分屏视图：顶部为计划面板（plan pane），中部为工具调用流（tool-call stream），底部为 Token 预算（token budget）。添加 Ctrl-C 中断功能，在程序退出前触发 `SessionEnd` 钩子（hook）。

2. **计划状态。** 定义一个类型化的 `TodoWrite` 数据结构（包含待处理/进行中/已完成状态及备注）。模型在每一轮（turn）都将完整状态作为工具调用进行重写——禁止其进行增量修改。将计划持久化至 `.agent/state.json`，以便在程序崩溃后能够恢复。

3. **工具接口。** 定义六个工具：`read_file`、`edit_file`（带差异预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时设置）、`git`（支持 status / diff / commit / push）。通过 MCP StreamableHTTP 协议暴露这些工具，使调度框架（harness）实现传输层无关（transport-agnostic）。每个工具返回截断后的输出（每次调用限制为 4k Token）。

4. **沙箱封装。** 每个任务都会启动一个 E2B 沙箱（sandbox）。使用 `git worktree add -b agent/$TASK_ID` 创建一个新分支。所有工具调用均在沙箱内部执行，无法访问宿主机文件系统。

5. **钩子（Hook）。** 实现全部八种 2026 版钩子类型。至少接入四个用户自定义钩子：(a) `PreToolUse` 破坏性命令防护，拦截工作树（worktree）之外的 `rm -rf` 命令；(b) `PostToolUse` Token 核算；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终的追踪数据包（trace bundle）。

6. **评估循环。** 克隆 SWE-bench Pro Python 数据集中包含 30 个问题的子集。针对每个问题运行你的调度框架。在首次通过率（pass@1）、单任务轮次（turns-per-task）和单任务成本（$-per-task）指标上与 mini-swe-agent（最小基线）进行对比。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 设定硬性上限：50 轮次、200k 上下文（context）、每任务 5 美元。当上下文达到 150k 标记时，`PreCompact` 钩子会将较早的轮次总结为历史状态块，从而在保留计划完整性的同时为新观测数据腾出空间。

8. **提交拉取请求（PR）。** 任务成功后，最后一步是执行 `git push` 并调用 GitHub API 创建 PR，在 PR 正文中附带计划与差异摘要。

## 使用方法

$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41

## 正式发布

交付的技能模块位于 `outputs/skill-terminal-coding-agent.md`。给定仓库路径和任务描述后，它将在沙箱（Sandbox）中运行完整的规划-执行-观察循环（Plan-Act-Observe Loop），并返回 PR URL 以及追踪数据包（Trace Bundle）。本压轴项目的评分标准如下：

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 与基线对比 | 在 30 个匹配的 Python 任务上，对比你的编排器（Harness）与 mini-swe-agent 的表现 |
| 20 | 架构清晰度 | 规划/执行/观察的分离、钩子（Hook）接口、工具模式（Tool Schema）——对照 Live-SWE-agent 布局进行评审 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令防护通过红队（Red-team）测试 |
| 20 | 可观测性（Observability） | 追踪完整性（100% 的工具调用均被记录跨度），每轮对话的 Token 核算 |
| 15 | 开发者体验（Developer UX） | 冷启动（Cold-start）< 2s，崩溃恢复后能继续执行计划，Ctrl-C 能在工具执行中途干净地取消 |
| **100** | | |

## 练习

1. 将底层模型从 Claude Sonnet 4.7 替换为在 vLLM 上部署的 Qwen3-Coder-30B。对比 pass@1 和单任务成本（$-per-task）。报告开源模型表现不佳的具体场景。

2. 添加一个 `reviewer` 子智能体（Sub-agent），在提交 PR 前读取代码差异（Diff），并可请求修订循环。测量误报审查是否会导致 SWE-bench 通过率低于单智能体基线（提示：通常会的）。

3. 对沙箱进行压力测试：编写一个尝试 `curl` 外部 URL 的任务，以及一个尝试在工作树（Worktree）外部写入的任务。确认两者均被 `PreToolUse` 钩子拦截。记录这些尝试。

4. 使用较小的模型（Haiku 4.5）实现 `PreCompact` 摘要功能。测量在 3 倍压缩率下，计划保真度（Plan Fidelity）的损失程度。

5. 将 MCP 的 StreamableHTTP 传输协议（Transport）替换为 stdio。对冷启动和单次调用延迟（Latency）进行基准测试。为纯本地使用场景选出更优方案。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| Harness | “智能体循环” | 围绕模型运行的代码，负责调度工具、维护计划状态并执行预算限制 |
| Hook | “智能体事件监听器” | 由用户编写的脚本，由编排器在八个生命周期事件之一触发时运行 |
| Worktree | “Git 沙箱” | 位于独立路径的关联 Git 检出副本；可随时丢弃且不影响主克隆仓库 |
| TodoWrite | “计划状态” | 模型在每轮对话中重写的待办/进行中/已完成事项的强类型列表 |
| StreamableHTTP | “MCP 传输协议” | 2026 版 MCP 修订：支持双向流的长连接 HTTP；用于替代 SSE |
| Token ceiling | “上下文预算” | 每轮或每次会话的输入+输出 Token 上限；触发压缩或终止机制 |
| pass@1 | “单次尝试通过率” | 在不重试或偷看测试集的情况下，首次运行即解决的 SWE-bench 任务比例 |

## 扩展阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 提供的参考评测框架（reference harness）
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 与 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — 用于 SWE-bench 评测框架对比的最小基线（minimal baseline）
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2% 的得分
- [OpenCode](https://opencode.ai) — 开源评测框架，获 11.2 万星标
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本综合项目（capstone）所针对的评估基准
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP 与能力元数据（capability metadata）
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 用于工具调用（tool calls）与 Token 使用量（token usage）的 Span 结构（span schema）
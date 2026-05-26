# Claude Agent SDK：子代理（Subagents）与会话存储（Session Store）

> Claude Agent SDK 是 Claude Code 运行框架（harness）的库形式。提供内置工具、用于上下文隔离（context isolation）的子代理、钩子（hooks）、W3C 追踪传播（W3C trace propagation）以及会话存储（session store）的功能对齐。Claude Managed Agents 则是面向长时间运行异步任务的托管替代方案。

**类型：** 学习 + 构建
**语言：** Python（标准库（stdlib））
**前置条件：** 第 14 阶段 · 01（智能体循环（Agent Loop）），第 14 阶段 · 10（技能库（Skill Libraries））
**时长：** 约 75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API（raw API））与 Claude Agent SDK（运行框架形态（harness shape））之间的区别。
- 描述子代理 —— 并行化（parallelization）与上下文隔离 —— 以及何时应当使用它们。
- 列举 Python SDK 会话存储的接口（session store surface）（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个基于标准库的运行框架，包含内置工具、具备隔离上下文的子代理生成（subagent spawning）、生命周期钩子（lifecycle hooks）以及会话存储。

## 问题背景

原始的大语言模型 API（LLM API）仅支持单次请求-响应往返（round-trip）。而生产级智能体需要工具执行（tool execution）、MCP 服务器（MCP servers）、生命周期钩子、子代理生成、会话持久化（session persistence）以及追踪传播（trace propagation）。Claude Agent SDK 以库的形式提供这一完整架构 —— 即 Claude Code 所使用的同一套运行框架，现开放供自定义智能体使用。

## 核心概念

### 客户端 SDK (Client SDK) 与智能体 SDK (Agent SDK)

- **客户端 SDK (`anthropic`)。** 原始消息 API (Raw Messages API)。由你完全掌控循环、工具与状态。
- **智能体 SDK (`claude-agent-sdk`)。** 内置工具执行、MCP 连接 (MCP connections)、钩子 (hooks)、子智能体 (subagent) 生成与会话存储 (session store)。将 Claude Code 循环作为库使用。

### 内置工具 (Built-in tools)

该 SDK 开箱即用提供 10 余种工具：文件读写、Shell 命令、`grep`、`glob`、网页抓取等。自定义工具可通过标准工具模式 (tool-schema) 接口进行注册。

### 子智能体 (Subagents)

Anthropic 官方文档中明确了其两大用途：

1. **并行化 (Parallelization)。** 并发执行独立任务。例如“为这 20 个模块分别查找测试文件”即可拆分为 20 个并行的子智能体任务。
2. **上下文隔离 (Context isolation)。** 子智能体使用独立的上下文窗口 (context window)，仅将结果返回给编排器 (orchestrator)，从而保留编排器的上下文预算。

Python SDK 近期新增功能：`list_subagents()` 与 `get_subagent_messages()`，用于读取子智能体的对话记录 (transcripts)。

### 会话存储 (Session store)

与 TypeScript 版本保持协议一致：

- `append(session_id, message)` — 添加一轮对话。
- `load(session_id)` — 恢复对话。
- `list_sessions()` — 枚举会话列表。
- `delete(session_id)` — 删除会话，并级联删除关联的子智能体会话。
- `list_subkeys(session_id)` — 列出子智能体键值。

`--session-mirror`（命令行标志 (CLI flag)）可在流式传输过程中将会话记录镜像同步至外部文件，便于调试。

### 钩子 (Hooks)

可注册的生命周期钩子包括：

- `PreToolUse`、`PostToolUse` — 拦截或审计工具调用。
- `SessionStart`、`SessionEnd` — 初始化与清理。
- `UserPromptSubmit` — 在模型接收前处理用户输入。
- `PreCompact` — 在上下文压缩 (context compaction) 前执行。
- `Stop` — 智能体退出时执行清理。
- `Notification` — 旁路通知/警报。

钩子机制正是专业工作流（参考第 14 阶段课程）及类似系统实现横切行为 (cross-cutting behavior) 的方式。

### W3C 追踪上下文 (W3C trace context)

调用方活跃的 OpenTelemetry 跨度 (OTel spans) 会通过 W3C 追踪上下文标头传播至 CLI 子进程中。整个多进程追踪链路将在你的后端系统中显示为单一完整的追踪记录。

### Claude 托管智能体 (Claude Managed Agents)

托管替代方案（Beta 版请求头 (beta header) `managed-agents-2026-04-01`）。支持长时间运行的异步任务、内置提示词缓存 (prompt caching) 与内置压缩功能。以牺牲部分控制权换取托管基础设施的便利。

### 该模式的常见误区

- **子智能体过度生成 (Subagent over-spawn)。** 为 100 个微小任务生成 100 个子智能体，导致系统开销占据主导。应改用批处理。
- **钩子蔓延 (Hook creep)。** 各团队随意添加钩子，导致启动时间急剧膨胀。建议每季度审查一次钩子配置。
- **会话膨胀 (Session bloat)。** 会话不断累积导致体积增大。应结合 `list_sessions` 与会话过期策略进行管理。

## 动手实践 (Build It)

`code/main.py` 在标准库 (stdlib) 中实现了该 SDK 的架构形态 (shape)：

- `Tool`、`ToolRegistry`，内置 `read_file`、`write_file`、`list_dir`。
- `Subagent`（子智能体 (Subagent)）—— 私有上下文、隔离运行、返回结果。
- `SessionStore`（会话存储 (SessionStore)）—— 追加、加载、列表、删除、列出子键。
- `Hooks`（钩子 (Hooks)）—— `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 演示示例：主智能体 (main agent) 并行生成 3 个子智能体（各自隔离），聚合结果并持久化会话。

运行方式：

python3 code/main.py

追踪日志 (trace) 展示了子智能体的上下文隔离（编排器 (orchestrator) 上下文大小保持有界）、钩子执行以及会话持久化过程。

## 使用它

- **Claude Agent SDK**：适用于优先采用 Claude 且希望复用 Claude Code 架构形态 (harness shape) 的产品。
- **Claude Managed Agents**（托管智能体 (Managed Agents)）：适用于托管的长时间运行异步任务。
- **OpenAI Agents SDK**（第 16 课）：适用于优先采用 OpenAI 的对应方案。
- **LangGraph + 自定义工具**：如果你更倾向于使用基于图的状态机 (graph-shaped state machine) 架构。

## 部署它

`outputs/skill-claude-agent-scaffold.md` 提供了 Claude Agent SDK 应用的脚手架 (scaffold)，包含子智能体、钩子、会话存储、MCP 服务器 (MCP server) 接入以及 W3C 追踪上下文传播 (W3C trace propagation)。

## 练习

1. 添加一个子智能体生成器 (spawner)，将 20 个任务分批，每批由 5 个并行子智能体处理。对比测量编排器上下文大小与“每个任务一个智能体”方案的差异。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行速率限制 (rate-limit)（每个会话每分钟最多 5 次）。追踪其行为表现。
3. 将 `list_subkeys` 接入以渲染子智能体树状结构。深度嵌套时会呈现何种形态？
4. 将此示例移植到正式的 `claude-agent-sdk` Python 包中。工具注册 (tool registration) 机制会发生哪些变化？
5. 阅读 Claude Managed Agents 文档。在什么情况下你会从自托管 (self-hosted) 切换到托管服务？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 智能体 SDK (Agent SDK) | “将 Claude Code 作为库使用” | 架构形态 (harness shape)：包含工具、MCP、钩子、子智能体、会话存储 |
| 子智能体 (Subagent) | “子级智能体” | 独立上下文、自有预算；结果向上冒泡返回 |
| 会话存储 (Session store) | “对话数据库” | 持久化、加载、列表、删除对话轮次，支持子智能体级联 |
| 钩子 (Hook) | “生命周期回调” | 工具调用前后、会话、提示词提交、压缩、停止等阶段触发 |
| W3C 追踪上下文 (W3C trace context) | “跨进程追踪” | 父级跨度 (span) 传播至 CLI 子进程中 |
| 托管智能体 (Managed Agents) | “托管架构” | 由 Anthropic 托管的长时间运行异步任务 |
| `--session-mirror` | “转录镜像” | 在会话轮次流式输出时，将其同步写入外部文件 |
| MCP 服务器 (MCP server) | “工具接口层” | 附加到智能体的外部工具/资源来源 |

## 延伸阅读

- [Claude 智能体 SDK（Agent SDK）概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库（Library）形态
- [Anthropic：使用 Claude Agent SDK 构建智能体](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式（Production Patterns）
- [Claude 托管智能体（Managed Agents）概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案（Hosted Alternative）
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 同类对标方案（Counterpart）
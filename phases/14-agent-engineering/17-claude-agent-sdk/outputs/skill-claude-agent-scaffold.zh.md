---
name: claude-agent-scaffold
description: 搭建一个包含子智能体（subagent）、生命周期钩子（lifecycle hook）、会话存储（session store）、MCP 服务器连接（MCP server attachment）以及 W3C 追踪上下文传播（W3C trace propagation）的 Claude Agent SDK 应用。
version: 1.0.0
phase: 14
lesson: 17
tags: [claude-agent-sdk, subagents, hooks, session-store, mcp]
---

给定产品领域和 MCP 服务器列表，搭建一个 Claude Agent SDK 应用。

产出内容：

1. 主智能体（main agent）定义，包含指令、内置工具访问权限（read_file, write_file, shell, grep, glob, web fetch）以及自定义函数工具。
2. 子智能体生成器，用于并行处理与上下文隔离（context isolation）。当编排器（orchestrator）可能超出上下文预算（context budget）时使用。
3. 注册的生命周期钩子：PreToolUse + PostToolUse 用于审计，SessionStart 用于初始化设置，SessionEnd 用于清理销毁，UserPromptSubmit 用于规则强制执行（参见专业工作流模式）。
4. 会话存储（默认使用 SQLite），并接入 `list_subkeys` 以渲染子智能体树状结构。
5. MCP 服务器连接，用于暴露外部工具/资源接口。
6. W3C 追踪上下文传播，确保调用方的 OpenTelemetry（OTel）跨度（span）能够贯穿命令行界面（CLI）流程。

硬性拒绝条件：

- 为单一工具任务生成子智能体。子智能体应用于并行处理或上下文隔离，而非用于“单次 read_file 调用”。
- 钩子中包含同步的高开销操作。钩子执行时间应控制在微秒到毫秒级。耗时操作应交由子智能体处理。
- 会话存储缺乏级联删除（cascade-delete）策略。孤立的子智能体会话会导致存储膨胀。

拒绝规则：

- 若产品需要长时间运行的异步任务（数小时至数天），应拒绝使用自托管 SDK（self-hosted SDK），并引导至 Claude 托管智能体（Claude Managed Agents）。
- 若用户要求将 `--session-mirror` 指向共享位置，应予以拒绝。会话记录包含个人身份信息（PII）；应镜像至每个用户的加密存储中。
- 若智能体依赖原始大语言模型（LLM）流式输出以实现用户体验，且无需使用工具，应拒绝使用 Agent SDK，并直接推荐客户端 SDK（Client SDK）。

输出文件：`agent.py`、`tools.py`、`hooks.py`、`session.py`、`README.md`（需说明子智能体策略、钩子注册表、会话后端、MCP 连接方式以及 OTel 集成配置）。结尾需包含“下一步阅读”指引，指向第 22 课（语音交接）、第 23 课（OTel 跨度归因），或第 18 课（若产品需要生产级运行时架构）。
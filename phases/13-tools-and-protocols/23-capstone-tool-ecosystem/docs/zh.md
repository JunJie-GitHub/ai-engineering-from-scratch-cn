# 综合实战项目（Capstone）— 构建完整的工具生态系统

> 第 13 阶段讲解了各个独立组件。本综合实战项目将它们整合为一个具备生产级架构的系统：一个集成工具（tools）、资源（resources）、提示词（prompts）、任务（tasks）与用户界面（UI）的模型上下文协议（MCP）服务器，位于边缘的 OAuth 2.1 网关，基于角色的访问控制（RBAC）网关，多服务器客户端，智能体到智能体（A2A）子智能体调用，接入收集器的开放遥测（OTel）链路追踪，持续集成（CI）流水线中的工具投毒检测，以及 AGENTS.md 与 SKILL.md 配置包。完成本项目后，你将能够为每一项架构决策提供充分依据。

**类型：** 构建
**编程语言：** Python（标准库，端到端生态测试框架）
**前置要求：** 第 13 阶段 · 01 至 21
**预计耗时：** 约 120 分钟

## 学习目标

- 构建一个 MCP 服务器，对外暴露工具、资源、提示词，并集成一个基于 `ui://` 协议的应用任务。
- 在服务器前端部署 OAuth 2.1 网关，强制实施基于角色的访问控制（RBAC）与哈希值固定（pinned hashes）校验。
- 编写一个多服务器客户端，实现端到端的开放遥测（OTel）生成式人工智能（GenAI）属性链路追踪。
- 将部分工作负载委派给智能体到智能体（A2A）子智能体；验证其内部不透明性（opacity）得以保留。
- 使用 AGENTS.md 与 SKILL.md 打包整个技术栈，以便其他智能体能够调用并驱动该系统。

## 问题背景

交付“研究与报告”系统：

- 用户提问：“总结 2026 年关于智能体协议（agent protocols）引用量最高的三篇 arXiv 论文。”
- 系统响应：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要任务委派给专业的写作智能体；汇总结果；将交互式报告渲染为 MCP Apps 的 `ui://` 资源；将每一步操作记录至 OTel。

第 13 阶段涵盖的所有基础原语（primitives）均在此体现。这并非玩具项目——Anthropic（Claude Research 产品）、OpenAI（搭载 Apps SDK 的 GPTs）以及第三方厂商在 2026 年交付的生产级研究助手系统，其架构形态与此完全一致。

## 核心概念

### 架构

[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans

### 追踪层级

agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)

仅使用一个追踪 ID（Trace ID）。每个跨度（Span）均具备正确的 `gen_ai.*` 属性。

### 安全策略

- 采用 OAuth 2.1 + 代码交换证明密钥（PKCE），并通过资源指示符将受众（Audience）固定至网关。
- 网关负责保管上游凭据；用户端永远无法直接查看。
- 基于角色的访问控制（RBAC）：`alice` 拥有 `research:read` 与 `research:write` 权限，可调用所有工具。`bob` 仅拥有 `research:read` 权限，无法调用 `generate_report`。
- 固定描述清单（Pinned Description Manifest）：若检测到任何服务器的工具哈希值发生变更，则自动将其剔除。
- “双重规则”审计（Rule of Two Audit）：确保没有任何工具会同时组合不可信输入、敏感数据与关键性操作。

### 渲染

最终的 `generate_report` 任务会返回内容块以及一个 `ui://report/current` 资源。客户端宿主环境（如 Claude Desktop 等）将在沙盒内联框架（iframe）中渲染交互式仪表盘。该仪表盘包含排序后的论文列表、引用次数统计，以及一个按钮；当用户点击任意论文时，该按钮会调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 项目打包

整个系统以如下目录结构交付：

research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest

用户可通过 `docker compose up` 命令进行部署。Claude Code、Cursor、Codex 和 opencode 的用户只需调用 `run-research` 技能（Skill）即可驱动该系统。

### 第 13 阶段各课程的核心贡献

| 课程 | 综合项目中的应用 |
|--------|------------------------|
| 01-05 | 工具接口、提供商可移植性、并行调用、数据模式（Schema）、代码检查（Linting） |
| 06-10 | 模型上下文协议（MCP）原语、服务端、客户端、传输协议、资源与提示词 |
| 11-14 | 采样（Sampling）、根目录与引导机制（Elicitation）、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒防护、OAuth 2.1、网关与注册表 |
| 18 | 智能体到智能体（A2A）子智能体委派 |
| 19 | 开放遥测（OpenTelemetry/OTel）生成式 AI 追踪 |
| 20 | 面向大语言模型（LLM）层的路由网关 |
| 21 | `SKILL.md` 与 `AGENTS.md` 打包规范 |

## 使用它

`code/main.py` 将前几节课的设计模式整合为一个可运行的演示程序。全部基于标准库 (stdlib)，且均在单进程内运行，便于你通读完整代码。它完整执行了“研究与报告”场景的全流程：与网关 (gateway) 握手、模拟 OAuth 2.1 认证、合并工具列表 (tools/list)、将 `generate_report` 作为任务执行、向 writer 发起 A2A (Agent-to-Agent) 调用、返回 `ui://` 资源，以及输出 OTel (OpenTelemetry) 跨度 (span)。

重点关注以下内容：

- 所有跳转节点共享同一个追踪 ID (trace id)。
- 网关策略会阻止第二个用户执行写入操作。
- 任务生命周期经历 working → completed 状态，并同时返回文本和 `ui://` 内容。
- A2A 调用的内部状态对编排器 (orchestrator) 不可见。
- `AGENTS.md` 和 `SKILL.md` 是其他智能体 (agent) 复现该工作流所需的唯一文件。

## 交付成果

本节课将生成 `outputs/skill-ecosystem-blueprint.md` 文件。针对具体的产品需求（如研究、摘要生成或自动化），该技能 (skill) 会输出完整的架构设计：包括使用哪些 MCP (Model Context Protocol) 原语 (primitives)、哪些网关控制策略、哪些 A2A 调用、哪些遥测 (telemetry) 方案以及何种打包方式。

## 练习

1. 运行 `code/main.py`。观察单一的追踪 ID (trace id) 以及跨度 (span) 的嵌套方式。统计该演示程序涉及了 Phase 13 中的多少个原语 (primitives)。
2. 扩展示例：添加第二个后端 MCP 服务器（例如 `bibliography`），并确认网关将其工具合并到了同一命名空间 (namespace) 中。
3. 将模拟的 A2A writer 智能体替换为在子进程 (subprocess) 中运行的真实智能体。请使用 Lesson 19 的测试脚手架 (harness)。
4. 在编排器与大语言模型 (LLM) 之间的路由网关中，添加个人身份信息 (PII) 脱敏步骤。确认用户查询中的电子邮件地址已被清除。
5. 为负责维护该系统的队友编写一份 `AGENTS.md`。阅读时间应控制在五分钟以内，并提供他们在 Cursor 或 Codex 中驱动毕业项目 (capstone) 所需的全部信息。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 毕业项目 (Capstone) | “Phase-13 集成演示” | 使用所有原语的端到端系统 |
| 研究与报告 (Research and report) | “该场景” | 搜索、摘要生成与渲染模式 |
| 生态系统 (Ecosystem) | “所有组件的组合” | 服务器 + 客户端 + 网关 + 子智能体 + 遥测 + 打包 |
| 追踪层级 (Trace hierarchy) | “单一追踪 ID” | 每个跳转节点的跨度共享同一追踪 ID；通过跨度 ID 建立父子关系 |
| 网关签发令牌 (Gateway-issued token) | “传递式认证” | 客户端仅能看到网关的令牌；网关持有上游凭证 |
| 合并命名空间 (Merged namespace) | “所有工具置于单一扁平列表” | 网关处合并多服务器工具，冲突时添加前缀 |
| 不透明边界 (Opacity boundary) | “A2A 调用隐藏内部细节” | 子智能体的推理过程对编排器不可见 |
| 三层架构栈 (Three-layer stack) | “AGENTS.md + SKILL.md + MCP” | 项目上下文 + 工作流 + 工具 |
| 纵深防御 (Defense-in-depth) | “多层安全防护” | 哈希值固定校验 (Pinned hashes)、OAuth、基于角色的访问控制 (RBAC)、双人原则 (Rule of Two)、审计日志 |
| 规范合规矩阵 (Spec compliance matrix) | “我们交付物中符合规范的部分” | 将交付成果映射至 2025-11-25 版本需求的检查清单 |

## 延伸阅读

- [模型上下文协议 (MCP) — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 综合参考文档
- [MCP 博客 — 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议演进方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 智能体间通信协议 (A2A) v1.0 参考文档
- [开放遥测 (OpenTelemetry) — 生成式 AI 语义约定 (GenAI Semconv)](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 标准链路追踪规范
- [Anthropic — Claude 智能体 SDK (Agent SDK) 概览](https://code.claude.com/docs/en/agent-sdk/overview) — 生产级智能体运行时模式
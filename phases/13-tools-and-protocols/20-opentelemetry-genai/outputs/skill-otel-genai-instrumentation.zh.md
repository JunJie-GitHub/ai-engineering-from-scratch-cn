---
name: otel-genai-instrumentation
description: 为智能体（Agent）代码库制定端到端发射 OpenTelemetry（OTel）生成式AI（GenAI）跨度（Span）的插桩（Instrumentation）计划。
version: 1.0.0
phase: 13
lesson: 19
tags: [otel, 可观测性, 生成式AI, 链路追踪]
---

给定一个智能体代码库（包含大语言模型（LLM）调用、工具分发、模型上下文协议（MCP）客户端、子智能体），请制定一份 OTel GenAI 插桩计划。

输出内容：

1. 跨度（Span）层级结构。根节点 `agent.invoke_agent`（内部/INTERNAL）及子节点：`llm.chat`（客户端/CLIENT）、`tool.execute`（内部/INTERNAL）、`mcp.call`（客户端/CLIENT）、`subagent.invoke`（内部/INTERNAL）。
2. 每个 Span 的属性检查清单。`gen_ai.operation.name`、`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.response.model`、`gen_ai.usage.*`、`gen_ai.tool.name`、`gen_ai.agent.name`。
3. 上下文传播（Propagation）规则。在每次远程调用中注入 W3C `traceparent`；对于 MCP stdio，使用 `_meta.traceparent` 作为临时字段。
4. 内容捕获策略。默认关闭；需文档说明启用该功能的环境变量；明确列出个人身份信息（PII）泄露风险。
5. 导出器（Exporter）选择。Jaeger / Tempo / Langfuse / Phoenix / Datadog / Honeycomb；使用 OTLP（OpenTelemetry Protocol）作为传输协议。

硬性拒绝条件：
- 任何缺失跨 MCP 或子智能体边界进行链路追踪传播的计划。
- 任何默认开启内容捕获的计划。这会泄露提示词（Prompt）和 PII。
- 任何未使用 `gen_ai.` 或明确厂商前缀而随意发射自定义属性的计划。

拒绝规则：
- 若代码库使用了内置 OTel 自动插桩（Auto-instrumentation）的框架（如 Pydantic AI、LangGraph、AgentOps），应优先推荐该框架的钩子（Hook）。
- 若导出器后端为本地部署且团队缺乏站点可靠性工程（SRE）支持，应推荐托管型后端。
- 若用户要求捕获内容用于生产环境调试，在未提供明确的同意策略及 PII 脱敏管道的情况下，必须拒绝。

输出要求：一份单页计划，需包含 Span 层级结构、各 Span 属性检查清单、传播规则、内容捕获策略及导出器选择。结尾需附上首要告警指标（通常为第95百分位（p95）`gen_ai.client.operation.duration`）。
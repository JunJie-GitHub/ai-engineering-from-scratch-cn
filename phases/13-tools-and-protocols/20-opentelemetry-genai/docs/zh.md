# OpenTelemetry GenAI —— 端到端追踪工具调用

> 一个智能体（Agent）调用了五个工具、三个 MCP 服务器和两个子智能体。你需要一条贯穿所有这些操作的追踪链路（Trace）。OpenTelemetry GenAI 语义规范（Semantic Conventions，v1.37 及以上版本提供稳定属性）是 2026 年的行业标准，已获得 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 的原生支持。本课程将列出必需属性，梳理跨度（Span）层级结构（智能体 → 大语言模型（LLM） → 工具），并提供一个基于标准库的跨度发射器（Span Emitter），可直接接入任意 OTel 导出器（Exporter）。

**类型：** 构建实践
**语言：** Python（标准库、OTel 跨度发射器）
**前置条件：** 第 13 阶段 · 07（MCP 服务器）、第 13 阶段 · 08（MCP 客户端）
**耗时：** 约 75 分钟

## 学习目标

- 明确大语言模型（LLM）跨度和工具执行跨度所需的 OTel GenAI 属性。
- 构建涵盖智能体循环、LLM 调用、工具调用以及 MCP 客户端分发的追踪层级结构。
- 决定哪些内容需要捕获（需主动启用）与哪些内容需要脱敏（默认行为）。
- 将跨度发送至本地收集器（Collector，如 Jaeger、Langfuse），且无需重写工具代码。

## 问题背景

2026 年 2 月的一次调试记录：用户反馈“我的智能体有时需要 30 秒才能响应，有时却只需 3 秒。”缺乏追踪链路。日志仅显示了 LLM 调用，却未记录工具分发、MCP 服务器往返通信以及子智能体的执行情况。你只能靠猜测。最终你发现：某个 MCP 服务器在冷启动（Cold Start）时偶尔会发生挂起。

如果没有端到端追踪，你根本无法定位此类问题。而 OTel GenAI 正是为了解决这一痛点而生。

该规范于 2025 至 2026 年间在 OpenTelemetry 语义规范工作组中最终定稿。它定义了稳定的属性名称，确保 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的跨度数据。只需一次插桩（Instrumentation），即可对接任意后端（Backend）。

## 核心概念

### 跨度（Span）层级结构

agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)

整个结构嵌套在同一个追踪 ID（Trace ID）下。跨度 ID（Span ID）用于关联父子关系。

### 必需属性（Required Attributes）

根据 2025-2026 版语义约定（Semantic Conventions, semconv）：

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际提供服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 用于关联的提供商响应 ID。

对于工具跨度（Tool Span）：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体的调用 ID。
- `gen_ai.tool.description` — 工具描述（可选）。

对于智能体跨度（Agent Span）：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### 跨度类型（Span Kind）

- `SpanKind.CLIENT`：用于跨越进程边界的调用（如大语言模型提供商、MCP 服务器）。
- `SpanKind.INTERNAL`：用于智能体自身的循环步骤和工具执行。

### 按需内容捕获（Opt-in Content Capture）

默认情况下，跨度仅携带指标（Metric）和时序数据，不包含提示词（Prompt）或补全结果（Completion）。大型负载与个人身份信息（PII）默认处于关闭状态。如需包含内容，请设置环境变量 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 及特定的内容捕获环境变量。在生产环境启用前，请务必仔细评估。

### 跨度事件（Span Event）

可将 Token 级别的事件作为跨度事件添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在跨度内按时间顺序排列，便于进行详细回放。

### 导出器（Exporter）

OpenTelemetry（OTel）跨度可导出至：

- **Jaeger / Tempo**：开源软件（OSS），支持本地部署（On-premises）。
- **Langfuse**：专为大语言模型可观测性（LLM Observability）设计；可视化 Token 使用情况。
- **Arize Phoenix**：结合评估（Evaluation）与追踪（Tracing）功能。
- **Datadog**：商业软件；原生解析 `gen_ai.*` 属性。
- **Honeycomb**：面向列存储；查询友好。

它们均使用 OpenTelemetry 协议（OTLP）作为网络传输格式。您的代码无需关心底层差异。

### 跨 MCP 传播（Propagation across MCP）

当 MCP 客户端调用服务器时，需将 W3C `traceparent` 请求头注入到请求中。Streamable HTTP 支持标准请求头。标准输入输出（Stdio）原生不携带 HTTP 请求头；该规范 2026 年的路线图讨论了在 JSON-RPC 调用中添加 `_meta.traceparent` 字段。

在该功能正式发布前：请手动将 `traceparent` 包含在每个请求的 `_meta` 字段中。服务器将记录追踪 ID。

### 指标（Metric）

除跨度外，GenAI 语义约定还定义了以下指标：

- `gen_ai.client.token.usage` — 直方图（Histogram）。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

这些指标适用于无需单次调用详细数据的仪表盘。

### AgentOps 层（AgentOps Layer）

AgentOps（成立于 2024 年）专注于生成式 AI 可观测性（GenAI Observability）。它封装了主流框架（如 LangGraph、Pydantic AI、CrewAI），以自动发射 OTel 跨度。如果您的技术栈使用了受支持的框架，它将非常实用；否则，请使用手动插桩（Manual Instrumentation）。

## 使用方法
`code/main.py` 会向标准输出 (stdout) 发射符合 OpenTelemetry (OTel) 规范的跨度 (Span)（采用类似 OTLP-JSON 的格式），用于模拟一个调用大语言模型 (LLM)、调度两个工具 (Tool) 并完成一次模型上下文协议 (MCP) 往返交互的智能体 (Agent)。本示例不包含真实的导出器 (Exporter)——教程的重点在于跨度 (Span) 的结构与属性集。你可以将输出粘贴至兼容 OTLP 的查看器中，或直接阅读文本。

观察重点：
- 所有跨度 (Span) 共享同一个追踪 ID (Trace ID)。
- 父子层级关系通过 `parentSpanId` 进行编码。
- 必填的 `gen_ai.*` 属性均已填充。
- 内容捕获 (Content capture) 默认处于关闭状态；其中一个场景会通过环境变量 (Environment Variable) 将其开启。

## 交付成果
本教程将生成 `outputs/skill-otel-genai-instrumentation.md` 文件。给定一个智能体 (Agent) 代码库，该技能 (Skill) 会生成一份插桩计划 (Instrumentation Plan)：明确在何处添加跨度 (Span)、需要填充哪些属性，以及目标导出器 (Exporter) 的选择。

## 练习
1. 运行 `code/main.py`。统计跨度 (Span) 的数量，并区分哪些属于 CLIENT（客户端）类型，哪些属于 INTERNAL（内部）类型。
2. 开启内容捕获 (Content capture)（通过环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件是否出现。注意其对个人身份信息 (PII) 的潜在影响。
3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并在每次调用时将其作为直方图 (Histogram) 样本发射出去。
4. 将父级智能体 (Agent) 跨度 (Span) 中的 `traceparent` 传播到 MCP 请求的 `_meta.traceparent` 字段中。验证 MCP 服务器是否能获取到相同的追踪 ID (Trace ID)。
5. 阅读 OTel GenAI 语义约定 (Semantic Conventions) 规范。找出规范中列出但本教程代码未发射的一个属性，并将其添加进去。

## 关键术语
| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | 用于追踪 (Traces)、指标 (Metrics) 和日志 (Logs) 的开放标准 |
| GenAI semconv | "GenAI semantic conventions" | 用于 LLM / 工具 / 智能体 (Agent) 跨度 (Span) 的稳定属性命名规范 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性均共享此前缀 |
| Span | "定时操作" | 具有开始时间、结束时间和属性的工作单元 |
| Trace | "跨 Span 的谱系" | 共享同一追踪 ID (Trace ID) 的跨度 (Span) 树状结构 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 提示跨度 (Span) 的方向或角色 |
| OTLP | "OpenTelemetry Line Protocol" | 供导出器 (Exporter) 使用的网络传输格式 |
| Opt-in content | "提示词/补全内容捕获" | 默认关闭；需通过环境变量 (Environment Variable) 启用 |
| traceparent | "W3C 请求头" | 用于在服务间传播追踪上下文 |
| Exporter | "特定后端的发送器" | 负责将跨度 (Span) 发送至 Jaeger / Datadog 等后端的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 生成式 AI (GenAI) 跨度 (span)、指标 (metric) 与事件 (event) 的权威语义约定 (semantic convention)
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — 大语言模型 (LLM) 与工具执行 (tool-execution) 的跨度属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体 (agent) 级别的 `invoke_agent` 跨度
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — 托管于 GitHub 的权威参考源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产环境集成实操指南
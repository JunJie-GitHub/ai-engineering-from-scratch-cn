# OpenTelemetry GenAI 语义规范 (Semantic Conventions)

> OpenTelemetry 的 GenAI 特别兴趣组 (SIG)（2024 年 4 月成立）定义了智能体遥测 (agent telemetry) 的标准模式。各供应商在跨度 (Span) 名称、属性以及内容捕获规则上趋于统一，从而确保智能体追踪 (agent traces) 在 Datadog、Grafana、Jaeger 和 Honeycomb 等平台上具有完全一致的含义。

**类型：** 学习与实践
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 24（可观测性平台）
**预计耗时：** 约 60 分钟

## 学习目标

- 列举 GenAI 的跨度 (Span) 类别：模型/客户端 (model/client)、智能体 (agent)、工具 (tool)。
- 区分 `invoke_agent` 的客户端 (CLIENT) 与内部 (INTERNAL) 跨度 (Span)，并明确各自的适用场景。
- 列出 GenAI 的顶层属性：提供商名称 (provider name)、请求模型 (request model)、数据源 ID (data-source ID)。
- 解释内容捕获约定 (content-capture contract)：主动启用 (opt-in)、环境变量 `OTEL_SEMCONV_STABILITY_OPT_IN` 以及外部引用 (external-reference) 建议。

## 问题背景

各供应商各自定义跨度 (Span) 名称，导致运维团队不得不为每个框架单独构建监控面板。OpenTelemetry 的 GenAI 特别兴趣组 (SIG) 通过制定一项全生态统一遵循的标准，彻底解决了这一问题。

## 核心概念

### Span（跨度）类别

1. **模型/客户端 Span（Model / client spans）**。涵盖原始的大语言模型（LLM）调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）及框架模型适配器发出。
2. **智能体（Agent）Span**。包含 `create_agent`（构建智能体时）和 `invoke_agent`（运行智能体时）。
3. **工具（Tool）Span**。每次工具调用对应一个；通过父子关系与智能体 Span 关联。

### 智能体 Span 命名

- Span 名称：若已命名则为 `invoke_agent {gen_ai.agent.name}`；否则回退至 `invoke_agent`。
- Span 类型（Span kind）：
  - **CLIENT** — 适用于远程智能体服务（如 OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 适用于进程内智能体框架（如 LangChain、CrewAI、本地 ReAct）。

### 关键属性（Key attributes）

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 实际解析的模型（可能因路由策略与请求模型不同）。
- `gen_ai.agent.name` — 智能体标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于检索增强生成（RAG）：指明查询了哪个语料库或存储。

针对 Anthropic、Azure AI Inference、AWS Bedrock 和 OpenAI 存在特定于技术的约定。

### 内容捕获（Content capture）

默认规则：插桩（Instrumentations）默认不应捕获输入/输出。需通过以下方式显式启用（Opt-in）捕获：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产环境模式：将内容存储在外部（如 S3 或日志存储中），并在 Span 上记录引用（使用指针 ID，而非完整文本）。这是第 27 课中提到的内容投毒（content-poisoning）防御机制，已集成至可观测性（Observability）体系中。

### 稳定性（Stability）

截至 2026 年 3 月，大多数约定仍处于实验阶段。可通过以下配置启用稳定预览版：

OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental

Datadog v1.37+ 已将 GenAI 属性原生映射至其 LLM 可观测性（LLM Observability）架构中。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 该模式的常见错误实践

- **在 Span 中捕获完整提示词（prompts）**。运维人员可读的追踪（Traces）中会包含个人身份信息（PII）、密钥和客户数据。应改为外部存储。
- **缺少 `gen_ai.provider.name`**。若缺失归属信息，多提供商仪表板将无法正常工作。
- **Span 缺少父级链接**。导致孤立的工具 Span。务必始终传播上下文（Context）。
- **未设置稳定性显式启用（stability opt-in）**。后端升级时，您的属性可能会被重命名。

## 动手构建（Build It）

`code/main.py` 实现了一个符合 GenAI 约定的标准库 Span 发射器（span emitter）：

- 包含 GenAI 属性模式的 `Span`。
- 支持 `start_span` 和嵌套上下文的追踪器（Tracer）。
- 一个脚本化的智能体运行流程，会发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具对应的 Span，以及用于 LLM 调用的 `chat` Span。
- 一种内容捕获模式，将提示词外部存储并在 Span 上记录 ID。

运行方式：

python3 code/main.py

输出结果：包含所有必需 GenAI 属性的 Span 树，以及一个显示已启用内容引用的“外部存储”。

## 实际应用（Use It）

- **Datadog LLM Observability**（v1.37+）原生支持属性映射。
- **Langfuse / Phoenix / Opik**（第 24 课）—— 自动对该生态进行插桩（instrument）。
- **Jaeger / Honeycomb / Grafana Tempo** —— 原始 OTel 追踪数据（traces）；基于 GenAI 属性构建仪表盘。
- **Self-hosted（自托管）** —— 运行带有 GenAI 处理器（processor）的 OTel Collector。

## 部署上线

`outputs/skill-otel-genai.md` 将 OTel GenAI 跨度（spans）集成至现有智能体（agent）中，默认启用内容捕获（content-capture）并支持外部引用存储。

## 练习

1. 使用 `invoke_agent`（INTERNAL）及按工具划分的跨度（per-tool spans）对你的第 01 课 ReAct 循环进行插桩。将数据发送至 Jaeger 实例。
2. 在“仅引用（references only）”模式下添加内容捕获：将提示词（prompts）存入 SQLite，跨度属性仅携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其接入你第 09 课的 Mem0 搜索中。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，并验证你的属性未被收集器（Collector）重命名。
5. 构建仪表盘：仅基于 GenAI 属性分析“哪些工具错误与哪些模型相关”。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | “OpenTelemetry GenAI 小组” | 负责定义该模式的 OTel 工作组 |
| invoke_agent | “智能体跨度（Agent span）” | 表示智能体运行过程的跨度名称 |
| CLIENT span | “远程调用” | 用于调用远程智能体服务的跨度 |
| INTERNAL span | “进程内” | 用于进程内智能体运行的跨度 |
| gen_ai.provider.name | “提供商（Provider）” | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | “RAG 数据源” | 检索命中所对应的语料库/存储 |
| Content capture | “提示词日志记录（Prompt logging）” | 按需捕获消息；在生产环境中外部存储 |
| Stability opt-in | “预览模式” | 用于锁定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定（semantic conventions）](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 官方规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 默认生成 GenAI 跨度
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 内置 OTel 跨度支持
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) —— 支持 W3C 追踪上下文传播（trace context propagation）
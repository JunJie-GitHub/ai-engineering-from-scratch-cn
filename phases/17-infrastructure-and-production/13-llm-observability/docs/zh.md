# LLM 可观测性（Observability）技术栈选型

> 2026 年的可观测性市场分为两大类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估（Evals）、提示词管理（Prompt Management）和会话回放（Session Replays）捆绑在一起。网关/插桩工具（Helicone、SigNoz、OpenLLMetry、Phoenix）则专注于遥测数据（Telemetry）。Langfuse 核心采用 MIT 许可证，在开源与商业化之间取得了良好平衡（云端免费版每月 5 万次事件）。Phoenix 基于 Elastic License 2.0 协议，原生支持 OpenTelemetry —— 非常适合用于漂移（Drift）/检索增强生成（RAG）的可视化，但不适合作为持久化的生产环境后端。Arize AX 采用零拷贝（Zero-Copy）的 Iceberg/Parquet 集成方案，声称其成本比单体可观测性架构低 100 倍。LangSmith 在 LangChain/LangGraph 生态中处于领先地位，定价为 39 美元/用户/月，仅企业版支持私有化部署（Self-Host）。Helicone 基于代理（Proxy）模式，15-30 分钟即可完成配置，每月免费提供 10 万次请求，但在智能体追踪（Agent Traces）的深度上稍显不足。常见的生产环境架构模式：通过 OpenTelemetry 将网关工具（Helicone/Portkey）与独立的评估平台（Phoenix/TruLens）进行集成。

**Type:** 学习
**Languages:** Python（标准库，简易追踪采样模拟器）
**Prerequisites:** 第 17 阶段 · 08（推理指标），第 14 阶段（智能体工程）
**Time:** 约 60 分钟

## 学习目标

- 区分开发平台（捆绑功能：评估 + 提示词 + 会话）与网关/遥测工具（仅包含追踪 + 指标）。
- 梳理六大主流工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）的许可证类型、定价策略及其最佳适用场景。
- 解释 OpenTelemetry 胶水模式（OpenTelemetry-glue pattern），该模式允许你将网关工具与独立的评估平台结合使用。
- 指出 2026 年的核心成本差异点（Arize AX 的零拷贝方案与单体数据摄入（monolithic ingest）架构的对比），并说明其约 100 倍的成本差距。

## 问题背景

你上线了一个 LLM 功能。它能正常运行。但你完全无法洞察提示词失败、工具调用死循环、延迟回退、成本激增或提示词缓存命中率等问题。你在 Google 搜索“LLM 可观测性”，会跳出八个工具，它们都声称能解决同样的问题，且分属三个不同的价格档位。

它们解决的其实并非同一类问题。LangSmith 回答的是“这次 LangGraph 运行为何失败？”；Phoenix 回答的是“我的 RAG 流水线是否发生了数据漂移？”；Helicone 回答的是“哪个应用在疯狂消耗 Token？”；Langfuse 回答的是“我能把整套系统私有化部署吗？”工具不同，面向的受众也不同。

选型需要权衡四个维度：技术栈（LangChain？原生 SDK？多供应商？）、许可证接受度（仅限 MIT？接受 Elastic？商业许可也行？）、预算（免费层？100 美元/月？1000 美元/月？），以及私有化部署需求（必须？锦上添花？完全不需要？）。

## 核心概念

### 两大类别

**开发平台（Development platforms）**将可观测性（observability）与评估（evals）、提示词管理（prompt management）、数据集版本控制（dataset versioning）和会话回放（session replay）整合在一起。你可以运行实验，查看哪个提示词效果更好，并针对历史优胜提示词对新提示词进行数据集回归测试（dataset-regression）。代表工具有 LangSmith、Langfuse、Comet Opik。

**网关/遥测工具（Gateway/telemetry tools）**对推理调用（inference calls）进行插桩（instrument）——记录提示词、响应、Token 数量、延迟、模型和成本。代表工具有 Helicone、SigNoz、OpenLLMetry、Phoenix。这类工具设计极简，可通过 OpenTelemetry 与独立的评估工具组合使用。

### Langfuse —— 开源生态的平衡之选

- 核心代码采用 Apache / MIT 许可证；可通过 Docker 自行托管（self-host）。
- 云端免费版：每月 5 万次事件（events）。付费版：团队版 29 美元/月。
- 提供评估、提示词管理、调用链追踪（traces）和数据集功能。合理覆盖了上述四大开发平台特性。
- 最佳适用场景：你需要 LangSmith 级别的功能，但必须自行托管或坚持使用开源许可证。

### Phoenix (Arize) —— 遥测优先，原生支持 OpenTelemetry

- 采用 Elastic License 2.0 许可证；自行托管非常简单。
- 在检索增强生成（RAG）和漂移（drift）可视化方面表现出色。嵌入空间散点图（Embedding-space scatter plots）作为一等公民功能内置提供。
- 并非设计为持久化的生产环境后端——主要用于开发阶段的可观测性。
- 最佳适用场景：RAG 流水线开发、漂移调试，生产环境可搭配独立的网关使用。

### Arize AX —— 面向大规模场景的解决方案

- 商业软件。通过 Iceberg/Parquet 实现零拷贝（zero-copy）数据湖集成。
- 宣称在大规模场景下，成本比单体可观测性架构（monolithic observability）（如 Datadog 级别）低约 100 倍。其原理在于：你将调用链追踪数据以 Parquet 格式存储在自己的 S3 上，Arize 直接读取。
- 最佳适用场景：每日调用链追踪数据超过 1000 万条，已有数据湖基础设施，希望获得大语言模型（LLM）专属仪表盘，但不想承担 Datadog 级别的高昂定价。

### LangSmith —— 深度集成 LangChain/LangGraph

- 商业软件，39 美元/用户/月。仅企业版支持自行托管。
- 在 LangChain 和 LangGraph 技术栈中表现最佳。如果你未使用这两者，其吸引力会大打折扣。
- 最佳适用场景：团队已全面投入 LangChain 生态，且愿意为此付费。

### Helicone —— 基于代理的最小可行方案

- 只需将你的 `OPENAI_API_BASE` 替换为 Helicone 代理（proxy）地址，15-30 分钟即可完成配置。
- 采用 MIT 许可证；免费版每月 10 万次请求（req），付费版 20 美元/月起。
- 内置故障转移（failover）、缓存和速率限制功能——同时充当网关角色。
- 在智能体（agent）/多步调用链追踪方面的深度相对不足。
- 最佳适用场景：快速启动、单一技术栈应用，需要网关与可观测性二合一。

### Opik (Comet) —— 开源开发平台

- 采用 Apache 2.0 许可证，完全开源。
- 功能集与 Langfuse 类似，并继承了 Comet 的基因。
- 最佳适用场景：机器学习（ML）团队已在使用 Comet，希望在同一控制台内集成大语言模型可观测性。

### SigNoz —— 基于 OpenTelemetry 的全栈应用性能监控（APM）

- 采用 Apache 2.0 许可证。通过 OpenTelemetry 同时处理通用 APM 与大语言模型监控。
- 最佳适用场景：需要跨微服务与大语言模型调用的统一可观测性。

### 粘合剂：OpenTelemetry + 生成式 AI 语义规范（GenAI semantic conventions）

OpenTelemetry 于 2025 年底发布了生成式 AI 语义规范（包含 `gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens` 等字段）。消费 OpenTelemetry（OTel）数据的工具因此能够实现互操作。目前逐渐成型的生产环境模式如下：

1. 从每次大语言模型调用中，按照生成式 AI 规范输出 OTel 数据。
2. 将数据路由至网关（如 Helicone / Portkey）用于日常监控。
3. 双写（dual-ship）至评估平台（如 Phoenix / Langfuse）用于回归测试。
4. 归档至数据湖（如 Iceberg），以便通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误的层级进行插桩

在智能体框架内部进行插桩（例如直接添加 LangSmith 追踪）会导致你与该框架强耦合。而在 HTTP/OpenAI-SDK 层进行插桩（通过 OpenLLMetry 或你的网关）则具备更好的可移植性。

### 采样策略——你无法保留所有数据

当每日请求量超过 100 万次时，保留完整调用链的成本甚至会超过大语言模型调用本身的费用。建议按规则采样：100% 保留错误请求，100% 保留高成本请求，5% 保留成功请求。始终保留聚合数据；仅对长尾（long tail）请求保留原始数据。

### 关键数据备忘

- Langfuse 云端免费版：每月 5 万次事件。
- LangSmith：39 美元/用户/月。
- Helicone 免费版：每月 10 万次请求。
- Arize AX 宣称：在大规模场景下，成本比单体方案低约 100 倍。
- OpenTelemetry 生成式 AI 语义规范：2025 年发布，预计 2026 年广泛采用。

## 使用它

`code/main.py` 模拟了单日产生 100 万条追踪记录（trace）的场景，并对比了不同的数据保留策略（100% 全量采集、采样、采样 + 错误记录）。报告将列出各策略的存储成本及数据丢失情况。

## 交付它

本节将生成 `outputs/skill-observability-stack.md`。根据技术栈、规模、预算和许可证状况，选择相应的工具。

## 练习

1. 你的 LangChain 团队希望采用开源（OSS）且可自托管的可观测性（observability）方案。请在 Langfuse 和 Opik 之间做出选择并说明理由。
2. 在每日 500 万条追踪记录的规模下，若 Datadog 报价为每月 15 万美元，请计算采用 Arize AX 的盈亏平衡点（break-even）。
3. 设计一套 OpenTelemetry GenAI 属性集（attribute set），作为你所在组织规范中要求每次大语言模型（LLM）调用必须包含的内容。
4. 论证仅使用 Phoenix 是否足以满足生产环境（production）需求。在哪些情况下它无法满足要求？
5. Helicone 作为代理会带来 20 毫秒的额外开销（proxy overhead）。若 P99 首字延迟（TTFT）为 300 毫秒，该开销是否可接受？如果服务等级协议（SLA）要求为 100 毫秒呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| OpenLLMetry | “面向 LLM 的 OTel” | 用于大语言模型的开源 OpenTelemetry 插桩（instrumentation） |
| GenAI conventions | “OTel 属性” | 针对 LLM 调用的标准 OpenTelemetry 属性名称 |
| LangSmith | “LangChain 可观测性” | 与 LangChain 生态绑定的商业平台 |
| Langfuse | “开源版 LangSmith” | 采用 MIT 许可证的开源项目，功能集相似 |
| Phoenix | “Arize 开发工具” | 原生支持 OpenTelemetry 的开发与评估（eval）平台 |
| Arize AX | “规模化可观测性” | 商业级零拷贝（zero-copy）Iceberg/Parquet 可观测性方案 |
| Helicone | “代理可观测性” | 收集 LLM 遥测数据（telemetry）并提供网关功能的 HTTP 代理 |
| Opik | “Comet LLM” | Comet 推出的 Apache 2.0 开源开发平台 |
| Session replay | “追踪重放” | 完整重放包含工具调用的智能体（agent）会话 |
| Eval | “离线测试” | 在标注数据集上运行候选模型或提示词（prompt） |

## 延伸阅读

- [SigNoz — 2026 年顶级 LLM 可观测性工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 配置 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)
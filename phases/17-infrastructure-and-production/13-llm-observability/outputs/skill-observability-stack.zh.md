---
name: observability-stack
description: 根据技术栈、规模、预算和许可证要求，选择大语言模型（LLM）可观测性技术栈（observability stack）（开发平台 + 网关 + 可选的扩展层），并定义 OpenTelemetry 生成式人工智能（GenAI）属性集。
version: 1.0.0
phase: 17
lesson: 13
tags: [可观测性, langfuse, langsmith, phoenix, arize, helicone, opik, opentelemetry, genai-conventions]
---

根据所选技术栈（LangChain / DSPy / 原生 SDK）、规模（每日追踪数 (traces/day)）、预算、许可证要求（仅限 MIT 许可证 vs 允许商业许可证）以及自托管需求，制定一份可观测性（observability）方案。

输出内容：

1. 开发平台选择。Langfuse（开源软件 (OSS)）、LangSmith（优先支持 LangChain 的商业平台）、Opik（Comet 开源版）或不使用。需结合技术栈与许可证要求进行论证。
2. 网关/遥测选择。Helicone（代理 + 网关）、SigNoz（全链路应用性能监控 (APM)）、OpenLLMetry（纯 OpenTelemetry (OTel)）。若已在使用 AI 网关（第 17 阶段 · 第 19 课），请指明集成方案。
3. 扩展/数据湖层。可选；使用 Arize AX 或原始 Iceberg 进行长期分析，使用 Phoenix 监控检索增强生成 (RAG) 漂移。
4. OTel GenAI 规范。指定最小属性集：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.request.temperature`、`gen_ai.response.finish_reasons`，以及组织特定属性（`tenant_id`、`user_id`、`task`）。
5. 采样策略。100% 错误请求、100% 高成本请求（>0.10 美元/次）、N% 成功请求采样率。原始数据保留窗口（14 天 / 30 天 / 90 天）。聚合数据保留时间更长。
6. 告警机制。必须配置告警的五项指标：错误率、P99 首字延迟 (TTFT)、单次请求成本、提示词缓存命中率、拒绝率。

硬性拒绝项：
- 仅在特定框架 SDK 内部进行埋点，且未提供 OpenTelemetry (OTel) 回退方案。拒绝——会导致框架锁定。
- 针对非受监管工作负载，以 Datadog 级别定价（>500 美元/月）保留 100% 的追踪数据。拒绝——建议采用采样。
- 忽略 OpenTelemetry GenAI 规范。拒绝——2026 年的互操作性要求必须遵循该规范。

拒绝规则：
- 若每日追踪数 > 500 万，且团队坚持使用 Datadog 进行全量数据保留，在未提供成本预测的情况下予以拒绝。
- 若团队仅限 MIT 许可证，却选择了 LangSmith，予以拒绝——Langfuse 是符合 MIT 许可证的等效替代方案。
- 若团队尚未部署 AI 网关，且选择 Helicone 同时作为网关和可观测性平台，予以接受——该代理在约 500 次请求/秒 (RPS) 以内可兼任网关（第 17 阶段 · 第 19 课涵盖网关扩展规模）。

输出要求：一份单页方案，需明确开发平台、网关、扩展层（如有）、OTel 属性集、采样规则及五项告警指标。结尾需附上唯一用于指示技术栈漂移的指标：过去 7 天内具备完整 OTel GenAI 属性的 LLM 调用占比。
---
name: llm-observability
description: 构建一个自托管的大语言模型可观测性（LLM Observability）仪表板，用于接收 OpenTelemetry GenAI 跨度（Spans）、运行评估（Evals），并在五分钟内捕获注入的回归问题（Regressions）。
version: 1.0.0
phase: 19
lesson: 11
tags: [综合项目, 可观测性, otel, langfuse, phoenix, 评估, 漂移, clickhouse]
---

针对至少涵盖六个 SDK 系列（OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM）的生产环境大语言模型流量，部署一套自托管的可观测性平台。该平台需接收符合 OTLP GenAI 语义约定（GenAI-semconv）的跨度，运行模型评估，检测数据漂移（Drift），并触发告警。

构建计划：

1. 配置 OpenTelemetry Collector，包含 OTLP HTTP 接收器、尾部采样（Tail-sampling）处理器（保留 100% 错误、10% 成功、100% 高毒性/个人身份信息（PII）数据），以及导出至 ClickHouse 与 S3 的导出器。
2. 设计 ClickHouse 跨度表结构，以映射 GenAI 语义约定：包含 `gen_ai.system`、`gen_ai.request.model`、`usage.input/output_tokens`、`latency_ms`、`user_id`、`app_id`，以及用于存储提示词（Prompts）与补全内容（Completions）的 JSON 字段包。
3. 部署 Postgres 元数据存储，用于管理应用、用户、会话及标注队列。
4. 在每个 SDK 系列的客户端应用上启用 OpenLLMetry 自动插桩（Auto-instrumentation）；验证标准跨度（Canonical Spans）能否正确上报。
5. 基于采样后的追踪数据（Traces），定时调度 DeepEval、RAGAS 与 Phoenix 评估器套件；针对 PII 和偏离策略（Off-policy）行为定制大语言模型裁判（LLM-judge）。
6. 每周对聚合的提示词嵌入向量（Prompt Embeddings）运行 PSI / KL 漂移检测器；告警阈值设为 0.2。
7. 配置 Prometheus 导出器以汇总评估分数与延迟百分位数（Latency Percentiles）；通过 Alertmanager 将告警路由至 Slack（警告级别）与 PagerDuty（严重级别）。
8. 开发基于 Next.js 15 App Router 的仪表板：包含概览、追踪搜索与瀑布图（Waterfall）、评估趋势、漂移图表及告警面板。
9. 回归探测：注入一种有 1% 概率泄露伪造社会安全号码（SSNs）的响应模式；测量平均恢复时间（MTTR，即从告警触发到解决的时间）。

评估标准：

| Weight | Criterion | Measurement |
|:-:|---|---|
| 25 | 跨度表结构覆盖率 | 生成标准 GenAI 跨度的 SDK 系列数量（目标 6+） |
| 20 | 评估准确性 | DeepEval / RAGAS 评分与人工标注数据集的对比结果 |
| 20 | 仪表板用户体验 | 针对注入回归问题的平均恢复时间（MTTR）（目标低于 5 分钟） |
| 20 | 成本与扩展性 | 在无积压情况下持续接收 1k spans/sec 的流量 |
| 15 | 告警与漂移检测 | Prometheus/Alertmanager 链路完成端到端验证 |

硬性否决项：

- 跨度表结构中使用了 OpenTelemetry GenAI 语义约定之外的自定义属性名称。
- 尾部采样策略丢弃了错误数据（一种公认的反模式（Anti-pattern））。
- 评估任务以全量接收速率运行且未进行采样（成本不可接受）。
- 仪表板仅展示“延迟”而未区分 p50/p95/p99 百分位数。

拒绝规则：

- 若无个人身份信息（PII）脱敏策略（Redaction Policy），则拒绝持久化存储提示词或补全内容。
- 若未针对每个 SDK 进行标准跨度回归测试，则拒绝宣称“多 SDK 支持”。
- 若未设定基线窗口（Baseline Window），则拒绝交付漂移检测功能；零样本漂移（Zero-shot Drift）检测毫无意义。

交付物：一个代码仓库，需包含 Collector 配置、ClickHouse 表结构、Next.js 15 仪表板、评估任务、漂移检测器、告警链路、带有标注回归问题的 1 万条追踪演示数据集，以及一份说明文档。文档需记录针对注入 PII 回归问题的平均恢复时间（MTTR），并列出在迭代过程中使 MTTR 下降的前三项仪表板用户体验改进。
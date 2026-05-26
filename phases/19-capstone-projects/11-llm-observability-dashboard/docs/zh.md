# 综合项目 11 — LLM 可观测性与评估仪表盘

> Langfuse 转向了开放核心 (Open-core) 模式。Arize Phoenix 发布了 2026 年 GenAI 语义规范 (GenAI Semantic Conventions) 映射。Helicone 和 Braintrust 均进一步深耕了按用户维度的成本归因 (Cost Attribution)。Traceloop 的 OpenLLMetry 已成为事实上的 SDK 插桩 (Instrumentation) 标准。生产环境的典型架构为：使用 ClickHouse 存储追踪数据 (Traces)，Postgres 存储元数据，Next.js 构建 UI，并部署一支由评估任务 (Eval Jobs)（如 DeepEval、RAGAS、LLM-judge）组成的小型作业集群，对采样追踪数据进行处理。请构建一个自托管 (Self-hosted) 仪表盘，至少从四个 SDK 家族接入数据，并演示如何在五分钟内捕获一次故意注入的回归缺陷 (Regression)。

**类型：** 综合项目
**编程语言：** TypeScript（UI）、Python / TypeScript（数据接入 + 评估）、SQL（ClickHouse）
**前置要求：** 第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P11 · P13 · P17 · P18
**预计耗时：** 25 小时

## 问题描述

2026 年，所有在生产环境承载流量的 AI 团队都会在模型旁部署一套可观测性层 (Observability Plane)。这涵盖了成本归因、幻觉检测 (Hallucination Detection)、数据漂移监控 (Drift Monitoring)、越狱信号 (Jailbreak Signal) 识别、SLO 仪表盘 (Service Level Objective Dashboards) 以及 PII 泄露告警 (PII Leak Alerts)。开源参考项目——Langfuse、Phoenix 和 OpenLLMetry——已统一采用 OpenTelemetry GenAI 语义规范作为数据接入模式 (Ingest Schema)。现在，你只需使用一个 SDK 即可对 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 进行插桩，并输出兼容的跨度 (Spans) 数据。

你将构建一个自托管仪表盘，用于从至少四个 SDK 家族接入数据，在采样追踪数据上运行一组评估任务，检测数据漂移并触发告警。验收标准为：当系统遭遇故意注入的回归缺陷（例如某个提示词开始输出 PII 信息）时，仪表盘需能在五分钟内捕获该异常并触发告警。

## 核心概念

数据接入采用 OTLP HTTP 协议。SDK 会生成符合 GenAI 语义规范的跨度，包含 `gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions` 等属性。跨度数据将写入 ClickHouse 以进行列式分析 (Columnar Analytics)；元数据（用户、会话、应用）则存入 Postgres。

评估任务以批处理作业 (Batch Jobs) 的形式在采样追踪数据上运行。DeepEval 负责评估忠实度 (Faithfulness)、毒性 (Toxicity) 和答案相关性 (Answer Relevance)。当追踪数据携带检索上下文 (Retrieval Context) 时，RAGAS 负责评估检索指标。自定义的 LLM 裁判 (LLM-judges) 用于执行领域特定检查（如 PII 泄露、偏离策略的回复）。评估作业会将结果以评估跨度 (Eval Spans) 的形式写回同一个 ClickHouse 实例，并与父级追踪数据 (Parent Trace) 关联。

漂移检测 (Drift Detection) 会持续监控嵌入空间分布 (Embedding-space Distributions) 随时间的变化（基于提示词嵌入向量的 PSI 或 KL 散度 (KL Divergence)），同时跟踪评估分数的趋势。告警信息将推送至 Prometheus Alertmanager，随后路由至 Slack 或 PagerDuty。前端 UI 采用 Next.js 15 结合 Recharts 构建。

## 架构设计

production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)

## 技术栈

- 数据接入（Ingest）：OpenTelemetry SDK + 生成式 AI 语义规范（GenAI semantic conventions）；OTLP HTTP 传输协议
- 收集器（Collector）：OpenTelemetry Collector 配合尾部采样处理器（tail-sampling processor，用于成本控制）
- 存储（Storage）：ClickHouse 用于存储跨度数据（spans），Postgres 用于存储元数据（metadata），S3 用于原始事件归档（raw event archive）
- 评估（Evals）：DeepEval、RAGAS 0.2、Arize Phoenix 评估器包（evaluator pack）、自定义大语言模型裁判（custom LLM-judge）
- 漂移检测（Drift）：每周对池化提示词嵌入（pooled prompt embeddings）进行 PSI / KL 散度计算（基于 sentence-transformers）
- 告警（Alerting）：Prometheus Alertmanager -> Slack / PagerDuty
- 用户界面（UI）：Next.js 15 App Router + Recharts + 服务端动作（server actions）
- 开箱即用的 SDK 支持：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建指南

1. **采集器配置。** 配置 OpenTelemetry 采集器 (OpenTelemetry Collector)，搭载 OTLP HTTP 接收器 (OTLP HTTP receiver)、保留 100% 错误追踪 (errored traces) 和 10% 成功追踪的尾部采样器 (tail-sampler)，以及指向 ClickHouse 和 S3 的导出器 (exporters)。

2. **ClickHouse 表结构。** 创建 `spans` 表，其列结构对齐 GenAI 语义约定 (GenAI semconv)：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，并附加用于存储长负载的 JSON 字段。为 `user_id` 和 `app_id` 添加二级索引。

3. **SDK 覆盖率测试。** 使用各 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）编写一个小型客户端应用，并启用 OpenLLMetry 自动插桩 (auto-instrument)。验证每个 SDK 均能生成符合规范的 GenAI 跨度 (GenAI spans) 并成功写入 ClickHouse。

4. **评估任务。** 定时任务读取过去 15 分钟的采样追踪数据，并运行 DeepEval 的忠实度 (faithfulness)、毒性 (toxicity) 和答案相关性 (answer relevance) 评估。输出结果为评估跨度 (eval spans)，并与父级追踪关联。

5. **自定义 LLM 裁判 (LLM-judge)。** 个人身份信息 (PII) 泄露检测裁判：针对模型回复，调用防护 LLM (guard LLM) 对 PII 泄露概率进行打分。高分回复将进入分类处理队列 (triage queue)。

6. **漂移检测。** 每周任务计算本周汇总的提示词嵌入 (prompt embeddings) 与过去 4 周基线之间的群体稳定性指数 (PSI)。若 PSI 超过阈值，则触发告警。

7. **仪表盘。** 基于 Next.js 15 构建，包含以下页面：概览（每秒跨度数、单用户成本、p95 延迟）、追踪（搜索 + 瀑布图）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链路。** Prometheus 导出器 (Prometheus exporter) 读取评估分数聚合数据与延迟百分位数；Alertmanager 将警告路由至 Slack，将严重违规事件路由至 PagerDuty。

9. **回归探测。** 注入缺陷：被测聊天机器人开始以 1% 的概率泄露伪造的社会安全号码 (SSNs)。测量平均恢复时间 (MTTR)：从缺陷部署到 Slack 告警触发的时间。

## 使用指南

$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com

## 交付

`outputs/skill-llm-observability.md` 为最终交付物。针对任意 LLM 应用，该仪表盘可接入其追踪数据、运行评估、在发生漂移时告警，并在 Next.js 中展示单用户成本明细。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 追踪模式覆盖率 | 生成标准 GenAI 跨度的 SDK 家族数量（目标：6+） |
| 20 | 评估准确性 | DeepEval / RAGAS 评分与人工标注数据集的对比 |
| 20 | 仪表盘用户体验 | 注入回归缺陷后的平均恢复时间 (MTTR)（目标：5 分钟以内） |
| 20 | 成本 / 扩展性 | 在无积压情况下持续以 1k 跨度/秒 的速率接入数据 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链路端到端验证 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩（instrumentation）。验证规范跨度（canonical spans）是否已正确写入 ClickHouse，并携带如实反映的 `gen_ai.*` 属性。

2. 在相同的追踪数据（traces）上，将 DeepEval 替换为 Phoenix 评估器（evaluators）。测量两个评估引擎之间的评分漂移（score drift）。

3. 优化漂移检测器（drift detector）：按应用 ID（app-id）而非全局计算群体稳定性指数（PSI）。展示各应用的漂移轨迹（drift trails）。

4. 新增“用户影响”页面：使用迷你折线图（sparklines）展示单用户成本（cost-per-user）与单用户失败率（failure-rate-per-user）。

5. 构建尾部采样策略（tail-sampling policy）：完整保留毒性评分（toxicity）> 0.5 的 100% 追踪数据，并对其余数据执行 10% 的分层采样（stratified sampling）。评估由此引入的采样偏差（sampling bias）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GenAI 语义约定（GenAI semconv） | “OTel LLM 属性” | 2025 年 OpenTelemetry 规范中针对 LLM 跨度（span）属性的定义（涵盖系统、模型、Token 等） |
| 尾部采样（Tail sampling） | “追踪后采样” | 收集器（Collector）在追踪数据完成后决定保留或丢弃该数据（可预先检查错误） |
| 群体稳定性指数（PSI） | “总体稳定性指标” | 用于对比两个分布的漂移度量指标；通常 > 0.2 即表示存在显著漂移 |
| LLM 裁判（LLM-judge） | “以模型进行评估” | 使用一个 LLM 根据既定评分标准（如忠实度、毒性、PII 等）对另一个 LLM 的输出进行打分 |
| 尾部采样策略（Tail-sampling policy） | “保留规则” | 决定保留或丢弃哪些追踪数据的规则；通常结合错误状态与采样率 |
| 评估跨度（Eval span） | “关联评估追踪” | 携带评估分数的子跨度（child span），并与原始 LLM 调用跨度相关联 |
| 单用户成本（Cost per user） | “单位经济效益” | 在特定时间窗口内归因于特定 user_id 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) —— 参考级开放核心可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) —— 备选参考方案，具备强大的漂移检测支持
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) —— 自动插桩（auto-instrumentation）SDK 系列
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 数据接入规范（ingest schema）
- [Helicone](https://www.helicone.ai) —— 备选托管型可观测性服务
- [Braintrust](https://www.braintrust.dev) —— 备选评估优先（eval-first）平台
- [ClickHouse 文档](https://clickhouse.com/docs) —— 列式跨度（span）存储方案
- [DeepEval](https://github.com/confident-ai/deepeval) —— 评估器库
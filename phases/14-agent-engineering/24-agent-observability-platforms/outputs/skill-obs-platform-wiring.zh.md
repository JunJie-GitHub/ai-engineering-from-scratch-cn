---
name: obs-platform-wiring
description: 选择一个可观测性平台 (Observability Platform)（Langfuse, Phoenix, Opik, Datadog），并将追踪 (Traces)、评估 (Evals) 和提示词版本 (Prompt Versions) 接入现有智能体 (Agent)。
version: 1.0.0
phase: 14
lesson: 24
tags: [可观测性, langfuse, phoenix, opik, datadog, 追踪]
---

给定智能体运行时 (Agent Runtime) 和产品需求，选择一个可观测性平台并搭建接入框架。

决策：

1. 需要集中管理提示词 (Prompt Management) 与会话回放 (Session Replay) -> **Langfuse**。
2. 需要深度检索增强生成 (RAG) 相关性分析与漂移/异常检测 (Drift/Anomaly Detection) -> **Phoenix**。
3. 需要自动化提示词优化 (Prompt Optimization) 与个人身份信息防护护栏 (PII Guardrails) -> **Opik**。
4. 已在使用 Datadog -> **Datadog LLM Observability**（自 v1.37+ 起原生支持生成式 AI 映射）。
5. 需要避开 Elastic License v2.0 (ELv2) 许可证 -> **Langfuse** (MIT) 或 **Opik** (Apache 2.0)；若为纯开源软件 (OSS) 分发，请避免使用 Phoenix。

产出物：

1. OpenTelemetry 生成式 AI 遥测埋点 (OTel GenAI Instrumentation)（第 23 课）——这是通用底层基础。
2. 平台专属 SDK 或 OTel 导出器 (OTel Exporter) 配置。
3. 针对您业务领域的大语言模型裁判评估量规 (LLM-Judge Rubric)（事实准确性、范围覆盖、语气、拒绝回答质量）。
4. 将提示词版本控制 (Prompt Versioning) 接入追踪数据（Langfuse），或配置追踪聚类 (Trace Clustering)（Phoenix），或定义实验 (Experiment Definitions)（Opik）。
5. 日志内容防护护栏 (Guardrails)：个人身份信息脱敏 (PII Redaction)、密钥清理 (Secret Scrubbing)。
6. 仪表盘 (Dashboards)：会话健康度、故障分类体系 (Failure Taxonomy)、延迟分布、单次会话成本。

硬性拒绝项：

- 未经评估 (Evals) 即上线。仅依赖追踪等同于昂贵的日志记录。
- 使用自行编写且无外部验证的大语言模型裁判 (LLM-Judge)。CRITIC 模式 (CRITIC Pattern)（第 05 课）：裁判模型需要外部工具进行事实锚定 (Factual Grounding)。
- 在 Span 主体 (Span Bodies) 中存储个人身份信息 (PII)。必须始终使用外部存储 + 引用 ID (Reference IDs)。

拒绝规则：

- 若用户要求“一个平台解决所有问题”，请拒绝并提供上述决策建议。没有任何单一平台能在所有三个维度上占据主导地位。
- 若产品未为每个智能体任务设定验收标准 (Acceptance Criteria)，请拒绝交付评估模块。大语言模型裁判依赖评估量规，而量规的制定依赖于产品决策。
- 若用户要求“不采样，全量捕获”，请拒绝。追踪数据量与流量呈线性增长；在规模化场景下必须进行采样（头部采样 Head-based 或尾部采样 Tail-based）。

输出文件：`instrumentation.py`、`judge.py`、`dashboards.md`、`README.md`，用于说明平台选择依据、评估量规、采样策略及事件响应流程。结尾需包含“下一步阅读”指引，指向第 30 课（评估驱动开发 Eval-Driven Development）或第 26 课（故障模式分类体系 Failure-Mode Taxonomy）。
# 智能体可观测性 (Agent Observability)：Langfuse、Phoenix、Opik

> 三大开源智能体可观测性平台主导了 2026 年的市场。Langfuse（MIT 许可证）—— 月安装量超 600 万次，涵盖链路追踪 (Tracing)、提示词管理 (Prompt Management)、评估 (Evals) 与会话回放 (Session Replay)。Arize Phoenix（Elastic 2.0 许可证）—— 专注于智能体专属的深度评估、检索增强生成 (RAG) 相关性分析、OpenInference 自动插桩 (Auto-instrumentation)。Comet Opik（Apache 2.0 许可证）—— 提供自动化提示词优化、安全护栏 (Guardrails) 以及基于大语言模型裁判 (LLM-judge) 的幻觉检测。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 第 23 课（OTel GenAI）
**Time:** 约 45 分钟

## 学习目标

- 列举三大顶级开源智能体可观测性平台及其开源许可证。
- 区分各平台的核心优势：Langfuse（提示词管理 + 会话管理）、Phoenix（检索增强生成 + 自动插桩）、Opik（优化 + 安全护栏）。
- 解释为何到 2026 年，89% 的组织报告已部署智能体可观测性方案。
- 使用标准库实现一条从链路追踪到仪表板的流水线，并集成大语言模型裁判评估。

## 问题背景

OTel GenAI（第 23 课）为你提供了数据模式 (Schema)。但你仍需要一个平台来摄入跨度数据 (Spans)、执行评估、存储提示词版本，并暴露性能回退 (Regressions)。这三大竞争者各自侧重于开发生命周期的不同环节。

## 核心概念

### Langfuse (MIT)

- 每月 SDK 安装量超 600 万次，GitHub 星标数超 1.9 万。
- 核心功能：追踪（Tracing）、带版本控制与测试场（Playground）的提示词管理（Prompt Management）、评估（Evaluation，含大模型即裁判/LLM-as-judge、用户反馈、自定义评估）、会话回放（Session Replay）。
- 2025 年 6 月：原商业模块（大模型即裁判、标注队列、提示词实验、测试场）已以 MIT 协议开源。
- 最适用场景：具备紧密提示词管理闭环的端到端可观测性（Observability）。

### Arize Phoenix (Elastic License 2.0)

- 更深入的智能体（Agent）专属评估：追踪聚类、异常检测、检索增强生成（RAG）的检索相关性。
- 原生支持 OpenInference 自动插桩（Auto-instrumentation）。
- 可与托管版 Arize AX 搭配用于生产环境。
- 无提示词版本控制——定位为与更广泛平台配合使用的漂移/行为回归检测工具。
- 最适用场景：RAG 相关性、行为漂移、异常检测。

### Comet Opik (Apache 2.0)

- 通过 A/B 实验实现提示词自动优化。
- 护栏机制（Guardrails，含个人身份信息/PII 脱敏、主题约束）。
- 基于大模型裁判的幻觉检测。
- Comet 官方基准测试数据：Opik 完成日志记录与评估耗时 23.44 秒，而 Langfuse 为 327.15 秒（差距约 14 倍）——厂商基准测试数据仅供参考方向。
- 最适用场景：优化闭环、自动化实验、护栏机制执行。

### 行业数据

根据 Maxim（2026 年实地分析）：89% 的组织已部署智能体可观测性；质量问题是生产环境的首要障碍（32% 的受访者提及）。

### 如何选择

| 需求 | 推荐选择 |
|------|------|
| 集成提示词管理的一体化方案 | Langfuse |
| 深度 RAG 评估 + 漂移检测 | Phoenix |
| 自动化优化 + 护栏机制 | Opik |
| 开放许可协议，无 ELv2 限制 | Langfuse (MIT) 或 Opik (Apache 2.0) |
| 集成 Datadog / New Relic | 任意一款——均支持导出 OTel（OpenTelemetry） |

### 常见误区

- **缺乏评估策略。** 仅有追踪而无评估，不过是昂贵的日志记录。
- **未结合事实依据的自研大模型裁判。** 适用 CRITIC 模式（第 05 课）——裁判模型需要借助外部工具进行事实核查。
- **提示词版本未与追踪记录关联。** 当生产环境出现性能回退时，你将无法通过二分法（Bisect）定位导致问题的提示词版本。

## 动手构建

`code/main.py` 实现了一个基于标准库（Standard Library）的追踪收集器与大模型裁判评估器：

- 接收生成式 AI（GenAI）格式的跨度（Span）数据。
- 按会话分组，标记失败运行（触发护栏、低置信度评估）。
- 通过脚本实现的大模型裁判，根据评分细则对智能体响应进行打分。
- 类似仪表盘的汇总视图：失败率、主要失败原因、评估分数分布。

运行方式：

python3 code/main.py

输出结果：各会话的评估分数与失败分类，其展示形式与 Langfuse/Phoenix/Opik 的输出一致。

## 投入使用

- **Langfuse**：支持自托管或云端部署；通过 OTel 或其官方 SDK 接入。
- **Arize Phoenix**：支持自托管；自动插桩 OpenInference。
- **Comet Opik**：支持自托管或云端部署；提供自动化优化闭环。
- **Datadog LLM Observability**：适用于已在使用 Datadog 的运维与机器学习混合团队。

## 交付上线

`outputs/skill-obs-platform-wiring.md` 用于选择一个平台，并将追踪链路 (traces)、评估结果 (evaluations) 与提示词版本 (prompt versions) 接入现有的智能体 (agent)。

## 练习

1. 将一周的 OpenTelemetry 追踪数据 (OTel traces) 导出至 Langfuse 云端（免费套餐）。哪些会话 (sessions) 失败了？原因是什么？
2. 为你的业务领域编写一套大语言模型裁判评分标准 (LLM-judge rubric)（涵盖事实准确性、语气、范围遵循度）。在 50 条追踪数据上进行测试。
3. 对比 Langfuse 的提示词版本控制 (prompt versioning) 与 Phoenix 的追踪数据聚类 (trace clustering) 功能。哪一种能更快地帮你定位故障原因？
4. 阅读 Opik 的护栏文档 (guardrail docs)。将个人身份信息脱敏护栏 (PII redaction guardrail) 接入你的某次智能体运行记录 (agent runs) 中。
5. 在你的语料库 (corpus) 上对这三款工具进行基准测试 (benchmark)。忽略厂商公布的数据，亲自测量实际表现。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 链路追踪 (Tracing) | “跨度收集器” | 采集 OpenTelemetry / SDK 跨度 (spans)；按会话 (session) 建立索引 |
| 提示词管理 (Prompt management) | “提示词内容管理系统” | 与追踪数据关联的版本化提示词 |
| 大模型裁判 (LLM-as-judge) | “自动化评估” | 使用独立的大语言模型根据评分标准对智能体输出进行打分 |
| 会话回放 (Session replay) | “链路回放” | 逐步回溯历史运行记录以进行调试 |
| 检索增强生成相关性 (RAG relevancy) | “检索质量” | 检索到的上下文是否与查询意图匹配 |
| 追踪聚类 (Trace clustering) | “行为分组” | 将相似运行记录聚类以检测模型漂移 |
| 护栏强制执行 (Guardrail enforcement) | “日志记录时的策略” | 对已记录的内容进行个人身份信息/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) — 链路追踪、评估、提示词管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) — 自动插桩、漂移检测
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化与护栏
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 这三款工具共同遵循的数据模式
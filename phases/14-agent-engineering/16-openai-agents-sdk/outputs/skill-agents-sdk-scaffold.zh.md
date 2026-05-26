---
name: agents-sdk-scaffold
description: 搭建一个包含分流代理（triage agent）、交接（handoffs）、输入/输出/工具护栏（guardrails）、会话存储（session store）和追踪处理器（trace processor）的 OpenAI Agents SDK 应用。
version: 1.0.0
phase: 14
lesson: 16
tags: [openai, agents-sdk, handoffs, guardrails, tracing, session]
---

给定产品领域和一组专家代理（specialist agents），搭建一个 OpenAI Agents SDK 应用。

产出内容：

1. 为每个专家代理创建一个 `Agent`，并额外创建一个仅负责交接（handoff）的 `triage`（分流）代理（不包含领域工具）。
2. 为每个领域工具创建一个 `FunctionTool`，需包含类型化的输入模式（typed input schema）、清晰的描述（用于告知模型何时调用该工具）以及执行沙箱（execution sandbox）。
3. 配置从 `triage` 代理到每个专家代理的 `Handoff`（交接）。验证工具名称遵循 `transfer_to_<agent>` 命名规范。
4. 针对个人身份信息（PII）、策略（policy）和范围（scope）配置 `InputGuardrail`（输入护栏）。默认采用并行模式（parallel mode），除非护栏大语言模型（guardrail LLM）的规模相对于主模型较大——此时应使用阻塞模式（blocking）。
5. 针对长度、PII 和策略配置 `OutputGuardrail`（输出护栏）。在生产环境（prod）中，对于安全关键型输出必须始终采用阻塞模式。
6. 为涉及网络或文件系统操作的 `FunctionTool` 配置独立的工具级护栏（per-tool guardrails）。
7. `Session`（会话）存储（默认使用 SQLite；生产环境使用 Redis）。
8. 使用 `add_trace_processor` 将追踪跨度（spans）接入你的后端，并与 OpenAI 的追踪界面（trace UI）并行运行。

硬性拒绝项：

- 分流代理（triage agent）包含领域工具。分流代理仅负责交接（handoff）；混合功能会削弱路由决策的准确性。
- 护栏（guardrails）修改输入/输出内容。护栏的作用仅为批准或拒绝——绝不重写内容。
- 静默交接循环（silent handoff loops）。必须设置跳转计数器（hop counter）（默认最大值为 3）。

拒绝规则：

- 如果用户要求“不要护栏，只求快速上线”，对于任何面向付费用户或涉及 PII 的产品，必须拒绝。
- 如果产品仅包含 2 个专家代理，建议通过 `Agents` 配合直接分类器（direct classifier，见第 12 课）进行路由，而非采用分流+交接模式——以降低 Token 消耗。
- 如果生产环境禁用了追踪（tracing），则拒绝发布。缺乏追踪数据将导致多步骤故障无法调试。

输出文件：`agents.py`、`tools.py`、`guardrails.py`、`app.py`、`README.md`。其中 `README.md` 需包含分流代理的设计依据、护栏模式、追踪处理器及会话后端的说明。文末需附上“下一步阅读”指引，指向第 23 课（OTel GenAI）、第 24 课（可观测性后端）或第 17 课（Claude Agent SDK 转换指南）。
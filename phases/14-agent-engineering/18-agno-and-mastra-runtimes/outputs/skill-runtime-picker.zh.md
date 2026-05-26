---
name: runtime-picker
description: 根据给定的技术栈（Tech Stack）、延迟预算（Latency Budget）和运维形态（Operational Shape），选择合适的生产级智能体运行时（Production Agent Runtime）（如 Agno、Mastra、LangGraph 或提供商 SDK）。
version: 1.0.0
phase: 14
lesson: 18
tags: [agno, mastra, langgraph, runtime, selection]
---

根据给定的技术栈、延迟预算、所需原语（Primitives）及运维形态，选择合适的运行时。

决策指南：

1. Python + FastAPI + 每秒处理数千个短生命周期智能体（Short-lived Agents） -> **Agno**。
2. TypeScript + Next.js/Vercel + 统一的多提供商集成（Unified Multi-provider） -> **Mastra**。
3. 持久化状态（Durable State）、显式图结构（Explicit Graph）、故障恢复（Resume-on-Failure） -> **LangGraph**（第 13 课）。
4. 以 Claude 为核心的产品，需要 Claude Code 的控制架构（Harness Shape） -> **Claude Agent SDK**（第 17 课）。
5. 以 OpenAI 为核心的产品，需要任务交接（Handoffs）+ 安全护栏（Guardrails）+ 链路追踪（Tracing） -> **OpenAI Agents SDK**（第 16 课）。
6. 多智能体协作、Actor 模型并发（Actor-model Concurrency）、故障隔离（Fault Isolation） -> **AutoGen v0.4** / **Microsoft Agent Framework**（第 14 课）。
7. 基于角色的协作或事件驱动的确定性工作流（Event-driven Deterministic Workflows） -> **CrewAI** 的 Crew 或 Flow 模式（第 15 课）。
8. 均不符合 -> 直接调用 API + 使用第 01 课中的标准库循环（Stdlib Loop）。

交付物：

- 一份简短的决策文档：包含技术栈、延迟目标、所需原语及观察到的权衡取舍（Trade-offs）。
- 基于所选运行时的最小化脚手架（Scaffold）代码。
- 若当前已使用其他运行时，需提供迁移方案。

明确否决项：

- 当每个请求仅包含一次慢速调用时，仅凭“性能”选择 Agno 或 Mastra。性能通常并非瓶颈所在。
- 在 Python 单体仓库（Monorepo）中无合理理由地引入 TypeScript 运行时。混合语言编写的智能体代码会带来额外的运维负担（Operational Tax）。
- 为无状态的短时任务选择 LangGraph。其检查点机制（Checkpointer）会引入额外开销，而简单工作流（第 12 课）可避免此问题。

拒绝规则：

- 若用户要求“同时部署全部五种运行时以作对比”，请予以拒绝。应基于实际工作负载进行基准测试（Benchmark）；框架厂商提供的基准测试数据仅作趋势性参考（Directional）。
- 若用户希望自行托管 Mastra 的 `ee/` 功能，请予以拒绝并指引其查阅许可协议（License Terms）条款。
- 若产品需要长时间运行的异步任务（Async Work）（数小时至数天），请拒绝自建方案，并引导至 Claude Managed Agents 或基于队列的架构（Queue-based Architecture）（第 29 课）。

输出要求：决策文档 + 脚手架代码 + README。结尾需附上“下一步阅读”指引，指向第 24 课（可观测性/Observability）与第 29 课（生产级运行时/Production Runtimes），以了解框架之上的运维层架构。
# Agno 与 Mastra：生产级运行时 (Production Runtimes)

> Agno（Python）与 Mastra（TypeScript）是面向 2026 年的生产级运行时 (Production Runtime) 组合。Agno 专注于微秒级智能体 (Agent) 实例化与无状态 FastAPI 后端。Mastra 则基于 Vercel AI SDK 底层架构，提供智能体、工具、工作流、统一模型路由 (Unified Model Routing) 以及复合存储 (Composite Storage) 功能。

**Type:** 学习
**Languages:** Python, TypeScript
**Prerequisites:** 第 14 阶段 · 01（智能体循环 (Agent Loop)），第 14 阶段 · 13（LangGraph）
**Time:** 约 45 分钟

## 学习目标

- 明确 Agno 的性能目标及其适用场景。
- 列举 Mastra 的三大核心原语 (Primitives) —— 智能体 (Agents)、工具 (Tools)、工作流 (Workflows) —— 以及其支持的服务器适配器 (Server Adapters)。
- 解释为何基于会话作用域 (Session-Scoped) 的无状态 FastAPI 后端是 Agno 推荐的生产部署方案。
- 根据现有技术栈（Python 优先 vs TypeScript 优先）在 Agno 与 Mastra 之间做出选择。

## 问题背景

LangGraph、AutoGen 和 CrewAI 属于重量级框架。对于希望“仅保留智能体循环，追求极致速度，并无缝集成至自有运行时”的团队而言，Agno（Python）或 Mastra（TypeScript）是理想之选。两者均通过舍弃部分框架内置的原语，换取了更高的执行效率以及与周边技术栈更紧密的契合度。

## 核心概念

### Agno

- Python 运行时（Runtime），前身为 Phi-data。
- “无需图（Graphs）、链（Chains）或复杂的模式——只需纯 Python。”
- 官方文档中的性能目标：智能体（Agent）实例化约 2μs，每个智能体内存占用约 3.75 KiB，支持约 23 家模型提供商。
- 生产环境路径：无状态（Stateless）且基于会话作用域（Session-scoped）的 FastAPI 后端。每次请求都会启动一个全新的智能体；会话状态存储在数据库中。
- 原生支持多模态（Multimodal，含文本、图像、音频、视频、文件）及智能体 RAG（Retrieval-Augmented Generation，检索增强生成）。

当每秒需要处理数千个短生命周期智能体（如聊天消息扇入、评估流水线）时，这些速度指标至关重要。但如果单个智能体运行长达 10 分钟，这些指标的影响就微乎其微了。

### Mastra

- 基于 TypeScript 构建，底层依赖 Vercel AI SDK。
- 三大核心原语（Primitives）：**智能体（Agents）**、**工具（Tools）**（基于 Zod 类型校验）、**工作流（Workflows）**。
- 统一模型路由（Model Router）——支持 94 家提供商的 3,300+ 个模型（截至 2026 年 3 月）。
- 复合存储架构：内存、工作流、可观测性（Observability）数据可对接不同后端；大规模可观测性推荐使用 ClickHouse。
- 采用 Apache 2.0 许可证，但 `ee/` 目录下的代码遵循源码可用（Source-available）的企业许可证。
- 提供 Express、Hono、Fastify、Koa 的服务器适配器；深度集成 Next.js 和 Astro。
- 内置 Mastra Studio（访问 localhost:4111）用于调试。
- 在 1.0 版本发布时（2026 年 1 月），GitHub 星标数超 2.2 万，npm 周下载量超 30 万。

### 定位

两者均无意成为 LangGraph。它们的竞争焦点在于：

- **语言契合度。** Agno 适合 Python 优先的团队；Mastra 适合 TypeScript 优先的团队。
- **运行时体验。** Agno = 近乎零开销；Mastra = 深度融入 Vercel 生态。
- **可观测性。** 两者均支持集成 Langfuse/Phoenix/Opik（第 24 课），但 Mastra Studio 是官方原生工具。

### 何时选择各框架

- **Agno** —— Python 后端、大量短生命周期智能体、对性能要求严苛、使用 FastAPI 的团队。
- **Mastra** —— TypeScript 后端、部署于 Next.js / Vercel、需要统一的多提供商模型路由、偏好 Zod 类型校验工具。
- **LangGraph**（第 13 课）—— 当持久化状态（Durable State）和显式图推理（Graph Reasoning）比原始速度更重要时。
- **OpenAI / Claude Agent SDK** —— 当你希望直接采用模型提供商产品化的架构形态时（第 16–17 课）。

### 该模式的常见误区

- **为性能而性能。** 当每个请求仅包含一次较慢的智能体调用时，仅仅因为“2μs”听起来很酷就选择 Agno。此时开销根本不是瓶颈。
- **生态锁定。** Mastra 对 Vercel 风格的深度集成在 Vercel 上是优势，但在其他平台上可能成为劣势。
- **企业许可证混淆。** Mastra 的 `ee/` 目录采用源码可用许可证，而非 Apache 2.0。如果计划 Fork（分支开发），请务必仔细阅读许可证条款。

## 动手构建

本课主要为对比性质——单一代码示例难以充分展现两个框架的全貌。请参阅 `code/main.py` 中的对照示例：一个极简的“运行智能体、流式输出、持久化会话”流程被实现了两次（一次采用 Agno 风格，一次采用 Mastra 风格）。

运行方式：

python3 code/main.py

两次运行将生成结构不同但功能等效的追踪记录（Traces）。

## 投入使用

- **Agno** — 需要高性能且契合 FastAPI 架构的 Python 后端。
- **Mastra** — 支持众多提供商（providers）并内置工作流原语（workflow primitives）的 TypeScript 后端。
- 两者均提供官方可观测性钩子（observability hooks）。两者均可与 Langfuse 集成。

## 交付上线
`outputs/skill-runtime-picker.md` 会根据技术栈（stack）、延迟预算（latency budget）和运维形态（operational shape），自动选择 Agno、Mastra、LangGraph 或特定提供商的 SDK。

## 练习
1. 阅读 Agno 的文档。将标准库（stdlib）的 ReAct 循环（ReAct loop，见第 01 课）移植到 Agno。哪些部分消失了？哪些部分保留了下来？
2. 阅读 Mastra 的文档。将相同的循环移植到 Mastra。工具类型定义（tool typing）方面有何变化（Zod 对比无类型）？
3. 基准测试（Benchmark）：测量你的技术栈中智能体实例化延迟（agent instantiation latency）。Agno 的 2μs 延迟对你的工作负载（workload）有影响吗？
4. 设计迁移方案（migration）：如果你一直在 Python 中运行 CrewAI，迁移到 Agno 会导致哪些功能失效？
5. 阅读 Mastra 的 `ee/` 许可证条款（license terms）。哪些限制会影响开源分支（open-source fork）？

## 关键术语
| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agno | “快速的 Python 智能体” | 无状态会话级智能体运行时（stateless session-scoped agent runtime） |
| Mastra | “基于 Vercel AI SDK 的 TypeScript 智能体” | 智能体（Agents）+ 工具（Tools）+ 工作流（Workflows）+ 模型路由（Model Router） |
| Unified Model Router | “多提供商访问” | 跨 94 家提供商、支持 3,300+ 模型的单一客户端 |
| Composite storage | “多后端” | 记忆（Memory）、工作流和可观测性数据分别存储于不同的后端 |
| Mastra Studio | “本地调试器” | 运行在 localhost:4111 的 UI，用于内省（introspecting）智能体 |
| Source-available | “非开源（Not OSS）” | 许可证允许阅读源码，但限制商业用途 |

## 延伸阅读
- [Agno Agent Framework 文档](https://www.agno.com/agent-framework) — 性能目标、FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) — 原语、服务器适配器（server adapters）、模型路由（Model Router）
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图（stateful-graph）替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成中引用的可观测性对比（observability comparisons）
# 生产环境运行时（Production Runtimes）：队列、事件、定时任务（Cron）

> 生产环境中的智能体（Agent）运行于六种运行时形态（Runtime Shapes）之上：请求-响应（Request-Response）、流式（Streaming）、持久化执行（Durable Execution）、基于队列的后台任务（Queue-based Background）、事件驱动（Event-driven）以及定时调度（Scheduled）。在选择框架之前，应先确定运行时形态。可观测性（Observability）在每种形态中都起着至关重要的底层支撑作用。

**类型：** 学习
**语言：** Python (stdlib)
**前置条件：** 第 14 阶段 · 13 (LangGraph)，第 14 阶段 · 22 (语音)
**时长：** 约 60 分钟

## 学习目标

- 列举六种生产环境运行时形态，并将其与相应的框架/产品模式进行匹配。
- 解释为何持久化执行对长周期任务（Long-horizon Tasks）至关重要。
- 描述事件驱动运行时，并说明 Claude 托管智能体（Claude Managed Agents）的适用场景。
- 阐述为何可观测性是多步智能体（Multi-step Agents）架构中的关键支撑。

## 问题所在

生产环境中的智能体会以 Jupyter Notebook 无法暴露的方式发生故障：例如在第 37 步出现网络超时、用户在语音通话中途挂断、机器重启导致定时任务（Cron Job）中断，或后台工作进程（Background Worker）内存溢出。运行时形态决定了哪些故障是可以恢复的。

## 核心概念

### 请求-响应 (Request-Response)
- 同步 HTTP。用户需等待任务完成。
- 仅适用于短时任务（<30秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性 (Observability)：标准 HTTP 访问日志 + OpenTelemetry (OTel) 跨度 (Spans)。

### 流式传输 (Streaming)
- 使用服务器发送事件 (SSE) 或 WebSocket 实现渐进式输出。
- LiveKit 将其扩展至 WebRTC，用于语音/视频场景（第 22 课）。
- 技术栈：任何支持流式传输的框架 + 能够处理 SSE/WS 的前端。
- 可观测性：分块耗时、首字延迟 (First-Token Latency)、尾部延迟 (Tail Latency)。

### 持久化执行 (Durable Execution)
- 每一步后保存状态检查点；失败时自动恢复。
- AutoGen v0.4 的 Actor 模型将故障隔离在单个智能体 (Agent) 内（第 14 课）。
- LangGraph 的核心差异化特性（第 13 课）。
- 当步骤数量未知且恢复成本较高时，此模式必不可少。

### 基于队列/后台任务 (Queue-based / Background)
- 任务进入队列，由工作节点 (Workers) 拾取处理，结果通过 Webhook 或发布/订阅 (Pub/Sub) 机制返回。
- 对于长周期智能体至关重要（根据 Anthropic 的计算机使用公告，每个任务包含数十到数百个步骤）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）或自定义方案。
- 可观测性：队列深度、单任务延迟分布、死信队列 (DLQ) 大小。

### 事件驱动 (Event-driven)
- 智能体订阅触发器：如新邮件、PR 合并请求开启、定时任务触发等。
- Claude Managed Agents 开箱即用地支持此模式（第 17 课）。
- CrewAI Flows（第 15 课）用于构建事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动的延迟、智能体处理延迟。

### 定时调度 (Scheduled)
- 类似 Cron 的智能体，按固定周期运行。
- 结合持久化执行，确保夜间运行的任务失败后能在下一个调度周期自动恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；或托管服务（Render cron、Vercel cron）。

### 2026 年部署模式
- **CrewAI Flows** 用于事件驱动的生产环境。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入式集成。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音服务（第 22 课）。
- **Claude Managed Agents** 用于托管长时异步任务。

### 可观测性是核心支撑
如果没有 OpenTelemetry GenAI 跨度（第 23 课）配合 Langfuse/Phoenix/Opik 后端（第 24 课），你将无法调试在第 40 步失败的多步智能体。这在生产环境中绝非可选项。它决定了你是“快速定位并修复问题”，还是“只能增加日志从头重新运行”。

### 生产环境运行时的常见失败点
- **架构选型错误。** 为耗时 5 分钟的任务选择请求-响应模式。导致用户超时挂起；工作节点堆积；重试引发雪崩。
- **缺少死信队列。** 队列工作节点未配置死信机制。失败的任务直接消失。
- **后台任务不透明。** 后台智能体运行未导出追踪数据。故障在用户反馈前完全不可见。
- **忽略持久化状态。** 任何运行时间超过 30 秒且无法承受重启成本的任务，都必须采用持久化执行。

## 开始构建

`code/main.py` 是一个标准库（stdlib）多形态（multi-shape）演示：

- 请求-响应端点（Request-response endpoint，普通函数）。
- 流式处理程序（Streaming handler，生成器）。
- 基于队列的工作器（Queue-based worker），带死信队列（Dead Letter Queue, DLQ）。
- 事件触发器注册表（Event trigger registry）。
- 定时调度器（Cron-shaped scheduler）。

运行方式：

python3 code/main.py

输出：五条追踪记录（traces），展示每种形态在同一任务上的行为。相同的智能体（agent）逻辑，不同的外部封装。持久化执行（Durable execution，第六种形态）有意留到第 13 课结合 LangGraph 检查点机制（checkpointing）进行讲解。

## 使用场景

- **请求-响应（Request-response）** 适用于聊天式用户体验。
- **流式（Streaming）** 适用于渐进式响应。
- **持久化（Durable）** 适用于长周期任务。
- **队列（Queue）** 适用于批处理/异步/长时间运行任务。
- **事件（Event）** 适用于智能体响应性交互。
- **定时任务（Cron）** 适用于系统维护（记忆整合、评估、成本报告）。

## 部署上线

`outputs/skill-runtime-shape.md` 为任务选择合适的运行时形态，并配置可观测性（observability）需求。

## 练习

1. 将第 01 课的 ReAct 循环迁移到你技术栈中的所有六种形态。哪种形态最适合哪个产品界面？
2. 为基于队列的演示添加死信队列（DLQ）。模拟 10% 的任务失败率；展示 DLQ 的大小。
3. 编写一个由定时任务触发的评估智能体，每晚针对当天排名前 20 的追踪记录运行。
4. 实现带背压（backpressure）的流式传输：如果客户端响应慢，则暂停智能体。这会如何与轮次预算（turn budget）交互？
5. 阅读 Claude 托管智能体（Claude Managed Agents）文档。在什么情况下你会将自托管的长周期智能体迁移到托管服务？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------------|------------------------|
| 请求-响应（Request-response） | “同步” | 用户等待；仅适用于短任务 |
| 流式（Streaming） | “SSE / WS” | 渐进式输出；更好的用户体验；可观测每个数据块（chunk）的延迟 |
| 持久化执行（Durable execution） | “从失败中恢复” | 状态已检查点化；从最后一步重启 |
| 基于队列（Queue-based） | “后台任务” | 生产者/工作池/死信队列（DLQ） |
| 事件驱动（Event-driven） | “基于触发器” | 智能体对外部事件作出反应 |
| 死信队列（DLQ） | “死信队列” | 失败任务的暂存区 |
| Claude 托管智能体（Claude Managed Agents） | “托管框架” | Anthropic 托管的长时间运行异步服务，带缓存与上下文压缩（compaction） |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行详情
- [Claude 托管智能体概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管的长时间运行异步服务
- [Anthropic：推出计算机使用功能](https://www.anthropic.com/news/3-5-models-and-computer-use) — “每个任务包含数十到数百个步骤”
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — Actor 模型（Actor-model）故障隔离
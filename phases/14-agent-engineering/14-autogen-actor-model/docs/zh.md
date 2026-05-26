# AutoGen v0.4：参与者模型（Actor Model）与智能体框架（Agent Framework）

> AutoGen v0.4（微软研究院，2025年1月）围绕参与者模型重新设计了智能体编排（Agent Orchestration）机制。支持异步消息交换、事件驱动型智能体、故障隔离与原生并发。目前该框架已进入维护模式，其继任者 Microsoft Agent Framework（微软智能体框架）已于2025年10月进入公开预览阶段。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** 第14阶段 · 01（智能体循环（Agent Loop））、第14阶段 · 12（工作流模式（Workflow Patterns））
**预计耗时：** 约75分钟

## 学习目标

- 阐述参与者模型的核心概念：将智能体视为参与者，消息作为唯一的进程间通信（IPC）方式，以及每个参与者的故障隔离机制。
- 列举 AutoGen v0.4 的三层 API 架构——核心层（Core）、智能体对话层（AgentChat）与扩展层（Extensions），并说明各自的用途。
- 解释为何将消息投递与消息处理解耦能够实现故障隔离与原生并发。
- 使用 Python 标准库实现一个参与者运行时（Actor Runtime），并将一个双智能体代码审查流程迁移至该运行时上。

## 问题所在

大多数智能体框架采用同步架构：一个智能体生成输出，另一个智能体消费输入，整个过程依赖于调用栈。一旦出现故障，整个调用栈便会崩溃。并发能力往往是事后附加的，而分布式部署则需要重写大量代码。

AutoGen v0.4 的解决方案是引入参与者模型。每个智能体都是一个拥有独立收件箱的参与者，消息是它们之间唯一的交互方式。运行时环境将消息投递与消息处理解耦，使得故障仅局限于单个参与者。并发成为原生特性，而分布式部署仅需更换底层传输协议即可。

## 核心概念

### 参与者（Actors）

一个参与者（Actor）包含：

- 私有状态（private state）（外部无法直接访问）。
- 收件箱（inbox）（消息队列）。
- 处理器（handler）：`receive(message) -> effects`，其中 `effects`（效应/操作）可以是“回复”、“发送给其他参与者”、“生成新参与者”、“更新状态”、“停止自身”。

两个参与者之间无法共享内存，它们只能通过发送消息进行通信。

### AutoGen v0.4 中的三层 API 架构

1. **核心层（Core）**。底层参与者框架。包含 `AgentRuntime`、`Agent`、`Message`、`Topic`。支持异步消息交换与事件驱动（event-driven）。
2. **AgentChat**。任务驱动的高层 API（用于替代 v0.2 中的 `ConversableAgent`）。包含 `AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **扩展层（Extensions）**。集成组件——支持 OpenAI、Anthropic、Azure、工具调用（tools）与记忆模块（memory）。

### 解耦的重要性

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 `agent_a`，直到 `agent_b` 返回结果。而在 v0.4 中，`send(agent_b, msg)` 仅将消息放入 `agent_b` 的收件箱后立即返回，由运行时（runtime）在后续进行投递。这带来了三大优势：

- **故障隔离（Fault isolation）**。Agent B 崩溃不会导致 Agent A 崩溃——运行时会在 B 的处理器中捕获故障，并决定后续操作（如记录日志、重试或转入死信队列）。
- **天然并发（Natural concurrency）**。可同时处理多条处于传输中的消息；各参与者能够并发处理各自的收件箱。
- **分布式就绪（Distribution-ready）**。无论参与者运行在同一进程内还是远程主机上，“收件箱 + 传输层”都采用相同的抽象模型。

### 拓扑结构（Topologies）

- **RoundRobinGroupChat**。智能体按固定顺序轮流发言。
- **SelectorGroupChat**。由一个选择器智能体根据对话上下文决定下一个发言者。
- **Magentic-One**。用于网页浏览、代码执行和文件处理的参考多智能体团队架构。基于 AgentChat 构建。

### 可观测性（Observability）

内置 OpenTelemetry 支持。每条消息都会生成一个追踪跨度（span）；工具调用会携带 `gen_ai.*` 属性，符合 2026 版 OTel GenAI 语义规范（Semantic Conventions）（参见第 23 课）。

### 当前状态：维护模式

截至 2026 年初：AutoGen v0.7.x 已稳定，适用于研究与原型开发。微软已将主要开发重心转移至 Microsoft Agent Framework（2025 年 10 月 1 日开启公开预览；1.0 正式版目标于 2026 年第一季度末发布）。AutoGen 的设计模式可无缝向前兼容——参与者模型才是其经得起时间考验的核心设计理念。

## 动手构建

`code/main.py` 实现了一个基于标准库的参与者运行时：

- `Message`：包含 `sender`、`recipient`、`topic`、`body` 的类型化负载（payload）。
- `Actor`：抽象基类，需实现 `receive(message, runtime)` 方法。
- `Runtime`：事件循环（event loop），包含共享队列、消息投递与故障隔离机制。
- 双参与者演示示例：`ReviewerAgent` 负责审查代码，`ChecklistAgent` 执行检查清单；两者通过消息交换直至达成共识。

运行方式：

python3 code/main.py

运行日志将展示消息投递过程、单个参与者的模拟故障（且不会导致另一个参与者崩溃），以及双方最终收敛至一致的评审结论。

## 使用它

- **AutoGen v0.4/v0.7**（维护阶段）— 适用于研究、原型设计与多智能体（Multi-Agent）模式，状态稳定。
- **Microsoft Agent Framework**（公开预览版）— 未来的演进方向；在更新后的 API 中延续了相同的 Actor 模型（Actor Model）理念。
- **LangGraph 集群拓扑**（第 13 课）— 通过共享工具交接（Shared-Tool Handoffs）实现类似的模式。
- **自定义 Actor 运行时**（Custom Actor Runtime）— 适用于需要特定传输层（如 NATS、RabbitMQ、gRPC）的场景。

## 交付上线

`outputs/skill-actor-runtime.md` 会为指定的多智能体任务生成一个最小化的 Actor 运行时，并附带一个团队模板（轮询 `RoundRobin` 或选择器 `Selector`）。

## 练习

1. 添加死信队列（Dead-Letter Queue, DLQ）：当处理器抛出异常时，将失败的消息暂存以供人工检查。在你的练习项目中，DLQ 被触发的频率有多高？
2. 实现 `SelectorGroupChat`：由一个选择器 Actor 根据对话状态决定由谁来处理下一条消息。
3. 添加分布式传输层：将进程内队列替换为基于 HTTP 的 JSON 服务器，使 Actor 能够在独立的进程中运行。
4. 为每条消息接入 OpenTelemetry 追踪跨度（OTel Span）（或使用空操作占位符）。按照第 23 课的要求，输出 `gen_ai.agent.name` 和 `gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的练习项目迁移至正式的 `autogen_core` API。在生产环境中，你忽略了哪些关键细节？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Actor | “智能体（Agent）” | 私有状态 + 收件箱 + 处理器；无共享内存 |
| Message | “事件（Event）” | 类型化的负载（Payload）；Actor 之间交互的唯一方式 |
| Inbox | “邮箱（Mailbox）” | 每个 Actor 专属的待处理消息队列 |
| Runtime | “智能体宿主（Agent Host）” | 负责消息路由并隔离故障的事件循环 |
| Topic | “通道（Channel）” | Actor 之间具名的发布-订阅路由 |
| Fault isolation | “任其崩溃（Let it crash）” | 单个 Actor 故障不会导致其他 Actor 崩溃 |
| RoundRobinGroupChat | “固定轮转团队” | 智能体按顺序轮流执行 |
| SelectorGroupChat | “上下文路由团队” | 选择器决定下一个执行者 |
| Magentic-One | “参考团队” | 用于处理网页、代码和文件的多智能体小队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 架构重构文章
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 基于图结构的替代方案
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的追踪跨度（Span）
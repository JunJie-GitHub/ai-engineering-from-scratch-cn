# 生产环境扩展 —— 队列、检查点与持久化

> 将多智能体系统（multi-agent system）扩展至数千个并发运行实例需要**持久化执行（durable execution）**。LangGraph 的运行时会以 `thread_id` 为键，在每个超级步骤（super-step）后写入检查点（checkpoint）（默认使用 Postgres）；工作进程（worker）崩溃会释放租约（lease），随后由另一个工作进程恢复执行。智能体可以无限期休眠以等待人类输入。**MegaAgent**（arXiv:2408.09955）为每个智能体运行了一个包含三种状态（空闲 / 处理中 / 响应中）的生产者-消费者队列（producer-consumer queue），并采用双层协调机制（组内聊天 + 组间管理员聊天）。在 LLM 流式传输场景中，**纤程/异步（Fiber/async）** 优于“每个任务一个线程”的模式：线程在等待 token 生成时 99% 的时间处于空闲状态，而纤程会在 I/O 操作时协作式让出控制权。反方观点：Ashpreet Bedi 在《Scaling Agentic Software》一文中主张，在负载证明有必要之前，应坚持使用 **FastAPI + Postgres + 无其他组件** 的架构——简单架构的实际承载能力往往超出预期。本课时将构建一个持久化检查点日志、一个带状态转换的每智能体工作队列、一个异步与线程的对比演示，并最终落脚于务实的“从简起步”原则。

**Type:** 学习 + 构建
**Languages:** Python（标准库、`asyncio`、`sqlite3`）
**Prerequisites:** 第 16 阶段 · 09（并行群体网络），第 16 阶段 · 13（共享内存）
**Time:** 约 75 分钟

## 问题

一个原型多智能体系统在一台笔记本电脑上运行，三个智能体共享一个内存中的事件循环（event loop）。当你将其迁移至生产环境时：

- 智能体有时会运行数小时（例如长时间研究、人在回路（human-in-the-loop）等待）。
- 工作进程（worker process）崩溃。重启会导致状态丢失。
- 峰值负载是平均值的 10 倍；你需要水平扩展（horizontal scaling）。
- 用户按智能体运行次数付费；你需要精确一次语义（exactly-once semantics）来进行计费。

内存中的事件循环无法满足上述任何需求。你需要在底层引入一个持久化执行层（durable execution layer）。2026 年的标准方案包括：

1. 带检查点的工作流引擎（workflow engine）（如 Temporal、LangGraph 运行时）。
2. 带状态存储（state store）的消息队列（message queue）（如 Postgres + SQS/RabbitMQ）。
3. 参与者模型框架（Actor-model frameworks）（如 MegaAgent 的每智能体生产者-消费者模式）。
4. 自研的 FastAPI + Postgres 架构（Bedi 的主张）。

本课时将逐一构建这些方案的精简版。

## 概念

### 持久化执行（Durable Execution）模式

持久化执行引擎会在每个“步骤”（LangGraph 中称为超步 super-step）之后保存完整的程序状态。发生崩溃时：

worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects

实现该模式的要求如下：

- **可序列化状态（Serializable State）**。所有智能体（Agent）状态必须可持久化。包含活跃数据库连接的函数闭包无法存活。
- **确定性恢复（Deterministic Resume）**。在相同的状态和输入下，智能体必须产生相同的动作（或针对大语言模型 LLM 调用，交由外部确定性预言机处理）。
- **幂等副作用（Idempotent Side Effects）**。外部调用（工具调用、支付等）必须是幂等的，或使用去重键（deduplication key）。

LangGraph 在每个超步后写入检查点（checkpoint）；Temporal 在每个活动（activity）后写入；Restate 使用事件溯源日志（event-sourced journals）。三者均实现了相同的模式。

### LangGraph 的运行时（Runtime）

每个智能体都有一个 `thread_id`；状态是一个类型化字典（typed dict）；每个超步会向检查点表写入一行数据。恢复时，运行时从最后一个检查点开始重放，而非从头开始。智能体可以调用 `interrupt()` 等待人类输入；此时运行时会持久化状态并释放工作节点（worker）。当输入到达时，任意工作节点均可恢复执行。

这是截至 2026 年 4 月的参考生产级架构设计。

### MegaAgent 的每智能体队列（Per-Agent Queue）

arXiv:2408.09955 描述了一项规模实验：在单个集群中运行数千个并发智能体。架构如下：

agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)

这种双层协调机制使得组内对话可以密集进行，而组间通信保持稀疏——这是在数千个智能体规模下保持成本呈线性增长的常用模式。

### 异步（Async）与每任务一线程（Thread-per-Job）

大语言模型调用属于 I/O 密集型（I/O-bound）。等待下一个 token 的线程在 99% 的时间内处于空闲状态。每个线程约消耗 1MB 内存；在 10,000 个并发调用下，仅栈空间就会占用 10GB。

纤程（Fibers，如 Python `asyncio`、Go goroutines、Rust `tokio`）会在 I/O 操作时协作式让出控制权。同样的 10,000 个调用可以轻松容纳在单个进程内。在 LLM 智能体规模下，异步并非一种优化手段，而是基础架构本身。

例外情况：CPU 密集型（CPU-bound）的后处理（如嵌入向量计算、分词器技巧）仍然需要线程或进程。请将 I/O 层与 CPU 计算层分离。

### Bedi 的反方观点

《Scaling Agentic Software》（Ashpreet Bedi，2026）指出，大多数团队在尚未测量实际负载之前就进行了过度设计。务实的默认方案如下：

- FastAPI + Postgres。
- 每次智能体运行对应数据库中的一行；状态通过乐观并发控制（optimistic concurrency）就地更新。
- 通过 `pg_notify` 或简单的 Celery worker 处理后台任务。
- 在应用代码中实现重试策略。

对于可管理任务下约 100 个并发智能体运行的负载，这通常已完全足够。仅在实测发现性能瓶颈或失败时再进行架构升级。

核心原则：仅在遇到简单架构无法解决的具体问题时，才引入持久化执行框架。过早采用只会将时间浪费在无法带来实际收益的繁琐流程上。

### 精确一次语义（Exactly-Once Semantics）

对于涉及付费的智能体运行，你需要实现“实际精确一次”（exactly-once effective，即至少一次投递 at-least-once delivery + 幂等消费者 idempotent consumer）。工程实践如下：

- **每次运行使用去重键**。将其包含在每一次副作用调用中。
- **发件箱模式（Outbox Pattern）**。副作用先写入数据库表，再由独立进程执行。两个步骤均需保证幂等。
- **补偿事务（Compensating Transactions）**。当副作用执行成功但其状态记录写入失败时，调度补偿操作。

这些属于数据库工程模式，并非 LLM 专属。LLM 带来的额外开销仅在于调用速度较慢；其余部分均为标准的分布式系统设计。

### 彩虹部署（Rainbow Deployment）

Anthropic 的多智能体研究系统采用了“彩虹部署”：多个版本的智能体运行时并发运行，从而避免在每次代码部署时强制终止长时间运行的智能体。对新版本仅对部分流量进行金丝雀发布（Canary）；待旧版本中的智能体执行完毕后，再逐步退役旧版本。

这是长运行有状态系统的标准做法；2026 年的演进在于智能体可能持续运行数小时，因此部署周期必须能够兼容这一特性。

### 标准生产环境检查清单

- 持久化状态（检查点、快照，或发件箱 + 可重放日志）。
- 幂等副作用。
- 用于 LLM 调用的异步 I/O 层。
- 带去重机制的至少一次投递。
- 针对有状态工作负载的彩虹/金丝雀部署。
- 可观测性（Observability）：每智能体追踪（traces）、超步审计、重试计数器。

## 构建它

`code/main.py` 实现了以下功能：

- `CheckpointStore` —— 基于 SQLite 的检查点日志，以线程 ID（thread-id）作为键。每个超步（super-step）会追加一行记录。
- `run_with_checkpoint(agent, thread_id)` —— 模拟运行中途崩溃；第二个 Worker（工作进程）将从最后一个检查点恢复执行。
- `AgentQueue` —— 每个智能体（Agent）独立的空闲（Idle）/ 处理中（Processing）/ 响应（Response）状态机，附带一个小型工作队列。
- `demo_async_vs_threads()` —— 分别通过 `asyncio` 和多线程运行 500 个并发的模拟“大语言模型（LLM）调用”；报告实际耗时（wall-clock）和峰值内存（近似值）。

运行方式：

python3 code/main.py

预期输出：模拟崩溃后检查点恢复成功；异步版本在 1 秒内处理完 500 个并发调用；多线程版本耗时数秒，且每个并发单元的内存占用高出数个数量级。

## 使用它

`outputs/skill-scaling-advisor.md` 提供了关于持久化执行（durable-execution）方案的选择建议：FastAPI + Postgres、LangGraph 运行时（runtime）、Temporal 或自定义方案。具体选择需根据负载、状态保留需求以及部署频率进行权衡。

## 部署上线

生产环境加固的标准实践：

- **从简起步（Bedi 法则）。** 使用 FastAPI + Postgres，直到实际测量到性能瓶颈或故障为止。
- **优化前先做好全链路监控（Instrument everything）。** 记录每次运行的延迟直方图、单步耗时、重试次数及失败分类。
- **针对副作用采用发件箱模式（Outbox pattern）。** 尤其适用于支付和外部 API 调用。
- **彩虹部署（Rainbow deploys）。** 部署期间绝不要中断正在运行的智能体任务。
- **当遇到特定问题时再引入持久化执行引擎（durable-execution engines，如 Temporal / LangGraph / Restate）：** 例如长达数小时的人机协同（human-in-the-loop）等待、跨区域协调、或复杂的重试/补偿策略。
- **I/O 层采用异步（Async）架构。** 仅将多线程用于 CPU 密集型（CPU-bound）的后处理任务。

## 练习

1. 运行 `code/main.py`。验证检查点恢复功能是否正常；测量异步与多线程在并发性能上的差异。
2. 实现一个**发件箱（outbox）**表：每次工具调用先写入发件箱，再由独立的 goroutine/任务执行。通过重复运行两次工具调用来验证其幂等性（idempotency）。
3. 模拟**彩虹部署（rainbow deploy）**：同时运行两个版本的运行时；将一半的新 `thread_id` 路由到每个版本；确认旧版本上正在运行的线程不会被中断。
4. 阅读 LangGraph 的运行时文档（链接见下文）。找出该运行时的哪些功能在手动实现的 FastAPI + Postgres 版本中最难/最耗时复刻。这是否构成采用该框架的理由，还是可以暂缓？
5. 阅读 MegaAgent（arXiv:2408.09955）第 3 节。文中明确提出了双层协调机制（组内协调 + 组间管理聊天）。请构思如何将其映射到包含两个队列族（queue families）的消息队列架构中。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 持久化执行 (Durable Execution) | “持久化程序状态” | 引擎在每个超步 (Super-step) 后写入状态；崩溃恢复具有确定性。 |
| 超步 (Super-step) | “事务边界” | 检查点之间的工作单元。LangGraph 术语。 |
| `thread_id` | “智能体运行标识符” | 用于绑定检查点与恢复逻辑的键。 |
| 幂等性 (Idempotency) | “可安全重试” | 重复执行副作用操作所产生的结果与单次尝试完全一致。 |
| 发件箱模式 (Outbox Pattern) | “解耦副作用” | 将操作意图写入数据表；由独立的执行器负责执行并标记完成。 |
| 至少一次交付 (At-least-once Delivery) | “可能出现重复” | 消息队列语义；通过去重键 (dedup key) 可使消费者实现实际上的仅一次处理 (effective-once)。 |
| 彩虹部署 (Rainbow Deploy) | “版本重叠” | 在长时间运行的工作负载期间，多个运行时版本并发共存。 |
| 异步纤程 (Async Fiber) | “协作式让出” | 用户态并发机制；对于 I/O 密集型负载，其开销远低于传统线程。 |
| 检查点 (Checkpoint) | “状态快照” | 超步边界处的序列化状态；是恢复执行的关键依据。 |

## 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph 运行时架构设计
- [MegaAgent](https://arxiv.org/abs/2408.09955) — 面向单个智能体的生产者-消费者队列；支持数千个并发智能体的双层协调机制
- [Matrix](https://arxiv.org/abs/2511.21686) — 以消息队列作为协调底座的去中心化框架
- [Temporal docs](https://docs.temporal.io/) — 实现持久化执行的参考级工作流引擎
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产环境实战经验，涵盖彩虹部署等实践
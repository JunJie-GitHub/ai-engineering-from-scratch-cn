# 长时间运行的后台智能体：持久化执行（Durable Execution）

> 生产环境中的长周期智能体（long-horizon agents）不会运行在 `while True` 循环中。每次大语言模型（LLM）调用都会转化为具备检查点（checkpoint）、重试（retry）和重放（replay）机制的活动（activity）。Temporal 的 OpenAI Agents SDK 集成已于 2026 年 3 月正式发布（GA）。Claude Code Routines（Anthropic）可在无需持久化本地进程的情况下，按计划调度 Claude Code 的调用。会话会在等待人类输入时暂停，能够跨越部署存活，并通过 `thread_id` 键从最新的检查点恢复。在这些全新的人机交互体验背后，是一个经典的设计模式——工作流编排（workflow orchestration）——只是引入了一个新的输入要素：将 LLM 调用视为非确定性活动（non-deterministic activities），在系统恢复时必须进行确定性重放。

**类型：** 学习
**语言：** Python（标准库，极简的持久化执行状态机）
**前置知识：** 第 15 阶段 · 10（权限模式），第 15 阶段 · 01（长周期智能体）
**预计耗时：** 约 60 分钟

## 核心问题

假设有一个运行时长为四小时的智能体。它调用了三次工具，两次向用户发起提示，并进行了四十次 LLM 调用。运行到一半时，其所在的主机重启了。会发生什么？

- 在朴素的 `while True` 循环中：所有进度都会丢失。任务将从头开始重新运行。那三次工具调用（带有真实的副作用）会再次执行。用户会被再次询问他们已经批准过的事项。四十次 LLM 调用会被重复计费。
- 采用持久化执行（durable execution）时：任务将从最近的检查点恢复。已完成的活动不会重新执行；其结果会从持久化日志（durable log）中重放。用户无需再次批准已确认的事项。已发生的 LLM 调用也不会被重复计费。

这正是工作流引擎（如 Temporal、Cadence、Uber 的 Cherami）在过去十年中一直采用的模式。不同之处在于，LLM 调用现在成为了一种活动——具有非确定性、高成本且带有副作用——并且它们能够完美契合这一模式。

本课贯穿始终的主题：长周期任务的可靠性会随时间衰减（METR 观察到“35 分钟衰减”（35-minute degradation）现象——成功率随任务时长呈近似二次方下降）。持久化执行使得任务运行时长能够突破可靠性曲线的限制。如果设计得当，这是一种安全容错的新范式；如果设计不当，则会导致不安全的失败。

## 核心概念

### 活动（Activity）、工作流（Workflow）与重放（Replay）

- **工作流（Workflow）**：确定性的编排代码。用于定义活动的执行顺序、分支逻辑以及等待条件。必须具备确定性，以便能够从事件日志中重放，而不会出现意外的偏差。
- **活动（Activity）**：非确定性且可能失败的工作单元。例如大语言模型（LLM）调用、工具调用、文件写入或 HTTP 请求。每个活动都会记录其输入参数，并在完成后记录输出结果。
- **事件日志（Event Log）**：持久化的底层存储。记录每个活动的启动、完成、失败、重试，以及工作流的每一次决策。
- **重放（Replay）**：在恢复运行时，工作流代码会从头开始执行；所有已完成的活动将直接返回已记录的结果，而不会重新执行。只有尚未完成的活动才会被实际运行。

这与 React 基于虚拟 DOM 重新渲染，或 Git 根据提交记录重建工作树的原理如出一辙。编排器（Orchestrator）的确定性正是实现低成本持久化的关键。

### 为什么 LLM 调用符合该模式

LLM 调用具有以下特征：
- 非确定性（温度参数 > 0；即使温度参数为 0，不同模型版本间也会存在漂移）。
- 成本高昂（涉及资金消耗与延迟）。
- 可能失败（受限于速率限制或超时）。
- 具有副作用（若其调用了外部工具）。

这完全符合“活动”的特征。将每次 LLM 调用封装为活动，即可实现指数退避重试（Exponential Backoff Retry）、跨重启的检查点（Checkpoint）机制，以及用于调试的可重放追踪记录。

### 基于 `thread_id` 的检查点（Checkpoint）

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 以及 Claude Code Routines 均采用了相同的 API 设计模式：通过 `thread_id`（或等效标识符）来标识会话；每次状态转换都会持久化到后端存储（默认使用 PostgreSQL，开发环境使用 SQLite，缓存使用 Redis）；恢复运行时则读取最新的检查点。

后端存储的选择至关重要：

- **PostgreSQL**：具备持久性、支持查询，且在部署后数据不丢失。LangGraph 的默认选项。
- **SQLite**：仅限本地开发使用；跨主机时数据会丢失。
- **Redis**：速度极快，但除非配置了 AOF 或快照，否则数据是临时的。
- **Cloudflare Durable Objects**：透明分布式架构；通过唯一键进行作用域隔离；数据可存活数小时至数周。

### 将人类输入作为一等公民状态

“先提议后提交”（Propose-then-commit，第 15 课）需要一种持久的“等待人类输入”状态。工作流会在此暂停，外部队列将挂起该请求，待审批通过后，工作流将从该确切位置恢复执行。若无持久化机制，这只能做到尽力而为；有了持久化，即使审批在夜间完成，工作流也能在次日清晨无缝接续。

### 35 分钟可靠性衰减现象

METR 的研究表明，所有被测的 Agent 类别在连续运行约 35 分钟后，均会出现可靠性衰减。任务时长翻倍，故障率大致会增至四倍。持久化执行（Durable Execution）无法解决此问题，但它允许系统在超出可靠性支持范围的情况下继续运行。安全的实践模式是：将持久化与检查点机制结合，在重新进入时要求全新的人工介入（Human-In-The-Loop, HITL）；同时配合预算熔断开关（Budget Kill Switch，第 13 课），无论实际运行时间多长，均对总计算量进行硬性限制。

### 何时不应采用持久化执行

- 运行时间短于几分钟且无需人类输入的场景。此时系统开销大于收益。
- 严格只读的信息检索任务。
- 正确性要求必须在单个上下文窗口内端到端完成的任务（例如部分推理任务或一次性生成任务）。

## 使用方法

`code/main.py` 使用 Python 标准库实现了一个最小化的持久化执行引擎（durable-execution engine）。它支持：

- `@activity` 装饰器，用于将输入和输出记录到 JSON 事件日志（event log）中。
- 一个用于按顺序编排活动（activity）的工作流（workflow）函数。
- 一个 `run_or_replay(workflow, event_log)` 函数，用于重放（replay）已完成的活动，而无需重新执行它们。

驱动程序模拟了一个包含三个活动的工作流，在运行中途发生崩溃，并展示了两种情况：(a) 朴素重试（naive retry）会重新执行所有内容，而 (b) 重放机制仅运行缺失的活动。

## 部署上线

`outputs/skill-durable-execution-review.md` 审查了一个拟议的长周期智能体（agent）部署方案，以确保其符合正确的持久化执行架构：活动、确定性（determinism）、检查点（checkpoint）后端、人工输入状态以及恢复时的人工介入（HITL-on-resume）策略。

## 练习

1. 运行 `code/main.py`。观察朴素重试与重放机制在活动执行次数上的差异。更改崩溃点，并展示重放次数如何随之变化。

2. 修改该示例引擎以显式使用 `thread_id`。模拟两个共享该引擎的并发会话（session），并确认它们的事件日志不会发生冲突。

3. 选取示例引擎中的一个活动。引入一个非确定性因素（non-determinism）（例如在工作流决策中使用墙钟时间戳）。演示重放时出现的分歧。解释真实引擎如何处理此问题（副作用注册（side-effect registration）、`Workflow.now()` API）。

4. 阅读 LangChain 的《Runtime behind production deep agents》文章。列出运行时（runtime）持久化的所有状态，并指明每个状态所覆盖的故障模式（failure mode）。

5. 为一个耗时 6 小时的自主编码任务设计检查点策略。你在何处设置检查点？崩溃恢复（resume-on-crash）的流程是怎样的？哪些环节需要全新的人工介入（HITL）？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| 工作流（Workflow） | “智能体的脚本” | 确定性编排代码；可从事件日志中重放 |
| 活动（Activity） | “一个步骤” | 非确定性单元（如大语言模型调用、工具调用）；执行前后均会记录日志 |
| 事件日志（Event log） | “底层存储” | 每次状态转换的持久化记录 |
| 重放（Replay） | “恢复运行” | 重新运行工作流；已完成的活动直接返回已记录的结果，无需重新执行 |
| 检查点（Checkpoint） | “存档点” | 以 `thread_id` 为键的持久化状态；恢复时采用最新状态覆盖原则 |
| `thread_id` | “会话密钥” | 用于界定持久化状态范围的标识符 |
| 35 分钟性能衰减（35-minute degradation） | “可靠性衰退” | METR 研究指出：成功率随任务时长呈近似二次方下降 |
| 非确定性（Non-determinism） | “重放漂移” | 墙钟时间、随机数、大语言模型输出；必须作为副作用进行注册 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK：智能体循环（Agent Loop）](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 预算（Budget）、交互轮次（Turns）与状态恢复（Resume）语义。
- [Microsoft — 智能体框架（Agent Framework）：人在回路（Human-in-the-Loop）与检查点机制（Checkpointing）](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent` 结构（Shape）。
- [LangChain — 生产级深度智能体（Deep Agents）背后的运行时（Runtime）](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体的运行时（Runtime）要求。
- [OpenAI Agents SDK 与 Temporal 集成（Trigger.dev 公告）](https://trigger.dev) — 大语言模型（LLM）调用的活动（Activity）结构（Shape）。
- [Anthropic — 实践中智能体自主性（Agent Autonomy）的评估](https://www.anthropic.com/research/measuring-agent-autonomy) — 35 分钟性能衰减（Degradation）参考基准。
# LangGraph：有状态图（Stateful Graphs）与持久化执行（Durable Execution）

> LangGraph 是 2026 年底层有状态编排（low-level stateful orchestration）的参考标准。智能体（Agent）本质上是一个状态机（state machine）；节点（nodes）即函数；边（edges）代表状态转换（transitions）；状态（state）是不可变的（immutable），且在每一步执行后都会生成检查点（checkpoint）。系统可从任意故障点精确恢复（resume）至中断前的位置。

**类型：** 学习 + 实践
**编程语言：** Python (stdlib)
**前置知识：** 第 14 阶段 · 01（智能体循环），第 14 阶段 · 12（工作流模式）
**预计耗时：** 约 75 分钟

## 学习目标

- 阐述 LangGraph 的核心模型：基于不可变状态的状态机、函数节点、条件边以及步骤后检查点。
- 列举文档强调的四大核心能力：持久化执行、流式处理、人机协同以及全面记忆。
- 解释 LangGraph 支持的三种编排拓扑结构：监督者模式、点对点/蜂群模式以及分层模式（嵌套子图）。
- 使用标准库实现一个具备不可变状态、条件边以及检查点/恢复循环的状态图。

## 痛点分析

智能体与工作流面临一个共同难题：当一个包含 40 个步骤的运行流程在第 38 步失败时，开发者期望从第 38 步恢复，而非从头开始。次级状态模型往往迫使运维人员围绕那些默认每次都是全新运行的库，手动拼凑重试逻辑。

LangGraph 的设计方案是：将状态视为一等公民的类型化对象，状态变更必须显式声明，且在每个节点执行后都会持久化检查点。恢复执行仅需调用一次 `load_state(session_id)`。

## 核心概念

### 图（Graph）

图由以下要素定义：

- **状态类型（State type）**。一种类型化字典（或 Pydantic 模型），供每个节点读取和修改。
- **节点（Nodes）**。纯函数 `(state) -> state_update`。函数返回后，更新内容会合并到状态中。
- **边（Edges）**。节点之间的条件或直接转换路径。
- **入口与出口**。`START` 和 `END` 哨兵节点（sentinel nodes）用于标记边界。

示例：一个包含 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体——将路由工作流抽象为图结构。

### 持久化执行（Durable execution）

每个节点返回后，运行时（runtime）会将状态序列化并写入检查点存储（checkpointer，如 SQLite、Postgres、Redis 或自定义实现）。若在第 N 步发生故障，运行时可通过 `resume(session_id)` 恢复执行，并携带精确的状态从第 N+1 步继续。

LangGraph 文档明确强调了该特性在生产环境中的重要性，并列举了 Klarna、Uber、摩根大通（J.P. Morgan）等用户。其核心优势并非图结构本身，而是“图结构 + 检查点机制”使得故障恢复的成本大幅降低。

### 流式输出（Streaming）

每个节点均可产出部分输出。图会将每个节点的增量事件（per-node-delta events）流式传输给调用方，以便在图运行过程中实时更新用户界面。

### 人在回路（Human-in-the-loop）

在节点之间检查并修改状态。典型实现方式：在关键节点前暂停执行，将状态呈现给人类操作者，接收修改后继续执行。由于状态已被序列化，检查点机制使得这一过程变得十分便捷。

### 记忆（Memory）

分为短期记忆（单次运行内——状态中保存的对话历史）和长期记忆（跨运行——通过检查点持久化，并结合独立的长期存储）。LangGraph 可通过工具与外部记忆系统（如 Mem0 或自定义系统）集成。

### 三种拓扑结构（Topologies）

1. **监督者模式（Supervisor）**。由中心路由大语言模型（LLM）将任务分发给专业子智能体（subagents）。可使用 `langgraph-supervisor` 中的 `create_supervisor()` 实现（不过 LangChain 团队在 2026 年建议直接通过工具调用（tool calls）来实现，以便更好地控制上下文）。
2. **蜂群/对等模式（Swarm / peer-to-peer）**。智能体通过共享的工具接口直接交接任务，无需中心路由。
3. **层级模式（Hierarchical）**。监督者管理子监督者，通过嵌套子图（nested subgraphs）实现。

### 该模式的常见陷阱

- **检查点粒度过小**。仅对对话轮次（conversation turns）设置检查点会导致工具状态和记忆写入无法恢复。必须对完整状态进行序列化。
- **非确定性节点**。恢复执行的前提是相同的节点输入能产生相同的状态更新。必须捕获随机种子、物理时钟（wall-clock）以及外部 API 的响应。
- **过度使用条件边**。如果图中的每条边都是条件边，它将变成一个难以推理的状态机（state machine）。建议优先采用线性链式结构，仅在必要时添加分支。

## 动手构建（Build It）

`code/main.py` 实现了一个基于标准库的有状态图（stateful graph）：

- `State` —— 一个包含 `messages`、`step`、`route`、`output`、`human_approval` 的类型化字典（typed dict）。
- `Node` —— 可调用对象（callable），接收状态并返回更新字典。
- `StateGraph` —— 包含节点、边、条件边（conditional edge）、运行（run）与恢复（resume）功能。
- `SQLiteCheckpointer`（内存模拟版）—— 在每个节点执行后序列化状态；通过 `load(session_id)` 恢复状态。
- 一个演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行方式：

python3 code/main.py

执行轨迹（trace）显示：首次运行在人工审批节点（human gate）处暂停，状态被持久化（persistence）后，恢复运行（resume）并生成最终输出。

## 使用方式

- **LangGraph** —— 官方参考实现，已具备生产环境可用性。可使用 `create_react_agent`、`create_supervisor`，或自行构建图。
- **AutoGen v0.4**（第 14 课）—— 面向高并发场景的 Actor 模型（actor model）替代方案。
- **Claude Agent SDK**（第 17 课）—— 提供托管框架，内置会话存储（session store）。
- **自定义实现** —— 当需要精确控制状态结构（state shape）或检查点后端（checkpointer backend）时使用。

## 部署与交付

`outputs/skill-state-graph.md` 可在任意目标运行时中生成符合 LangGraph 架构的状态图，并已内置检查点机制（checkpointing）与恢复（resume）逻辑。

## 练习

1. 当分类置信度低于设定阈值时，添加一条从 `classify` 到 `end` 的条件边（conditional edge）。在人工手动设置 `route` 后恢复运行。
2. 将模拟的 SQLite 检查点替换为真实的 SQLite 检查点（checkpointer）。测量每一步的序列化开销。
3. 实现并行边（parallel edges）：两个节点并发运行，通过自定义归约器（reducer）合并结果。在此场景下，不可变状态（immutable state）能带来什么优势？
4. 阅读 `langgraph-supervisor` 参考文档。将该示例移植到 `create_supervisor`。对比两者的执行轨迹（trace）结构。
5. 添加流式输出（streaming）：每个节点在运行时逐步产出部分状态。实时打印接收到的增量数据（deltas）。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| State graph（状态图） | “将智能体视为状态机” | 类型化状态 + 节点 + 边 + 归约器 |
| Checkpointer（检查点） | “持久化后端” | 在每个节点后序列化状态；支持断点恢复 |
| Reducer（归约器） | “状态合并器” | 将当前状态与节点更新进行合并的函数 |
| Conditional edge（条件边） | “分支” | 根据状态函数动态选择的边 |
| Subgraph（子图） | “嵌套图” | 作为节点嵌入到另一个图中的图结构 |
| Durable execution（持久化执行） | “故障恢复” | 携带精确状态从最后一个成功节点重启 |
| Supervisor（监督器） | “路由大模型” | 负责调度专业子智能体的中央分发器 |
| Swarm（蜂群/多智能体协作） | “P2P 智能体” | 智能体通过共享工具进行交接；无中央路由 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档
- [langgraph-supervisor 参考文档](https://reference.langchain.com/python/langgraph/supervisor/) — 监督模式 (Supervisor Pattern) API
- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — Actor 模型 (Actor Model) 替代方案
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — 会话存储 (Session Store) 与子智能体 (Subagents)
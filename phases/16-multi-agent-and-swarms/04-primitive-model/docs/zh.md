# 多智能体原语模型 (Multi-Agent Primitive Model)

> 2026 年发布的每一个多智能体框架 (Multi-Agent Framework) —— AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework —— 都是四维设计空间中的一个坐标点。仅包含四个原语 (Primitive)：智能体 (Agent)、任务交接 (Handoff)、共享状态 (Shared State) 和编排器 (Orchestrator)。本课将从零开始构建它们，基于这四个原语运行一个示例系统，然后将所有主流框架映射到相同的坐标轴上，让你只需一段话就能读懂任何新发布的框架。

**类型：** 学习
**语言：** Python (stdlib)
**前置知识：** 第 14 阶段（智能体工程），第 16 阶段 · 01（为什么使用多智能体）
**预计时间：** 约 60 分钟

## 问题

每六个月就会有一个新的多智能体框架发布。2023 年的 AutoGen，2024 年的 CrewAI，2024 年的 LangGraph 和 OpenAI Swarm，2025 年 4 月的 Google ADK，以及 2026 年 2 月的 Microsoft Agent Framework RC。每一份新闻稿都宣称自己是“最合适的抽象 (Abstraction)”。

如果你试图逐个学习它们，很快就会精疲力竭。它们的 API 看起来各不相同。文档对“智能体”的定义也莫衷一是。一个框架将其共享内存称为“黑板 (Blackboard)”，另一个称为“消息池 (Message Pool)”，第三个则称为“状态图 (StateGraph)”。你开始怀疑这个领域只是在频繁更迭。

事实并非如此。在营销话术之下，这四个原语是稳定不变的。只需掌握一次，你就能用一段话读懂每一个新框架。

## 概念

### 四大原语 (Primitives)

1. **智能体 (Agent)** —— 系统提示词 (System Prompt) 加上工具列表。无状态 (Stateless)；每次运行都从其系统提示词和当前消息历史开始。
2. **交接 (Handoff)** —— 控制权从一个智能体到另一个智能体的结构化转移。在机制上，表现为返回新智能体的工具调用，或遵循特定条件的图边 (Graph Edge)。
3. **共享状态 (Shared State)** —— 任何可供多个智能体读取（有时可写入）的数据结构。例如消息池、黑板 (Blackboard)、键值存储、向量内存。
4. **编排器 (Orchestrator)** —— 决定下一个发言者的实体。选项包括：显式图（确定性）、大语言模型 (LLM) 发言选择器（软性）、上一发言者的交接调用（OpenAI Swarm），或基于队列的调度器（群体架构）。

这就是完整的设计空间。每个框架仅为每个维度选择默认值；其余部分仅是表层语法。

### 2026 年各框架如何映射到该模型

| 框架 | 智能体 (Agent) | 交接 (Handoff) | 共享状态 (Shared State) | 编排器 (Orchestrator) |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用者的问题 | LLM 的下一次交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的发言选择器 | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 任务输出链式传递 | 管理 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` 归约器 (Reducer) | 图，确定性 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程 / 上下文 | 模式特定 |
| Google ADK | 智能体 + A2A 卡片 | A2A 任务 | A2A 工件 | 宿主决定 |

表层差异看似巨大。底层逻辑：相同的四个控制旋钮。

### 为何这很重要

一旦看清这些原语，框架对比就变成了一份简短的检查清单：

- 编排器是信任 LLM 进行路由（Swarm），还是将路由硬编码在代码中（LangGraph）？
- 共享状态是完整历史记录（GroupChat），还是投影视图（StateGraph 归约器）？
- 智能体能否修改彼此的提示词（CrewAI 管理器），还是仅能进行交接（Swarm）？

这三个问题能解答 80% 关于“哪个框架适合特定问题”的疑问。你将不再盲目寻找“最佳的多智能体框架”，而是开始针对你真正关心的维度进行设计。

### 无状态洞察

除共享状态外，所有原语都是无状态的。智能体是（提示词，工具）的函数。交接是函数调用。编排器是调度器。**系统中唯一有状态的组件就是共享状态。** 所有棘手的 Bug 都潜伏于此：记忆投毒（第 15 课）、消息排序、版本控制、写入竞争。

隐藏共享状态的框架（Swarm）将问题推给了调用者。集中管理共享状态的框架（LangGraph 检查点、AutoGen 消息池）使其可被检查，但将协调成本转移到了共享状态的实现上。

### 单个原语的解剖结构

#### 智能体 (Agent)

Agent = (system_prompt, tools, model, optional_name)

无记忆。无状态。拥有相同系统提示词和工具的两个智能体可以互换。所有看似属于单个智能体的状态，实际上都存在于共享状态或交接协议中。

#### 交接 (Handoff)

Handoff = (from_agent, to_agent, reason, payload)

三种主流实现方式：

- **函数返回** —— 工具返回下一个智能体。这是 OpenAI Swarm 的模式。智能体在其工具模式 (Schema) 中携带路由逻辑。
- **图边** —— LangGraph。边是声明式的。LLM 生成一个值；条件选择下一个节点。
- **发言选择** —— AutoGen GroupChat。选择器函数（有时本身就是一个 LLM 调用）读取消息池并挑选下一个发言者。

#### 共享状态 (Shared State)

SharedState = { messages: [], artifacts: {}, context: {} }

最低限度是一个消息列表。通常更多：结构化工件（CrewAI 任务输出）、类型化上下文（LangGraph 归约器）、外部记忆（MCP、向量数据库）。

两种拓扑结构：**完整池**（每个智能体都能看到所有消息）和**投影视图**（智能体只能看到其角色范围内的视图）。完整池实现简单但扩展性差。投影视图扩展性好，但需要预先进行模式设计。

#### 编排器 (Orchestrator)

Orchestrator = ({state, last_speaker}) -> next_agent

四种变体：

- **静态** —— 图在构建时即固定（LangGraph 确定性、CrewAI 顺序模式）。
- **LLM 选择** —— LLM 读取消息池并挑选下一个发言者（AutoGen、CrewAI 层级模式）。
- **交接驱动** —— 当前智能体通过调用交接工具来决定（Swarm）。
- **队列驱动** —— 工作节点从共享队列中拉取任务；没有明确的下一个发言者（群体架构、Matrix）。

### 框架之间的差异

一旦原语确定，剩余的设计决策包括：

- **记忆策略** —— 临时存储与持久化检查点 (Checkpointing)（LangGraph 检查点器）。
- **安全边界** —— 谁有权批准交接（人在回路 Human-in-the-loop）。
- **成本核算** —— 每个智能体的 Token 预算。
- **可观测性** —— 追踪交接过程、持久化状态以供回放。

所有这些均可在原语之上实现。它们本身都不是新的原语。

## 构建它

`code/main.py` 使用约 150 行 Python 标准库代码实现了这四个原语（primitives）。此处未接入真实的大语言模型（LLM）——每个智能体（agent）均采用脚本化策略（scripted policy），以便将重点保持在协调结构（coordination structure）上。

该文件导出了以下内容：

- `Agent` —— 一个包含名称、系统提示词、工具和策略函数的数据类（dataclass）。
- `Handoff` —— 一个用于返回新智能体的函数。
- `SharedState` —— 一个线程安全（thread-safe）的消息池（message pool）。
- `Orchestrator` —— 包含三种变体的编排器（orchestrator）：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

该演示程序将相同的三智能体流水线（pipeline）（研究 → 撰写 → 评审）依次通过所有三种编排器类型运行，并在最后打印消息池。你可以看到，输出结果的差异仅在于*由谁决定下一步*；而在各次运行中，智能体与共享状态（shared state）完全一致。

运行方式：

python3 code/main.py

预期输出：三次编排器运行，每种模式各一次。每次运行都会打印最终的消息池。如果研究智能体提前判定任务完成，由交接驱动的运行将访问更少的智能体——这正是大语言模型路由（LLM-routing）权衡机制的缩影。

## 使用它

`outputs/skill-primitive-mapper.md` 是一项技能（skill），用于读取任意多智能体代码库或框架文档，并返回四原语映射（mapping）关系。在新框架发布时运行它，可在深入阅读文档前快速获得一段话的概要理解。

## 交付使用

在采用新框架之前，先为其编写原语映射。如果无法编写，说明文档不完整，或者该框架正在发明第五种原语（这种情况很少见——请检查是否存在你未曾见过的共享状态变体）。

将该映射固定在你的架构文档中。当新团队成员加入时，在发送 API 文档之前，先发送这份映射文档。当框架版本更新时，对比映射文件的差异，而不是去读更新日志。

## 练习

1. 使用不同的智能体策略运行 `code/main.py` 三次。观察编排器的选择如何影响实际运行的智能体。
2. 实现第四种编排器类型：一种由队列驱动的类型，智能体通过轮询共享状态来获取任务。可能会发生什么死锁（deadlock），你该如何检测它？
3. 参考 LangGraph 快速入门指南（https://docs.langchain.com/oss/python/langgraph/workflows-agents），将其重写为四原语形式。LangGraph 的哪些抽象（abstraction）是 1:1 映射的，哪些只是便捷封装（wrapper）？
4. 阅读 OpenAI Swarm 教程（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出 Swarm 对四种原语中的哪一种提供了最易用（ergonomic）的设计，又将哪一种推给了调用方（caller）处理。
5. 在本文表格中找出一个完全隐藏共享状态的框架。解释当智能体需要在交接过程中进行协调且无法重新读取历史记录时，会出现什么问题。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|----------------|------------------------|
| 智能体 (Agent) | “带工具的大语言模型” | 由 `(system_prompt, tools, model)` 构成的三元组。无状态 (Stateless)。 |
| 交接 (Handoff) | “控制权转移” | 一种结构化调用，用于指定下一个智能体及可选的负载数据 (Payload)。三种实现方式：函数返回、图边 (Graph Edge)、发言者选择 (Speaker Selection)。 |
| 共享状态 (Shared State) | “记忆” / “上下文” | 多智能体系统中唯一具备状态的部分。通常表现为消息池或黑板 (Blackboard)。 |
| 编排器 (Orchestrator) | “协调器” | 负责决定下一个执行者的实体。可采用静态图 (Static Graph)、大语言模型选择器 (LLM Selector)、交接驱动 (Handoff-Driven) 或队列驱动 (Queue-Driven)。 |
| 原语 (Primitive) | “抽象” | 每个框架都会进行参数化的四个核心维度之一。并非框架的专属功能。 |
| 消息池 (Message Pool) | “共享聊天记录” | 包含完整历史记录的共享状态。逻辑易于理解，但扩展性较差。 |
| 投影状态 (Projected State) | “作用域视图” | 面向特定角色的共享状态视图。扩展性良好，但需要进行数据模式设计 (Schema Design)。 |
| 发言者选择 (Speaker Selection) | “下一个谁发言” | 一种编排器模式，由特定函数（通常是大语言模型）从一组智能体中挑选下一个执行者。 |

## 进一步阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 对交接驱动型编排 (Handoff-Driven Orchestration) 最清晰的阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat 结合发言者选择是大语言模型选择型编排 (LLM-Selected Orchestration) 的参考实现
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 基于图边的编排与基于归约器 (Reducer) 的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — 基于角色-目标-背景设定的智能体，支持顺序/层级流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — 微软将 v0.4 转入维护阶段后，活跃的 AutoGen v0.2 分支
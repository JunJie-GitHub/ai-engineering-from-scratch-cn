# 智能体框架 (Agent Framework) 权衡 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架推销的都是同一个演示案例（研究智能体生成报告），隐藏的也是同一个缺陷（状态结构 (state schema) 与编排层 (orchestration layer) 发生冲突）。请选择其抽象概念 (abstraction) 与你问题形态相匹配的框架；其余部分不过是你需要重复编写的胶水代码。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第11阶段 · 09（函数调用 (Function Calling)），第11阶段 · 16（LangGraph）
**Time:** 约45分钟

## 问题所在

你面临的任务需要多次调用大语言模型 (LLM)。它可能是一个研究工作流（规划、搜索、总结、引用）。也可能是一个代码审查流水线（解析差异、提出批评、生成补丁、验证结果）。还可能是一个多轮对话助手，负责预订航班、撰写邮件和提交报销单。于是你选择了一个框架。

三天后，你发现该框架的抽象泄漏 (abstraction leak) 问题暴露无遗。CrewAI 提供了角色定义，但当“研究员”需要将结构化计划交给“撰稿人”时，框架却处处掣肘。AutoGen 支持智能体之间的对话，但缺乏一等公民状态 (first-class state)，导致你的检查点只能是对话日志的 pickle 序列化对象。LangGraph 提供了状态图 (state graph)，但要求你在明确智能体行为之前，就必须为每一次状态转移 (transition) 命名。Agno 提供了单智能体原语 (single-agent primitive)，但当你尝试将其扇出 (fan out) 到三个并发工作节点 (concurrent workers) 时，它就会报错崩溃。

解决之道并非“挑选最好的框架”。而是让框架的核心抽象与你问题的形态相匹配。本课程将为你绘制这张选型地图。

## 核心概念

![智能体框架矩阵：核心抽象 vs 问题形态](../assets/framework-matrix.svg)

四大框架主导了 2026 年的技术格局。它们的核心抽象（Core Abstraction）各不相同。

| 框架 | 核心抽象 | 最佳适用场景 | 最不适用场景 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` —— 类型化状态（Typed State）、节点、条件边（Conditional Edges）、检查点（Checkpointer）。 | 具有显式状态和人在回路（Human-in-the-loop）中断的工作流；需要时间旅行调试（Time-travel Debugging）的生产级智能体。 | 拓扑结构未知的、松散的、基于角色的头脑风暴。 |
| **CrewAI** | `Crew` —— 角色（目标、背景故事）、任务、流程（顺序或层级）。 | 具有简短线性/层级计划的、基于角色扮演或人设驱动的工作流。 | 超出团队轮次历史之外的任何有状态（Stateful）场景；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 —— 两个或多个智能体轮流对话，直到满足退出条件。 | 多智能体*对话*（师生、提议者-批评者、执行者-审查者），思维从聊天中涌现。 | 具有已知有向无环图（DAG）的确定性工作流；任何需要在重启后保持持久状态的场景。 |
| **Agno** | `Agent` —— 单个大语言模型（LLM）+ 工具 + 记忆，可组合成团队。 | 快速构建的单体智能体和轻量级团队；强大的多模态（Multi-modality）能力和内置存储驱动。 | 具有自定义归约器（Reducer）的深度、显式分支图。 |

### “抽象”的实际含义

框架的核心抽象，就是你在向他人推介架构时，在白板上画出的那个核心图示。

- **LangGraph** → 你画的是一个图（Graph）。节点代表步骤，边代表状态转换，每个点的状态对象都是类型化的。其心智模型是状态机（State Machine）。
- **CrewAI** → 你画的是组织架构图。每个角色都有岗位描述，由管理者负责任务分发。其心智模型是一个由专家组成的小型团队。
- **AutoGen** → 你画的是 Slack 私信界面。两个智能体互相发消息；如果需要协调员，第三个智能体加入。其心智模型是聊天。
- **Agno** → 你画的是一个带有外挂工具的独立方框。将多个方框并排放置即组成团队。其心智模型是“开箱即用的智能体”。

### 状态管理问题

状态（State）管理是大多数框架在生产环境中表现不佳的症结所在。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型）、按字段归约、一等公民级的检查点（支持 SQLite/Postgres/Redis）。恢复、中断和时间旅行调试均为内置功能。*（参见第 11 阶段 · 16。）*
- **CrewAI。** 状态通过 `context` 字段以字符串形式在任务间流转，或通过 `output_pydantic` 进行结构化。开箱即无团队级持久化存储；若需团队在重启后存活，需自行外挂存储方案。
- **AutoGen。** 状态即聊天记录及用户自定义的 `context`。对话记录可持久化；但任意工作流状态除非自行编写适配器，否则无法持久化。
- **Agno。** 通过 `storage=` 参数为 `Agent` 绑定内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）—— 对话会话和用户记忆自动持久化。它并非完整的图检查点，而是一个会话存储（Session Store）。

### 分支控制问题

任何非平凡的智能体都会产生分支。由谁来决定分支走向至关重要。

- **LangGraph** —— 由你通过条件边决定。路由是一个带有命名分支的 Python 函数。分支在编译后的图中是一等公民；检查点会记录实际走过的分支路径。
- **CrewAI** —— 在层级模式下由管理者决定；在顺序模式下由你在构建时决定。路由隐含在任务列表中；除了管理者的提示词（Prompt）外，没有一等公民级的“条件判断”。
- **AutoGen** —— 由智能体通过聊天决定。分支走向由“下一个谁发言”自然涌现。`GroupChatManager` 负责选择下一位发言者；你可以手写 `speaker_selection_method`，但默认由大语言模型驱动。
- **Agno** —— 由智能体决定下一步调用哪个工具。团队支持协调器/路由器/协作者模式；超出此范围的分支逻辑由开发者自行负责。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任意开放遥测（OpenTelemetry）导出器实现可观测性。每次节点转换都是一个追踪跨度（Trace Span）；检查点同时可作为可重放的追踪记录。LangSmith 是官方首选方案；Langfuse/Phoenix 也提供适配器。
- **CrewAI** —— 自 2025 年底起提供一等公民级的 OpenTelemetry 支持；已集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen** —— 通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 提供连接器。追踪粒度为“每条智能体消息”，而非“每个节点”。
- **Agno** —— 内置 `monitoring=True` 标志及 OpenTelemetry 导出器；与 Langfuse 深度集成以追踪会话。

### 成本与延迟

这四大框架都会为每次调用增加额外开销（框架逻辑、验证、序列化）。开销大致递增顺序为：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架额外执行的大语言模型路由量。CrewAI 的层级管理者会消耗 Token 来决定下一步由谁执行；AutoGen 的 `GroupChatManager` 同理。LangGraph 仅在你显式编写 `llm.invoke` 的地方消耗 Token。Agno 的单体智能体路径则非常轻量。

当单次运行成本至关重要时，应优先选择显式路由（如 LangGraph 的边、AutoGen 的 `speaker_selection_method`），而非由大语言模型动态选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器（Retrievers）、大语言模型。提供一等公民级的模型上下文协议（Model Context Protocol, MCP）适配器（工具可作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具均可适配接入。通过 `allow_delegation=True` 实现团队间的任务委派。
- **AutoGen** → `FunctionTool` 可封装任意 Python 可调用对象；提供 MCP 适配器。在智能体间交互模式上与 AG2 生态紧密耦合。
- **Agno** → 使用 `@tool` 装饰器或继承 BaseTool 子类；提供 MCP 适配器；工具可在不同智能体和团队间共享。

## 核心技能

> 你能用一句话解释清楚，为什么某个特定框架适合解决某个特定的智能体（Agent）问题。

构建前检查清单：

1. **绘制架构图（Draw the shape）。** 这是一个图结构（Graph，含类型化状态与命名状态转换）？角色扮演模式（专家交接任务）？对话模式（智能体持续交互直至完成）？还是带工具调用的单智能体？
2. **确定分支决策者。** 开发者决定分支 → LangGraph。管理型智能体决定 → CrewAI 层级模式。对话中自然涌现 → AutoGen。工具调用驱动 → Agno。
3. **评估状态管理需求。** 是否需要检查点恢复（Resume-from-Checkpoint）？时间回溯（Time-Travel）？运行中人工中断（Human Interrupts）？如果是，LangGraph 是默认选择；Agno 的会话（Sessions）可覆盖对话级状态（Conversation-Scoped State）。
4. **评估成本预算。** 由大语言模型（LLM）动态选择路由会在每轮交互中消耗额外 Token。如果智能体每天运行数千次，优先使用显式路由（Explicit Routing）。
5. **评估框架开销（Framework Overhead）。** 每个框架都会引入额外的依赖。如果任务只是两次 LLM 调用加一个工具，直接写 30 行原生 Python 代码即可；不使用框架的开销永远是最小的。

在你能清晰画出图结构、组织架构图、对话流或智能体模块之前，不要急于引入框架。不要选择一个迫使你为了实际需求而去与其状态模型“死磕”的框架。

## 决策矩阵

| 问题形态 | 推荐框架 | 原因 |
|---------------|---------------------|-----|
| 带类型化状态、人工审批、长时运行的工作流有向无环图（DAG） | LangGraph | 提供一等公民级（First-Class）状态管理、检查点机制、中断功能与时间回溯。 |
| 角色分工明确的研究/写作流水线 | CrewAI（顺序模式）或 LangGraph 子图 | CrewAI 能低成本表达“一任务一角色”；当分支逻辑变复杂时，可借助 LangGraph 进行扩展。 |
| 提议者-批评者或师生对话模式 | AutoGen | 双智能体对话是其原生架构。 |
| 带工具、会话和记忆的单智能体 | Agno | 配置最轻量，内置存储与记忆功能。 |
| 包含数千个并行扇出（Fanout）与归约器（Reducer）的任务 | LangGraph + `Send` | 唯一提供一等公民级并行分发原语（Parallel Dispatch Primitive）的框架。 |
| 快速原型验证，无需绑定框架 | 原生 Python + 提供商 SDK | 不使用框架就是最快的“框架”。 |

## 练习

1. **简单。** 针对同一任务——“调研 Anthropic 总部信息，撰写 200 字简报并引用来源”——分别在 LangGraph（四个节点 (node)：规划、搜索、撰写、引用）和 CrewAI（三个角色 (role)：研究员、撰稿人、编辑）中实现。报告每次运行的 Token 成本 (token cost) 与代码行数。
2. **中等。** 在 AutoGen（研究员 ↔ 撰稿人对话，编辑通过 `GroupChat` 加入）和 Agno（单个智能体 (agent) 配备 `search_tools` 和 `write_tools`，外加会话存储 (session store)）中实现相同任务。从以下三个维度对四种实现方案进行排名：(a) 单次运行成本，(b) 崩溃后恢复 (resume after crash) 能力，(c) 在撰写步骤前注入人工审批 (human approval) 的能力。
3. **困难。** 编写一个决策树 (decision tree) 脚本 `pick_framework.py`，接收简短的问题描述（JSON 格式：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），并返回带有一句话理由的推荐结果。在你自行设计的六个测试用例上验证该脚本。

## 关键术语

| 术语 | 业界常说 | 实际含义 |
|------|-----------------|-----------------------|
| 编排 (Orchestration) | “智能体如何协同” | 决定下一个运行哪个节点/角色/智能体的逻辑层。 |
| 持久化状态 (Durable state) | “重启后恢复” | 能够抵御进程终止的状态，通常绑定至检查点 (checkpoint) 或会话存储。 |
| 大模型选择的路由 (LLM-selected routing) | “让模型决定” | 由规划大模型 (planner LLM) 在每轮中选择下一步；灵活性高，但每次决策都会消耗 Token。 |
| 显式路由 (Explicit routing) | “开发者决定” | 由 Python 函数或静态边 (static edge) 选择下一步；成本低且可审计。 |
| 团队 (Crew) | “CrewAI 团队” | 将角色、任务与流程（顺序或层级）绑定为一个可独立运行的单元。 |
| 群组聊天 (GroupChat) | “AutoGen 的多智能体聊天” | 由发言选择器 (speaker selector) 管理的 N 个智能体之间的对话。 |
| 团队 (Team, Agno) | “多智能体 Agno” | 在一组智能体之上运行的路由/协调/协作模式。 |
| 状态图 (StateGraph) | “LangGraph 的图” | 包含类型化状态 (typed-state)、节点、条件边 (conditional-edge) 和检查点器 (checkpointer) 的基础原语。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 状态图（StateGraph）、检查点（checkpointers）、中断（interrupts）、时间回溯（time-travel）。
- [CrewAI 文档](https://docs.crewai.com/) — 团队（Crews）、工作流（Flows）、智能体（Agents）、任务（Tasks）、流程（Processes）。
- [AutoGen 文档](https://microsoft.github.io/autogen/) — 可对话智能体（ConversableAgent）、群聊（GroupChat）、团队（teams）、工具（tools）。
- [Agno 文档](https://docs.agno.com/) — 智能体（Agent）、团队（Team）、工作流（Workflow）、存储（storage）、记忆（memory）。
- [Anthropic — 构建高效智能体（2024年12月）](https://www.anthropic.com/research/building-effective-agents) — 模式库（pattern library）（涵盖提示词链（prompt chaining）、路由（routing）、并行化（parallelization）、编排器-工作者（orchestrator-workers）、评估器-优化器（evaluator-optimizer）），与框架无关（framework-agnostic）。
- [Yao 等人，《ReAct：协同推理与行动》（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 所有框架都在此基础上进行封装的底层原语（primitive）。
- [Wu 等人，《AutoGen：通过多智能体对话赋能下一代大语言模型应用》（2023）](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文。
- [Park 等人，《生成式智能体：人类行为的交互式模拟》（UIST 2023）](https://arxiv.org/abs/2304.03442) — CrewAI 风格的角色设定栈（persona stacks）所依托的角色扮演（role-play）基础。
- 第 11 阶段 · 16（LangGraph） — 本课程用作基准对比的框架。
- 第 11 阶段 · 19（Reflexion） — 一种能无缝映射至 LangGraph，但在 CrewAI 中适配较为生硬的模式。
- 第 11 阶段 · 22（生产环境可观测性（Production observability）） — 如何为你所选的任意框架进行插桩监控（instrument）。
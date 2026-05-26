# 编排模式（Orchestration Patterns）：主管（Supervisor）、群体（Swarm）与分层（Hierarchical）

> 在 2026 年的各类框架中，反复出现四种编排模式：主管-工作节点（supervisor-worker）、群体/对等网络（swarm / peer-to-peer）、分层（hierarchical）以及辩论（debate）。Anthropic 的指导原则是：“关键在于构建适合你需求的系统。”从简单开始；仅当单个智能体（agent）结合五种工作流模式（workflow patterns）仍无法满足需求时，再引入拓扑结构（topology）。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置条件：** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 25（多智能体辩论）
**耗时：** 约 60 分钟

## 学习目标

- 说出四种反复出现的编排模式及其适用场景。
- 描述 LangChain 在 2026 年的建议：基于工具调用的监督（tool-call-based supervision）与主管库（supervisor libraries）的对比。
- 解释 Anthropic 的“构建合适系统”原则，以及它如何作为选择拓扑结构的门槛。
- 使用标准库（stdlib）针对同一个脚本化大语言模型（LLM）实现全部四种模式。

## 问题背景

团队往往在尚未真正需要时就急于采用“多智能体（multi-agent）”架构。这四种模式在各类框架中反复出现；一旦你能准确命名它们，就能选出最合适的一种——或者干脆完全跳过拓扑结构。

## 核心概念

### 主管-工作者模式（Supervisor-Worker）
- 由一个中心路由大语言模型（LLM）将任务分发给专家智能体（Specialist Agents）。
- 决策逻辑包括：循环回自身、移交（Handoff）给专家智能体、或终止流程。
- 专家智能体之间不直接通信；所有路由请求均必须经过主管。

框架：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 LangChain 建议：** 通过直接工具调用（Tool Calls）而非 `create_supervisor` 来实现监督控制。这能提供更精细的上下文工程（Context Engineering）粒度——由你精确决定每个专家智能体所能获取的上下文信息。

### 蜂群/对等模式（Swarm / Peer-to-Peer）
- 智能体通过共享的工具层（Tool Surface）直接进行任务移交。
- 无中心路由节点。
- 延迟低于主管模式（网络跳数更少）。
- 系统行为更难追踪与推理（缺乏单一控制点）。

框架：LangGraph swarm topology、OpenAI Agents SDK handoffs（适用于所有智能体均可相互移交的场景）。

### 层级模式（Hierarchical）
- 主管管理子主管，子主管再进一步管理底层工作者。
- 在 LangGraph 中通过嵌套子图实现；在 CrewAI 中通过嵌套团队实现。
- 能够扩展至大规模智能体集群，但代价是系统运维复杂度显著上升。

适用场景：当单个主管的上下文预算（Context Budget）无法容纳所有专家智能体的描述信息时。

### 辩论模式（Debate）
- 并行提议者（Parallel Proposers）+ 迭代交叉评审（Iterative Cross-Critique）（参见第25课）。
- 严格来说不属于智能体编排（Orchestration）范畴，更偏向于结果验证（Verification），但在各类框架中常被作为一种拓扑结构（Topology）选项提供。

### CrewAI 的 Crew 与 Flow 模式对比
CrewAI 明确规范了两种部署模式：
- **Flow**：适用于确定性事件驱动自动化（推荐作为生产环境的起步方案）。
- **Crew**：适用于基于角色的自主协作。

该分类维度与上述四种模式相互独立（正交），但可与拓扑结构对应：Flow 通常对应主管模式或层级模式；Crew 通常对应配备 LLM 路由器的主管模式。

### Anthropic 的指导原则
“在大语言模型（LLM）领域取得成功，并非在于构建最复杂的系统，而在于构建最契合你实际需求的系统。”

架构选型决策顺序：
1. 单智能体（Single Agent）+ 工作流模式（Workflow Patterns）（参见第12课）——建议从此入手。
2. 主管-工作者模式（Supervisor-Worker）——适用于拥有 2-4 个专家智能体的场景。
3. 蜂群模式（Swarm）——当系统延迟的优先级高于推理过程的可解释性时。
4. 层级模式（Hierarchical）——仅在单个主管的上下文预算（Context Budget）无法满足需求时采用。
5. 辩论模式（Debate）——当输出准确性的优先级高于计算成本时。

### 模式应用中的常见陷阱
- **拓扑优先思维（Topology-First Thinking）。** 尚未明确多智能体（Multi-Agent）架构究竟要解决什么业务问题，就盲目宣称“我们需要多智能体”。
- **蜂群模式中的任务乒乓移交（Bouncing Handoffs）。** 出现 A -> B -> A -> B 的无限循环。应引入跳数计数器（Hop Counters）进行限制。
- **虚假层级（Fake Hierarchy）。** 为了迎合“企业级”概念硬凑三层架构，实际仅有两个团队在运作。应果断扁平化/合并。

## 动手实践 (Build It)

`code/main.py` 使用标准库（stdlib）针对一个脚本化的大语言模型（LLM）实现了全部四种模式：

- `Supervisor`（监督者模式）—— 中央路由（router）。
- `Swarm`（群体模式）—— 点对点直接交接。
- `Hierarchical`（层级模式）—— 监督者的监督者。
- `Debate`（辩论模式）—— 并行提议者 + 批判。

每种模式都处理相同的三意图任务（退款 / 缺陷 / 销售）。执行轨迹（trace）的形态各不相同。

运行方式：

python3 code/main.py

输出：各模式的执行轨迹（trace）与操作计数（op count）。监督者模式最清晰；群体模式路径最短；层级模式嵌套最深；辩论模式开销最大。

## 实际应用

- 使用 **LangGraph** 实现监督者与层级模式（嵌套子图）。
- 使用 **OpenAI Agents SDK** 实现“工具化交接”（监督者架构）。
- 使用 **CrewAI Flow** 实现生产级确定性工作流。
- 采用**自定义**方案实现辩论模式，或当你需要精确控制时。

## 交付上线

`outputs/skill-orchestration-picker.md` 负责选择拓扑结构（topology）并完成实现。

## 练习

1. 通过移除路由器（router），将监督者-工作者（supervisor-worker）架构转换为群体模式。哪些功能会失效？哪些会得到改善？
2. 为群体模式添加跳数计数器（hop counter）：在 3 次交接后拒绝请求。这能否捕获 A->B->A 的循环反弹问题？
3. 为包含 12 个专家的领域构建两级层级系统。如果不采用嵌套结构，上下文预算（context budget）会在何处耗尽？
4. 在贴近生产环境的负载下对四种模式进行性能剖析（profile）。在各项指标（延迟、成本、准确率、可调试性）上，哪种模式胜出？
5. 阅读 Anthropic 的《构建高效智能体》（Building Effective Agents）文章。将你现有的生产工作流分别映射到这四种模式之一。是否存在无法清晰映射的情况？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Supervisor-worker（监督者-工作者） | “路由器 + 专家” | 中央大语言模型（LLM）将任务分发给专家；专家之间不直接通信 |
| Swarm（群体模式） | “点对点” | 通过共享工具直接交接；无中央路由器 |
| Hierarchical（层级模式） | “监督者的监督者” | 针对大规模节点采用嵌套子图结构 |
| Debate（辩论模式） | “提议者 + 批判者” | 并行提议，交叉批判（第 25 课） |
| Tool-call-based supervision（基于工具调用的监督） | “不依赖库的监督者” | 将监督者实现为直接的工具调用，以控制上下文 |
| Crew（团队） | “自主团队” | CrewAI 基于角色的协作模式 |
| Flow（工作流） | “确定性工作流” | CrewAI 事件驱动的生产模式 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 五种模式 + 智能体与工作流对比
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 监督者、群体与层级模式
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew 与 Flow 对比
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — 辩论模式
# 交接（Handoffs）与例程（Routines）——无状态编排（Stateless Orchestration）

> OpenAI 的 Swarm（2024 年 10 月）将多智能体编排（Multi-Agent Orchestration）提炼为两个基本原语（Primitives）：**例程（Routines）**（将指令与工具整合为系统提示词（System Prompt））和**交接（Handoffs）**（一个返回另一个智能体（Agent）的工具）。无需状态机（State Machine），也无需分支领域特定语言（Domain-Specific Language, DSL）——大语言模型（Large Language Model, LLM）通过调用正确的交接工具来实现路由（Routing）。OpenAI Agents SDK（2025 年 3 月）是其面向生产环境（Production）的继任者。Swarm 本身仍是最清晰的概念参考——其全部源代码仅几百行。该模式之所以迅速流行，是因为其 API 接口（API Surface）大致可概括为“智能体（Agent）= 提示词（Prompt）+ 工具（Tools）；交接（Handoff）= 返回智能体的函数”。局限性在于：它是无状态的（Stateless），因此记忆（Memory）管理由调用方负责。

**类型：** 学习与构建
**语言：** Python（标准库）
**前置要求：** 第 16 阶段 · 04（基础模型）
**耗时：** 约 60 分钟

## 问题

每个多智能体框架都要求你学习其领域特定语言（DSL）：例如 LangGraph 的节点（Nodes）与边（Edges）、CrewAI 的群组（Crews）与任务（Tasks）、AutoGen 的群聊（GroupChat）与管理器（Managers）。这些 DSL 确实是有效的抽象（Abstractions），但它们往往让系统显得比实际需要的更臃肿。

Swarm 则反其道而行之：直接利用模型已有的工具调用（Tool-Calling）能力。交接操作转化为工具调用。编排器（Orchestrator）即为当前持有对话的智能体。状态机则隐式地存在于各智能体的系统提示词中。

## 概念

### 两个原语 (Primitives)

**例程 (Routine)**。用于定义智能体 (Agent) 角色及其可用工具的系统提示词 (System Prompt)。你可以将其理解为一组作用域明确的指令：“你是一个分诊智能体；如果用户咨询退款事宜，请将任务交接给退款智能体。”

**交接 (Handoff)**。智能体可以调用的一种工具，调用后会返回一个新的智能体对象。Swarm 运行时 (Runtime) 会检测到该智能体返回值，并在下一轮对话中切换当前活跃的智能体。

这就是该框架的全部抽象概念。

def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)

分诊智能体的系统提示词会引导其根据用户消息选择正确的交接工具。大语言模型 (LLM) 的工具调用 (Tool Calling) 机制负责完成路由分发。

### 为何它能迅速走红

- **精简的 API**。仅需掌握两个核心概念。
- **复用模型现有能力**。工具调用在各家服务商中均已达到生产级可用标准。
- **无需状态机 (State Machine) 负担**。你无需显式描述状态图，智能体的提示词本身已说明了交接对象。

### 无状态 (Stateless) 设计的取舍

Swarm 在多次运行之间是明确无状态的。框架仅在单次运行期间保留消息历史，不会持久化任何数据。记忆管理、上下文连续性、长周期任务——这些全部由调用方自行解决。

在生产环境版本（OpenAI Agents SDK，2025 年 3 月）中，这是主要改进点之一：该 SDK 在保留交接原语的同时，增加了内置的会话管理 (Session Management)、护栏机制 (Guardrails) 和追踪 (Tracing) 功能。

### Swarm/交接机制的适用场景

- **分诊模式**。一线智能体将用户路由至专业智能体。
- **基于技能的交接**。“如果任务需要编写代码，则调用编程智能体；如果需要调研，则调用研究智能体。”
- **简短、有边界的对话**。客户支持、FAQ 转工单、简单工作流等。

### Swarm 的局限性

- **需要共享记忆的长会话**。交接操作会将对话状态重置为新智能体的提示词加上历史记录。若无调用方自行管理的记忆机制，智能体之间无法保持持久状态。
- **并行执行**。交接是串行进行的——每次只能切换一个活跃智能体。若需并行处理，必须由调用方协调多个 Swarm 运行实例。
- **审计与回放**。无状态运行难以精确重现；大语言模型的交接选择具有非确定性。

### OpenAI Agents SDK（2025 年 3 月）

该生产级后继版本新增了以下功能：

- **会话状态**。跨运行周期的持久化线程。
- **护栏机制**。输入/输出验证钩子。
- **追踪功能**。记录每一次工具调用和交接操作。
- **交接过滤器**。控制交接时传递的上下文内容。

交接原语得以保留，并在其外围补充了面向生产环境的易用性设计。

### Swarm 与 GroupChat 的对比

两者均采用大语言模型驱动的路由机制，但核心差异在于**由谁决定下一个发言者**：

- GroupChat：由外部选择器（函数或大语言模型）指定下一个发言者。
- Swarm：当前智能体通过调用交接工具自行选定继任者。

Swarm 是“智能体自主决定下一步”；GroupChat 是“管理器决定下一步”。Swarm 的决策逻辑内嵌于活跃智能体的工具调用中；而 GroupChat 的决策逻辑则集中在 `GroupChatManager` 中。

## 构建实现

`code/main.py` 从零实现了 Swarm（多智能体协作框架）：包含一个 Agent（智能体）数据类、一套交接机制（handoff mechanism，即工具返回新的 Agent 对象），以及一个用于检测智能体切换的运行循环（run loop）。

演示示例：一个分诊智能体（triage agent）将请求路由至退款、销售或支持专家。每位专家拥有各自的工具。运行循环会打印每次交接记录。

运行：

python3 code/main.py

## 使用它

`outputs/skill-handoff-designer.md` 为给定任务设计交接拓扑结构：定义存在哪些智能体、它们可以调用哪些交接操作，以及传递哪些上下文（context）。

## 部署上线

检查清单：

- **交接日志记录（Handoff logging）。** 每次交接都会写入一条追踪事件（trace event），包含来源智能体、目标智能体及上下文快照。
- **上下文传递规则（Context transfer rules）。** 决定交接时传递什么内容：完整历史记录（开销较大）、最近 N 条消息，或摘要。
- **交接护栏（Guardrail on handoff）。** 向具有不同工具权限的专家进行交接时必须进行身份验证——否则提示词注入（prompt injection）可能强制触发非预期的交接。
- **循环检测（Loop detection）。** 两个智能体来回交接是常见的故障模式；可通过简单的最近 K 次环形检查进行识别。
- **回退智能体（Fallback agent）。** 如果交接目标不存在，则回退至安全的默认智能体。

## 练习

1. 运行 `code/main.py`，将请求分诊至退款智能体。确认第二轮对话的活跃智能体（active agent）为退款智能体。
2. 添加循环检测规则：如果相同的两个智能体连续交接 3 次，则强制退出。设计相应的回退机制。
3. 阅读 OpenAI Agents SDK 关于交接过滤器（handoff filters）的文档。实现一个“交接时摘要”版本：在接收智能体接管前，由交出智能体将上下文压缩为要点摘要。
4. 将 Swarm 的交接机制与 GroupChatManager（群组聊天管理器）选择器进行对比。哪种模式更容易加剧提示词注入问题？为什么？
5. 阅读 Swarm 指南（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出 Swarm 做出的一项明确设计决策，并说明 OpenAI Agents SDK 是对其进行了修改还是保留。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 常规配置（Routine） | “智能体提示词” | 系统提示词（system prompt）+ 工具列表。定义角色及可用的交接操作。 |
| 交接（Handoff） | “转移给另一个智能体” | 活跃智能体可调用的一个工具，该工具返回一个新的 Agent。运行时会切换活跃智能体。 |
| 无状态（Stateless） | “运行间无记忆” | Swarm 不持久化任何数据；记忆管理由调用方负责。 |
| 活跃智能体（Active agent） | “当前谁在发言” | 当前持有对话的智能体。交接操作会改变此状态。 |
| 上下文传递（Context transfer） | “交接时传递什么” | 决定接收智能体能看到哪些历史记录的策略：完整历史、最近 N 条或摘要。 |
| 交接循环（Handoff loop） | “智能体来回踢皮球” | 故障模式：两个智能体不断互相交接。 |
| OpenAI Agents SDK | “生产级 Swarm” | 2025 年 3 月发布的继任版本；在交接原语（handoff primitive）之上增加了会话、护栏和追踪功能。 |
| 交接过滤器（Handoff filter） | “交接关卡” | SDK 提供的功能，用于在交接边界检查和修改上下文。 |

## 延伸阅读

- [OpenAI Cookbook — 智能体（Agent）编排：固定流程与交接（Handoff）](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 权威参考说明
- [OpenAI Swarm 仓库](https://github.com/openai/swarm) — 原始实现，保留作为概念性参考
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 支持会话（Session）与追踪（Tracing）的生产级后继版本
- [Anthropic 关于 Claude 中交接机制的说明](https://docs.anthropic.com/en/docs/claude-code) — Claude Code 子智能体（Subagent）如何通过 `Task` 实现类似交接的模式
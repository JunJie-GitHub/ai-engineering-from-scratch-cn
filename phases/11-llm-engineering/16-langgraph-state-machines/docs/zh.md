# LangGraph — 智能体的状态机 (State Machines for Agents)

> 手写的 ReAct 循环 (ReAct Loop) 本质上就是一个 `while True`。而在 LangGraph 中编写的 ReAct 循环则是一个图结构，你可以对其进行检查点保存、中断、分支和状态回溯。智能体本身并未改变，改变的是包裹它的运行框架 (Harness)。

**类型：** 构建
**语言：** Python
**前置条件：** 第 11 阶段 · 09（函数调用 (Function Calling)），第 11 阶段 · 14（模型上下文协议 (Model Context Protocol)）
**预计耗时：** 约 75 分钟

## 问题所在

你部署了一个函数调用智能体 (Function Calling Agent)。它在前三轮对话中运行良好，但随后出现了问题：模型尝试调用了一个返回 500 错误的工具，用户在任务中途改变了主意，或者智能体在未经人工确认的情况下擅自决定退款。`while True:` 循环没有任何钩子 (Hooks)。你无法暂停它，无法回滚它，也无法分支去探索“如果模型当初选择了另一个工具会怎样”。一旦你将它从演示阶段推向实际交付，这个智能体就会变成一个黑盒，结果非成即败。

一旦看清这一点，下一步就显而易见了。智能体本身已经是一个状态机 (State Machine)——由系统提示词 (System Prompt)、消息历史 (Message History)、待处理工具调用 (Pending Tool Calls) 以及下一步动作共同构成。我们需要将这个状态机显式化：用节点 (Nodes) 表示“模型思考”、“工具执行”、“人工审批”，用边 (Edges) 表示它们之间的条件转移。一旦图结构变得显式，运行框架就能免费获得四项核心能力：检查点机制 (Checkpointing，在步骤间保存状态)、中断机制 (Interrupts，暂停以等待人工介入)、流式传输 (Streaming，流式输出 Token 和中间事件) 以及状态回溯 (Time-travel，回滚到先前的状态并尝试不同的分支)。

LangGraph 正是提供这一抽象的库。它并非 LangChain 传统意义上的智能体框架（“给你一个 AgentExecutor，祝你好运”）。它是一个具备一等公民级状态 (First-class State)、一等公民级持久化 (First-class Persistence) 和一等公民级中断 (First-class Interrupts) 的图运行时 (Graph Runtime)。智能体循环是你“画”出来的，而不是你“手写”出来的。

## 核心概念

![LangGraph StateGraph：节点、边和检查点记录器](../assets/langgraph-stategraph.svg)

状态图（StateGraph）包含三个核心要素。

1. **状态（State）**。一个类型化字典（TypedDict 或 Pydantic 模型），在图中流转。每个节点接收完整状态并返回部分更新，LangGraph 会使用每个字段对应的归约器（reducer）进行合并——对于需要累积的列表使用 `operator.add`，默认情况下则为覆盖。
2. **节点（Nodes）**。Python 函数 `state -> partial_state`。每个节点代表一个离散步骤：“调用模型”、“运行工具”或“生成摘要”。
3. **边（Edges）**。节点之间的转换路径。静态边指向固定位置。条件边接收一个路由函数 `state -> next_node_name`，使图能够根据模型输出进行分支。

你需要对图进行编译。编译过程会绑定拓扑结构、附加检查点记录器（checkpointer）（可选，但在生产环境中至关重要），并返回一个可运行对象（runnable）。你通过传入初始状态和 `thread_id` 来调用它。执行的每一步都会持久化一个检查点，其键为 `(thread_id, checkpoint_id)`。

### 四大核心能力

**检查点记录（Checkpointing）**。每次节点转换都会将新状态写入存储（测试时使用内存，生产环境使用 Postgres/Redis/SQLite）。通过再次使用相同的 `thread_id` 调用图即可恢复执行。图会从暂停的位置继续运行。

**中断（Interrupts）**。使用 `interrupt_before=["human_review"]` 标记某个节点，执行会在该节点运行前暂停。状态会被持久化保存。你的 API 可以向用户返回“等待审批”。后续向相同的 `thread_id` 发送包含 `Command(resume=...)` 的请求即可恢复执行。

**流式传输（Streaming）**。`graph.stream(state, mode="updates")` 会实时产出状态增量。`mode="messages"` 会在模型节点内流式传输大语言模型（LLM）的 Token。`mode="values"` 则产出完整快照。你可以根据需要选择在 UI 中展示的内容。

**时间回溯（Time-travel）**。`graph.get_state_history(thread_id)` 返回完整的检查点日志。将任意历史 `checkpoint_id` 传入 `graph.invoke`，即可从该节点分叉执行。这非常适合用于调试（“如果模型当时选择了工具 B 会怎样？”）以及重放生产环境轨迹的回归测试。

### 归约器（Reducer）是关键

每个状态字段都对应一个归约器。大多数默认设置已经足够——新值会覆盖旧值。但消息列表需要使用 `operator.add`，以便新消息追加而非替换。并行边会通过归约器合并它们的更新。如果两个节点同时更新 `messages`，而你忘记使用 `Annotated[list, add_messages]`，第二个节点的更新会静默覆盖前者，导致丢失一半的对话轮次。归约器是该库中唯一需要细致处理的部分；只要配置正确，其余部分就能自然组合。

### 四节点实现 ReAct 图

一个生产级的 ReAct 智能体（ReAct agent）由四个节点和两条边构成：

1. `agent` —— 使用当前消息历史调用大语言模型。返回助手消息（可能包含 `tool_calls`）。
2. `tools` —— 执行最后一条助手消息中的所有 `tool_calls`，并将工具结果作为工具消息追加。
3. 从 `agent` 出发的条件边：如果最后一条消息包含 `tool_calls`，则路由至 `tools`，否则路由至 `END`。
4. 从 `tools` 返回 `agent` 的静态边。

仅此而已。你只需大约 40 行代码，即可获得具备检查点记录、中断和流式传输功能的完整 ReAct 循环（思考 → 行动 → 观察 → 思考 → …）。

### StateGraph 与 Send（扇出）

`Send(node_name, state)` 允许节点分发并行子图。例如：智能体决定同时查询三个检索器。每个 `Send` 都会生成目标节点的并行执行实例；它们的输出会通过状态归约器进行合并。这就是 LangGraph 在不依赖底层线程原语的情况下，实现编排器-工作者（orchestrator-workers）模式的方式。

### 子图（Subgraphs）

已编译的图可以作为另一个图中的节点。外部图将其视为单个节点；内部图则拥有独立的状态和检查点。团队正是通过这种方式构建监督者-工作者（supervisor-worker）智能体：监督者图将用户意图路由至按领域划分的工作者子图。

## 构建

### 步骤 1：状态（State）与节点（Node）

from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())

`add_messages` 是一个归约函数（Reducer），它使消息列表能够累积而非覆盖。忘记配置它是 LangGraph 中最常见的错误。

### 步骤 2：使用线程（Thread）运行

config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以将这些数据流式传输（Stream）到用户界面，让用户实时看到“智能体（Agent）正在思考… 正在调用 search_web… 已获取结果… 正在回答。”

### 步骤 3：添加人机协同中断（Human-in-the-loop Interrupt）

标记一个节点，使其在运行前暂停执行。

app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})

状态、检查点（Checkpoint）和线程在中断期间都会持久化保存。除了执行期间，没有任何数据会驻留在内存中。

### 步骤 4：用于调试的时间回溯（Time-travel）

history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward

将 `None` 作为输入传入，会从指定的检查点开始重放；传入一个具体值，则会在恢复执行前将其作为更新追加到该检查点的状态中。这样你就可以在不重新运行整个对话的情况下，复现智能体运行出错的过程。

### 步骤 5：为生产环境替换检查点保存器（Checkpointer）

from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)

框架已内置 SQLite、Redis 和 Postgres 支持。`MemorySaver` 仅适用于测试环境。任何需要在重启后持久化的数据，都应使用真正的存储后端。

## 核心技能

> 构建智能体（Agent）时应采用图（Graph）结构，而非 `while True` 循环。

在引入 LangGraph 之前，请先进行 60 秒的快速设计：

1. **命名节点（Node）。** 每一个独立的决策或产生副作用（Side-effect）的操作都应作为一个节点。例如“智能体思考”、“工具执行”、“审核员批准”、“响应流式输出”。如果无法列出这些节点，说明该任务尚未具备智能体的形态。
2. **声明状态（State）。** 使用最简化的 `TypedDict`，并为每个列表字段配置一个归约器（Reducer）。不要将所有内容都塞进 `messages` 中；应将任务特定的字段（如工作 `plan`、`budget` 计数器、`retrieved_docs` 列表）提升至顶层。
3. **绘制边（Edge）。** 除非下一步依赖于模型输出，否则边应为静态的。每条条件边都需要一个带有命名分支的路由函数（Router Function）。
4. **提前选择检查点保存器（Checkpointer）。** 测试时使用 `MemorySaver`，其他场景使用 Postgres/Redis/SQLite。切勿在未配置检查点保存器的情况下发布——没有它，就无法实现恢复（Resume）、中断（Interrupt）或时间回溯（Time-travel）。
5. **在工具运行前决定中断点，而非之后。** 审批逻辑应设置在进入产生副作用节点的边上，以便在造成损害前取消；验证逻辑应设置在模型输出后的边上，以便以较低成本拒绝错误的调用。
6. **默认启用流式输出（Streaming）。** UI 使用 `mode="updates"`，模型节点内的词元级（Token-level）流式输出使用 `mode="messages"`，评估时的完整快照使用 `mode="values"`。

拒绝发布未配置检查点保存器的 LangGraph 智能体。拒绝发布在副作用发生*之后*才触发中断的智能体。拒绝发布 `messages` 字段未将 `add_messages` 设为归约器的智能体。

## 练习

1. **简单。** 使用计算器工具和网络搜索工具实现上述四节点 ReAct 图。验证 `list(app.get_state_history(config))` 在两轮对话中至少返回四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，并将结构化的 `plan: list[str]` 写入状态。让 `agent` 将计划步骤标记为已完成。如果在检查点恢复后 `plan` 丢失（归约器配置错误），则测试失败。
3. **困难。** 构建一个主管图（Supervisor Graph），使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间进行路由。每个子图拥有独立的状态和检查点保存器。在外层图上添加 `interrupt_before=["writer"]`，以便人工审批研究简报。确认从先前的检查点进行时间回溯时，仅重新运行分叉的分支。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 状态图 (StateGraph) | “LangGraph 图” | 在编译前用于添加节点和边的构建器对象。 |
| 归约器 (Reducer) | “字段如何合并” | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认为覆盖，`add_messages` 为追加。 |
| 会话 (Thread) | “对话 ID” | 一个 `thread_id` 字符串，用于限定单个会话的所有检查点范围。 |
| 检查点 (Checkpoint) | “暂停的状态” | 节点转换后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| 中断 (Interrupt) | “暂停以等待人工介入” | `interrupt_before` / `interrupt_after` 在节点边界停止执行；使用 `Command(resume=...)` 恢复。 |
| 时间回溯 (Time-travel) | “从先前步骤分叉” | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点开始向前重放。 |
| 发送 (Send) | “并行子图调度” | 节点可返回的构造函数，用于生成目标节点的 N 个并行执行实例。 |
| 子图 (Subgraph) | “作为节点的已编译图” | 在另一个图中作为节点使用的已编译 StateGraph；保留其自身的状态作用域。 |

## 进一步阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 关于状态图 (StateGraph)、归约器 (reducers)、检查点存储器 (checkpointers) 和中断 (interrupts) 的权威参考。
- [LangGraph 核心概念：状态、归约器、检查点存储器](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课程所采用的心智模型，直接源自官方文档。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) — 详细介绍 Postgres/SQLite/Redis 存储、检查点命名空间及线程 ID。
- [LangGraph 人机协同 (Human-in-the-loop)](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — 涵盖 `interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及状态编辑模式。
- [Yao 等人，《ReAct：在语言模型中协同推理与行动》（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 每个 LangGraph 智能体 (agent) 均实现的范式；阅读此文以了解推理轨迹的设计原理。
- [Anthropic — 构建高效智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 探讨应优先选择何种图结构（链式、路由、编排器-工作节点、评估器-优化器）及其适用场景。
- 第 11 阶段 · 09（函数调用 (Function Calling)） — 每个 LangGraph 智能体节点都会复用的工具调用原语。
- 第 11 阶段 · 14（模型上下文协议 (Model Context Protocol)） — 通过 MCP 适配器接入 LangGraph `ToolNode` 的外部工具发现机制。
- 第 11 阶段 · 17（智能体框架权衡 (Agent framework tradeoffs)） — 何时应优先选择 LangGraph 而非 CrewAI、AutoGen 或 Agno。
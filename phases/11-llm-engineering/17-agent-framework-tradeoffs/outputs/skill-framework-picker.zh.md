---
name: framework-picker
description: 根据抽象层级与问题形态的匹配度，为智能体任务选择 LangGraph、CrewAI、AutoGen、Agno 或纯 Python。
version: 1.0.0
phase: 11
lesson: 17
tags: [langgraph, crewai, autogen, agno, 智能体框架, 任务编排, 决策矩阵]
---

根据任务描述（问题形态、每次运行的大语言模型（LLM）调用总数、分支模式、持久化与恢复需求、人在回路（human-in-the-loop）检查点、并行扇出（parallel fanout）、会话记忆、预期每日运行量），输出以下内容：

1. 形态匹配。用一句话指出匹配的抽象模式：图（graph，带类型状态、命名转换）、组织架构图（org chart，专家角色、管理器路由交接）、聊天（chat，智能体持续对话直至完成）、带工具的单一智能体。若无法选定其一，说明该任务尚未具备智能体形态；请停止并对其进行拆解。
2. 分支决策权。由谁决定下一步：开发者（显式边）、管理器 LLM（CrewAI 层级模式）、对话涌现式（AutoGen GroupChat）、工具调用自路由（Agno）。如适用，请注明 LLM 选择路由时每轮（per-turn）的 Token 消耗。
3. 状态预算。确认是否需要重启后恢复（resume-after-restart）、时间回溯（time-travel）或人工中断。若需要，LangGraph 在状态优先（state-first）抽象方面胜出；Agno 仅覆盖会话级（session-scoped）记忆。
4. 框架选择。输出 `langgraph`、`crewai`、`autogen`、`agno` 或 `plain_python` 之一。附上一句话理由，将上述形态与状态的答案映射到该框架的核心原语（core primitive）上。
5. 降级方案（Escape hatch）。若每日运行量超过 10_000 次，或任务为无状态且 LLM 调用不超过两次，则建议改用纯 Python 配合提供商 SDK。对于小型任务，不使用框架反而是最快的“框架”。

对于具有已知有向无环图（DAG）的确定性工作流，拒绝推荐 AutoGen；其 `GroupChatManager` 会消耗 Token 来选择发言者，而这些连接本可由开发者静态配置。CrewAI 确实支持通过 `output_pydantic` / `output_json` 输出结构化任务结果（参见 [docs.crewai.com/en/concepts/tasks](https://docs.crewai.com/en/concepts/tasks)），但其 `context` 通道仍会流入下一个任务的提示词字符串中。当工作流依赖原始 `context` 在未配置上述输出模式的情况下跨任务传递结构化状态时，应反对使用 CrewAI。对于仅需两次调用的摘要任务，应反对使用 LangGraph；`StateGraph` 的开销纯属额外负担。当任务需扇出至 4 个以上具有归约（reducer）语义的并行子工作节点时，应反对使用 Agno；Agno 提供了一个 `Parallel` 模块，其输出会合并为以步骤名称为键的字典（参见 [docs-v1.agno.com/workflows_2/overview](https://docs-v1.agno.com/workflows_2/overview) 和 [docs.agno.com/workflows/access-previous-steps](https://docs.agno.com/workflows/access-previous-steps)），但它并未暴露类似 LangGraph 的 `Send` 风格扇出与归约原语。

示例输入：“长期运行的研究工作流：规划、扇出至三个检索器、综合整理、人工审批简报、撰写报告、引用来源。崩溃后必须恢复。生产环境每日运行 50 次。”

示例输出：
- 形态：图（graph）。带类型的计划、三个并行检索器、综合与撰写步骤间的命名转换。
- 分支：开发者通过条件边决定。无需每轮管理器 LLM。
- 状态：需要恢复与人工中断。必须使用 LangGraph。
- 框架：langgraph。状态管理、`Send` 扇出、`interrupt_before` 以及 `PostgresSaver` 均为一等公民（first-class）。
- 降级方案：不适用。每日 50 次运行远低于纯 Python 阈值，且该工作流状态依赖过重，不适合脱离框架运行。
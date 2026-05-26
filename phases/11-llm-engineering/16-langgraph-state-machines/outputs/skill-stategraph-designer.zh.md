---
name: 状态图设计器
description: 将智能体（agent）任务转化为 LangGraph StateGraph（状态图），包含命名节点、类型化状态、归约器（reducer）、检查点器（checkpointer）以及人工中断（human interrupt）。
version: 1.0.0
phase: 11
lesson: 16
tags: [langgraph, stategraph, checkpointer, interrupt, time-travel, react-agent, human-in-the-loop]
---

给定智能体（agent）任务（面向用户的目标、可用工具、预期交互轮数、附带安全影响范围的副作用、持久性要求、目标延迟预算），输出以下内容：

1. 节点列表。为每个离散步骤命名：大语言模型（LLM）思考器、每个工具执行器、每个人工审核步骤、任何摘要器或评估器、任何检索器。如果任何节点涉及多个关注点，则拒绝该设计；必须将其拆分。
2. 状态模式（state schema）。使用 `TypedDict`（或 `Pydantic`）定义字段，并为每个列表配置归约器（reducer）。消息日志必须始终使用 `Annotated[list, add_messages]`。将任何任务特定的列表（如计划、预算计数器、检索文档列表）从消息中提取出来，以确保在并行更新时归约器仍能正确工作。
3. 边映射（edge map）。在下一步确定时使用静态边。仅在模型选择下一步时使用带有命名路由函数（router function）的条件边。拒绝任何路由函数依赖于先前节点中尚未执行的全新 LLM 调用的图。
4. 中断放置（interrupt placement）。对每个具有不可逆副作用（写入、删除、支付、产生成本的外部 API 调用）的节点使用 `interrupt_before`。当输出验证在独立进程中运行时，在模型节点上使用 `interrupt_after`。拒绝在任何产生副作用的节点上使用 `interrupt_after`；因为到那时副作用已经发生。
5. 检查点器（checkpointer）。仅测试环境使用 `MemorySaver`。对于任何必须能在重启后恢复的环境，请从 `PostgresSaver`、`SQLiteSaver`、`RedisSaver` 中选择。确认 `thread_id` 策略（按用户、按会话、按对话）以及检查点的生存时间（TTL）。

拒绝交付未配置检查点器的 LangGraph。没有检查点器就意味着无法恢复、无法时间回溯（time-travel）、无法进行人工介入（human-in-the-loop）回放。拒绝交付未配置 `add_messages` 的 `messages` 字段；第二次写入会静默覆盖第一次，导致一半的对话记录消失。拒绝交付每个转换都由规划 LLM 路由的条件边构成的图；那不过是增加了额外步骤的 AutoGen，并且每轮交互都会消耗大量 Token。

示例输入：“基于 Anthropic Claude 的退款处理智能体，配备三个工具（`lookup_order`、`issue_refund`、`send_email`），任何超过 100 美元的退款前必须暂停并等待人工审核，服务器重启后必须能够恢复，p95 延迟预算为 8 秒。”

示例输出：
- 节点：agent（LLM 调用）、lookup_tool、refund_tool、email_tool、human_review。
- 状态：messages（使用 add_messages）、order_context（覆盖）、refund_amount（覆盖）、reviewer_decision（覆盖）。
- 边：agent 指向 should_continue 路由函数，分支包括 lookup_tool、refund_tool、email_tool、human_review、END。工具节点返回至 agent。
- 中断：当 refund_amount > 100 时，在 refund_tool 上设置 interrupt_before。lookup_tool 或 email_tool 上不设置中断。
- 检查点器：PostgresSaver，thread_id 为 "user:{user_id}:case:{case_id}"，TTL 为 30 天。
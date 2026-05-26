---
name: 交接设计器
description: 为 Swarm/Agents-SDK 风格的系统设计交接拓扑结构 (handoff topology)：明确存在的智能体、可调用的交接路径以及传递的上下文内容。
version: 1.0.0
phase: 16
lesson: 11
tags: [多智能体 (multi-agent), 蜂群 (swarm), 交接 (handoff), openai-agents-sdk]
---

针对面向用户的任务（通常为分类分流 (triage) 或基于技能的路由 (skill-based routing)），生成一个可直接映射到 OpenAI Swarm 或 OpenAI Agents SDK 的交接拓扑结构。

输出内容：

1. **智能体清单 (Agent roster)**。每个智能体需包含：名称、一句话职责说明、可用工具，以及可交接给的其他智能体。
2. **交接函数 (Handoff functions)**。每个智能体的工具签名 (tool signatures)。每个交接函数需返回一个目标智能体 (Agent)。
3. **上下文传递策略 (Context transfer policy)**。针对每条交接边 (handoff edge)：传递完整历史记录、最近 N 条消息，或摘要快照。需提供理由。
4. **安全护栏 (Guardrails)**。每个智能体的输入验证（允许哪些提示词触发向敏感专家智能体的交接），以及在需要时对交接进行身份验证。
5. **循环检测 (Loop detection)**。用于检测乒乓循环 (ping-pong)（例如，“A 交接给 B；B 又交接回 A”连续发生超过一次）的规则。
6. **降级行为 (Fallback behavior)**。如果交接目标缺失（智能体被移除或身份验证失败），由哪个智能体接管该会话。
7. **会话/记忆方案 (Session / memory plan)**。明确是否使用 Agents SDK 会话、调用方管理的记忆，或完全不使用记忆。

硬性拒绝条件：

- 任何缺乏循环检测机制的交接设计。
- 将完整历史记录传递给具有不同工具权限的专家智能体的交接函数（存在安全风险）。
- 假设 Swarm 为无状态行为却又要求多轮记忆的设计——应改用 Agents SDK 会话。

拒绝规则：

- 如果任务需要并行执行，则拒绝使用 Swarm，并推荐改用监督者模式 (supervisor)（第 05 课）。
- 如果任务需要确定性审计/回放 (deterministic audit/replay) 功能，则拒绝并推荐 LangGraph 静态图。
- 如果任务仅为简单的阶段型有向无环图 (DAG)（研究 → 编码 → 审查），则推荐改用 CrewAI 顺序模式。

输出要求：一份单页的交接简报。结尾需附上安全说明，阐述提示词注入 (prompt injection) 如何触发非预期的交接，以及哪些安全护栏可阻止此类情况。
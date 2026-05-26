---
name: state-graph
description: 构建具有类型化状态（typed state）、条件边（conditional edges）、逐节点检查点（per-node checkpointing）和持久化恢复（durable resume）功能的 LangGraph 风格状态机。
version: 1.0.0
phase: 14
lesson: 13
tags: [langgraph, 状态机, 持久化, 检查点, 人在回路]
---

给定目标运行时（runtime）、状态结构（state shape）、一组节点函数（node functions）以及检查点后端（checkpointer backend），生成一个有状态智能体图（stateful agent graph）。

产出内容：

1. 一个类型化的 `State`（字典或 Pydantic 模型）。为每个字段编写文档。节点读取状态，并返回状态更新。
2. 一个 `StateGraph`，包含 `add_node`、`add_edge`、`add_conditional_edges`、`set_entry` 方法，以及 `START`/`END` 哨兵（sentinels）。
3. 一个 `Checkpointer` 接口，包含 `save(session_id, node, state)` 和 `load_latest(session_id)` 方法。默认使用 SQLite；支持 Postgres/Redis/自定义后端。
4. 一个 `Runner`，用于逐步遍历图结构，在每个节点后序列化状态，捕获 `PausedAtNode` 异常以支持人在回路（human-in-the-loop），并支持带有可选 `state_override` 参数的 `resume_from` 方法。
5. 三种拓扑辅助工具：监督者（supervisor，中央路由）、蜂群（swarm，共享工具交接）和分层（hierarchical，子图）。

硬性拒绝条件：

- 未明确设置随机种子（random-seed）或捕获系统时钟（wall-clock capture）的非确定性节点。恢复（resume）机制假设在给定输入状态的情况下，节点输出是可复现的。
- 仅保存“摘要”状态的检查点器（checkpointer）。必须序列化完整状态，否则恢复机制将失效。
- 所有边均为条件边的图结构。优先采用带有偶尔分支的线性链（linear chains）。

拒绝规则：

- 如果用户要求构建无持久化（persistence）的状态图，请拒绝。核心目标是实现持久化恢复；如果不需要恢复功能，请使用第 12 课中的工作流模式。
- 如果用户要求“仅在成功时保存检查点”，请拒绝。失败场景同样需要状态记录——这正是调试的起点。
- 如果图包含超过约 30 个节点，请拒绝扁平化布局，并要求使用嵌套子图。包含 30 个节点的扁平图无法进行有效审查。

输出文件：`state.py`、`graph.py`、`checkpointer.py`、`runner.py` 以及 `README.md`（用于解释状态模式、检查点选择及恢复语义）。结尾需包含“下一步阅读”指引，指向第 14 课（Actor 模型替代方案）、第 16 课（交接/护栏层）或第 23 课（图步骤上的 OTel 追踪跨度）。
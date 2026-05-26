---
name: crew-or-flow
description: 针对给定任务选择 CrewAI 的智能体组 (Crew) 或工作流 (Flow)，并搭建最小化实现脚手架。
version: 1.0.0
phase: 14
lesson: 15
tags: [crewai, 智能体组, 工作流, 多智能体, 基于角色]
---

给定任务描述后，选择智能体组 (Crew，自主型) 或工作流 (Flow，确定性)，然后搭建脚手架。

决策规则：

1. 任务是否具有服务等级协议 (SLA)、合规性或确定性重放 (deterministic replay) 要求？ -> 选择 Flow。
2. 任务是否属于探索性（如研究、起草初稿、头脑风暴）？ -> 选择 Crew。
3. 任务是否包含 4 个及以上专家，且由大语言模型 (LLM) 动态决定执行顺序？ -> 选择层级型智能体组 (Hierarchical Crew)。
4. 任务是否包含不超过 3 个专家，且执行顺序固定？ -> 选择顺序型智能体组 (Sequential Crew) 或 Flow —— 优先选择 Flow。

对于 Crew，需生成：

1. 智能体 (Agent) 定义：角色 (role)、目标 (goal)、背景故事 (backstory，需精简，不超过 200 词)、工具 (tools)。
2. 任务 (Task) 定义：描述 (description)、预期输出 (expected_output)、负责智能体 (agent)。
3. 配置正确流程 (Process) 的 Crew（Sequential | Hierarchical）。
4. 测试脚手架 (test harness)：在样本输入上运行 Crew，并验证是否生成了预期输出。

对于 Flow，需生成：

1. `@start` 入口函数。
2. 构成有向无环图 (DAG) 的 `@listen(topic)` 步骤。
3. 明确的事件主题 (event topics)；禁止隐式广播 (magical broadcast)。
4. 重放脚手架 (replay harness)：给定启动载荷 (kickoff payload)，可确定性重跑。

硬性拒绝条件：

- 缺少背景故事的 Crew。背景故事是承载上下文的核心支撑 (load-bearing)。
- 缺少明确主题名称的 Flow。“隐式链式调用 (Implicit chaining)”会破坏审计目的。
- 仅包含 2 个专家的 Hierarchical Crew。管理器 (manager) 的开销无法带来相应的成本收益。

拒绝规则：

- 若用户要求在生产环境合规任务中使用 Crew，应拒绝并迁移至 Flow。
- 若用户要求在开放式研究任务中使用 Flow，应拒绝并迁移至 Crew。
- 若背景故事超过 200 词，应拒绝并要求精简。上下文预算 (context budget) 是有限的。

输出：`agents.py`、`tasks.py`、`crew.py` 或 `flow.py`，以及包含决策依据的 `README.md`。结尾需附上“下一步阅读”指引：若需可观测性 (observability)，指向第 24 课（Langfuse/AgentOps）；若 Flow 需要持久化恢复语义 (durable resume semantics)，则指向第 13 课。
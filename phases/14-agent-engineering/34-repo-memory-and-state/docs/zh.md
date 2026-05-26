# 仓库记忆与持久化状态

> 聊天历史记录是易失的，而代码仓库是持久的。工作台将智能体（Agent）状态存储在带版本控制的文件中，确保下一次会话、下一个智能体以及下一位审查者都能从同一事实来源（source of truth）读取数据。

**类型：** 构建
**语言：** Python（标准库 + 可选 `jsonschema`）
**前置条件：** 第 14 阶段 · 32（最小化工作台）
**耗时：** 约 60 分钟

## 学习目标

- 明确哪些内容应存入仓库记忆，哪些应保留在聊天历史记录中。
- 为 `agent_state.json` 和 `task_board.json` 编写 JSON 模式（JSON Schema）。
- 构建一个状态管理器（State Manager），用于原子性地加载、验证、变更和持久化状态。
- 利用模式在错误写入破坏工作台之前将其拦截。

## 问题背景

智能体结束一次会话后，聊天窗口关闭。当新会话开启并询问从何开始时，模型表示“让我先查看文件”，却读取了过时的笔记，并重新执行了已完成的工作。更糟糕的是，由于无人告知该文件已处理完毕，它可能会直接覆盖重写已完成的文件。

工作台的解决方案是引入仓库记忆：状态以 JSON 文件的形式存在于代码仓库中，遵循既定模式编写，采用原子化方式持久化，并在代码审查中便于进行差异对比（diff）。聊天仅是临时数据流，而代码仓库才是权威记录系统（system of record）。

## 核心概念

flowchart LR
  Agent[Agent Loop] --> Manager[StateManager]
  Manager --> Schema[agent_state.schema.json]
  Schema --> Validate{valid?}
  Validate -- yes --> Write[agent_state.json]
  Validate -- no --> Reject[refuse + raise]
  Write --> Manager

### 仓库记忆应包含的内容

| 应包含 | 不应包含 |
|---------|-----------------|
| 当前活跃的任务 ID | 原始聊天转录文本 |
| 本次会话中修改过的文件 | Token 级别的推理轨迹 |
| 智能体做出的假设 | “用户似乎很沮丧” |
| 未解决的阻塞问题 | 采样生成的补全结果 |
| 下一步行动 | 特定厂商的模型 ID |

判断标准在于持久性：三个月后在持续集成（CI）重新运行时，这些数据是否仍有价值？如果是，则存入仓库；如果否，则仅作为遥测数据（telemetry）处理。

### 模式优先的状态管理

JSON 模式（JSON Schema）即契约。缺乏它时，每个智能体都会随意新增字段，每位审查者都要重新适应数据结构，每个 CI 脚本都必须为历史版本编写特殊处理逻辑。有了它，任何不符合规范的写入都会被直接拒绝。

该模式涵盖以下内容：

- 必填键。
- 允许的 `status` 取值。
- 禁止的值（例如数组字段不允许为 `null`）。
- 模式约束（例如任务 ID 需匹配正则表达式 `T-\d{3,}`）。
- 用于数据迁移的版本字段。

### 原子化写入

状态写入必须能够抵御部分故障：先写入临时文件，执行 `fsync` 同步，再通过重命名覆盖目标文件。状态文件是事实来源，一个写入不完整的文件比完全没有文件更糟糕。

### 数据迁移

当模式发生变更时，需随模式版本升级一同发布迁移脚本。状态文件中包含 `schema_version` 字段；若管理器无法迁移该版本对应的文件，将拒绝加载。

## 动手实践

`code/main.py` 实现了：

- `agent_state.schema.json` 和 `task_board.schema.json`。
- 仅使用标准库的验证器（JSON Schema 的子集：required、type、enum、pattern、items）。
- 采用原子化临时文件与重命名写入（atomic temp-and-rename writes）机制的 `StateManager.load`、`StateManager.update` 和 `StateManager.commit` 方法。
- 一个演示程序，用于变更状态、持久化、重新加载并验证往返一致性（round-trip）。

运行方式：

python3 code/main.py

该脚本会生成 `workdir/agent_state.json` 和 `workdir/task_board.json`，在两个回合中对其进行状态变更，并在每一步打印验证后的状态。

## 实际生产环境中的模式

以下四种模式将本教程的最小可行实现，升级为能够支撑多智能体单体仓库（multi-agent monorepo）稳定运行的方案。

**原子化临时文件与重命名写入（atomic temp-and-rename writes）并非可选项。** 2026 年 3 月的一份 Hive 项目缺陷报告清晰地记录了该故障模式：`state.json` 通过 `write_text()` 写入，且异常被捕获并静默处理。部分写入导致会话在恢复时面对损坏的状态却毫无提示。正确的修复方案始终是：在与目标文件相同的目录下使用 `tempfile.mkstemp` 创建临时文件，写入数据，执行 `fsync`，最后通过 `os.replace` 完成替换（在 POSIX 和 Windows 上均为原子重命名操作）。本教程中的 `atomic_write` 正是采用此逻辑。

**为每次非幂等（non-idempotent）的工具调用添加幂等键（idempotency keys）。** 如果智能体在调用工具后、保存检查点（checkpoint）前发生崩溃，恢复机制会重试该工具调用。这对读取操作是安全的，但对发送邮件、数据库插入或文件上传等操作则存在风险。应对模式：在执行前将每个工具调用的 ID 记录到 `pending_calls.jsonl` 中。重试时检查该 ID；若已存在，则跳过实际调用并直接使用缓存结果。Anthropic 和 LangChain 均在 2026 年的指南中强调了这一点；LangGraph 的检查点持久化机制出于相同原因也会保留待写入的数据。

**将大型产物（artifacts）与状态分离。** 不要将 CSV 文件、长对话记录或生成的文件存储在 `agent_state.json` 中。应将产物保存为独立文件（或上传至对象存储），仅在状态中保留其路径。这样检查点能保持轻量与快速，而产物则可独立增长。

**采用事件溯源（event sourcing）进行审计，使用快照（snapshots）实现恢复。** 每次状态变更时追加到事件日志（`state.events.jsonl`）；定期将状态快照保存至 `state.json`。恢复时先读取快照，然后重放快照时间戳之后的所有事件。这会占用更多磁盘空间，但允许你逐字重放智能体的决策过程——在调试长周期（long-horizon）任务时至关重要。这与 PostgreSQL 内部用于预写式日志（WAL）的架构如出一辙。

**执行模式迁移（schema migrations）或拒绝加载。** `schema_version` 整数即为版本契约。当管理器加载未知版本的文件时，将直接拒绝读取。在更新模式版本时附带迁移脚本；`tools/migrate_state.py` 会在每次启动时以幂等方式运行。

## 使用指南

在生产环境中：

- **LangGraph 检查点器（LangGraph checkpointers）**。核心理念一致，仅存储介质不同。该检查点器会将图状态（graph state）持久化至 SQLite、Postgres 或自定义后端（backend）。当检查点器发生故障且你需要手动读取状态时，本课所讲授的模式（schema）正是你所需的工具。
- **Letta 记忆块（Letta memory blocks）**。具有结构化模式（structured schemas）的持久化块（Phase 14 · 08）。采用相同的规范，但作用域限定于长期运行的智能体角色（personas）。
- **OpenAI Agents SDK 会话存储（OpenAI Agents SDK session store）**。支持可插拔后端，具备模式感知能力。本课中的状态文件即充当本地文件后端。

## 交付上线

`outputs/skill-state-schema.md` 会生成项目专属的 JSON Schema 对（状态 + 看板）、一个已接入原子写入（atomic writes）的 Python `StateManager`，以及一个迁移脚手架（migration scaffold），以确保下一次模式升级不会破坏工作台。

## 练习

1. 添加 `last_human_touch` 时间戳。拒绝智能体在人类编辑后五秒内的任何写入操作。
2. 扩展验证器以支持 `oneOf`，使任务可以是构建任务或评审任务，并各自拥有不同的必填字段。
3. 添加 `schema_version` 字段，并编写从 v1 到 v2 的迁移脚本（将 `blockers` 重命名为 `risks`）。
4. 将存储后端从本地文件迁移至 SQLite。保持 `StateManager` API 完全不变。
5. 让两个智能体针对同一状态文件运行，并制造 50 毫秒的写入竞争（write race）。会出现什么问题？原子重命名（atomic rename）又是如何挽救局面的？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 仓库记忆（Repo memory） | “笔记文件” | 存储在仓库受版本控制的文件中、受模式约束的状态 |
| 模式优先（Schema-first） | “验证输入” | 在写入前定义契约，拒绝状态漂移 |
| 原子写入（Atomic write） | “直接重命名” | 先写入临时文件，执行 fsync，再重命名，从而避免部分失败导致的数据损坏 |
| 迁移（Migration） | “模式升级” | 将 vN 状态转换为 v(N+1) 状态的脚本 |
| 记录系统（System of record） | “单一事实来源” | 工作台视为权威依据的工件 |

## 延伸阅读

- [JSON Schema 规范](https://json-schema.org/specification.html)
- [LangGraph 检查点器（checkpointers）](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Letta 内存块（memory blocks）](https://docs.letta.com/concepts/memory)
- [Fast.io，AI 智能体（AI Agent）状态检查点（State Checkpointing）：实用指南](https://fast.io/resources/ai-agent-state-checkpointing/) — 采用模式优先（schema-first）与幂等性（idempotency）的检查点方案
- [Fast.io，AI 智能体工作流状态持久化（Workflow State Persistence）：2026 最佳实践](https://fast.io/resources/ai-agent-workflow-state-persistence/) — 并发控制（concurrency control）、TTL（Time To Live）与事件溯源（event sourcing）
- [Hive Issue #6263 — 非原子性（non-atomic）state.json 写入被静默忽略（silently ignored）](https://github.com/aden-hive/hive/issues/6263) — 真实项目中的故障模式（failure mode）
- [eunomia，检查点/恢复系统（Checkpoint/Restore Systems）：演进、技术与应用](https://eunomia.dev/blog/2025/05/11/checkpointrestore-systems-evolution-techniques-and-applications-in-ai-agents/) — 将源自操作系统历史的检查点/恢复（CR）原语（CR primitives）应用于智能体
- [Indium，2026 年长周期运行 AI 智能体（Long-Running AI Agents）的 7 种状态持久化策略](https://www.indium.tech/blog/7-state-persistence-strategies-ai-agents-2026/)
- [Microsoft Agent Framework，上下文压缩（Compaction）](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction) — 厂商检查点管理器（vendor checkpoint manager）
- 第 14 阶段 · 08 — 内存块与休眠期计算（sleep-time compute）
- 第 14 阶段 · 32 — 本课所模式化（schematizes）的三文件基础配置
- 第 14 阶段 · 40 — 基于同一模式（schema）解析的交接数据包（handoff packets）
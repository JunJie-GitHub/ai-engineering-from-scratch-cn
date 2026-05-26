# 检查点与回滚 (Checkpoints and Rollback)

> 每次图状态转换 (graph-state transition) 都会被持久化。当工作节点 (worker) 崩溃时，其租约 (lease) 过期，另一个工作节点将从最新检查点 (checkpoint) 恢复执行。Cloudflare Durable Objects 可在数小时或数周内保持状态。先提议后提交 (Propose-then-commit)（第 15 课）为每个操作定义了回滚计划 (rollback plan)。操作后验证 (Post-action verification) 形成闭环。《欧盟人工智能法案》第 14 条 (EU AI Act Article 14) 强制要求高风险系统 (high-risk systems) 必须具备有效的人工监督 (human oversight)——在实践中，这意味着检查点必须可查询，回滚必须经过演练，且审计轨迹 (audit trail) 必须在部署后得以保留。典型的故障模式是：如果没有幂等键 (idempotency keys) 和前置条件检查 (precondition checks)，在瞬时故障 (transient failure) 后的重试可能会导致已批准的操作被重复执行 (double-execute)。而操作后验证正是用来捕获此类问题的。

**Type:** 学习
**Languages:** Python（标准库，检查点与回滚状态机）
**Prerequisites:** 第 15 阶段 · 12（持久化执行），第 15 阶段 · 15（先提议后提交）
**Time:** 约 60 分钟

## 核心问题

持久化执行 (Durable execution)（第 12 课）使崩溃的智能体 (agent) 能够恢复执行。先提议后提交（Propose-then-commit）（第 15 课）使已批准的操作具备可审计性 (auditable)。本课将两者结合：当一个已批准的操作执行到一半时发生崩溃并恢复，会发生什么？回滚 (rollback) 何时触发？基于何种状态执行？

实际系统中的实现方式各不相同：

- **LangGraph** 将每次图状态转换持久化到 PostgreSQL 中。当工作节点崩溃时，租约释放，另一个工作节点将从最新检查点恢复执行。工作流会在 `interrupt()` 处暂停，该状态本身也会被持久化。
- **Cloudflare Durable Objects** 可在数小时或数周内保持基于键的状态。将计算与已批准操作的存储部署在同一位置 (co-locate)。
- **Microsoft Agent Framework** 在工作流 API 中暴露了 `Checkpoint` 原语；重放 (replay) 结合幂等性 (idempotency) 可覆盖重试场景。

在所有情况下，真正有效的组合是：幂等键 (idempotency key，防止重复执行) + 前置条件检查 (precondition check，确保状态仍与批准时一致) + 操作后验证 (post-action verify，确认副作用 (side effect) 确实已发生) + 验证失败时触发回滚。

## 核心概念

### 每次状态转换均需持久化

图状态转换（graph-state transition）是指将工作流（workflow）从一个命名状态推进至另一个命名状态的任何步骤。初级实现仅在特定的提交点（commit point）进行持久化（persist）；而生产级实现（production implementation）会对每一次转换进行持久化。其代价（仅多出几次写入操作）相对于可靠性提升（重放（replay）可落在任意位置，租约恢复（lease recovery）精准）而言微乎其微。

### 租约恢复

当工作节点（worker）崩溃时，工作流并不会丢失；租约（lease，即声明该节点正在执行当前运行实例的短期凭证）仅会自然过期。另一个工作节点会拾取最新的检查点（checkpoint）并恢复执行。正是租约机制使得生产系统能够在滚动部署（rolling deploy）期间平稳运行，且不会丢失进行中的任务（in-flight work）。

### 幂等性结合前置条件

仅靠幂等性（idempotency）是不够的。试想：一个工作流被批准执行“当余额 > $1000 时，从 A 向 B 转账 $100”。该工作流已提交，但在执行中途崩溃并恢复。如果仅检查幂等键（idempotency key）并恢复执行，转账只会运行一次（正确）。但假设在崩溃与恢复之间，A 的余额因另一个工作流降至 $500。此时幂等性检查依然通过，但前置条件（precondition）检查却失败了。若无前置条件检查，我们将导致透支。

每个关键操作都需要同时具备以下两者：

- **幂等键（Idempotency key）**：防止重复执行（double-execute）。
- **前置条件检查（Precondition check）**：确认当前状态仍与批准时的状态一致。

### 操作后验证

“工具返回了 200”并不等于验证。真正的验证会重新读取目标状态，并确认副作用（side effect）确实已发生。常见模式如下：

- 数据库更新：执行 `UPDATE ... RETURNING *`，然后断言返回的行与预期状态匹配。
- 邮件发送：提交后检查发件箱（sent-folder）中是否存在该邮件 ID。
- 文件写入：重新读取文件并计算其哈希值。
- API 调用：对目标资源执行后续的 `GET` 请求。

如果验证失败，工作流将进入已知的异常状态（known-bad state）。此时将触发回滚（rollback）。

### 回滚计划

在“先提议后提交”（propose-then-commit，第 15 课）模式中，每个关键操作都必须附带回滚计划。类型包括：

- **带内回滚（In-band rollback）**：直接逆转副作用（例如在 `INSERT` 后执行 `DELETE`，或在发送邮件后发送 `Send-correction-email`）。
- **补偿事务（Compensating transaction）**：执行一个新操作以抵消原操作的影响（标准的 SAGA 模式）。
- **带外回滚（Out-of-band rollback）**：向人工发出警报，暂停工作流，保留异常状态以供调查。

无操作回滚（no-op rollback，即“我们无法撤销此操作”）必须在提议阶段明确声明。对于没有回滚方案的操作，在提交时需要更强的人工介入（HITL）机制（第 15 课的质询与响应（challenge-and-response）挑战）。

### 《欧盟人工智能法案》第 14 条的实操解读

第 14 条要求高风险系统具备“有效的人工监督”。在工程实操层面，开发者通常将其解读为：

- 检查点（checkpoint）必须可供审计员查询。
- 回滚流程必须经过演练（至少进行一次端到端测试）。
- 审计轨迹（audit trail）必须在部署后依然保留（检查点后端不能是临时性的）。
- 验证失败必须触发告警，而非仅静默记录日志。

如果一个工作流在提交中途崩溃、恢复执行，并在缺乏“验证 + 回滚”路径的情况下完成了副作用，它将无法通过第 14 条的合规性检验。

### 典型的故障模式：重复执行

该领域最常见的生产事故如下：

1. 操作获批，分配幂等键 `k`。
2. 提交开始，执行操作，返回 200。
3. 工作流在持久化“已提交”状态前崩溃。
4. 工作流恢复；发现状态为“已获批但未提交”；重新执行。
5. 副作用触发两次。

缓解措施：在执行前持久化一个“进行中”（in-flight）意图，使用幂等键执行操作，然后仅在操作后验证成功后才标记为“已提交”。如果操作已触发但状态写入失败，系统应进行验证并在必要时重新触发。如果状态写入成功但操作失败，系统将通过恢复路径进行验证并确保仅执行一次。

## 实践应用

`code/main.py` 实现了一个具备**幂等性 (idempotency)**、**前置条件 (preconditions)**、**验证 (verify)** 和**回滚 (rollback)** 功能的检查点工作流 (checkpointed workflow)。驱动程序 (driver) 模拟了四种场景：正常运行、崩溃后重试（由幂等性机制捕获）、前置条件失败（工作流中止且不触发执行）、验证失败（触发回滚）。

## 交付上线

`outputs/skill-rollback-rehearsal.md` 为拟议的工作流设计了一项回滚演练测试 (rollback-rehearsal test)，并审计检查点后端以确保审计追踪 (audit-trail) 的持久化。

## 练习

1. 运行 `code/main.py`。验证上述四种场景。针对“提交期间崩溃”的情况，确认在多次重试中该操作仅精确触发一次。

2. 修改“先标记完成，再执行操作”的模式，使状态写入在操作执行之后触发。重新运行崩溃场景。统计重复触发的操作次数。

3. 为某个具体的生产环境操作（例如“向 Slack 频道发送消息”）设计回滚方案。将其分类为带内 (in-band)、补偿型 (compensating) 或带外 (out-of-band)，并说明选择理由。

4. 选取一个你熟悉的工作流。识别其中的每一个状态转换 (state transition)。为每个转换标注持久化要求 (durability requirement)（持久化 / 不持久化）。统计当前尚未进行持久化的转换数量。

5. 回滚演练测试：设计一个端到端测试 (end-to-end test)，运行真实的工作流，人为使其崩溃，并确认回滚路径被正确触发。该测试需要断言 (assert) 哪些内容？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| 检查点 (Checkpoint) | “存档点” | 图状态 (graph-state) 的每次转换都会持久化到可靠的存储 (durable store) 中 |
| 租约 (Lease) | “工作节点声明” | 工作节点正在执行某次运行的短期声明；崩溃时自动失效 |
| 前置条件 (Precondition) | “状态闸门” | 断言当前状态仍与已批准的操作保持一致 |
| 操作后验证 (Post-action verify) | “重读检查” | 确认副作用 (side effect) 确实在目标系统中发生 |
| 带内回滚 (In-band rollback) | “直接撤销” | 使用逆操作来撤销副作用 |
| 补偿事务 (Compensating transaction) | “SAGA 撤销” | 执行一个新操作以抵消原始操作的影响 |
| 先标记完成 (Mark-as-done-first) | “状态写入顺序” | 在从提交操作返回前，先持久化已提交的状态 |
| 第 14 条 (Article 14) | “欧盟《人工智能法案》人工监督” | 运维层面：可查询的检查点、经过演练的回滚、可审计的追踪记录 |

## 延伸阅读

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 检查点原语 (checkpoint primitives) 与租约恢复 (lease recovery)。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — 将 Durable Objects 作为状态底层架构 (state substrate)。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 监管基线。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长周期工作流 (long-horizon workflows) 的可靠性框架。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Claude Code Routines 的工作流形态。
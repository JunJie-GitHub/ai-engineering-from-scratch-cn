# 人在回路（Human-in-the-Loop）：先提议后提交（Propose-Then-Commit）

> 2026 年关于人在回路（Human-in-the-Loop, HITL）的共识非常明确。它并非“智能体询问，用户点击批准”。而是先提议后提交（Propose-Then-Commit）：提议的操作会附带幂等键（idempotency key）持久化至持久化存储（durable store）中；向审核者呈现操作意图、数据血缘（data lineage）、涉及的权限、影响范围（blast radius）以及回滚计划；仅在收到明确确认（positive acknowledgement）后才提交；执行后进行验证，以确认实际影响（side effect）确实发生。LangGraph 的 `interrupt()` 结合 PostgreSQL 检查点机制、Microsoft Agent Framework 的 `RequestInfoEvent`，以及 Cloudflare 的 `waitForApproval()` 均实现了相同的架构形态。典型的失败模式是橡皮图章式审批（rubber-stamp approval）：用户未经审核便直接点击“批准？”。文档中记载的缓解措施是采用附带明确检查清单的质询与响应（challenge-and-response）机制。

**类型：** 学习
**语言：** Python（标准库，带幂等性的先提议后提交状态机）
**前置条件：** 第 15 阶段 · 12（持久化执行（Durable execution）），第 15 阶段 · 14（安全绊线（Tripwires））
**耗时：** 约 60 分钟

## 问题

智能体执行某项操作。用户必须做出决定：批准或不批准。如果决策是瞬间完成的，那它很可能算不上真正的审核。如果决策是结构化的，虽然速度较慢，但更加可靠。工程上的核心问题在于：如何让结构化审核成为阻力最小的路径。

2023 年时期的人在回路（HITL）模式是一种同步提示：“智能体希望向 X 发送内容为 Y 的邮件——是否批准？”用户点击“批准”。所有人都觉得系统是安全的。但在实践中，这种界面极易导致盲目审批：用户审批速度极快，审批记录几乎无法预测风险；而当智能体出错时，审计日志中只会留下一长串用户早已毫无印象的批准记录。

2026 年的模式——先提议后提交（Propose-Then-Commit）——将人在回路机制迁移至持久化底层架构，附加结构化元数据，并要求进行明确确认提交。每个主流智能体 SDK 都提供了相应的实现版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。尽管 API 名称各异，但其底层架构模式完全一致。

## 核心概念

### 提议-提交状态机 (Propose-then-Commit State Machine)

1. **提议 (Propose)。** 智能体 (Agent) 生成一个提议的操作。该提议会持久化存储到持久化存储 (Durable Store)（如 PostgreSQL、Redis、Durable Object）中。包含以下内容：
   - 意图 (Intent)（智能体为何执行此操作）
   - 数据血缘 (Data Lineage)（何种数据源促成了此提议）
   - 涉及的权限 (Permissions Touched)（触及了哪些作用域/文件/端点）
   - 影响范围 (Blast Radius)（最坏情况下的影响）
   - 回滚计划 (Rollback Plan)（若已提交，如何撤销）
   - 幂等键 (Idempotency Key)（每个提议唯一；重复提交将返回相同记录）
2. **展示 (Surface)。** 审核员 (Reviewer) 查看包含所有元数据的提议。审核员必须是人类（而非智能体自我审核）。
3. **提交 (Commit)。** 明确确认。操作开始执行。
4. **验证 (Verify)。** 执行完成后，系统会回读并确认产生的副作用 (Side Effect)。若验证步骤失败，系统将进入已知的异常状态，并触发告警。

### 幂等键 (Idempotency Key)

若缺乏幂等键，在发生瞬时故障 (Transient Failure) 后进行重试，可能导致已批准的操作被重复执行。具体示例：用户批准了“从 A 向 B 转账 100 美元”。网络出现短暂波动。工作流 (Workflow) 触发重试。用户仅批准了一次，但转账却执行了两次。幂等键将批准操作与单一、唯一的副作用绑定；第二次执行将变为空操作 (No-op)。

这与 Stripe 和 AWS API 采用的幂等模式相同。Microsoft Agent Framework 文档中明确指出，该模式可复用于智能体审批流程。

### 持久性：为何审批状态能跨越进程生命周期

审批等待区 (Approval Waiting Room) 是智能体不直接持有的一段状态。此时工作流处于暂停状态（参见第 12 课）。当审批结果到达时，工作流将从该确切断点恢复。这正是 LangGraph 将 `interrupt()` 与 PostgreSQL 检查点 (Checkpointing) 结合使用，而非仅依赖内存状态的原因——即使两天后才收到审批，工作流依然完好无损。

### 橡皮图章式审批与挑战-响应缓解机制

人机协同 (Human-in-the-Loop, HITL) 的默认 UI（“批准”/“拒绝”按钮）往往导致快速审批，缺乏实质性审核。文档记录的缓解措施：采用挑战-响应 (Challenge-and-Response) 检查清单，要求审核员对特定问题给出肯定答复后，“批准”按钮才会启用。具体形式如下：

- “您是否清楚此操作涉及哪些资源？[ ]”
- “您是否已确认影响范围在可接受范围内？[ ]”
- “若操作失败，您是否已制定回滚计划？[ ]”

这并非为了官僚主义而设，而是一种强制机制 (Forcing Function)。无法勾选这些选项的审核员，要么请求进一步澄清（升级处理），要么直接拒绝（安全默认值）。Anthropic 的智能体安全研究明确引用了基于检查清单的 HITL 作为缓解橡皮图章式审批模式的有效手段。

### 何为关键性操作

并非所有操作都需要走提议-提交流程。2026 年指南建议：

- **关键性操作 (Consequential Actions)**（始终需要 HITL）：不可逆写入、金融交易、对外通信、生产环境数据库变更、破坏性文件系统操作。
- **可逆操作 (Reversible Actions)**（视情况需要 HITL）：本地文件编辑、预发环境 (Staging Environment) 变更、具备明确回滚路径的可逆写入。
- **读取与检查操作 (Reads and Inspections)**（无需 HITL）：读取文件、列出资源、调用只读 API。

### 操作后验证

“提交已执行”并不等同于“副作用已生效”。网络分区 (Network Partition) 和竞态条件 (Race Conditions) 可能导致工作流误判为成功，而后端实际并未完成持久化。验证步骤会在提交后重新读取目标资源以进行确认。这与数据库事务中使用 `RETURNING` 子句，或在 AWS 中执行 `PutObject` 后调用 `GetObject` 的模式一致。

### 《欧盟人工智能法案》第 14 条

该法案第 14 条强制要求对欧盟境内的高风险 AI 系统实施有效的人类监督 (Human Oversight)。“有效”绝非装饰性要求。监管措辞明确排除了橡皮图章式审批模式。在 Microsoft Agent Governance Toolkit 合规文档中，结合挑战-响应机制的提议-提交流程，是能够通过第 14 条审查的标准架构。

## 使用它

`code/main.py` 使用 Python 标准库实现了一个提议-提交（propose-then-commit）状态机。持久化存储（durable store）为一个 JSON 文件。幂等键（idempotency key）是 `(thread_id, action_signature)` 的哈希值。驱动程序（driver）模拟了三种场景：正常的审批流程、瞬态故障（transient failure）后的重试（必须避免重复执行（double-execute）），以及默认的“橡皮图章”式审批（rubber-stamp）与“质询-响应”（challenge-and-response）流程的对比。

## 交付它

`outputs/skill-hitl-design.md` 审查了一个拟议的人机协同（HITL）工作流，评估其是否符合“提议-提交”架构，并标记出缺失的元数据（metadata）、幂等性（idempotency）、验证（verification）或质询-响应层。

## 练习

1. 运行 `code/main.py`。确认已批准提议的重试会复用持久化记录，且不会重新执行。接着，修改幂等键以包含时间戳，并演示重试会导致重复执行。

2. 在提议记录中扩展一个 `rollback` 字段。模拟一次验证步骤失败的执行过程。展示回滚机制如何自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。找出该 API 包含但当前示例引擎（toy engine）缺失的一个元数据字段。将其添加进去，并说明它能防范什么风险。

4. 为某项具体操作（例如“发布到公开的 Twitter 账号”）设计一份质询-响应检查清单。审核者必须回答哪三个问题？为什么是这三个？

5. 挑选一个仅需同步（synchronous）“批准？”提示即可满足需求的场景（无需持久化存储）。解释原因，并指出你所接受的风险类别（risk class）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| 提议-提交（Propose-then-commit） | “两阶段审批” | 持久化提议 + 明确提交 + 验证 |
| 幂等键（Idempotency key） | “重试安全令牌” | 每个提议唯一；第二次执行不产生任何操作（no-op） |
| 数据血缘（Data lineage） | “来源追溯” | 导致该提议的具体原始内容 |
| 爆炸半径（Blast radius） | “最坏情况” | 操作出错时的影响范围 |
| 橡皮图章式审批（Rubber-stamp） | “快速审批” | 未经实质审查直接点击“批准” |
| 质询-响应（Challenge-and-response） | “强制检查清单” | 审核者必须对特定问题给出明确确认 |
| `RequestInfoEvent` | “MS Agent Framework 基础原语” | 带有结构化元数据的持久化人机协同（HITL）请求 |
| `interrupt()` / `waitForApproval()` | “框架基础原语” | LangGraph / Cloudflare 中架构相同的等效实现 |

## 延伸阅读

- [Microsoft Agent Framework — 人在回路 (Human-in-the-Loop)](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`，持久化审批。
- [Cloudflare Agents — 人在回路](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 与持久化对象 (Durable Objects)。
- [Anthropic — 实践中衡量智能体自主性 (agent autonomy)](https://www.anthropic.com/research/measuring-agent-autonomy) — 将人在回路 (HITL) 作为缓解长程风险 (long-horizon risk) 的措施。
- [欧盟《人工智能法案》— 第14条：人类监督 (Human oversight)](https://artificialintelligenceact.eu/article/14/) — 高风险系统 (high-risk systems) 的监管基线。
- [Anthropic — Claude 宪法（2026年1月）](https://www.anthropic.com/news/claudes-constitution) — 围绕监督机制的宪法框架。
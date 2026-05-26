# 运行时反馈循环 (Runtime Feedback Loops)

> 无法看到真实命令输出的智能体（Agent）只能靠猜测。反馈运行器（Feedback Runner）会捕获标准输出（stdout）、标准错误（stderr）、退出码（exit code）和执行耗时，并将其存入结构化记录中，供下一轮对话读取。这样，智能体就能基于事实做出反应，而不是基于自己对事实的预测。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 32（最小工作台），第 14 阶段 · 35（初始化脚本）
**预计耗时：** 约 50 分钟

## 学习目标

- 区分运行时反馈（Runtime Feedback）与可观测性遥测数据（Observability Telemetry）。
- 构建一个反馈运行器，用于封装 Shell 命令并持久化结构化记录。
- 以确定性方式截断大型输出，确保循环保持在 Token 预算（Token Budget）范围内。
- 在缺少反馈时拒绝推进循环。

## 问题背景

智能体声称“正在运行测试”，下一条消息却显示“所有测试通过”。而实际情况是根本没有执行任何测试。智能体可能凭空捏造了输出，或者执行了命令却从未读取结果，又或者读取了结果却静默截断了报错行。

反馈运行器正是为了消除这一差距而生。每条命令都必须经过该运行器处理。每条记录都会包含执行的命令、捕获的标准输出与标准错误、退出码、实际耗时（wall-clock duration）以及智能体的一行备注。智能体在下一轮对话中读取该记录，而验证关卡（Verification Gate）则在任务结束时统一读取这些记录。

## 核心概念

flowchart LR
  Agent[Agent Loop] --> Runner[run_with_feedback.py]
  Runner --> Shell[subprocess]
  Shell --> Capture[stdout / stderr / exit / duration]
  Capture --> Record[feedback_record.jsonl]
  Record --> Agent
  Record --> Gate[Verification Gate]

### 反馈记录包含的内容

| 字段 | 重要性说明 |
|-------|----------------|
| `command` | 精确的 argv 参数，避免 Shell 展开带来的意外 |
| `stdout_tail` | 末尾 N 行内容，采用确定性截断 |
| `stderr_tail` | 末尾 N 行内容，与标准输出分离 |
| `exit_code` | 明确的成功/失败信号 |
| `duration_ms` | 暴露缓慢的探测任务或失控进程 |
| `started_at` | 用于回放的时间戳 |
| `agent_note` | 智能体记录的一行预期说明 |

### 确定性截断机制

一份 50 MB 的日志足以摧毁整个循环。运行器会在头部和尾部进行截断，并插入 `...truncated N lines...` 标记。该过程是确定性的，确保相同的输出始终生成相同的记录。不进行随机采样，因为智能体需要查看的关键部分（最终报错、最终摘要）都位于日志尾部。

### 反馈与遥测数据的区别

遥测数据（Telemetry，参考第 14 阶段 · 23 及 OTel GenAI 规范）供人类操作员跨时间段审查运行记录。而反馈数据专为当前运行的下一轮对话服务。两者虽然共享部分字段，但存储于不同的文件中，且数据保留策略（Retention）也不同。

### 无反馈则拒绝推进

如果运行器在捕获退出状态前发生错误，记录中将包含 `exit_code: null` 和 `error: <reason>`。智能体循环必须拒绝在退出码为 `null` 时宣称成功。没有退出状态，就没有进展。

## 动手构建

`code/main.py` 实现了以下功能：

- `run_with_feedback(command, agent_note)`：封装了 `subprocess.run`，捕获标准输出（stdout）/标准错误（stderr）/退出状态码（exit）/执行时长（duration），进行确定性截断，并追加至 `feedback_record.jsonl`。
- 一个小型加载器（loader），用于将 JSONL 数据流式读取为 Python 列表。
- 一个演示程序，运行三条命令（成功、失败、慢速），并打印每条命令的最后一条记录。

运行方式：

python3 code/main.py

输出结果：三条反馈记录将追加至 `feedback_record.jsonl`，每条命令的最后一条记录会直接打印在终端中。在多次重新运行期间使用 `tail` 命令查看该文件，即可观察到循环累积的过程。

## 实际生产环境中的模式
以下三种模式足以让该运行器（runner）达到生产级标准并正式发布。

**写入时脱敏，而非读取时。** 任何涉及标准输出或标准错误的记录都可能泄露敏感信息。该运行器在追加至 JSONL 之前内置了脱敏（redaction）处理流程：过滤掉匹配 `^Bearer `、`password=`、`api[_-]?key=`、`AKIA[0-9A-Z]{16}`（AWS）或 `xox[baprs]-`（Slack）的行。在读取时进行脱敏极易引发隐患（foot-gun）；攻击者实际接触的是磁盘上的文件。应每季度根据生产运行时（runtime）实际观测到的密钥格式，对脱敏规则进行审计。

**采用轮转策略，而非单一文件。** 将 `feedback_record.jsonl` 的单文件大小限制为 1 MB；超出限制时轮转至 `.1`、`.2`，并丢弃 `.5`。智能体（agent）的循环仅读取当前文件，因此运行时开销是可控的。完整的轮转文件集将归档至 CI 制品存储（artifact storage）中。若不进行轮转，该文件将在每次加载器调用时成为性能瓶颈。

**为重试链设置父命令 ID。** 每条记录都会分配 `command_id`；重试操作会携带 `parent_command_id` 以指向上一次尝试。审查者的“失败尝试”列表（第 14 阶段 · 40）以及验证网关（verification gate）的审计流程均会沿此链路追踪。若缺少此关联，重试操作看起来就像是独立的成功执行，审计记录也会掩盖失败历史。

## 使用指南
生产环境模式示例：

- **Claude Code Bash 工具。** 该工具已内置对标准输出、标准错误、退出状态码和执行时长的捕获功能。本课程中的运行器是适用于任何智能体产品的框架无关（framework-agnostic）等效实现。
- **LangGraph 节点。** 使用运行器封装任意 Shell 节点，使记录能够持久化保存于图状态（graph state）之外。
- **CI 日志。** 将 JSONL 数据流式传输至 CI 制品存储库；审查者无需重新运行会话即可回放任意命令。

该运行器是一个轻量级封装层，由于它独立定义了记录的数据结构，因此能够平滑跨越每一次框架迁移。

## 发布部署
`outputs/skill-feedback-runner.md` 会生成项目专属的 `run_with_feedback.py`，其中包含合理的截断配额（truncation budget）配置、与工作台（workbench）对接的 JSONL 写入器，以及供智能体在每次执行轮次中读取的加载器。

## 练习

1. 为每条记录添加 `cwd` 字段，以便区分从不同目录执行的相同命令。
2. 添加一个脱敏（redaction）步骤，用于剔除匹配 `^Bearer ` 或 `password=` 的行。在测试夹具记录（fixture record）上进行测试。
3. 通过文件轮转（file rotation）生成 `.1`、`.2` 等文件，将 `feedback_record.jsonl` 的总大小限制在 1 MB 以内。请论证该轮转策略的合理性。
4. 添加 `parent_command_id` 字段，使重试链（retry chains）可见：明确哪个命令生成了下一个命令所消费的输入。
5. 将 JSONL 数据流接入一个轻量级终端用户界面（TUI），以高亮显示最新的非零退出码（non-zero exit）。明确该 TUI 在审查中必须具备的八项关键功能。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 反馈记录（Feedback record） | “运行日志” | 包含命令、输出、退出状态和持续时间的结构化 JSONL 条目 |
| 尾部截断（Tail truncation） | “裁剪日志” | 确定性捕获头部与尾部内容，确保记录符合 Token 预算（token budget）限制 |
| 遇空拒绝（Refuse-on-null） | “缺失数据时阻塞” | 当 `exit_code` 为空时，循环不得继续推进 |
| 智能体备注（Agent note） | “预期标签” | 智能体在读取结果前写下的一行预测内容 |
| 遥测分流（Telemetry split） | “两个日志文件” | 为下一轮交互提供反馈，为操作员提供遥测数据 |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Guardrails AI x MLflow — deterministic safety, PII, quality validators](https://guardrailsai.com/blog/guardrails-mlflow) — 脱敏模式作为回归测试
- [Aport.io, Best AI Agent Guardrails 2026: Pre-Action Authorization Compared](https://aport.io/blog/best-ai-agent-guardrails-2026-pre-action-authorization-compared/) — 工具调用前后的数据捕获
- [Andrii Furmanets, AI Agents in 2026: Practical Architecture for Tools, Memory, Evals, Guardrails](https://andriifurmanets.com/blogs/ai-agents-2026-practical-architecture-tools-memory-evals-guardrails) — 可观测性界面（observability surfaces）
- 第 14 阶段 · 23 — 遥测侧的 OTel GenAI 规范
- 第 14 阶段 · 24 — 智能体可观测性平台（Langfuse、Phoenix、Opik）
- 第 14 阶段 · 33 — 声明完成前必须获取反馈的规则
- 第 14 阶段 · 38 — 读取 JSONL 的验证关卡
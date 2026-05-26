# 范围契约与任务边界

> 模型并不知道工作应在何处结束。范围契约（Scope Contract）是一个针对每个任务的文件，它明确规定了工作的起点、终点，以及当工作越界时如何回滚。该契约将“保持在范围内”从一种期望转变为一项可执行的检查。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 32（最小工作台），第 14 阶段 · 33（规则即约束）
**耗时：** 约 50 分钟

## 学习目标

- 编写一份范围契约，供智能体（Agent）在任务开始时读取，供验证器（Verifier）在任务结束时读取。
- 明确允许修改的文件、禁止触碰的文件、验收标准、回滚计划以及审批边界。
- 实现一个范围检查器（Scope Checker），用于将代码差异（Diff）与契约进行比对，并标记违规行为。
- 使范围蔓延（Scope Creep）变得可见、自动化且可审查。

## 问题背景

智能体容易产生范围蔓延。任务本是“修复登录漏洞”，但最终的代码差异（Diff）却涉及了登录路由、邮件辅助模块、数据库驱动、README 文档以及发布脚本。每一次修改在当时看来都有合理的理由，但组合在一起后，实际变更的内容已与最初评审的范围截然不同。

范围蔓延是智能体工作中最缺乏监控的故障模式，因为智能体在叙述每一步时都出于“善意”。解决之道并非使用更严格的提示词（Prompt），而是在磁盘上保存一份明确承诺内容的契约，并通过检查机制将实际结果与承诺进行比对。

## 核心概念

flowchart LR
  Task[Task] --> Contract[scope_contract.json]
  Contract --> Agent[Agent Loop]
  Agent --> Diff[final diff]
  Diff --> Checker[scope_checker.py]
  Contract --> Checker
  Checker --> Verdict{in scope?}
  Verdict -- yes --> Verify[Verification Gate]
  Verdict -- no --> Block[block + open question]

### 范围契约应包含的内容

| 字段 | 用途 |
|-------|---------|
| `task_id` | 关联看板上的任务 |
| `goal` | 一句可供评审人员验证的目标描述 |
| `allowed_files` | 智能体允许写入的文件通配符（Globs） |
| `forbidden_files` | 智能体即使误操作也绝不能触碰的文件通配符 |
| `acceptance_criteria` | 用于证明任务完成的测试命令或断言语句 |
| `rollback_plan` | 若需中止任务，操作人员可执行的一段回滚说明 |
| `approvals_required` | 超出范围且需要人工明确签字批准的操作 |

缺少 `forbidden_files` 的契约是不完整的。明确“不可为”的边界构成了契约的一半。

### 使用通配符而非绝对路径

实际代码仓库中的文件经常会被移动。将契约绑定到通配符（如 `app/**/*.py`、`tests/test_signup*.py`），可确保跨会话的重构操作不会导致契约失效。

### 回滚机制属于范围的一部分

列出回滚步骤会迫使契约编写者提前思考可能出现的风险。一份无法回滚的契约，就不应该被批准。

### 范围检查即差异检查

智能体生成代码差异（Diff）。检查器读取该差异、允许的通配符、禁止的通配符，以及已执行的验收命令列表。每一项违规都会被标记为具体发现项，验证网关（Verification Gate）可据此拒绝通过。

## 动手实践

`code/main.py` 实现了以下功能：

- `scope_contract.json` 模式（schema）（JSON Schema 的子集，支持 glob 数组）。
- 一个差异解析器（diff parser），用于将受影响的文件列表与运行命令列表转换为 `RunSummary`。
- 一个 `scope_check` 检查器，用于根据合约返回 `(violations, in_scope, off_scope)` 结果。
- 两次演示运行：一次保持在范围内，一次发生范围蔓延（scope creep）。检查器会精确标记出蔓延的文件及原因。

运行方式：

python3 code/main.py

输出结果：合约内容、两次运行记录、每次运行的判定结果，以及保存的 `scope_report.json` 文件。

## 实际生产环境中的模式

一位实践者在调用智能体（agent）前使用 YAML 编写范围合约（即“specsmaxxing”实践）后报告称，在未更改智能体的情况下，三周内陷入死胡同（rabbit-hole）的发生率从 52% 降至 21%。真正发挥作用的是合约，而非模型本身。以下三种模式能够确保这一收益持续生效。

**违规预算（violation budget），而非二元失败。** `agent-guardrails`（通过 MCP 被 Claude Code、Cursor、Windsurf、Codex 使用的开源合并门禁（merge gate））为每个任务提供 `violationBudget`：预算范围内的轻微范围偏离将以警告形式提示；仅当超出预算时，合并门禁才会拒绝。需配合 `violationSeverity: "error" | "warning"` 使用。设置预算正是“能够顺利交付的门禁”与“因团队反感而被禁用的门禁”之间的关键区别。

**基于路径族的严重性不对称。** 对 `docs/**` 的范围外写入通常设为 `warn`（警告）；而对 `scripts/**`、`migrations/**`、`config/prod/**` 的范围外写入则始终设为 `block`（拦截）。这种不对称性必须定义在合约中，而非运行时（runtime）中，因为它是项目特定的，且会随任务变化。

**时间与网络预算，与文件预算并列。** `time_budget_minutes` 字段用于限制物理时钟时间（wall clock）；若无重新审批，运行时将拒绝继续执行。基于主机名的 `network_egress` 允许列表（allowlist）可防止智能体静默调用任务范围之外的外部 API。这些同样属于范围维度；仅靠文件通配符（file glob）是必要但不充分的条件。

**多合约合并语义（最小权限原则（least privilege））。** 当两个范围合约同时生效时（例如一个项目级合约加上一个任务级合约），合并规则如下：**取交集** `allowed_files`（两个合约都必须允许该路径），**取并集** `forbidden_files`（任一合约禁止即生效），`time_budget_minutes` 取最严格值（最小值），`approvals_required` 累加。`network_egress` 中，`None` 表示不强制执行，`[]` 表示全部拒绝，`[...]` 作为允许列表；在合并时，`None` 让位于另一侧的配置，两个列表取交集，而全部拒绝（deny-all）保持不变。应在合约模式（schema）中明确声明这些规则，以确保合并过程是机械化且可审查的。

## 使用指南

**生产模式（Production patterns）：**

- **Claude Code 斜杠命令（slash commands）。** `/scope` 命令用于编写契约并将其固定为会话上下文（session context）。子代理（subagents）在执行操作前会先读取该契约。
- **GitHub PR。** 将契约以 JSON 文件形式附加在 PR 描述中，或作为已提交的构建产物（artifact）。持续集成（CI）会针对合并差异（merge diff）运行范围检查器（scope checker）。
- **LangGraph 中断（interrupts）。** 范围违规（scope violation）会触发中断；处理程序（handler）会询问人工决策者：是需要扩展契约范围，还是让代理回退操作。

契约会随任务流转。当任务关闭时，该契约将被归档至 `outputs/scope/closed/` 目录下。

## 发布（Ship It）

`outputs/skill-scope-contract.md` 可根据任务描述生成范围契约（scope contract），并生成一个支持通配符模式（glob-aware）的检查器。该检查器会在持续集成（CI）流程中针对每次代理生成的代码差异（agent diff）自动运行。

## 练习（Exercises）

1. 添加 `network_egress` 字段，用于列出允许访问的外部主机。若运行过程尝试访问其他主机，则直接拒绝执行。
2. 扩展检查器逻辑，使其对 `docs/**` 路径采用软失败（fail soft）策略，而对 `scripts/**` 路径采用硬失败（fail hard）策略。请说明这种不对称设计的合理性。
3. 让契约通过静态规则集（不依赖大语言模型/LLM）从 `goal` 字段自动推导 `allowed_files`。在遇到第一个边界情况（edge case）时会出现什么问题？
4. 添加 `time_budget_minutes` 字段，一旦实际耗时（wall clock）超过该预算，则拒绝继续执行。
5. 针对同一份代码差异（diff）同时运行两份契约。当两份契约均适用时，应采用何种正确的合并语义（merge semantics）？

## 关键术语（Key Terms）

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 范围契约（Scope contract） | “任务简报” | 针对单个任务的 JSON 文件，列出允许/禁止访问的文件、验收标准及回滚方案 |
| 范围蔓延（Scope creep） | “它还改了……” | 在同一任务中修改了契约范围之外的文件 |
| 回滚计划（Rollback plan） | “我们可以回退” | 用于中止操作的一页纸运维操作手册（operator runbook） |
| 审批边界（Approval boundary） | “需要签字确认” | 契约中明确列出、必须经过人工显式批准才能执行的操作 |
| 差异检查（Diff check） | “路径审计” | 将实际修改的文件与契约中定义的通配符规则（globs）进行比对 |

## 延伸阅读（Further Reading）

- [LangGraph 人在回路中断机制 (Human-in-the-Loop Interrupts)](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [OpenAI Agents SDK 工具审批策略 (Tool Approval Policies)](https://platform.openai.com/docs/guides/agents-sdk)
- [logi-cmd/agent-guardrails — 合并门禁 (Merge Gates) 与作用域验证 (Scope Validation)](https://github.com/logi-cmd/agent-guardrails) — 违规配额 (Violation Budgets)、严重性分级 (Severity Tiers)
- [Dev|Journal，通过智能体契约测试 (Agent Contract Testing) 防止 AI 智能体配置漂移 (Configuration Drift)](https://earezki.com/ai-news/2026-05-05-i-built-a-tiny-ci-tool-to-keep-ai-agent-configs-from-drifting-in-my-repo/) — 无外部依赖的 `--strict` 模式
- [智能体编程 (Agentic Coding) 并非陷阱（生产环境日志）](https://dev.to/jtorchia/agentic-coding-is-not-a-trap-i-answered-the-viral-hn-post-with-my-own-production-logs-33d9) — 规范过度优化 (Specsmaxxing) 数据记录：52% → 21%
- [OpenCode 权限通配符 (Permission Globs)](https://opencode.ai/docs/agents/) — 细粒度的单权限作用域控制
- [Knostic，AI 编程智能体安全：威胁模型 (Threat Models) 与防护策略](https://www.knostic.ai/blog/ai-coding-agent-security) — 将作用域纳入最小权限原则 (Principle of Least Privilege)
- [Augment Code，AI 规范模板](https://www.augmentcode.com/guides/ai-spec-template) — 三级边界系统（必须执行/询问确认/禁止执行）
- 第 14 阶段 · 27 — 与作用域锁定配合的提示词注入 (Prompt Injection) 防御机制
- 第 14 阶段 · 33 — 该契约针对各任务特化的规则集
- 第 14 阶段 · 38 — 检查器上报结果的验证门禁
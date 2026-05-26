# 多会话交接 (Multi-Session Handoff)

> 会话即将结束，但工作并未停止。交接包 (handoff packet) 是将“智能体工作了一小时”转化为“下一会话在第一分钟即可高效产出”的关键产物。应有意识地构建它，而非事后补救。

**类型：** 构建
**语言：** Python (标准库)
**前置条件：** 第14阶段 · 34（Repo Memory），第14阶段 · 38（Verification），第14阶段 · 39（Reviewer）
**耗时：** 约50分钟

## 学习目标

- 明确每个交接包所需的七个字段。
- 基于工作台产物 (workbench artifacts) 自动生成交接包，无需手动撰写长篇大论。
- 将庞大的反馈日志 (feedback logs) 精简为适合交接包大小的摘要。
- 确保下一会话的首个操作具有确定性。

## 问题背景

会话结束。智能体表示“很好，我们取得了进展。”下一会话开启。新的智能体询问“我们上次进行到哪了？”前一个智能体的回答已不复存在。新的智能体只能重新摸索、重复执行相同的命令、向人类重复提出相同的问题，并耗费三十分钟来恢复上一个会话最后三十秒的上下文。

糟糕的交接会在任务存续期间的每个会话中持续产生代价。解决方案是在会话结束时自动生成一个数据包，记录：变更内容、变更原因、尝试过的操作、失败项、剩余工作，以及下次会话的首要任务。

## 核心概念

flowchart LR
  State[agent_state.json] --> Generator[generate_handoff.py]
  Verdict[verification_report.json] --> Generator
  Review[review_report.json] --> Generator
  Feedback[feedback_record.jsonl] --> Generator
  Generator --> Handoff[handoff.md + handoff.json]
  Handoff --> Next[Next Session]

### 交接包必备的七个字段

| 字段 | 解答的问题 |
|-------|---------------------|
| `summary` | 一段话概括已完成的工作 |
| `changed_files` | 变更文件一览 |
| `commands_run` | 实际执行的命令 |
| `failed_attempts` | 尝试过的操作及失败原因 |
| `open_risks` | 下一会话可能遇到的潜在风险及其严重程度 |
| `next_action` | 下一会话需执行的首个具体步骤 |
| `verdict_pointer` | 指向验证报告 (verification report) 与审查报告 (review report) 的路径 |

`next_action` 字段是承载核心作用的关键。缺少 `next_action` 的交接包只是一份状态报告，而非真正的交接。

### 交接包应自动生成，而非手动编写

手动编写的交接包在任务繁重时最容易被跳过。生成器会读取工作台产物并输出交接包。智能体的职责是让工作台处于生成器可总结的状态，而非亲自撰写摘要。

### 两种格式：人类可读与机器可读

`handoff.md` 供人类阅读，`handoff.json` 供下一会话的智能体加载。两者源自同一批源产物。若内容出现分歧，以 JSON 为准。

### 反馈日志精简

完整的 `feedback_record.jsonl` 可能包含数百条记录。交接包仅保留最后 K 条记录以及所有退出码非零 (non-zero exit) 的条目。下一会话可按需加载完整日志，但交接包本身保持精简。

## 构建

`code/main.py` 实现了：

- 一个加载器，负责将状态、裁决、审查和反馈收集到单个 `WorkbenchSnapshot` 中。
- 一个 `generate_handoff(snapshot) -> (markdown, payload)` 函数。
- 一个过滤器，用于筛选最后 K 条反馈记录以及所有非零退出状态。
- 一个演示运行脚本，会在脚本同级目录下生成 `handoff.md` 和 `handoff.json` 文件。

运行方式：

python3 code/main.py

输出：打印交接主体内容，并在磁盘上生成上述两个文件。

## 实际生产模式

Codex CLI、Claude Code 和 OpenCode 各自采用了不同的上下文压缩（compaction）方案；而结构化的交接数据包（handoff packet）则构建于这三者之上。

**压缩策略各异，但数据包模式（schema）保持一致。** Codex CLI 的 `POST /v1/responses/compact` 接口返回的是服务器端不透明的 AES 加密数据块（针对 OpenAI 模型的快速路径）；其回退方案是生成本地“交接摘要”，并作为 `_summary` 用户角色消息追加。Claude Code 在上下文（context）使用率达到 95% 时执行五阶段渐进式压缩。OpenCode 则采用基于时间戳的消息隐藏机制，外加包含 5 个标题的大语言模型（LLM）摘要。三种不同的机制，解决的是同一个需求：将压缩后保留的核心信息序列化为可移植的产物（artifact）。该数据包正是这一产物。

**全新会话交接并非上下文压缩。** 压缩旨在延长当前会话；而交接则是干净地结束当前会话并开启下一个。Hermes Issue #20372 的界定（2026 年 4 月）切中要害：当原位压缩（in-place compression）开始导致质量下降时，智能体（agent）应生成精简的交接数据，结束当前会话，并在全新的上下文中恢复工作。该数据包正是实现低成本状态过渡的关键。常见的错误是持续压缩直至输出质量崩溃；正确的做法是预留预算，尽早执行干净利落的交接。

**每个分支和主题仅保留一个活跃交接记录。** 多智能体（multi-agent）协同的失败往往源于过时的交接数据，而非模型输出质量差。务必包含 `branch`、`last_known_good_commit` 字段，以及状态为 `active | superseded | archived` 的 `status` 字段。过时的交接记录应归档；只有活跃的记录才能驱动下一次会话。这正是“交接即笔记”与“交接即状态”的本质区别。

**在上下文预算消耗至 50-75% 时收尾，而非触及上限。** 手写模式手册（CLAUDE.md + HANDOVER.md）的实践表明，在上下文预算消耗至 50-75% 时结束会话，效果远优于 95%。数据包生成器能在压缩伪影（compression artifacts）污染源状态之前干净地运行。在上下文完整时生成成本极低；而当模型已开始迷失方向时，代价则十分高昂。

## 使用方式

生产环境实践：

- **会话结束钩子（Session-end hook）。** 当用户关闭聊天窗口时，运行时（runtime）会触发该生成器。生成的数据包将保存至 `outputs/handoff/<session_id>/` 目录。
- **PR 模板。** 生成器输出的 Markdown 内容可直接作为拉取请求（PR）的描述正文。审查者无需额外打开五个文件即可掌握全貌。
- **跨智能体交接。** 使用一款产品（如 Claude Code）进行开发，随后切换至另一款产品（如 Codex）继续。该数据包充当了通用语言（lingua franca）。

该数据包体积小、结构规范且生成成本低。随着会话次数的增加，节省的成本将呈复利式累积。

## 发布

`outputs/skill-handoff-generator.md` 会生成一个针对项目工件路径（artifact paths）定制的生成器、一个在会话结束时触发运行的钩子（hook），以及供下一个智能体（agent）在启动时读取的 `handoff.json` 结构定义（schema）。

## 练习

1. 添加一个 `assumptions_to_validate` 字段，用于显式展示构建器（builder）已记录但评审者（reviewer）评分未超过 1 的所有假设。
2. 针对运行失败与运行成功的情况，采用不同的策略精简反馈摘要（feedback summary）。请论证这种不对称处理的合理性。
3. 增加一个“向人类提问”列表。请明确一个问题被纳入交接包（handoff packet）而非直接作为聊天消息发送的判定阈值是什么？
4. 确保生成器具备幂等性（idempotent）：连续运行两次应生成完全相同的交接包。为实现这一目标，哪些要素必须保持稳定？
5. 添加一个“下次会话前置条件（next session prereqs）”章节，精确列出下一个会话在执行操作前必须加载的所有工件（artifacts）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 交接包（Handoff packet） | “会话摘要” | 承载七个字段的生成工件，同时包含 Markdown 与 JSON 格式 |
| 下一步行动（Next action） | “首先要做什么” | 启动下一个会话的唯一具体步骤 |
| 反馈精简（Feedback trim） | “日志摘要” | 最近 K 条记录，以及所有非零退出状态（non-zero exit）的条目 |
| 状态报告（Status report） | “我们做了什么” | 缺少 `next_action` 字段的文档；虽具参考价值，但不构成交接包 |
| 结论指针（Verdict pointer） | “凭证” | 指向验证与评审报告的路径，用于实现可追溯性 |

## 延伸阅读

- [Anthropic，面向长时间运行智能体（Agent）的有效运行框架](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [OpenAI Agents SDK 中的上下文交接（Handoff）](https://platform.openai.com/docs/guides/agents-sdk/handoffs)
- [Codex 博客，Codex CLI 上下文压缩（Context Compaction）：架构、配置与长会话管理](https://codex.danielvaughan.com/2026/03/31/codex-cli-context-compaction-architecture/) — POST /v1/responses/compact 接口与本地回退（Fallback）机制
- [Justin3go，卸下沉重记忆：Codex、Claude Code 与 OpenCode 中的上下文压缩](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode) — 三家供应商压缩方案对比
- [JD Hodges，Claude 交接提示词：如何在跨会话中保持上下文（2026）](https://www.jdhodges.com/blog/ai-session-handoffs-keep-context-across-conversations/) — CLAUDE.md + HANDOVER.md，占用 50-75% 的上下文预算（Context Budget）
- [Mervin Praison，多智能体（Multi-Agent）编码会话中的交接管理：在不丢失连续性的前提下刷新上下文](https://mer.vin/2026/04/managing-handoffs-in-multi-agent-coding-sessions-fresh-context-without-losing-continuity/) — 基于分布式系统架构的视角
- [Hermes Issue #20372 — 当压缩风险过高时自动触发新会话交接](https://github.com/NousResearch/hermes-agent/issues/20372)
- [Hermes Issue #499 — 上下文压缩质量全面优化](https://github.com/NousResearch/hermes-agent/issues/499) — Codex CLI 中面向交接的提示词设计
- [Microsoft Agent Framework，上下文压缩](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction)
- [OpenCode，上下文管理与压缩](https://deepwiki.com/sst/opencode/2.4-context-management-and-compaction)
- [LangChain，面向智能体的上下文工程（Context Engineering）](https://www.langchain.com/blog/context-engineering-for-agents)
- 第 14 阶段 · 34 — 生成器（Generator）读取的状态文件
- 第 14 阶段 · 38 — 数据包（Packet）指向的验证结论
- 第 14 阶段 · 39 — 打包进数据包中的审查报告
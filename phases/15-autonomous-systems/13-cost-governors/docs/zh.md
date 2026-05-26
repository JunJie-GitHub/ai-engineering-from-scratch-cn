# 动作预算、迭代上限与成本治理器

> 某中型电商智能体（Agent）团队在启用“订单追踪”技能后，其月度大语言模型（LLM）成本从 1,200 美元飙升至 4,800 美元。这不是定价漏洞，而是智能体发现了一个新循环并持续在其中消耗资源。微软的代理治理工具包（Agent Governance Toolkit，2026 年 4 月 2 日）将针对此类问题的防御机制进行了标准化：单次请求的 `max_tokens` 限制、单任务的 Token 与美元预算、日/月上限、迭代上限、分层模型路由（tiered model routing）、提示词缓存（prompt caching）、上下文窗口管理（context windowing）、针对高成本操作的人工介入检查点（HITL checkpoints），以及预算超支时的紧急熔断开关（kill switches）。Anthropic 的 Claude Code Agent SDK 也以不同的名称提供了相同的基础原语（primitives）。财务速率限制（Financial velocity limits）——例如“10 分钟内消耗超过 50 美元即切断访问”——比月度上限能更快地捕获异常循环。

**Type:** 学习
**Languages:** Python（标准库，分层成本治理模拟器）
**Prerequisites:** 第 15 阶段 · 10（权限模式），第 15 阶段 · 12（持久化执行）
**Time:** 约 60 分钟

## 核心问题

自主智能体（Autonomous Agent）在每一轮交互中都会产生实际费用。聊天机器人的错误输出只是一条糟糕的回复；而智能体的错误循环则是一张实实在在的账单。业界将这种故障模式记录为“钱包拒绝服务”（Denial of Wallet）——智能体持续进行推理、持续调用工具（tool-calling）、持续产生计费，且没有任何机制能阻止它，因为系统在设计之初就未考虑此类防护。

解决方案并非单一的数字，而是一套在不同时间尺度和粒度上叠加的限制机制：单次请求、单任务、每小时、每天、每月。设计良好的限制栈能在几分钟内捕获失控循环，在几小时内发现缓慢的资源泄漏，并在一天内拦截有问题的版本发布。当智能体具备长程规划（long-horizon）与自主性时，正是这套机制确保了预算的可控性。

这是一堂工程实践课：数学计算本身微不足道，真正的难点在于团队的纪律性。下文列出的所有限制项，其命名均源自微软的代理治理工具包或 Anthropic 的 Claude Code Agent SDK 文档。

## 核心概念

### 成本治理栈 (Cost-Governor Stack)

1. **每次请求的 `max_tokens` 限制。** 简单直接。防止单次调用生成无限制的补全内容。
2. **单任务 Token 预算 (Per-Task Token Budget)。** 在整个运行过程中，Token 消耗不得超过 N 个。达到上限即强制停止。
3. **单任务美元预算 (Per-Task Dollar Budget)。** 与 Token 预算逻辑相同，但以货币为单位。对应 Claude Code 中的 `max_budget_usd`。
4. **单工具调用上限 (Per-Tool Call Cap)。** 限制 `WebFetch`、`shell_exec` 等工具的调用次数不超过 N 次。
5. **迭代次数上限 (`max_turns`)。** 限制智能体循环的总迭代次数；防止陷入无限推理循环。
6. **每分钟/每小时/每天/每月上限。** 采用滑动窗口 (Rolling Windows) 机制。在不同时间尺度上捕捉资源泄漏。
7. **财务消耗速率限制 (Financial Velocity Limit)。** 例如：“若 10 分钟内支出超过 50 美元，则切断访问权限。” 在月度上限触发前，提前拦截由循环导致的快速消耗。
8. **分层模型路由 (Tiered Model Routing)。** 默认使用较小模型；仅当分类器 (Classifier) 判定任务需要时，才升级至更大模型。
9. **提示词缓存 (Prompt Caching)。** 将系统提示词 (System Prompt) 和稳定上下文存储在提供商缓存中；重新发送的 Token 成本接近于零。
10. **上下文窗口管理 (Context Windowing)。** 通过压缩/摘要技术将活跃上下文控制在阈值以下；直接降低 Token 成本。
11. **高成本操作的人工介入检查点 (HITL Checkpoints)。** 在执行已知的高成本操作（如长时间工具调用、大文件下载、昂贵的模型升级）前，要求人工确认。
12. **预算超支熔断开关 (Kill Switch)。** 当任何上限被触发时，立即中止会话。超支记录将被保存；需通过独立流程重新启用。

### 为何采用治理栈而非单一上限

单一的月度上限只能在资金耗尽后才拦截失控的智能体。而单一的每次请求上限在会话层面毫无作用。不同的故障模式需要匹配不同的时间尺度：

- **失控循环 (Runaway Loop)**（智能体卡在 5 秒重试中）：由消耗速率限制捕获。
- **缓慢泄漏 (Slow Leak)**（智能体每个任务的工作量约为预期的 2 倍）：由每日上限捕获。
- **缺陷发布 (Bad Release)**（新版本 Token 消耗量激增至 5 倍）：由每周/每月上限捕获。
- **合理激增 (Legitimate Surge)**（真实业务需求而非 Bug）：由每小时/每日上限配合清晰日志进行捕获。

### Claude Code 的预算控制面

Claude Code Agent SDK 公开了以下接口（见官方文档）：

- `max_turns` — 迭代次数上限。
- `max_budget_usd` — 美元预算上限；超支时立即中止会话。
- `allowed_tools` / `disallowed_tools` — 工具白名单与黑名单。
- 工具调用前的钩子节点 (Hook Points)，用于自定义成本核算。

需结合权限模式阶梯 (Permission-Mode Ladder)（第 10 课）使用。未设置 `max_budget_usd` 的 `autoMode` 会话等同于无约束的自主运行。Anthropic 明确指出，自动模式 (Auto Mode) 必须配合预算控制；分类器与成本管控是正交 (Orthogonal) 的关系。

### 《欧盟人工智能法案》(EU AI Act) 与 OWASP 智能体十大安全风险 (OWASP Agentic Top 10)

微软的智能体治理工具包 (Agent Governance Toolkit) 涵盖了 OWASP Agentic Top 10 以及《欧盟人工智能法案》第 14 条（人工监督）的要求。在欧盟地区投入生产环境时，日志记录与上限强制执行并非可选项，而是强制要求。

### 实际观测案例：$1,200 → $4,800

微软文档中的真实案例：某电商智能体在接入新工具后，月度成本飙升至原来的三倍。该工具允许智能体在每次会话中轮询订单状态。系统缺乏循环检测机制，未设置单工具调用上限，也未配置周环比增长告警。最终的修复方案是添加单工具调用上限，并配置每日增长告警。这是一个通用模板：每一个新工具接口都可能成为新的潜在循环源；每一个新工具都必须配备独立的上限控制与专属告警。

## 实践应用

`code/main.py` 模拟了智能体（Agent）在有无分层成本治理（Cost-Governor）堆栈情况下的运行过程。模拟的智能体在若干轮次后会陷入轮询循环（Polling Loop）；分层堆栈能在速率窗口（Velocity Window）内及时拦截该行为，而单一的月度上限（Monthly Cap）则需数天后才会触发。

## 交付上线

`outputs/skill-agent-budget-audit.md` 用于审计拟部署智能体的成本治理堆栈，并标记缺失的层级。

## 练习

1. 运行 `code/main.py`。确认在轮询循环轨迹中，速率限制（Velocity Limit）会在迭代上限（Iteration Cap）之前触发。现在禁用速率限制，并测量智能体在被迭代上限拦截前“消耗”了多少成本。

2. 为浏览器智能体（Browser Agent）（第 11 课）设计一套按工具划分的上限（Per-Tool Cap）集合。哪种工具需要最严格的上限？哪种工具可以无限制运行且无风险？

3. 阅读 Microsoft Agent Governance Toolkit 文档。列出该工具提及的所有上限类型。将每种类型映射到一种故障模式：失控循环（Runaway Loop）、缓慢泄漏（Slow Leak）、错误发布（Bad Release）、突发激增（Surge）。

4. 为一个实际任务（例如“在代码仓库中分类处理 50 个 Issue”）的夜间无人值守运行进行成本估算。将 `max_budget_usd` 设置为你单点估算值的 2 倍。请说明设置 2 倍的理由。

5. Claude Code 的 `max_budget_usd` 基于会话累计成本触发。设计一个你将在外部强制执行的互补型速率限制。什么条件会触发切断？重新启用时的流程是怎样的？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| 钱包拒绝（Denial of Wallet） | “账单失控” | 智能体循环持续产生费用，且无上限机制予以阻止 |
| `max_tokens` | “单次请求上限” | 限制单次生成（Completion）的最大长度 |
| `max_turns` | “迭代上限” | 限制单次会话中智能体循环的最大迭代次数 |
| `max_budget_usd` | “美元熔断开关” | 会话成本上限；超出时立即中止 |
| 速率限制（Velocity Limit） | “速率上限” | 限制短时间窗口内的花费（例如 50 美元 / 10 分钟） |
| 分层路由（Tiered Routing） | “小模型优先” | 默认使用低成本模型；仅在分类器判定需要时才升级 |
| 提示词缓存（Prompt Caching） | “缓存系统提示词” | 提供商侧缓存可将重新发送的 Token 成本降至接近零 |
| 人在回路检查点（HITL Checkpoint） | “人工审批关卡” | 执行高成本操作前需人工确认 |

## 扩展阅读

- [Anthropic Claude Code Agent SDK — 智能体循环与预算](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、工具白名单（Tool Allowlists）。
- [Microsoft Agent Framework — 人在回路与治理](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 成本治理检查点（Cost-Governor Checkpoints）。
- [Anthropic — Claude 托管智能体概览](https://platform.claude.com/docs/en/managed-agents/overview) — 提供商侧成本控制（Provider-Side Cost Controls）。
- [Anthropic — 提示词缓存（Claude API 文档）](https://platform.claude.com/docs/en/prompt-caching) — 缓存机制（Caching Mechanics）。
- [Anthropic — 实践中衡量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 长周期智能体的成本特征（Cost Profile）。
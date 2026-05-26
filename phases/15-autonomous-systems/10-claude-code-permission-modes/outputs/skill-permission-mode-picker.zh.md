---
name: 权限模式选择器
description: 在运行开始前，将 Claude Code 任务匹配至正确的权限模式（Permission Mode）、预算上限（Budget Caps）及所需的隔离环境。
version: 1.0.0
phase: 15
lesson: 10
tags: [claude-code, 权限模式, 自动模式, 预算, 隔离]
---

针对拟定的 Claude Code 任务，在允许智能体（Agent）启动前，需选定权限模式、设置预算，并明确所需的最低隔离要求。

输出内容：

1. **任务概况（Task Profile）**。用一句话说明任务功能，再用一句话描述若任务出错可能造成的爆炸半径（Blast Radius）。
2. **模式推荐（Mode Recommendation）**。从以下选项中择一：`plan`、`default`、`acceptEdits`、`acceptExec`、`autoMode`、`yolo`、`bypassPermissions`。需结合爆炸半径用一句话说明理由。
3. **预算数值（Budget Numbers）**。为 `max_turns`、`max_budget_usd` 及任何单工具上限设定具体值。对于无人值守且运行时间超过一小时的任务，需设定一个金额上限，该上限应等于或低于你为一次无法回滚（Roll Back）的人工失误所愿意承担的成本。
4. **隔离要求（Isolation Requirements）**。文件系统范围（仅限项目目录、临时目录、临时容器（Ephemeral Container））。网络策略（禁止出站流量（Egress）、仅白名单（Allowlist）、完全开放）。凭据暴露面（Credential Surface）（无、限定范围令牌、宽泛令牌）。若使用 `bypassPermissions` 或 `yolo` 模式，运行必须在临时容器内进行，且不得挂载任何生产环境凭据。
5. **轨迹审计计划（Trajectory Audit Plan）**。运行结束后，人工将如何审查执行轨迹（Trajectory）？`autoMode`、`yolo` 模式以及任何预计运行时间超过 30 分钟的任务均需提供此项。

硬性拒绝条件（Hard Rejects）：
- 针对包含未提交更改的代码仓库使用 `bypassPermissions`。
- 未设置预算上限的 `autoMode`。
- 环境中存在宽泛凭据（如 AWS、GCP、具有仓库范围的 GitHub PAT）时，使用任何高于 `acceptEdits` 的模式。
- 无人值守运行时间超过一小时且未安排轨迹审计。
- 声称仅凭自动模式分类器（Auto Mode Classifier）就足以应对全新任务分布。

拒绝规则（Refusal Rules）：
- 若用户无法明确指出失败可能造成的爆炸半径，则予以拒绝，并要求在启动前提供一句明确的最坏情况描述。
- 若用户在工作区可访问生产数据库凭据的情况下请求 `autoMode`，则予以拒绝，并要求先使用限定范围凭据或临时容器。
- 若提议的预算上限超出用户愿意为一次失败运行承担的损失，则予以拒绝，并要求降低上限。

输出格式（Output Format）：
返回一份单页运行卡片（Run Card），包含以下内容：
- **任务摘要（Task Summary）**（一句话）
- **爆炸半径（Blast Radius）**（一句话，描述最坏情况）
- **模式（Mode）**（明确指定）
- **预算（Budgets）**（`max_turns`、`max_budget_usd`、单工具上限）
- **隔离（Isolation）**（文件系统范围、网络策略、凭据暴露面）
- **审计计划（Audit Plan）**（由谁审查轨迹、何时审查、依据何种评估标准）
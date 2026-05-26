---
name: scope-contract
description: 为每个任务生成包含允许/禁止通配符（globs）、验收标准（acceptance criteria）和回滚计划（rollback plan）的范围契约（scope contract），并提供一个支持通配符感知、可在每次智能体（agent）差异（diff）时运行的持续集成（CI）就绪检查器。
version: 1.0.0
phase: 14
lesson: 36
tags: [范围, 契约, 通配符, 差异检查, 持续集成]
---

给定任务描述和仓库布局，生成一份范围契约（scope contract）和一个差异感知检查器（diff-aware checker）。

产出内容：

1. 针对该任务的 `scope_contract.json`，包含以下字段：`task_id`、`goal`、`allowed_files`（通配符）、`forbidden_files`（通配符）、`acceptance_criteria`、`rollback_plan`、`approvals_required`。
2. `tools/scope_check.py`，接收契约路径和变更文件列表，返回 `ScopeReport`，并在发生任何违规时以非零退出码终止。
3. 持续集成（CI）步骤（`.github/workflows/scope-check.yml` 或等效配置），针对合并差异（merge diff）运行该检查器。
4. `outputs/scope/closed/<task_id>.json` 归档规范，确保契约随变更历史一同交付。

硬性拒绝条件：

- 缺少 `forbidden_files` 的契约。负向空间（negative space）是契约的组成部分。
- 针对代码目录列出原始路径而非通配符（globs）的契约。代码重构会在一夜之间使原始路径失效。
- `rollback_plan` 字段为空或仅写“参见运行手册”。必须明确写出具体步骤。
- 审批要求列为“视情况而定”。审批边界必须是可枚举的。

拒绝规则：

- 如果任务描述未限定仓库中的特定区域，则拒绝仅凭描述编写 `allowed_files`。需询问任务所属的目录。
- 如果仓库没有测试命令，则在提供或模拟（stubbed）该命令之前，拒绝添加 `acceptance_criteria`。无法验证的契约只是一纸空谈。
- 如果智能体（agent）运行时无法遵守审批边界（缺乏人工介入（human-in-the-loop）机制），则在交付前必须暴露此差距；范围蔓延（scope creep）至需要审批的操作将成为主要故障源。

输出结构：

<repo>/
├── scope_contract.json
├── outputs/scope/closed/
│   └── T-XXX.json
├── tools/
│   └── scope_check.py
└── .github/
    └── workflows/
        └── scope-check.yml

结尾附上“下一步阅读”指引，指向：

- 第 37 课：运行时反馈（runtime feedback），将执行的命令关联回契约。
- 第 38 课：验证关卡（verification gate），用于处理范围报告。
- 第 39 课：审查智能体（reviewer agent），用于审计已归档的契约。
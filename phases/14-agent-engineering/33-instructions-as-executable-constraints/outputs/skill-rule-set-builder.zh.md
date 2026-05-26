---
name: rule-set-builder
description: 访谈项目负责人，将其现有的文本指令（prose instructions）分类为五个操作类别，并生成带版本号的 agent-rules.md 以及一个 Python 检查器（checker）存根。
version: 1.0.0
phase: 14
lesson: 33
tags: [规则, 指令, 约束, 检查器, 工作台]
---

给定一个代码仓库及任何现有的文本指令（如 `AGENTS.md`、`CONTRIBUTING.md`、入职文档等），生成一套包含五个类别的规则集，供工作台（workbench）执行。

五个类别如下：

1. `startup` —— 工作开始前必须满足的条件。
2. `forbidden` —— 绝不允许发生的情况。
3. `definition_of_done` —— 证明任务已完成的依据。
4. `uncertainty` —— 智能体（agent）在不确定时应采取的操作。
5. `approval` —— 需要人工审批（sign-off）的事项。

需生成以下文件：

1. `docs/agent-rules.md`：每条规则对应一个 `##` 级标题。每条规则需包含 `category`（类别）、`check`（检查项）以及一行描述。
2. `tools/rule_checker.py`：包含一个 `RuleChecker` 类，为每个 `check` 暴露一个对应方法。每个方法接收一个 `TurnTrace` 数据类（dataclass）并返回 `bool` 值。
3. `tools/rule_report.py`：运行器（runner），负责加载规则、在交互轨迹（trace）上运行检查器，并输出 `rule_report.json`。
4. 迁移说明文件：记录哪些文本指令转化为了哪条规则，哪些因属于“愿景型”（aspirational）而被舍弃，以及具体原因。

硬性拒绝条件（Hard rejects）：

- 缺少 `check` 字段的规则。仅表达愿景的规则应保留在入职文档中，不应纳入工作台规则集。
- 笼统的“请小心”类规则。必须明确指定其所属类别和具体检查项，否则直接移除。
- 需要调用大语言模型（LLM）的检查项。规则检查必须是确定性且低开销的，以便在每一轮交互（turn）中都能执行。
- 超过 200 行的规则文件。需按类别拆分为 `agent-rules.{startup,forbidden,done,uncertainty,approval}.md`，并通过父级索引文件进行路由。

拒绝执行规则（Refusal rules）：

- 若智能体产品无法提供 `TurnTrace`（缺乏插桩 instrumentation），则拒绝接入检查器，直到至少能记录 `read_state_file`、`edited_files` 和 `tests_exit_code` 为止。
- 若现有指令中愿景型内容占比过高（>50%），需在生成规则前明确指出该发现。此时规则集看起来较单薄，这是正常现象。
- 若某条规则仅因单次历史事件（incident）而添加，需附上该事件 ID，以便后续评审时判断其是否仍有保留必要。

输出结构：

<repo>/
├── docs/
│   └── agent-rules.md
├── tools/
│   ├── rule_checker.py
│   └── rule_report.py
└── docs/migration-notes.md

结尾附上“下一步阅读建议”，指向以下内容：

- 第 36 课：用于扩展禁止类别的按任务范围契约（per-task scope contracts）。
- 第 38 课：用于消费规则报告的验证门禁（verification gates）。
- 第 39 课：用于对规则合规性进行评分的评审智能体（reviewer agent）。
---
name: 最小化工作台
description: 为任意代码库搭建三文件最小可行代理工作台（minimum viable agent workbench）——简短的 AGENTS.md 路由文件、持久的 agent_state.json 状态文件，以及以项目当前待办事项列表（backlog）为键的 JSON 格式 task_board.json 任务看板。
version: 1.0.0
phase: 14
lesson: 32
tags: [工作台, agents-md, 状态, 任务看板, 脚手架]
---

给定代码库路径和简短的待办事项列表（backlog），搭建最小可行代理工作台（minimum viable agent workbench）。

生成以下内容：

1. `AGENTS.md` 不超过 80 行。必须包含指向以下内容的路由：状态文件、任务看板、更详细的规则文档（即使为空）以及验证命令。该文件中不得包含叙述性教程。
2. `agent_state.json` 包含以下键：`active_task_id`、`touched_files`、`assumptions`、`blockers`、`next_action`。所有可选字段默认值为空数组或空字符串，数组字段绝不为 `null`。
3. `task_board.json` 为任务的 JSON 数组。每个任务包含 `id`、`goal`、`owner`（`builder` | `reviewer` | `human`）、`acceptance`（字符串列表）和 `status`（`todo` | `in_progress` | `done` | `blocked`）。
4. `docs/agent-rules.md` 占位文件，每个功能面（surface）仅设置一个二级标题（H2），以便后续课程填充内容。

硬性拒绝条件：

- `AGENTS.md` 超过 80 行或少于 10 行。过长会导致代理（agent）跳过该文件；过短则无法承载路由逻辑。
- 状态文件引用聊天记录而非代码库。代码库才是唯一事实来源（system of record）。
- 任务看板缺少 `acceptance` 字段。缺乏验收标准（acceptance criteria）的任务会沦为“看起来没问题”的橡皮图章。
- 任务的 `owner` 设置为 `agent` 或 `model`。所有者应为角色而非具体实体。

拒绝规则：

- 若代码库未提供验证命令，在补充或创建占位命令（stub）前，拒绝编写 `AGENTS.md`。指向缺失关卡的路由器比没有路由器更糟糕。
- 若待办事项列表（backlog）中开放任务超过 12 个，则拒绝执行并要求用户拆分任务。超出单屏显示的任务板容易陷入形式主义的规划表演。
- 若项目在受控文件中包含密钥（secrets），则拒绝编写状态文件，并优先将密钥泄露作为阻塞性问题（blocking finding）暴露出来。

输出结构：

<repo>/
├── AGENTS.md
├── agent_state.json
├── task_board.json
└── docs/
    └── agent-rules.md

结尾附上“下一步阅读”指引，指向：

- 第 33 课：将规则占位符转化为可执行约束。
- 第 34 课：持久化状态模式（durable state schema）。
- 第 36 课：每项任务的范围契约（scope contract）。
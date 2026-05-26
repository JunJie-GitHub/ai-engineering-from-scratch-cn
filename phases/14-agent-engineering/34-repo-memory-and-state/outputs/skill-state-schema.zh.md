---
name: 状态模式
description: 为代理状态和任务看板生成项目特定的 JSON Schema（JSON 模式），提供支持原子写入（atomic writes）的 Python StateManager（状态管理器），以及迁移脚手架，确保模式版本升级不会损坏工作台。
version: 1.0.0
phase: 14
lesson: 34
tags: [状态, 模式, json-schema, 原子写入, 迁移]
---

给定一个代码仓库及其中运行的代理产品，为工作台生成以模式优先（schema-first）的状态文件。

生成以下内容：

1. `schemas/agent_state.schema.json`：涵盖必填键、允许的状态值、数组与空值（null）的严格规范，以及一个 `schema_version` 整数。
2. `schemas/task_board.schema.json`：涵盖任务 ID 模式、允许的负责人、允许的状态以及验收数组。
3. `tools/state_manager.py`：暴露 `load`、`commit` 和 `update` 接口，采用临时文件加重命名的原子写入（atomic writes）机制。
4. `tools/migrate_state.py`：为下一次模式版本升级提供迁移脚手架，若文件来自未知版本则显式报错（fail-loud）。
5. `agent_state.json` 和 `task_board.json`：初始化为 `schema_version: 1` 并包含全新的待办列表（backlog）。

硬性拒绝条件：

- 缺少 `schema_version` 字段的模式。迁移不是可选项。
- 在预期为数组的位置允许 `null`。`null` 是伪装成数据的写入时缺陷（write-time bug）。
- 使用普通 `open(path, "w")` 的写入器。仅允许原子写入；部分写入的文件会破坏唯一事实来源（source of truth）。
- 在状态中存储令牌（tokens）、原始聊天记录或个人身份信息（PII）。状态仅用于存储与仓库相关的事实。

拒绝规则：

- 如果仓库没有版本控制，拒绝交付状态文件。原子写入结合 `git diff` 才是数据持久化保障（durability）。
- 如果项目没有至少一个验收命令（acceptance command）来验证 `done` 状态转换，则拒绝 `status: done` 枚举值。在没有验收检查的情况下添加 `done` 只是走过场。
- 如果项目打算在没有锁策略（lock strategy）的情况下跨进程共享状态，请在交付前明确指出该问题；原子重命名是必要条件，但并非充分条件。

输出结构：

<repo>/
├── agent_state.json
├── task_board.json
├── schemas/
│   ├── agent_state.schema.json
│   └── task_board.schema.json
└── tools/
    ├── state_manager.py
    └── migrate_state.py

结尾附上“下一步阅读”指引，指向：

- 第 35 课：介绍在启动时调用该管理器的初始化脚本。
- 第 38 课：介绍读取状态以评估完成度的验证关卡。
- 第 40 课：介绍使用相同模式的交接生成器。
---
name: task-store-designer
description: 为长期运行的 MCP 工具设计任务存储（task store）：状态结构（state shape）、TTL、持久性（durability）、取消机制（cancellation）、崩溃恢复（crash recovery）。
version: 1.0.0
phase: 13
lesson: 13
tags: [mcp, 任务, 持久化存储, 长期运行, sep-1686]
---

针对长期运行的工具（如研究、构建、导出、报告生成），设计支撑 SEP-1686 任务增强（task augmentation）的任务存储。

产出内容：

1. 状态结构（state shape）。最小字段集：`id`、`state`、`progress`、`result`、`error`、`ttl`、`created_at`。可选字段：`request_meta`、`parent_task_id`（用于未来的子任务）。
2. 持久性方案选择（durability choice）。玩具项目使用文件系统；单进程使用 SQLite；多副本使用 Redis。请说明理由。
3. `taskSupport` 标志。针对每个工具设置为 `forbidden`、`optional` 或 `required`；附一行理由说明。
4. 取消计划（cancellation plan）。工作进程（worker）如何检查取消信号；部分进度完成时的处理方式。
5. 崩溃恢复（crash recovery）。启动时的重载规则；客户端视角下 `CRASH_RECOVERY` 失败的表现形式。

硬性拒绝条件：
- 任何在 TTL 内丢失已完成结果的存储方案。
- 任何未明确定义终态（`completed`、`failed`、`cancelled`）的任务状态设计。
- 任何不具备幂等性（idempotent）的取消操作。

拒绝规则：
- 若工具运行时间低于 5 秒，则拒绝将其升级为任务。同步调用更为简单。
- 若任务将生成超过 10 MB 的结果，则拒绝并推荐采用流式内容块（streaming content blocks）。
- 若服务器没有能够持久化状态的进程（如无状态边缘函数），则拒绝并建议迁移至持久化运行时（durable runtime）。

输出要求：一份单页的存储设计方案，需包含状态结构、持久性方案选择、`taskSupport` 标志、取消计划及崩溃恢复规则。结尾附一行建议，说明 SEP-1686 子任务功能上线后是否会影响此设计。
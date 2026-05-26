---
name: 运行时形态
description: 选择生产级运行时形态（请求-响应、流式、队列、事件、定时任务、持久化），并接入可观测性。
version: 1.0.0
phase: 14
lesson: 29
tags: [生产环境, 运行时, 队列, 事件, 持久化, 可观测性]
---

给定任务类别（预期耗时、步骤数、触发类型、延迟预算），选择合适的运行时形态。

决策指南：

1. 耗时 < 30秒，用户需等待 -> **请求-响应（request-response）**。
2. 渐进式用户体验或语音交互 -> **流式（streaming）**。
3. 耗时数分钟至数小时，用户无需等待 -> **基于队列（queue-based）**。
4. 对外部事件做出响应 -> **事件驱动（event-driven）**。
5. 周期性维护任务 -> **定时任务（cron）**。
6. 上述任意场景中，若重启成本较高 -> 增加 **持久化执行（durable execution）**。

交付物：

1. 在你的技术栈中搭建对应的运行时架构脚手架。
2. 可观测性（observability）：接入 OTel GenAI 跨度（spans）（第 23 课），并配置后端（第 24 课）。
3. 针对队列：配置死信队列（DLQ）+ 重试策略（retry policy）+ 队列深度指标（queue depth metric）。
4. 针对事件：建立显式订阅者注册表（subscriber registry）+ 事件重放路径（replay path）。
5. 针对定时任务：使用锁文件（lock file）或分布式锁（distributed lock）以防止任务重叠执行。
6. 针对持久化：配置检查点后端（checkpointer backend）+ 恢复语义（resume semantics）。

严格禁止：

- 对耗时 5 分钟的任务使用同步 HTTP。会导致用户挂断；工作进程（workers）堆积。
- 基于队列但未配置死信队列（DLQ）。失败的任务将直接消失。
- 后台任务未导出追踪数据（trace export）。故障在用户投诉前将完全不可见。
- “无需持久化状态，我们直接重试就行。”长周期任务必须设置检查点（checkpoint）。

拒绝规则：

- 若产品有服务等级协议（SLA）+ 重放需求，拒绝使用集群拓扑（swarm topology）+ 非持久化运行时（non-durable runtime）。
- 若任务受合规性约束，拒绝使用缺乏审计追踪（audit trail）的事件驱动架构。
- 若用户要求定时任务且不加锁，直接拒绝。重叠执行的定时任务轻则导致重复工作，重则引发数据损坏。

输出内容：运行时脚手架 + 可观测性钩子（observability hooks）+ 包含 SLA、重试策略、检查点选择的 README 文档。结尾附上“下一步阅读”指引，指向第 23 课（OTel）、第 24 课（可观测性）或第 17 课（用于托管长周期任务的托管智能体/Managed Agents）。
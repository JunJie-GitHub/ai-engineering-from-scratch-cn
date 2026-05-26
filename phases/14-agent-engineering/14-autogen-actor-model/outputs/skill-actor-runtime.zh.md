---
name: actor-runtime
description: 构建一个符合 AutoGen v0.4 架构的 Actor 运行时（Actor Runtime），具备私有状态、每个 Actor 独立收件箱、仅基于消息的进程间通信（IPC）、故障隔离（Fault Isolation）以及死信队列（Dead-Letter Queue）。
version: 1.0.0
phase: 14
lesson: 14
tags: [autogen, actor模型, 消息传递, 故障隔离, 死信队列]
---

针对多智能体任务，生成所需的 Actor 运行时及智能体 Actor。

产出内容：

1. 一个 `Message` 类型，包含 `sender`、`recipient`、`topic`、`body`、`mid` 字段。
2. 一个 `Actor` 基类，包含 `receive(message, runtime)` 方法。Actor 的状态必须为私有。
3. 一个 `Runtime`，包含共享队列、`send()`、`run_until_idle()` 以及死信队列（Dead-Letter Queue, DLQ）。处理器中的异常应路由至 DLQ，切勿向外传播。
4. 一个拓扑辅助组件：轮询调度（RoundRobin，固定轮换）、选择器（Selector，由大语言模型 LLM 决定下一个）或自定义广播。
5. 每条消息的可观测性钩子（Observability Hooks）：按照第 23 课的要求，发射包含 `gen_ai.agent.name` 和 `gen_ai.operation.name` 的 OpenTelemetry（OTel）跨度（Spans）。

严格拒绝项：

- 阻塞发送方直至接收方返回的同步消息传递机制。这是 v0.2 的模型，会破坏故障隔离。
- 跨 Actor 的共享可变状态。Actor 只能通过消息读取状态，或完全不读取。
- 会传播处理器异常的运行时。失败应归入 DLQ，确保其他 Actor 继续运行。

拒绝规则：

- 若任务仅涉及两个 Actor 且为固定的来回交互，则拒绝采用 Actor 架构，并建议改用提示词链（Prompt Chain，第 12 课）。当 Actor 数量 >=3 或存在异步并发时，引入 Actor 架构的额外开销才是合理的。
- 若用户出于“便于调试”的目的要求“同步模式”，则予以拒绝。建议改用日志记录与链路追踪（Tracing，第 23 课）。
- 若业务场景严格为单一专家模型的请求/响应模式，则建议采用路由机制（Routing，第 12 课），而非组建 Actor 团队。

输出文件：`message.py`、`actor.py`、`runtime.py`、`teams.py` 以及 `README.md`。其中 `README.md` 需说明 DLQ 策略、拓扑选择方案以及 OTel 跨度（Spans）的接入方式。文末需附上“下一步阅读建议”：若涉及 Actor 协商，指向第 25 课（多智能体辩论）；若需链路追踪，指向第 23 课（OTel）；若追求面向未来的运行时架构，则指向 Microsoft Agent Framework。
# 构建 MCP (Model Context Protocol) 客户端——发现、调用与会话管理

> 大多数关于 MCP 的资料都侧重于服务器端教程，而对客户端部分往往一笔带过。真正的复杂编排逻辑其实都集中在客户端代码中：包括进程派生 (Process Spawning)、能力协商 (Capability Negotiation)、跨多个服务器的工具列表合并 (Tool List Merging)、采样回调 (Sampling Callbacks)、重连机制 (Reconnection) 以及命名空间冲突解决 (Namespace Collision Resolution)。本课程将构建一个多服务器客户端，将三个不同的 MCP 服务器整合为一个扁平的工具命名空间，供模型调用。

**类型：** 构建
**语言：** Python（标准库，多服务器 MCP 客户端）
**前置条件：** 第 13 阶段 · 07（构建 MCP 服务器）
**时长：** 约 75 分钟

## 学习目标

- 将 MCP 服务器作为子进程 (Child Process) 启动，完成 `initialize` 流程，并发送 `notifications/initialized` 通知。
- 维护每个服务器的会话状态 (Session State)（能力、工具列表、最后接收到的通知 ID）。
- 将多个服务器的工具列表合并到单一命名空间中，并处理冲突。
- 将工具调用路由至对应的所属服务器，并重组响应结果。

## 问题背景

在实际的 AI 智能体宿主环境 (Agent Host)（如 Claude Desktop、Cursor、Goose、Gemini CLI）中，通常会同时加载多个 MCP 服务器。用户可能会同时运行文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端的职责包括：

1. 启动各个服务器进程。
2. 独立与各服务器完成握手。
3. 分别调用各服务器的 `tools/list` 接口，并将结果扁平化。
4. 当模型发出 `notes_search` 调用时，在合并后的命名空间中查找该工具，并将其路由至正确的服务器。
5. 在不阻塞主流程的情况下，处理来自任意服务器的通知（如 `tools/list_changed`）。
6. 在传输层故障 (Transport Failure) 时自动重连。

亲手实现上述所有逻辑，正是区分“玩具级原型”与“生产级可用服务”的关键。官方 SDK 虽然封装了这些功能，但你必须在脑海中建立起清晰的底层运作模型。

## 核心概念

### 子进程生成 (Child-process spawning)

使用 `subprocess.Popen` 并配置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式进行逐行读取。每个服务器对应一个独立进程；客户端为每个服务器维护一个 `Popen` 句柄。

### 单服务器会话状态 (Per-server session state)

每个服务器对应一个 `Session` 对象，用于保存以下信息：

- `process` — `Popen` 句柄。
- `capabilities` — 服务器在 `initialize` 阶段声明的能力。
- `tools` — 最后一次 `tools/list` 调用的结果。
- `pending` — 将请求 ID 映射到等待响应的异步占位符 (Promise/Future)。

请求本质上是异步的；当向服务器 A 发送 `tools/call` 时，即使服务器 B 正在处理调用，也不应造成阻塞。可采用带队列的线程模型或使用 `asyncio`。

### 合并命名空间 (Merged namespace)

当客户端汇总所有工具列表时，可能会出现名称冲突。例如，两个服务器可能都暴露了 `search` 工具。客户端有三种处理策略：

1. **按服务器名称添加前缀。** `notes/search`、`files/search`。语义清晰但不够美观。
2. **静默优先覆盖（先到先得）。** 后加载服务器的 `search` 会覆盖先加载的。存在风险，且会掩盖冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器并通知用户。对于安全敏感型主机最为稳妥。

Claude Desktop 采用按服务器添加前缀的方式。Cursor 采用冲突拒绝策略并返回明确的错误提示。VS Code MCP 同样采用按服务器添加前缀的方式。

### 路由分发 (Routing)

合并完成后，客户端会维护一个分发表，将 `tool_name -> session` 进行映射。模型按名称发起调用；客户端根据名称找到对应的会话，将 `tools/call` 消息写入该服务器的标准输入（stdin），随后等待响应。

### 采样回调 (Sampling callback)

如果服务器在 `initialize` 阶段声明了 `sampling` 能力，它可能会发送 `sampling/createMessage` 请求，要求客户端调用其大语言模型 (LLM)。客户端必须：

1. 在该采样请求完成前，阻塞对该服务器的其他请求；若实现支持并发，也可采用流水线处理 (pipeline)。
2. 调用其 LLM 提供商接口。
3. 将响应结果返回给服务器。

第 11 课将完整讲解采样流程。本课仅保留占位实现以保证结构完整。

### 通知处理 (Notification handling)

收到 `notifications/tools/list_changed` 表示需重新调用 `tools/list`。收到 `notifications/resources/updated` 表示若该资源正在使用中，需重新读取。通知消息不应产生响应——切勿尝试对其进行确认 (ack)。

客户端常见缺陷：当通知消息滞留在数据流中时，读取循环因等待 `tools/call` 响应而被阻塞。应使用后台读取线程将每条消息推入队列；主线程负责从队列中取出消息并进行分发。

### 重连机制 (Reconnection)

传输层可能发生故障：服务器崩溃、操作系统终止进程或标准输入输出 (stdio) 管道断裂。客户端通过检测标准输出（stdout）的 EOF 信号判定会话已失效。可选策略包括：

- 静默重启服务器并重新握手。适用于纯只读型服务器。
- 向用户暴露故障信息。适用于具有用户可见会话的有状态服务器。

第 13 阶段 · 09 节将详细讲解 Streamable HTTP 的重连语义；标准输入输出 (stdio) 模式则更为简单。

### 心跳保活 (Keepalive) 与会话 ID (Session ID)

Streamable HTTP 使用 `Mcp-Session-Id` 请求头。标准输入输出 (stdio) 模式没有会话 ID——进程身份本身即代表会话。心跳保活 (Keepalive pings) 是可选的；stdio 管道在空闲状态下不会断开。

## 使用它

`code/main.py` 会生成三个模拟的模型上下文协议 (Model Context Protocol) 服务器作为子进程，与每个服务器完成握手，合并它们的工具列表，并将工具调用路由至对应的服务器。这些“服务器”实际上是运行着简易响应器的其他 Python 进程（不包含真实的大语言模型 (Large Language Model)）。运行该脚本可观察到：

- 三次初始化过程，各自具备独立的能力集。
- 三个 `tools/list` 的结果被合并到一个包含 7 个工具的命名空间中。
- 基于工具名称的路由决策。
- 通过命名空间前缀机制避免的命名冲突。

值得关注的实现细节：

- `Session` 数据类 (dataclass) 清晰地维护了每个服务器的状态。
- 后台读取线程会逐行从标准输出 (stdout) 中出队数据，且不会阻塞主线程。
- 路由分发表 (dispatch table) 仅是一个简单的 `dict[str, Session]`。
- 冲突处理机制是显式的：当两个服务器声明了相同的名称时，后声明的服务器会被添加前缀进行重命名。

## 交付成果

本课程的产出文件为 `outputs/skill-mcp-client-harness.md`。给定一个声明式的 MCP 服务器列表（包含名称、命令和参数），该技能 (skill) 会生成一个控制框架 (harness)，用于启动这些服务器、合并工具列表，并交付一个具备冲突解决功能的路由函数。

## 练习

1. 运行 `code/main.py` 并观察服务器启动日志。使用 `SIGTERM` 信号终止其中一个模拟服务器进程，观察客户端如何检测到文件结束符 (EOF) 并将该会话标记为已失效。

2. 实现命名空间前缀功能。当两个服务器都暴露 `search` 工具时，将第二个重命名为 `<server>/search`。更新路由分发表并验证工具调用是否能正确路由。

3. 为服务器重启添加连接池风格的退避机制：连续失败时采用指数退避 (exponential backoff)，上限为 30 秒，在连续失败三次后向用户发送通知。

4. 设计一个支持 100 个并发 MCP 服务器的客户端架构。什么数据结构可以替代简单的路由分发字典？（提示：使用前缀树 (trie) 进行命名空间前缀管理，并结合每个服务器的工具数量指标。）

5. 将客户端移植至官方 MCP Python SDK。该 SDK 封装了 `stdio_client` 和 `ClientSession`。在保留多服务器路由功能的前提下，代码量应从约 200 行缩减至约 40 行。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| MCP 客户端 (MCP Client) | “智能体宿主” | 负责启动服务器并协调工具调用的进程 |
| 会话 (Session) | “单服务器状态” | 维护能力声明、工具列表及待处理请求的状态记录 |
| 合并命名空间 (Merged Namespace) | “单一工具列表” | 汇总所有活跃服务器工具名称的扁平化集合 |
| 命名空间冲突 (Namespace Collision) | “两个服务器同名工具” | 客户端需为冲突项添加前缀、予以拒绝或按先到先得原则处理 |
| 路由 (Routing) | “这个调用归谁处理？” | 根据工具名称将其分发至所属服务器的机制 |
| 后台读取器 (Background Reader) | “非阻塞标准输出” | 负责将服务器标准输出（stdout）持续消费并排入队列的线程或任务 |
| 采样回调 (Sampling Callback) | “大语言模型即服务” | 客户端用于响应服务器 `sampling/createMessage` 请求的处理程序 |
| `notifications/*_changed` | “核心原语变更” | 提示客户端必须重新发现或重新读取相关数据的信号 |
| 重连策略 (Reconnection Policy) | “服务器宕机时” | 传输层中断时的重启语义与恢复逻辑 |
| 标准输入输出会话 (Stdio Session) | “进程即会话” | 无独立会话 ID；子进程的生命周期即代表该会话的生命周期 |

## 进一步阅读

- [Model Context Protocol — 客户端规范](https://modelcontextprotocol.io/specification/2025-11-25/client) — 权威的客户端行为规范
- [MCP — 客户端快速入门指南](https://modelcontextprotocol.io/quickstart/client) — 基于 Python SDK 的 Hello World 客户端教程
- [MCP Python SDK — 客户端模块](https://github.com/modelcontextprotocol/python-sdk) — `ClientSession` 与 `stdio_client` 的参考实现
- [MCP TypeScript SDK — 客户端](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript 版本的对应实现
- [VS Code — 扩展中的 MCP](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单一编辑器宿主中多路复用多个 MCP 服务器
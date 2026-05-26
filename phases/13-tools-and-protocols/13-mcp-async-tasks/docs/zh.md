# 异步任务（Async Tasks，SEP-1686）—— 即时调用，延迟获取：面向长时间运行的工作

> 真实的智能体（Agent）工作通常需要数分钟到数小时：例如持续集成（CI）运行、深度研究综合分析与批量导出。同步工具调用会导致连接断开、超时或阻塞用户界面（UI）。于 2025-11-25 合并的 SEP-1686 引入了任务（Tasks）原语：任何请求均可被增强为任务，其结果可稍后获取或通过状态通知流式传输。漂移风险提示：任务功能在 2026 年上半年（H1 2026）仍处于实验阶段；软件开发工具包（SDK）接口仍在围绕该规范进行设计。

**类型：** 构建
**语言：** Python（标准库，异步任务状态机）
**前置条件：** 第 13 阶段 · 07（模型上下文协议（MCP）服务器），第 13 阶段 · 09（传输层）
**预计时间：** 约 75 分钟

## 学习目标

- 识别何时应将工具从同步调用升级为任务增强（Task Augmentation）模式（服务器端工作超过 30 秒）。
- 熟悉任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，确保系统崩溃时不会丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题背景

假设 `generate_report` 工具需要运行一个耗时数分钟的数据提取流水线。在同步模型下，可选方案如下：

1. 保持连接打开三分钟。远程传输层会断开连接；客户端会超时；用户界面会卡死。
2. 立即返回占位符；要求客户端轮询自定义端点。这破坏了模型上下文协议（MCP）的统一性。
3. 发送后不管（Fire-and-forget）；无法获取结果。

以上方案均不理想。SEP-1686 引入了第四种方案：任务增强（Task Augmentation）。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器会立即返回任务 ID。客户端随后轮询 `tasks/status`，并在完成后获取 `tasks/result`。服务器端状态在重启后依然保留。

## 核心概念

### 任务增强（Task Augmentation）

通过将 `params._meta.task.required: true`（或 `optional: true`，由服务器决定）设置为 `true`，请求即可升级为任务。服务器会立即返回以下响应：

{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}

`ttl`（生存时间，Time To Live）是服务器承诺保留状态的期限；超过该期限后，任务结果将被丢弃。

### 按工具启用（Per-tool Opt-in）

工具注解（Tool Annotations）可声明其对任务的支持情况：

- `taskSupport: "forbidden"` — 该工具始终以同步方式运行。适用于执行速度较快的工具。
- `taskSupport: "optional"` — 客户端可选择请求任务增强。
- `taskSupport: "required"` — 客户端必须使用任务增强。

例如，`generate_report` 工具应设置为 `required`，而 `notes_search` 工具则应设置为 `forbidden`。

### 状态（States）

working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled

状态机（State Machine）采用仅追加（append-only）模式：一旦任务进入 `completed`（已完成）、`failed`（失败）或 `cancelled`（已取消）状态，即视为终态（terminal），不可再变更。

### 方法（Methods）

- `tasks/status {taskId}` — 返回当前状态及进度提示。
- `tasks/result {taskId}` — 阻塞等待结果，若任务尚未完成则返回 404。
- `tasks/cancel {taskId}` — 幂等（idempotent）操作；若任务已处于终态，则忽略该请求。
- `tasks/list` — 可选方法；用于枚举当前活跃及近期已完成的任务。

### 状态变更流式推送（Streaming State Changes）

当服务器支持该功能时，客户端可订阅状态通知：

server -> notifications/tasks/updated {taskId, state, progress?}

采用流式接收而非轮询（Polling）的客户端能获得更佳的用户体验。轮询机制始终作为最基础的兼容方案提供支持。

### 持久化状态（Durable State）

规范要求声明支持任务的服务器必须实现状态持久化。服务器崩溃不应导致 `ttl` 内已完成的结果丢失。存储方案可涵盖 SQLite、Redis 或文件系统。第 13 课（Lesson 13）的测试框架（harness）即采用文件系统作为存储后端。

### 取消语义（Cancellation Semantics）

`tasks/cancel` 为幂等操作。若任务正在执行中，服务器将尝试中止（需检查执行器是否支持协作式取消）。若任务已处于终态，则该请求不执行任何操作（no-op）。

### 崩溃恢复（Crash Recovery）

当服务器进程重启时：

1. 加载所有已持久化的任务状态。
2. 将因进程意外终止而仍处于 `working` 状态的任务标记为 `failed`，并附带错误码 `CRASH_RECOVERY`。
3. 在 `ttl` 有效期内保留 `completed` / `failed` / `cancelled` 状态的任务数据。

### 异步任务与采样（Async Tasks plus Sampling）

任务本身可调用 `sampling/createMessage`。长耗时研究类任务正是基于此机制运行：服务器的任务线程按需对客户端模型进行采样（Sampling），同时客户端界面将任务显示为 `working` 状态，并定期更新进度。

### 为何处于实验阶段（Why this is experimental）

SEP-1686 规范已于 2025 年 11 月 25 日发布，但整体路线图仍指出三个待解决的问题：持久化订阅原语（durable subscription primitives）、子任务（subtasks，即父子任务关系）以及结果 TTL 的标准化。预计该规范将在 2026 年持续演进。生产环境代码应仅在常规场景下将任务功能视为稳定，并需做好防御性编程，以应对未来 SDK 针对子任务可能引入的变更。

## 使用指南

`code/main.py` 实现了一个持久化任务存储 (Durable Task Store)（基于文件系统）以及一个在后台线程 (Background Thread) 中运行的 `generate_report` 工具。客户端调用该工具后会立即获取任务 ID (Task ID)，在工作线程更新进度期间轮询 (Poll) `tasks/status`，完成后获取 `tasks/result`。取消功能已实现；通过终止工作线程并重新加载状态来模拟崩溃恢复 (Crash Recovery)。

重点关注：

- 任务状态 JSON 持久化存储至 `/tmp/lesson-13-tasks/<id>.json`。
- 工作线程更新 `progress` 字段；轮询可观察到进度推进。
- 客户端发起的取消操作会设置一个事件；工作线程检查该事件并提前退出。
- “崩溃”后重新加载状态时，会将进行中的任务标记为 `failed`，并附带 `CRASH_RECOVERY` 原因。

## 交付成果

本课时将生成 `outputs/skill-task-store-designer.md`。针对长时间运行的工具（如研究、构建、导出），该技能 (Skill) 会设计任务存储结构（状态形态、TTL、持久性），选择合适的 `taskSupport` 标志，并规划进度通知机制。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询其状态，然后获取结果。
2. 在任务运行中途添加 `tasks/cancel` 调用。验证工作线程是否正确响应取消请求，且状态变为 `cancelled`。
3. 模拟崩溃恢复：终止工作线程，重新启动加载器，观察 `CRASH_RECOVERY` 故障模式。
4. 将存储扩展至 SQLite。持久性优势保持不变；同时解锁更多查询选项（例如列出会话 X 中的所有任务）。
5. 阅读 MCP (Model Context Protocol) 2026 年路线图文章。找出最有可能在未来一年内影响 SDK API 设计的那个与任务 (Tasks) 相关的未解决议题 (Open Issue)。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Task | “长时间运行的工具调用” | 附加 `_meta.task` 以支持异步执行的请求 |
| SEP-1686 | “任务规范” | 于 2025-11-25 引入任务功能的规范演进提案 (Spec Evolution Proposal) |
| `_meta.task` | “任务信封” | 包含 id、state、ttl 的每个请求专属元数据 |
| taskSupport | “工具标志” | 每个工具对应的 `forbidden` / `optional` / `required` 标志 |
| `tasks/status` | “轮询方法” | 获取当前状态及可选的进度提示 |
| `tasks/result` | “获取结果” | 返回已完成的有效负载，若未完成则返回 404 |
| `tasks/cancel` | “停止它” | 幂等的取消请求 |
| ttl | “保留预算” | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | “状态推送” | 由服务器发起的状态变更事件 |
| Durable store | “防崩溃状态” | 基于文件系统 / SQLite / Redis 的持久化层 |

## 延伸阅读

- [MCP — GitHub SEP-1686 议题](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案及完整讨论
- [WorkOS — 面向 AI 智能体（AI Agent）工作流的 MCP 异步任务](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 设计详解与原理说明
- [DeepWiki — MCP 任务系统与异步操作](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 运行机制与状态机（State Machine）
- [FastMCP — 任务](https://gofastmcp.com/servers/tasks) — SDK 级别的任务实现模式
- [MCP 博客 — 2026 年路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 待解决问题及 2026 年优先事项（含子任务）
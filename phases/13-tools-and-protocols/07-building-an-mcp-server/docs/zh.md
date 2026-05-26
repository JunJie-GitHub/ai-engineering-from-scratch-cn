# 构建 MCP（Model Context Protocol）服务器 —— Python + TypeScript SDK

> 大多数 MCP 教程仅展示基于标准输入输出（stdio）的“Hello World”示例。而一个真正的服务器需要暴露工具（tools）、资源（resources）和提示词（prompts），处理能力协商（capability negotiation），发出结构化错误（structured errors），并在不同 SDK 间保持一致的行为。本课程将端到端地构建一个笔记服务器：涵盖标准库（stdlib）stdio 传输、JSON-RPC 分发（dispatch）、三大服务器原语（server primitives），以及一种纯函数风格（pure-function style），以便在进阶时能无缝迁移至 Python SDK 的 FastMCP 或 TypeScript SDK。

**Type:** 构建
**Languages:** Python（标准库，stdio MCP 服务器）
**Prerequisites:** 第 13 阶段 · 06（MCP 基础）
**Time:** 约 75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环（dispatch loop），从标准输入（stdin）读取 JSON-RPC 消息，并将响应写入标准输出（stdout）。
- 根据 JSON-RPC 2.0 规范及 MCP 的附加错误码，发出结构化错误响应。
- 将基于标准库的实现平滑升级至 FastMCP（Python SDK）或 TypeScript SDK，且无需重写工具逻辑。

## 问题背景

在使用远程传输（第 13 阶段 · 09）或认证层（第 13 阶段 · 16）之前，你需要先构建一个干净的本地服务器。本地模式即指标准输入输出（stdio）：服务器由客户端作为子进程启动，消息通过按行分隔的 stdin/stdout 进行传输。

2025-11-25 版规范明确规定，stdio 消息需编码为 JSON 对象，并使用显式的 `\n` 作为分隔符。此处不涉及服务器发送事件（SSE）；SSE 曾是旧的远程模式，并将于 2026 年年中移除（Atlassian 的 Rovo MCP 服务器已于 2026 年 6 月 30 日弃用该模式；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio 而言，每行一个 JSON 对象即构成了完整的线路传输格式（wire format）。

笔记服务器是一个很好的示例，因为它能充分演练全部三大服务器原语。工具负责执行状态变更（mutations）（`notes_create`）。资源用于暴露数据（`notes://{id}`）。提示词则提供模板（`review_note`）。本课程的设计模式可泛化至任何业务领域。

## 核心概念

### 调度循环 (Dispatch loop)

loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response

三条规则：

- 不要向标准输出 (`stdout`) 打印任何非 JSON-RPC 信封 (JSON-RPC envelope) 的内容。调试日志应输出到标准错误 (`stderr`)。
- 每个请求 (request) 都必须匹配一个携带相同 `id` 的响应 (response)。
- 通知 (notification) 绝不能被回复。

### 实现 `initialize`

def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }

仅声明你支持的功能。客户端依赖能力集 (capability set) 来控制功能开关。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，其中每个条目包含 `name`、`description` 和 `inputSchema`。`tools/call` 接收 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块 (content blocks) 具有类型。最常见的包括：

{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}

工具错误分为两种形式。协议级错误 (protocol-level errors)（如未知方法、参数错误）属于 JSON-RPC 错误。工具级错误 (tool-level errors)（调用有效但工具执行失败）则返回为 `{content: [...], isError: true}`。这使得模型能够在上下文中感知到失败。

### 实现资源 (Resources)

资源在设计上是只读的。`resources/list` 返回清单 (manifest)；`resources/read` 返回具体内容。统一资源标识符 (URI) 可以是 `file://...`、`http://...` 或自定义方案（如 `notes://`）。

当你将数据作为资源而非工具暴露时：

- 模型不会“调用”它；客户端可以根据用户请求将其注入到上下文中。
- 订阅 (subscriptions) 允许服务器在资源变更时推送更新（Phase 13 · 10）。
- Phase 13 · 14 通过 `ui://` 对此进行了扩展，以支持交互式资源。

### 实现提示词 (Prompts)

提示词是带有命名参数的模板。宿主程序 (host) 会将其呈现为斜杠命令 (slash-commands)。例如，`review_note` 提示词可能接收一个 `note_id` 参数，并生成一个多消息提示词模板，由客户端提供给其模型。

### 标准输入输出传输 (Stdio transport) 的注意事项

- 使用换行符分隔的 JSON。不使用长度前缀帧 (length-prefixed framing)。
- 不要缓冲。每次写入后调用 `sys.stdout.flush()`。
- 客户端控制生命周期。当标准输入 (`stdin`) 关闭（遇到 EOF）时，应干净地退出。
- 不要静默处理 `SIGPIPE` 信号；应记录日志并退出。

### 注解 (Annotations)

每个工具都可以携带 `annotations` 来描述安全属性：

- `readOnlyHint: true` — 纯读取操作，可安全重试。
- `destructiveHint: true` — 具有不可逆的副作用；客户端应要求确认。
- `idempotentHint: true` — 相同输入产生相同输出（幂等）。
- `openWorldHint: true` — 与外部系统交互。

客户端利用这些注解来决定用户体验（如确认对话框、状态指示器）和路由策略（Phase 13 · 17）。

### 进阶路径 (Graduation path)

`code/main.py` 中的标准库服务器代码大约 180 行。FastMCP (Python) 将相同的逻辑简化为装饰器风格：

from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...

TypeScript SDK 具有等效的结构。当你准备好时，进阶路径支持即插即用 (drop-in)；其核心概念（能力、调度、内容块）是完全一致的。

## 使用方法

`code/main.py` 是一个基于标准输入/输出（stdio）的完整笔记 MCP 服务器（MCP Server），仅依赖标准库（stdlib）。它负责处理 `initialize`、`tools/list`、`tools/call`（对应 `notes_list`、`notes_search`、`notes_create` 三个工具），处理每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示词（Prompt）。你可以通过管道传入 JSON-RPC 消息（JSON-RPC Messages）来驱动它：

echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py

关注要点：

- 调度器（Dispatcher）是一个以方法名为键的 `dict[str, Callable]` 字典。
- 每个工具执行器（Tool Executor）返回的是内容块（Content Block）列表，而非纯字符串。
- 当执行器抛出异常时，会设置 `isError: true`。

## 交付发布

本课时将生成 `outputs/skill-mcp-server-scaffolder.md` 文件。给定一个业务领域（如笔记、工单、文件或数据库），该技能（Skill）会生成脚手架（Scaffold）搭建出一个 MCP 服务器，合理划分工具（Tools）/ 资源（Resources）/ 提示词（Prompts），并提供 SDK 的演进路径。

## 练习

1. 运行 `code/main.py`，并使用手动构造的 JSON-RPC 消息驱动它。先测试 `notes_create`，再调用 `resources/read` 以获取新建的笔记。
2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端是否会弹出确认对话框（此操作需要真实的宿主环境；Claude Desktop 即可）。
3. 实现 `resources/subscribe`，使得每当笔记被修改时，服务器都会推送 `notifications/resources/updated` 通知。同时添加一个保活（Keepalive）任务。
4. 将服务器移植到 FastMCP。Python 文件应缩减至 80 行以内。底层通信行为（Wire Behavior）必须保持一致；请使用相同的 JSON-RPC 测试框架进行验证。
5. 阅读规范文档中的 `server/tools` 部分，找出本课时服务器中未实现的一个工具定义字段。（提示：有多个未实现的字段；任选其一并添加即可。）

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| MCP 服务器（MCP Server） | “暴露工具的东西” | 通过 stdio 或 HTTP 使用 MCP JSON-RPC 进行通信的进程 |
| stdio 传输（stdio Transport） | “子进程模型” | 由客户端启动的服务器；通过标准输入/输出（stdin/stdout）进行通信 |
| 调度器（Dispatcher） | “方法路由器” | 将 JSON-RPC 方法名映射到处理函数的字典/映射表 |
| 内容块（Content Block） | “工具结果片段” | 工具响应 `content` 数组中的类型化元素 |
| `isError` | “工具级失败” | 标识工具执行失败；用于与 JSON-RPC 协议错误区分 |
| 注解（Annotations） | “安全提示” | readOnly（只读）/ destructive（破坏性）/ idempotent（幂等）/ openWorld（开放世界）标志位 |
| FastMCP | “Python SDK” | 基于装饰器、构建在 MCP 协议之上的高级框架 |
| 资源 URI（Resource URI） | “可寻址数据” | 使用 `file://`、`db://` 或自定义协议标识资源的地址 |
| 提示词模板（Prompt Template） | “斜杠命令简介” | 由服务器提供、包含参数占位符以供宿主 UI 使用的模板 |
| 能力声明（Capability Declaration） | “功能开关” | 在 `initialize` 阶段声明的、针对各基础原语（Primitive）的标志位 |

## 扩展阅读

- [模型上下文协议（Model Context Protocol）— Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 官方参考 Python 实现
- [模型上下文协议 — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 配套的 TypeScript 实现
- [FastMCP — 服务器框架](https://gofastmcp.com/) — 面向 MCP 服务器的装饰器风格 Python API
- [MCP — 服务器快速入门指南](https://modelcontextprotocol.io/quickstart/server) — 基于任一 SDK 的端到端教程
- [MCP — 服务器工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — 针对 `tools/*` 消息的完整参考文档
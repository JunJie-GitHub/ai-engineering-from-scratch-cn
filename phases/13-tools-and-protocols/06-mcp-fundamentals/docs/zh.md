# MCP 基础 — 原语、生命周期与 JSON-RPC 基础

> 在 MCP 出现之前，每一次集成都是定制化的（one-off）。模型上下文协议（Model Context Protocol, MCP）最初由 Anthropic 于 2024 年 11 月发布，现由 Linux 基金会旗下的智能体人工智能基金会（Agentic AI Foundation）负责维护。该协议标准化了服务发现与调用流程，使得任意客户端都能与任意服务器通信。2025-11-25 版规范定义了六个原语（primitives）（服务端三个，客户端三个）、三阶段生命周期（lifecycle）以及 JSON-RPC 2.0 通信线路格式（wire format）。掌握这些内容后，本阶段后续 MCP 章节的阅读将水到渠成。

**Type:** 学习
**Languages:** Python（标准库，JSON-RPC 解析器）
**Prerequisites:** 第 13 阶段 · 01 至 05（工具接口与函数调用）
**Time:** 约 45 分钟

## 学习目标

- 列举全部六个 MCP 原语（服务端：工具 tools、资源 resources、提示词 prompts；客户端：根目录 roots、采样 sampling、引导 elicitation），并各举一个应用场景。
- 梳理三阶段生命周期（初始化 initialize、运行 operation、关闭 shutdown），并说明每个阶段由哪一方发送何种消息。
- 解析并生成 JSON-RPC 2.0 的请求、响应与通知报文封装（envelopes）。
- 解释 `initialize` 阶段的能力协商（capability negotiation）机制，以及缺失该机制会导致何种问题。

## 问题背景

在 MCP 问世前，每个使用工具的 AI 智能体（agent）都采用各自的私有协议。Cursor 拥有一套形似 MCP 但互不兼容的工具系统；Claude Desktop 内置了另一套；VS Code 的 Copilot 扩展又采用了第三套。如果一个团队开发了一个“Postgres 查询”工具，就必须针对不同的宿主 API 重复编写三次。想要复用该工具，只能靠复制代码。

其结果便是定制化集成方案如寒武纪大爆发般涌现，同时也给生态系统的演进速度设下了天花板。

MCP 通过标准化通信线路格式（wire format）彻底解决了这一痛点。单个 MCP 服务器即可无缝接入所有 MCP 客户端：包括 Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf 等，截至 2026 年 4 月客户端数量已突破 300 个。每月 SDK 下载量高达 1.1 亿次。公开服务器超过 10,000 个。2025 年 12 月，Linux 基金会通过新成立的智能体人工智能基金会正式接管了该协议的维护工作。

本阶段采用的规范版本为 **2025-11-25**。该版本新增了异步任务（async Tasks，SEP-1686）、URL 模式引导（URL-mode elicitation，SEP-1036）、带工具支持的采样（sampling with tools，SEP-1577）、渐进式作用域授权（incremental scope consent，SEP-835）以及 OAuth 2.1 资源指示符语义（OAuth 2.1 resource-indicator semantics）。第 13 阶段 · 09 至 16 将详细讲解这些扩展内容。本课仅聚焦于基础部分。

## 核心概念

### 三大服务端原语 (Server Primitives)

1. **工具 (Tools)。** 可执行的操作。遵循第 13 阶段 · 01 中相同的四步循环。
2. **资源 (Resources)。** 暴露的数据。可通过 URI 寻址的只读内容：`file:///path`、`db://query/...` 或自定义协议。
3. **提示词 (Prompts)。** 可复用的模板。宿主 UI 中的斜杠命令；服务端提供模板，客户端填充参数。

### 三大客户端原语 (Client Primitives)

4. **根路径 (Roots)。** 服务端被允许访问的 URI 集合。由客户端声明，服务端予以遵守。
5. **采样 (Sampling)。** 服务端请求客户端的模型执行文本补全 (completion)。使得服务端托管的代理循环 (agent loops) 无需服务端 API 密钥即可运行。
6. **引导输入 (Elicitation)。** 服务端在运行过程中向客户端用户请求结构化输入。通过表单或 URL 实现（SEP-1036）。

MCP 中的每一项能力都严格归属于这六类之一。第 13 阶段 · 10 至 14 将对它们进行深入讲解。

### 线路格式 (Wire Format)：JSON-RPC 2.0

每条消息都是一个包含以下字段的 JSON 对象：

- 请求 (Requests)：`{jsonrpc: "2.0", id, method, params}`。
- 响应 (Responses)：`{jsonrpc: "2.0", id, result | error}`。
- 通知 (Notifications)：`{jsonrpc: "2.0", method, params}` —— 不包含 `id`，且无需响应。

基础规范包含约 15 个方法，按原语分组。其中重要的包括：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务端至客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期 (Lifecycle)

**阶段 1：初始化 (initialize)。**

客户端发送 `initialize` 请求，携带其 `capabilities`（能力）和 `clientInfo`（客户端信息）。服务端响应自身的 `capabilities`、`serverInfo`（服务端信息）以及所支持的规范版本。客户端在解析完响应后发送 `notifications/initialized`。此后，双方均可根据协商好的能力发送请求。

**阶段 2：运行 (operation)。**

双向通信。客户端调用 `tools/list` 进行发现，随后调用 `tools/call` 执行。若服务端声明了该能力，则可发送 `sampling/createMessage`。当工具集发生变更时，服务端可发送 `notifications/tools/list_changed`。当用户更改根路径作用域时，客户端可发送 `notifications/roots/list_changed`。

**阶段 3：关闭 (shutdown)。**

任意一方关闭传输通道。MCP 中没有结构化的关闭方法；传输层（标准输入输出 stdio 或 Streamable HTTP，见第 13 阶段 · 09）负责传递连接终止信号。

### 能力协商 (Capability Negotiation)

`initialize` 握手过程中的 `capabilities` 即为双方契约。服务端示例：

{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}

服务端声明其可发送 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自身能力予以确认：

{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}

若客户端未声明 `sampling`，服务端则不得调用 `sampling/createMessage`。对称原则：若服务端未声明 `resources.subscribe`，客户端不得尝试订阅。

这正是防止生态碎片化 (Ecosystem Drift) 的关键。不支持采样的客户端依然是合法的 MCP 客户端；不调用 `sampling` 的服务端依然是合法的 MCP 服务端。它们只是不共同使用该功能而已。

### 结构化内容 (Structured Content) 与错误格式 (Error Shapes)

`tools/call` 返回一个包含类型化块 (typed blocks) 的 `content` 数组：`text`、`image`、`resource`。第 13 阶段 · 14 将 MCP 应用 (`ui://` 交互式 UI) 也加入了该列表。

错误使用 JSON-RPC 错误码。规范定义的补充项包括：`-32002` “未找到资源 (Resource not found)”、`-32603` “内部错误 (Internal error)”，以及通过 `error.data` 传递的 MCP 专属错误数据。

### 客户端能力与工具调用细节

一个常见的误区：`capabilities.tools` 表示客户端是否支持工具列表变更通知。客户端是否实际调用特定工具，是由其模型驱动的运行时决策，而非能力标志。能力标志属于规范层面的契约，而模型的选择是正交 (orthogonal) 的。

### 为何选择 JSON-RPC 而非 REST？

JSON-RPC 2.0（2010 年）是一种轻量级的双向协议。REST 是客户端发起的。MCP 需要服务端发起的消息（如采样、通知），因此具有对称请求/响应结构的 JSON-RPC 是自然之选。此外，JSON-RPC 能够干净地组合在 stdio 和 WebSocket/Streamable HTTP 之上，无需重新发明 HTTP 的请求结构。

## 使用方法

`code/main.py` 内置了一个极简的 JSON-RPC 2.0 解析器（parser）与生成器（emitter），随后手动逐步执行 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，并打印每条消息。此处不涉及真实的传输层，仅展示消息结构（message shapes）。请对照“延伸阅读”中链接的规范，验证每个消息信封（envelope）。

观察重点：

- `initialize` 双向声明能力（capabilities）；响应中包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目包含 `name`、`description` 和 `inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应中的 `content` 是一个由 `{type, text}` 块组成的数组。

## 产出物

本课时将生成 `outputs/skill-mcp-handshake-tracer.md`。给定一份类似 pcap 格式的 MCP 客户端-服务器交互记录，该技能（skill）会为每条消息标注其所属的原语（primitive）、生命周期阶段（lifecycle phase）以及所依赖的能力（capability）。

## 练习

1. 运行 `code/main.py`。找出发生能力协商（capability negotiation）的代码行，并描述如果服务器未声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息结构为：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 执行期间发出该消息，并确认客户端处理程序会显示进度条。

3. 通读 MCP 2025-11-25 规范全文——整份文档约 80 页。找出大多数服务器实际上并不需要的那个能力标志（capability flag）。提示：它与资源订阅（resource subscription）相关。

4. 在纸上草图推演一个假设的“定时任务（cron job）”功能应归属于哪种原语。（提示：服务器希望客户端在预定时间调用它。目前的六种原语均不适用。）MCP 的 2026 路线图中已包含针对此功能的草案 SEP（标准演进提案）。

5. 解析 GitHub 上某个开源 MCP 服务器的一份会话日志。统计请求（request）、响应（response）与通知（notification）消息的数量。计算生命周期（lifecycle）流量与操作（operation）流量各自所占的比例。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MCP | “模型上下文协议”（Model Context Protocol） | 用于模型发现与调用工具的开放协议 |
| 服务器原语（Server primitive） | “服务器暴露的内容” | 工具（操作）、资源（数据）、提示词（模板） |
| 客户端原语（Client primitive） | “客户端允许服务器使用的内容” | 根目录（Roots，作用域）、采样（Sampling，大语言模型回调）、引导（Elicitation，用户输入） |
| JSON-RPC 2.0 | “传输格式” | 对称的请求/响应/通知信封（Envelopes）结构 |
| `initialize` 握手 | “能力协商” | 首对消息；服务器与客户端声明各自支持的特性 |
| `tools/list` | “发现” | 客户端向服务器请求其当前的工具集 |
| `tools/call` | “调用” | 客户端请求服务器携带参数执行指定工具 |
| `notifications/*_changed` | “变更事件” | 服务器通知客户端其原语列表已发生变更 |
| 内容块（Content block） | “类型化结果” | 工具返回结果中的 `{type: "text" | "image" | "resource" | "ui_resource"}` |
| SEP | “规范演进提案”（Spec Evolution Proposal） | 具名草案提案（例如用于异步任务的 SEP-1686） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 官方规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024年11月发布博文
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾与 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 与 1724 提案摘要
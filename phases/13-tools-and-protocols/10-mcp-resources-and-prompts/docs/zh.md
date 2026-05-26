# MCP 资源（Resources）与提示词（Prompts）——超越工具（Tools）的上下文暴露

> 工具（Tools）占据了 MCP 90% 的关注度。另外两个服务器原语（server primitives）解决的是不同的问题。资源（Resources）用于暴露只读数据；提示词（Prompts）则将可复用的模板以斜杠命令（slash-commands）的形式暴露出来。许多服务器本应使用资源来替代将读取操作封装在工具中，并使用提示词来替代在客户端提示词中硬编码工作流。本课程将明确这一决策规则，并逐步讲解 `resources/*` 和 `prompts/*` 消息。

**Type:** 构建（Build）
**Languages:** Python（标准库，资源与提示词处理器）
**Prerequisites:** 第 13 阶段 · 07（MCP 服务器）
**Time:** 约 45 分钟

## 学习目标

- 针对特定领域，决定将某项能力暴露为工具、资源还是提示词。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 实现带有参数模板的 `prompts/list` 和 `prompts/get`。
- 识别宿主（host）何时将提示词呈现为斜杠命令，何时作为自动注入的上下文。

## 问题背景

一个初级的笔记应用 MCP 服务器会将所有功能都暴露为工具：`notes_read`、`notes_list`、`notes_search`。这种做法将每次数据访问都封装在由模型驱动的工具调用中。其后果包括：

- 模型必须为每一个可能受益于上下文的查询，决定是否调用 `notes_read`。
- 只读内容无法被订阅，也无法流式传输到宿主的侧边栏。
- 客户端 UI（如 Claude Desktop 的资源附加面板、Cursor 的“包含文件”选择器）无法直接展示这些数据。

正确的划分方式是：将数据作为资源暴露，将变更或计算操作作为工具暴露，将可复用的多步骤工作流作为提示词暴露。每种原语都有其对应的交互示能（UX affordance）和访问模式。

## 核心概念

### 工具（Tools）与资源（Resources）与提示词（Prompts）—— 决策规则

| 能力需求（Capability） | 基础原语（Primitive） |
|------------|-----------|
| 用户希望搜索、过滤或转换数据 | 工具（tool） |
| 用户希望宿主程序（host）将此数据作为上下文（context）包含进来 | 资源（resource） |
| 用户希望获得可重复运行的模板化工作流 | 提示词（prompt） |

指导原则：如果模型在每次处理相关查询时调用它都能获益，则它属于工具（tool）。如果用户将其附加到对话中能获益，则它属于资源（resource）。如果用户希望复用的单元是一个完整的多步骤工作流，则它属于提示词（prompt）。

### 资源（Resources）

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的标识符：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义协议）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 同时支持文本和二进制数据。二进制数据使用 `blob` 字段存储 Base64 编码的字符串，并附带 `mimeType`。

### 资源订阅（Resource subscriptions）

在能力声明（capabilities）中配置 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源发生变更时，服务器发送 `notifications/resources/updated {uri}` 通知。客户端随后重新读取数据。

使用场景：一个笔记服务器，其资源为磁盘上的文件；文件监听器（file watcher）触发更新通知；当文件在宿主程序（host）外部被编辑时，Claude Desktop 会重新将该文件拉取至上下文（context）中。

### 资源模板（Resource templates）（2025-11-25 新增）

`resourceTemplates` 允许你暴露参数化的 URI 模式：例如 `notes://{id}`，其中 `id` 作为自动补全的目标。客户端可在资源选择器中自动补全 `id`。

### 提示词（Prompts）

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示词（prompt）是一种模板，用于填充生成消息列表，供宿主程序（host）输入给模型。例如，一个 `code_review` 提示词接收 `file_path` 参数，并返回包含三条消息的序列：一条系统消息（system message）、一条包含文件内容的用户消息（user message），以及一条带有推理模板的助手启动消息（assistant kickoff）。

### 宿主程序与提示词（Hosts and prompts）

Claude Desktop、VS Code 和 Cursor 在聊天界面中将提示词暴露为斜杠命令（slash-commands）。用户输入 `/code_review` 并通过表单选择参数。服务器提供的提示词充当了“用户快捷方式”与“发送给模型的完整提示词”之间的契约。

并非所有客户端目前都支持提示词——请检查能力协商（capability negotiation）结果。如果服务器声明了提示词能力，但客户端不支持，则客户端根本不会显示这些斜杠命令。

### “列表已更改”通知（The "list changed" notification）

当资源或提示词集合发生变动时，两者都会发出 `notifications/list_changed` 通知。例如，一个刚刚导入了 20 条新笔记的笔记服务器会发出 `notifications/resources/list_changed`；客户端随后会重新调用 `resources/list` 以获取新增内容。

### 内容类型约定（Content type conventions）

文本类型：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制类型：`image/png`、`application/pdf`，并附带 `blob` 字段。
MCP 应用（MCP Apps，第 14 课）：在 `ui://` URI 中使用 `text/html;profile=mcp-app`。

### 动态资源（Dynamic resources）

资源 URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端按 URI 进行缓存，则该 URI 必须保持稳定。如果计算是一次性的，则 URI 应包含时间戳或随机数（nonce），以避免客户端缓存过期失效。

### 订阅与轮询（Subscriptions vs polling）

支持订阅的客户端通过 `notifications/resources/updated` 接收服务器推送。不支持订阅的旧版客户端或宿主程序则通过重新读取进行轮询。两者均符合规范。服务器的能力声明会告知客户端其支持哪种方式。

订阅的成本：服务器端需要维护每个会话的状态（记录谁订阅了什么）。需限制订阅集合的规模；已断开连接的客户端应设置超时机制。

### 提示词与系统提示词（Prompts vs system prompts）

MCP 中的提示词并非系统提示词（system prompts）。宿主程序的系统提示词（其自身的运行指令）与 MCP 提示词（由用户调用、服务器提供的模板）是并行存在的。行为规范的客户端绝不会让服务器提示词覆盖其自身的系统提示词，而是将它们分层叠加。

## 使用

`code/main.py` 在第 07 课的笔记服务器（notes server）基础上进行了扩展，新增了以下功能：

- 支持 `resources/subscribe` 的按笔记资源（per-note resources，如 `notes://note-1` 等）。
- 一个 `review_note` 提示词（prompt），可渲染为包含三条消息的模板。
- 一个文件监听器（file-watcher）模拟程序，当笔记被修改时会发出 `notifications/resources/updated` 事件。
- 一个 `notes://recent` 动态资源（dynamic resource），始终返回最新的五条笔记。

运行演示程序即可查看完整的工作流程。

## 交付

本课将生成 `outputs/skill-primitive-splitter.md` 文件。针对一个拟议的 MCP 服务器（MCP server），该技能（skill）会将各项能力分类为工具（tool）、资源（resource）或提示词（prompt），并附上分类依据。

## 练习

1. 运行 `code/main.py`。观察初始的资源列表，然后触发一次笔记编辑操作，并验证 `notifications/resources/updated` 事件是否正常触发。

2. 添加一个 `resources/list_changed` 事件发射器（emitter）：当创建新笔记时发送该通知，以便客户端重新发现资源。

3. 为 GitHub MCP 服务器设计三个提示词：`summarize_pr`、`triage_issue` 和 `release_notes`。每个提示词需包含参数结构（argument schemas）。提示词主体应无需进一步修改即可直接运行。

4. 选取第 07 课服务器中的一个现有工具，判断它应继续保持为工具，还是拆分为“资源+工具”的组合。请用一句话说明理由。

5. 阅读规范文档中的 `server/resources` 和 `server/prompts` 章节。找出 `resources/read` 中一个极少被填充但规范支持的字段。提示：查看资源内容上的 `_meta` 字段。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 资源（Resource） | “暴露的数据” | 宿主（host）可通过 URI 寻址读取的内容 |
| 资源 URI（Resource URI） | “数据指针” | 带有协议前缀的标识符（如 `file://`、`notes://` 等） |
| `resources/subscribe` | “监听变更” | 客户端主动订阅后，服务器针对特定 URI 推送的更新 |
| `notifications/resources/updated` | “资源已变更” | 向客户端发出的信号，表明已订阅的资源有了新内容 |
| 资源模板（Resource template） | “参数化 URI” | 带有补全提示的 URI 模式，供宿主选择器使用 |
| 提示词（Prompt） | “斜杠命令模板” | 带有参数占位符的命名多消息模板 |
| 提示词参数（Prompt arguments） | “模板输入” | 宿主在渲染前收集的带类型参数 |
| `prompts/get` | “渲染模板” | 服务器返回填充完毕的消息列表 |
| 内容块（Content block） | “类型化数据块” | `{type: text | image | resource | ui_resource}` |
| 斜杠命令交互（Slash-command UX） | “用户快捷方式” | 宿主将提示词以 `/` 开头的命令形式呈现给用户 |

## 进一步阅读

- [MCP — 概念：资源 (Resources)](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI、订阅 (subscriptions) 与模板 (templates)
- [MCP — 概念：提示词 (Prompts)](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示词模板 (prompt templates) 与斜杠命令 (slash-command) 集成
- [MCP — 服务器资源规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的 `resources/*` 消息参考 (message reference)
- [MCP — 服务器提示词规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的 `prompts/*` 消息参考
- [MCP — 协议信息站点：资源](https://modelcontextprotocol.info/docs/concepts/resources/) — 对官方文档进行扩展的社区指南 (community guide)
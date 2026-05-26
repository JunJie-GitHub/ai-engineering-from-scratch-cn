# 根路径 (Roots) 与用户交互 (Elicitation) — 作用域限定与执行中用户输入

> 硬编码路径会在用户打开不同项目时立即失效。当用户输入信息不足时，预填充的工具参数也会失效。根路径将服务器的作用域限定在用户控制的一组 URI 内；用户交互会在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。这是两种客户端原语，分别用于修复两种常见的 MCP 故障模式。SEP-1036（URL 模式用户交互，2025-11-25）在 2026 年上半年处于实验阶段——在依赖该功能前请务必检查 SDK 版本。

**类型：** 构建
**语言：** Python（标准库，根路径与用户交互演示）
**前置条件：** 第 13 阶段 · 07（MCP 服务器）
**耗时：** 约 45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器的文件操作限制在已声明的根路径集合内的 URI。
- 使用 `elicitation/create` 在工具调用中途向用户请求确认或结构化输入。
- 在表单模式与 URL 模式的用户交互之间进行选择（后者为实验性功能；需注意规范漂移风险）。

## 问题背景

笔记类 MCP 服务器在生产环境中常遇到的两个具体故障。

**路径假设失效。** 服务器代码基于 `~/notes` 编写。当用户在另一台机器上运行，且笔记实际位于 `~/Documents/Notes` 时，工具调用会静默失败（找不到文件），甚至更糟——写入错误的位置。

**缺失用户已知参数。** 用户要求“删除旧的 TPS 报告笔记”。模型调用 `notes_delete(title: "TPS report")`，但存在 2023、2024 和 2025 年三条匹配的笔记。工具无法自行猜测。直接返回“歧义”错误令人困扰；若对三条笔记全部执行删除则是灾难性的。

根路径解决第一个问题：客户端在 `initialize` 阶段声明服务器可访问的 URI 集合。用户交互解决第二个问题：服务器暂停工具调用，并发送 `elicitation/create` 请求，让用户选择具体操作哪一条。

## 核心概念

### 根 (Roots)

客户端在 `initialize` 阶段声明根列表：

{
  "capabilities": {"roots": {"listChanged": true}}
}

随后，服务器可以调用 `roots/list`：

{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}

服务器必须将根视为边界：拒绝任何超出根集合的文件读写操作。客户端不会强制执行此限制（因为服务器仍是用户信任的代码），但符合规范的服务器会遵守这一约定。

当用户添加或移除根时，客户端会发送 `notifications/roots/list_changed` 通知。服务器将重新调用 `roots/list` 并更新其访问边界。

### 为什么根是客户端原语 (Client Primitive)

根由客户端声明，因为它们代表了用户的授权模型。用户向 Claude Desktop 指示“允许此笔记服务器访问这两个目录”。服务器无法自行扩大该权限范围。

### 引导交互 (Elicitation)：表单模式默认值

`elicitation/create` 接收一个表单模式 (Form Schema) 加上一段自然语言提示：

{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}

客户端渲染表单，收集用户的回答后返回：

{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}

包含三种可能的操作：`accept`（用户已填写）、`decline`（用户关闭了表单）或 `cancel`（用户中止了整个工具调用）。

表单模式是扁平的——v1 版本不支持嵌套对象。SDK 通常会拒绝任何超过单层结构的复杂定义。

### 引导交互：URL 模式（SEP-1036，实验性）

于 2025-11-25 新增。服务器不再发送模式定义，而是直接发送一个 URL：

{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}

客户端会在浏览器中打开该 URL，等待操作完成，并在用户返回后返回结果。适用于 OAuth 流程、支付授权和文档签署等表单无法满足的场景。

兼容性风险提示：SEP-1036 的响应结构仍在最终确定中；部分 SDK 返回回调 URL，另一些则返回完成令牌 (Completion Token)。在生产环境中使用 URL 模式前，请务必查阅所用 SDK 的发布说明。

### 何时适合使用引导交互

- 在执行破坏性操作前获取用户确认（破坏性提示 + 引导交互）。
- 消除歧义（从 N 个匹配项中选择一个）。
- 首次运行设置（API 密钥、目录、偏好设置）。
- OAuth 风格流程（URL 模式）。

### 何时不应使用引导交互

- 填充模型本可通过自然语言询问的工具必需参数。应使用常规重新提示，而非引导交互对话框。
- 高频调用。引导交互会中断对话流程；请勿在循环中触发它。
- 服务器可在事后验证的任何内容。应先进行验证，返回错误，然后让模型通过文本向用户询问。

### 人在回路 (Human-in-the-loop) 桥梁

引导交互与采样 (Sampling) 相结合，共同实现了 MCP 的“人在回路”模型。服务器的智能体循环 (Agent Loop) 可以暂停以等待用户输入（引导交互）或模型推理（采样）。Phase 13 · 11 已涵盖采样内容；本课则聚焦引导交互。将两者结合，即可实现对循环中途的完整控制。

## 使用方法

`code/main.py` 对笔记服务器进行了以下扩展：

- 针对 `roots/list` 的响应：服务器在收到根目录列表变更通知后，会重新查询该接口。
- 一个 `notes_delete` 工具：当匹配到多个笔记时，使用 `elicitation/create` 进行消歧（Disambiguation）。
- 一个 `notes_setup` 工具：使用 URL 模式引导（URL-mode Elicitation）打开首次运行配置页面（模拟）。
- 边界检查机制：拒绝在已声明的根目录（Roots）范围之外的 URI 上执行操作。

该演示运行了三种场景：正常流程（Happy Path，仅匹配一项）、消歧场景（匹配三项，触发引导流程）、越权写入（Out-of-root-write，被拒绝）。

## 交付与发布

本课时将生成 `outputs/skill-elicitation-form-designer.md`。针对可能需要用户确认或消歧的工具，该技能（Skill）会负责设计引导表单的 Schema 以及消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟的用户回答能否正确路由回该工具。
2. 添加一个新工具 `notes_archive`，要求每次操作都必须经过引导确认（破坏性操作提示）。检查用户体验（UX）：这与模型在对话中重新询问相比有何不同？
3. 为首次运行的 OAuth 流程实现 URL 模式引导。注意漂移风险（Drift Risk），并添加 SDK 版本保护机制（SDK-version Guard）。
4. 扩展 `roots/list` 的处理逻辑：当收到通知时，服务器应原子性地重新读取并重新扫描可能已超出作用域的已打开文件句柄。
5. 阅读 GitHub 上关于 SEP-1036 的议题讨论串。找出一个会影响服务器如何处理 URL 模式回调（Callbacks）的未决问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Root（根目录） | “同意边界”（Consent boundary） | 客户端允许服务器访问的 URI |
| `roots/list` | “服务器请求作用域”（Server asks for scope） | 客户端返回当前的根目录集合 |
| `notifications/roots/list_changed` | “用户更改了作用域”（User changed scope） | 客户端发出根目录集合已变更的信号 |
| Elicitation（引导） | “调用中途询问用户”（Ask the user mid-call） | 由服务器发起的、请求结构化用户输入的机制 |
| `elicitation/create` | “该方法”（The method） | 用于发起引导请求的 JSON-RPC 方法 |
| Form mode（表单模式） | “由 Schema 驱动的表单”（Schema-driven form） | 在客户端 UI 中渲染为表单的扁平化 JSON Schema |
| URL mode（URL 模式） | “浏览器重定向”（Browser redirect） | SEP-1036 实验性功能；打开一个 URL 并等待响应 |
| `accept` / `decline` / `cancel` | “用户响应结果”（User response outcomes） | 服务器需要处理的三种分支逻辑 |
| Disambiguation（消歧） | “从中选一个”（Pick one） | 当工具匹配到 N 个候选项时常见的引导用例 |
| Flat form（扁平表单） | “仅包含顶层属性”（Top-level properties only） | 引导 Schema 不支持嵌套结构 |

## 扩展阅读

- [模型上下文协议 (MCP) — 客户端根目录 (roots) 规范](https://modelcontextprotocol.io/specification/draft/client/roots) — 标准根目录参考文档
- [MCP — 客户端引导 (elicitation) 规范](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 标准引导参考文档
- [思科 (Cisco) — MCP 引导、结构化内容 (structured content) 与 OAuth 增强功能的新动态](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增功能逐步指南
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式引导提案（实验性，存在规范漂移风险）
- [The New Stack — 引导机制如何将人在回路 (human-in-the-loop) 引入 AI 工具](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — 用户体验 (UX) 流程演示
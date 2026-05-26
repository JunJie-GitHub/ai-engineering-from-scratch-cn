# 函数调用深度解析 — OpenAI、Anthropic、Gemini

> 2024年，三大前沿大模型提供商在工具调用循环（tool-call loop）上趋于一致，但在其余实现细节上却分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 内容块。Gemini 采用 `functionDeclarations` 与唯一 ID 关联机制。本课程将三者进行横向对比（diff），确保你为某一平台编写的代码在移植到其他平台时能够无缝运行。

**类型：** 构建实践
**语言：** Python（标准库、模式转换器（schema translators））
**前置要求：** 第13阶段 · 01（工具接口）
**时长：** 约75分钟

## 学习目标

- 阐明 OpenAI、Anthropic 和 Gemini 在函数调用载荷（payload）结构上的三大差异（声明、调用、结果）。
- 将单一工具声明转换为三家提供商的格式，并预测严格模式（strict-mode）约束的差异点。
- 在各提供商的 API 中使用 `tool_choice` 参数，以强制调用、禁止调用或自动选择工具。
- 掌握各提供商的硬性限制（工具数量、模式深度、参数长度），以及触发限制时各自返回的错误特征（error signatures）。

## 核心问题

函数调用请求的数据结构因提供商而异。以下是来自 2026 年生产环境技术栈的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个需要你自行解析的 JSON 字符串。严格模式（`strict: true`）通过受限解码（constrained decoding）强制保证符合 JSON Schema 规范。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 字段已经是解析后的对象（而非字符串）。你需要回复一条新的 `user` 消息，其中包含 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中在 Gemini 3 及更高版本中，`id` 是唯一的，用于并行调用关联。你需要回复 `{functionResponse: {name, id, response}}`。

底层循环逻辑相同，但字段命名、嵌套层级、字符串与对象的约定以及关联机制各不相同。一个在 OpenAI 上开发天气智能体（agent）的团队，仅为了适配底层架构代码（plumbing），就需要花费两天时间移植到 Anthropic，再花一天移植到 Gemini。

本课程将构建一个转换器，将这三种格式统一为一种标准的工具声明，并在边缘侧进行路由分发。第13阶段 · 17 会将该模式进一步泛化，构建为一个 LLM 网关（LLM gateway）。

## 核心概念

### 通用结构

每个提供商（Provider）都需要以下五个要素：

1. **工具列表（Tool list）**。包含每个工具的名称、描述和输入模式（Input schema）。
2. **工具选择（Tool choice）**。强制使用特定工具、禁止使用工具，或交由模型自行决定。
3. **调用生成（Call emission）**。结构化输出，指明工具名称及参数。
4. **调用 ID（Call id）**。将响应与正确的调用关联起来（对并行调用至关重要）。
5. **结果注入（Result injection）**。一条消息或代码块，用于将执行结果回传至对应的调用。

### 结构差异（逐字段对比）

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明封装（Declaration envelope） | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 模式字段（Schema field） | `parameters` | `input_schema` | `parameters` |
| 响应容器（Response container） | 助手消息中的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型（Arguments type） | 字符串化 JSON | 解析后的对象 | 解析后的对象 |
| ID 格式（Id format） | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块（Result block） | 角色为 `tool`，包含 `tool_call_id` | 角色为 `user`，包含 `tool_result` 和 `tool_use_id` | `functionResponse`，包含匹配的 `id` |
| 强制使用工具（Force-a-tool） | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止使用工具（Forbid tools） | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格模式（Strict schema） | `strict: true` | 模式即契约（始终强制执行） | 请求级别的 `responseSchema` |

### 实际会遇到的限制

- **OpenAI**。每次请求最多 128 个工具。模式（Schema）深度限制为 5。参数字符串长度 ≤ 8192 字节。严格模式要求不能使用 `$ref`，不能存在重叠的 `oneOf`/`anyOf`/`allOf`，且所有属性都必须在 `required` 中列出。
- **Anthropic**。每次请求最多 64 个工具。模式深度理论上无限制，但实际建议不超过 10。无严格模式标志；模式本身即为契约，模型通常会严格遵守。
- **Gemini**。每次请求最多 64 个函数。模式类型采用 OpenAPI 3.0 的子集（与 JSON Schema 2020-12 略有差异）。自 Gemini 3 起，并行调用支持唯一 ID。

### `tool_choice` 行为

所有提供商均支持三种模式，仅命名不同。

- **自动（Auto）**。模型自行选择调用工具或生成文本。默认模式。
- **必需/任意（Required / Any）**。模型必须至少调用一个工具。
- **无（None）**。模型禁止调用任何工具。

此外，各提供商还拥有一种独有模式：

- **OpenAI**。通过名称强制指定特定工具。
- **Anthropic**。通过名称强制指定特定工具；`disable_parallel_tool_use` 标志用于区分单工具调用与多工具调用。
- **Gemini**。`mode: "VALIDATED"` 会强制所有响应经过模式验证器，无论模型原始意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认值）会在单条助手消息中发出多个调用。你需要执行所有这些调用，并回复一条包含工具角色的批量消息，其中每个 `tool_call_id` 对应一个条目。Anthropic 历史上仅支持单次调用；`disable_parallel_tool_use: false`（自 Claude 3.5 起为默认值）启用了多工具调用。Gemini 2 允许并行调用但未提供稳定的 ID；Gemini 3 引入了 UUID，使得乱序到达的响应也能准确关联。

### 流式传输（Streaming）

三者均支持流式工具调用，但传输格式（Wire format）有所不同：

- **OpenAI**。`tool_calls[i].function.arguments` 的增量数据块（Delta chunks）会逐步到达。你需要持续累积，直到收到 `finish_reason: "tool_calls"`。
- **Anthropic**。采用块开始（Block-start）/ 块增量（Block-delta）/ 块结束（Block-stop）事件。`input_json_delta` 数据块携带部分参数。
- **Gemini**。`streamFunctionCallArguments`（Gemini 3 新增）会发射带有 `functionCallId` 的数据块，从而允许多个并行调用交错传输。

第 13 阶段 · 03 节深入探讨了并行调用与流式重组。本节主要聚焦于声明格式与单次调用的结构。

### 错误处理与修复

参数无效错误的表现形式也有所不同。

- **OpenAI（非严格模式）**。模型返回 `arguments: "{bad json}"`，导致你的 JSON 解析失败。此时你需要注入一条错误消息并重新发起调用。
- **OpenAI（严格模式）**。验证在解码阶段进行；不会出现无效 JSON，但可能会触发 `refusal`（拒绝响应）。
- **Anthropic**。`input` 可能包含意外字段；模式仅作为参考建议。需在服务端进行验证。
- **Gemini**。OpenAPI 3.0 的一个特性：对象字段上的 `enum` 会被静默忽略；需自行实现验证。

### 转换器模式（Translator pattern）

代码中标准的工具声明如下所示（你可自行定义结构）：

Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)

只需三个小型函数即可将其转换为三家提供商的对应格式。`code/main.py` 中的测试框架（Harness）正是执行此操作，随后将模拟的工具调用往返传递至各提供商的响应结构中。无需网络连接——本节旨在讲解数据结构，而非 HTTP 通信。

生产环境团队通常会将此转换器封装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。第 13 阶段 · 17 节将提供一个网关，可在任意一家提供商之上暴露出符合 OpenAI 格式的 API。

## 使用它

`code/main.py` 定义了一个统一的（canonical）`Tool` 数据类（dataclass）以及三个转换器（translator），分别用于生成 OpenAI、Anthropic 和 Gemini 的声明 JSON（declaration JSON）。随后，它将针对每种结构手工构造的提供商响应解析为同一个统一的调用对象，从而证明三者的底层语义（semantics）完全一致。运行该脚本，并并排进行差异对比（diff）。

重点关注以下内容：

- 三个声明块仅在封装结构（envelope）和字段名称上存在差异。
- 三个响应块的差异在于调用信息所在的位置（顶层 `tool_calls`、`content[]` 块或 `parts[]` 条目）。
- 一个 `canonical_call()` 函数能够从这三种响应结构中提取出 `{id, name, args}`。

## 交付它

本实践将生成 `outputs/skill-provider-portability-audit.md`。针对某一提供商的函数调用（function-calling）集成，该技能模块会输出一份可移植性审计（portability audit）报告：指出其依赖了哪些特定提供商的限制、哪些字段需要重命名，以及迁移至其他提供商时哪些部分会失效。

## 练习

1. 运行 `code/main.py`，验证三个提供商的声明 JSON 是否都序列化了同一个底层的 `Tool` 对象。修改统一的工具定义以添加一个枚举（enum）参数，并确认只有 Gemini 转换器需要处理 OpenAPI 规范的特殊兼容问题。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，用于提取模型在执行 `list_tools` 或服务发现（discovery）调用后返回的工具列表。注意 OpenAI 原生并不支持此功能，请记录这一不对称性（asymmetry）。

3. 实现 `tool_choice` 转换逻辑：将统一的 `ToolChoice(mode="force", tool_name="x")` 映射为三种提供商的对应格式。随后处理 `mode="any"` 和 `mode="none"` 的映射。参考本课程的差异对比表进行核对。

4. 从三家提供商中任选其一，完整阅读其函数调用指南。在其模式规范（Schema spec）中找出一个另外两家不支持的字段。参考候选字段：OpenAI 的 `strict`、Anthropic 的 `disable_parallel_tool_use`、Gemini 的 `function_calling_config.allowed_function_names`。

5. 编写一个测试向量（test vector）：构造一个参数违反已声明模式（Schema）的工具调用。将其输入各提供商的验证器（validator）（可使用 `Lesson 01` 中的标准库验证器作为替代），并记录触发的具体错误。在文档中说明，若在生产环境中追求严格性（strictness），你会选择哪家提供商。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 函数调用 (Function Calling) | “工具使用 (Tool Use)” | 提供商级 API，用于结构化地输出工具调用请求 |
| 工具声明 (Tool Declaration) | “工具规范 (Tool Spec)” | 名称 + 描述 + JSON Schema 输入负载 |
| `tool_choice` | “强制 / 禁止” | 自动 / 必需 / 无 / 指定名称模式 |
| 严格模式 (Strict Mode) | “模式强制校验 (Schema Enforcement)” | OpenAI 的标志位，用于约束解码过程以严格匹配 Schema |
| `tool_use` 块 | “Anthropic 的调用结构” | 包含 id、name 和 input 的内联内容块 |
| `functionCall` 部分 | “Gemini 的调用结构” | `parts[]` 数组中的一个条目，包含 name、args 和 id |
| 字符串化参数 (Arguments-as-string) | “字符串化的 JSON” | OpenAI 将参数作为 JSON 字符串返回，而非对象 |
| 并行工具调用 (Parallel Tool Calls) | “单轮次多路分发” | 在一条助手消息中包含多个工具调用 |
| 拒绝响应 (Refusal) | “模型拒绝执行” | 仅在严格模式下返回的拒绝块，而非工具调用 |
| OpenAPI 3.0 子集 | “Gemini 的 Schema 特性” | Gemini 使用一种类似 JSON Schema 的方言，仅存在细微差异 |

## 延伸阅读

- [OpenAI — 函数调用指南](https://platform.openai.com/docs/guides/function-calling) — 权威参考文档，涵盖严格模式与并行调用
- [Anthropic — 工具使用概述](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 与 `tool_result` 块的语义说明
- [Google — Gemini 函数调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 及 OpenAPI 子集说明
- [Vertex AI — 函数调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级接口
- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式下的 Schema 强制校验详情
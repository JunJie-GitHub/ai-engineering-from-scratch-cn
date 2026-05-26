# 结构化输出 (Structured Output) — JSON Schema、Pydantic、Zod 与约束解码 (Constrained Decoding)

> 即使是最前沿的模型，“礼貌地要求模型返回 JSON”也有 5% 到 15% 的失败率。结构化输出通过约束解码 (Constrained Decoding) 填补了这一差距：模型在底层被禁止生成任何违反模式 (Schema) 的 Token。OpenAI 的严格模式 (Strict Mode)、Anthropic 的模式类型化工具调用 (Schema-typed Tool Use)、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 以及 Zod 的 `.parse`，都是同一理念的五种表层实现形式。本课程将构建模式验证器 (Schema Validator) 和严格模式契约，供学习者在所有生产级信息提取流水线 (Extraction Pipeline) 中使用。

**类型：** 构建
**语言：** Python（标准库，JSON Schema 2020-12 子集）
**前置条件：** 第 13 阶段 · 02（函数调用 (Function Calling) 深度解析）
**时长：** 约 75 分钟

## 学习目标

- 使用正确的约束条件（枚举 `enum`、最小/最大值 `min/max`、必填 `required`、正则匹配 `pattern`）为提取目标编写 JSON Schema 2020-12。
- 解释为何严格模式与约束解码所提供的保障不同于“生成后验证 (Validate After Generation)”。
- 区分三种失败模式：解析错误 (Parse Error)、模式违规 (Schema Violation) 与模型拒绝 (Model Refusal)。
- 交付具备类型化修复 (Typed Repair) 与类型化拒绝处理 (Typed Refusal Handling) 能力的信息提取流水线。

## 问题背景

一个读取采购订单邮件的智能体 (Agent) 需要将自由文本转换为 `{customer, line_items, total_usd}` 格式。现有三种方法。

**方法一：通过提示词要求 JSON (Prompt for JSON)。** “请以 JSON 格式回复，包含 customer、line_items、total_usd 字段。” 在前沿模型上成功率约为 85% 到 95%。存在六种失败情况：缺少大括号、末尾多余逗号、类型错误、幻觉字段 (Hallucinated Fields)、因 Token 限制被截断、以及泄露提示性文本（如“以下是您的 JSON：”）。

**方法二：生成后验证 (Validate After Generation)。** 自由生成内容，进行解析，对照模式进行验证，失败则重试。该方法可靠但成本高昂——每次重试都会产生费用，且截断错误每次发生都会额外消耗一轮对话。

**方法三：约束解码 (Constrained Decoding)。** 服务提供商在解码阶段强制执行模式约束。无效 Token 会在采样分布 (Sampling Distribution) 中被屏蔽。输出结果保证可解析且保证符合模式验证。失败情况收敛为单一模式：拒绝（模型判定输入内容不符合模式要求）。

到 2026 年，所有前沿模型提供商均已提供某种形式的第三种方法。

- **OpenAI。** 使用 `response_format: {type: "json_schema", strict: true}`，若模型拒绝生成，响应中将包含 `refusal` 字段。
- **Anthropic。** 对 `tool_use` 输入进行模式强制校验；不存在 `stop_reason: "refusal"`，但以 `end_turn` 结束且未发起工具调用即为拒绝信号。
- **Gemini。** 在请求级别使用 `responseSchema`；2026 年的 Gemini 已为特定类型提供 Token 级别的语法约束。
- **Pydantic AI。** 设置 `output_type=InvoiceModel` 将输出类型为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，用于根据 Zod 模式验证提供商的输出；可与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

核心共性：声明一次模式，端到端强制执行。

## 核心概念

### JSON Schema 2020-12 —— 通用规范

所有服务提供商均支持 JSON Schema 2020-12。你最常用的结构包括：

- `type`：取值为 `object`、`array`、`string`、`number`、`integer`、`boolean` 或 `null` 之一。
- `properties`：字段名到子模式 (subschema) 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合（枚举）。
- `minimum` / `maximum`（用于数字），`minLength` / `maxLength` / `pattern`（用于字符串）。
- `items`：应用于数组中每个元素的子模式。
- `additionalProperties`：设为 `false` 时将禁止额外字段（默认值因模式而异）。

OpenAI 的严格模式 (strict mode) 增加了三项要求：每个属性都必须列在 `required` 中、全局设置 `additionalProperties: false`，且不能包含未解析的 `$ref`。如果违反这些规则，API 将在请求时返回 400 错误。

### Pydantic —— Python 绑定库

Pydantic v2 通过 `model_json_schema()` 从类似数据类的模型中生成 JSON Schema。Pydantic AI 对此进行了封装，因此你只需编写：

class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal

随后，智能体框架 (agent framework) 会在边缘侧将该模式转换为 OpenAI 严格模式、Anthropic 的 `input_schema` 或 Gemini 的 `responseSchema`。模型的输出将作为类型化的 `Invoice` 实例返回。验证错误会抛出带有类型化错误路径的 `ValidationError`。

### Zod —— TypeScript 绑定库

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 中的等效方案。OpenAI 的 Node SDK 提供了 `zodResponseFormat(Invoice)` 方法，它会自动转换为 API 所需的 JSON Schema 负载 (payload)。

### 拒绝响应 (Refusals)

严格模式无法强制模型给出答案。如果输入内容不符合模式要求（例如“邮件是一首诗，而不是发票”），模型会返回一个包含原因的 `refusal` 字段。你的代码必须将其视为一种一等结果 (first-class outcome)，而非失败。拒绝响应也可用作安全信号：当模型被要求从受保护内容的邮件中提取信用卡号时，它会返回附带安全原因的拒绝响应。

### 开源环境中的约束解码 (Constrained Decoding)

开源权重 (open-weights) 实现通常采用以下三种技术：

1. **基于语法的解码 (Grammar-based decoding)**（`outlines`、`guidance`、`lm-format-enforcer`）：根据模式构建确定性有限自动机 (deterministic finite automaton)；在每一步生成时，屏蔽那些会违反该有限状态机 (FSM) 的 token 的 logits。
2. **结合 JSON 解析器的 Logit 屏蔽**：让流式 JSON 解析器与模型同步运行；在每一步计算有效的下一个 token 集合。
3. **带验证器的投机解码 (Speculative decoding)**：由轻量级草稿模型 (draft model) 提议 token，再由验证器强制执行模式约束。

商业提供商在后台通常会选择其中一种技术。截至 2026 年的最新技术表明，对于较短的结构化输出，其速度比纯文本生成更快；对于较长的输出，速度则大致相当。

### 三种失败模式

1. **解析错误 (Parse error)**：输出不是有效的 JSON。在严格模式下不会发生，但在非严格模式的提供商中仍可能出现。
2. **模式违规 (Schema violation)**：输出可被解析，但违反了模式定义。在严格模式下不会发生，但在非严格模式下很常见。
3. **拒绝响应 (Refusal)**：模型拒绝生成。必须将其作为类型化的结果进行处理。

### 重试策略

当处于非严格模式时（如 Anthropic 工具调用、非严格 OpenAI 或旧版 Gemini），恢复模式如下：

generate -> parse -> validate -> if fail, inject error and retry, max 3x

通常一次重试就足够了。三次重试可以捕获弱模型的偶发性错误 (flakes)。如果超过三次，则表明模式设计存在问题：模型在某些输入下无法满足该模式，此时需要修复提示词 (prompt) 或模式本身。

### 对小模型的支持

约束解码同样适用于小模型。在结构化任务中，采用语法约束的 30 亿参数 (3B) 开源模型，其表现优于仅使用原始提示词 (raw prompting) 的 700 亿参数 (70B) 模型。这正是结构化输出在生产环境中至关重要的主要原因：它将可靠性与模型规模解耦。

## 使用它

`code/main.py` 内置了一个基于标准库 (Standard Library) 的最小化 JSON Schema 2020-12 验证器 (JSON Schema 2020-12 Validator)，支持类型 (types)、必填 (required)、枚举 (enum)、最小/最大值 (min/max)、正则模式 (pattern)、数组项 (items) 和额外属性 (additionalProperties) 校验。它封装了一个 `Invoice` 模式 (Schema)，并将模拟的大语言模型 (LLM) 输出传入验证器，演示了解析错误 (Parse Error)、模式违规 (Schema Violation) 和拒绝路径 (Refusal Path)。在生产环境中，可将模拟输出替换为任意服务提供商的真实响应。

重点关注：

- 验证器会返回一个包含路径和消息的类型化 `[ValidationError]` 列表。这正是你希望在重试提示词 (Retry Prompt) 中暴露的数据结构。
- 拒绝分支不会触发重试。它会记录日志并返回类型化的拒绝结果。第 14 阶段 · 09 将拒绝结果用作安全信号 (Safety Signal)。
- 在对抗性测试输入上，`additionalProperties: false` 校验会被触发，这直观展示了严格模式 (Strict Mode) 为何能彻底阻断幻觉字段 (Hallucinated Fields) 的生成。

## 交付它

本课时将生成 `outputs/skill-structured-output-designer.md` 文件。给定一个自由文本提取目标（如发票、工单、简历等），该技能会生成一个兼容严格模式的 JSON Schema 2020-12，以及与之镜像对应的 Pydantic 模型 (Pydantic Model)，并预置了类型化的拒绝与重试处理存根 (Stub)。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，将其 `total_usd` 设为负数。确认验证器会通过 `minimum` 约束路径将其拒绝。

2. 扩展验证器以支持带鉴别器 (Discriminator) 的 `oneOf`。常见场景：`line_item` 要么是产品，要么是服务，通过 `kind` 字段进行标记。严格模式在此处有较为微妙的规则；请查阅 OpenAI 的结构化输出指南 (Structured Outputs Guide)。

3. 将相同的 Invoice 模式编写为 Pydantic 的 `BaseModel`，并将 `model_json_schema()` 的输出与你手动编写的模式进行对比。找出 Pydantic 默认设置但手动版本中遗漏的那个字段。

4. 测量拒绝率。构造十个不应被提取的输入（如歌词、数学证明、空白邮件），并在开启严格模式的真实服务提供商上运行它们。统计拒绝次数与幻觉输出的比例。这将作为你实现拒绝感知重试 (Refusal-Aware Retries) 的基准真相 (Ground Truth)。

5. 通读 OpenAI 的结构化输出指南。找出它在严格模式下明确禁止，但普通 JSON Schema 允许的那一种语法结构。然后设计一个非必需使用该结构的模式，并将其重构为严格模式兼容的版本。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | “模式规范” | 现代大模型服务商普遍支持的 IETF 草案（IETF-draft）模式方言 |
| 严格模式（Strict mode） | “保证符合模式” | OpenAI 的配置标志，通过约束解码（Constrained decoding）强制模型输出符合模式 |
| 约束解码（Constrained decoding） | “Logit 掩码（Logit masking）” | 解码阶段的强制机制，用于屏蔽无效的下一个 token |
| 拒绝响应（Refusal） | “模型拒绝” | 当输入无法适配模式时返回的类型化结果 |
| 解析错误（Parse error） | “无效 JSON” | 输出未能解析为 JSON；在严格模式下不会发生 |
| 模式违规（Schema violation） | “结构错误” | 虽成功解析，但违反了类型 / 必填项 / 枚举值 / 取值范围限制 |
| `additionalProperties: false` | “不允许额外字段” | 禁止出现未定义字段；OpenAI 严格模式的强制要求 |
| Pydantic BaseModel | “类型化输出” | 用于生成并校验 JSON Schema 的 Python 类 |
| Zod 模式（Zod schema） | “TypeScript 输出类型” | 用于校验服务商输出的 TypeScript 运行时模式 |
| 语法强制（Grammar enforcement） | “开源模型约束解码” | 基于有限状态机（FSM）的 Logit 掩码技术，常见于 outlines / guidance 等库 |

## 扩展阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝响应及模式要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布的公告，详细解释了解码保证机制
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 类型化的 `output_type` 绑定，可序列化适配各服务商
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 权威规范文档
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业级部署说明及严格模式注意事项
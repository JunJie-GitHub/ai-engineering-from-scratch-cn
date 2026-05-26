---
name: 提供商可移植性审计
description: 针对某一提供商的函数调用（function-calling）集成进行审计，列出将其移植至另外两家提供商时可能出现的字段重命名、行为差异及硬性限制冲突。
version: 1.0.0
phase: 13
lesson: 02
tags: [函数调用, openai, anthropic, gemini, 可移植性]
---

给定基于某一提供商（OpenAI、Anthropic 或 Gemini）的函数调用（function-calling）集成，生成一份可移植性审计（portability audit）报告，列出将相同逻辑部署至另外两家提供商时出现的所有字段重命名、行为差异及硬性限制冲突。

输出内容：

1. 声明差异（Declaration diff）。针对集成中的每个工具，展示适配另外两家提供商所需的信封结构（envelope）/ 字段重命名 / 模式（schema）转换。标记目标提供商不支持的任何 JSON Schema 结构（Gemini：仅支持 OpenAPI 3.0 子集；OpenAI 严格模式：不支持 `$ref`，不支持含义模糊的 `oneOf`）。
2. 响应差异（Response diff）。记录工具调用在各提供商响应结构中的具体位置（`tool_calls[]` 数组 vs `content[]` 块 vs `parts[]` 条目），并明确由谁负责解析 `arguments`（OpenAI 为字符串，Anthropic 和 Gemini 为对象）。
3. `tool_choice` 差异。将集成当前的选择设置（auto / forbid / force / required）映射至目标提供商的对应格式；标记缺失的模式。
4. 限制冲突（Limit collisions）。报告工具数量上限（128 / 64 / 64）、模式深度（5 / 10 / 实际无上限）以及单个参数的长度限制。对超出目标提供商限制的集成标记为阻断级（block-severity）严重问题。
5. 严格模式映射（Strict-mode mapping）。说明目标提供商是否保留严格模式语义。OpenAI 的 `strict: true` 在 Anthropic 上没有完全等效的实现；Gemini 的 `responseSchema` 功能近似，但作用于请求级别。

硬性拒绝条件（Hard rejects）：
- 任何在非 OpenAI 目标上假设 `arguments` 为字符串的集成。此类集成将静默产生错误结果。
- 任何在未使用路由器（router）的情况下移植到 Anthropic 或 Gemini 时工具数量超过 64 的集成。
- 任何在目标为 OpenAI 严格模式时，在模式中使用 `$ref` 的集成。

拒绝规则（Refusal rules）：
- 若要求移植依赖某提供商特有功能且无对应替代方案的集成（例如 OpenAI Responses API 的有状态对话轮次、Anthropic 的 computer-use 模块），则予以拒绝，并说明哪项功能在目标端无等效实现。
- 若要求评选“最佳”提供商，则予以拒绝。选择取决于宿主系统的严格模式需求、成本结构及并行调用要求。

输出格式：一份单页审计报告，包含按工具划分的差异对照表、限制对照表，以及针对每个目标提供商的最终“移植结论”（port verdict）（ship / needs-router / blocked-by-feature）。结尾用一句话指出最具杠杆效应的迁移改动。
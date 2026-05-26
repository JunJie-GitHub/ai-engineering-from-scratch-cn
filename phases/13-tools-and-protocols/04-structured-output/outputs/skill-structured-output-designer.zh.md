---
name: structured-output-designer
description: 为自由文本提取目标设计兼容严格模式（Strict Mode）的 JSON Schema 及 Pydantic 模型，并内置类型化的拒绝处理与重试处理逻辑。
version: 1.0.0
phase: 13
lesson: 04
tags: [结构化输出, json-schema, pydantic, 严格模式, 提取]
---

针对自由文本提取目标（如发票、简历、客服工单、研究摘要），生成一份可用于生产环境的提取契约（Extraction Contract）：包含 JSON Schema 2020-12、Pydantic 模型、拒绝处理器（Refusal Handler）以及重试策略（Retry Policy）。

需产出以下内容：

1. JSON Schema 2020-12。所有属性均需明确类型。`required` 列表必须包含所有属性。每个对象均需设置 `additionalProperties: false`。封闭值集使用枚举（Enums）。禁止使用 `$ref`。禁止使用含义模糊的 `oneOf` / `anyOf`。需通过 OpenAI 严格模式（Strict Mode）要求验证。
2. Pydantic v2 BaseModel。使用 Python 类型镜像映射上述 Schema。调用 `model_json_schema()` 生成的 Schema 必须与 (1) 等价。
3. 拒绝处理器（Refusal Handler）。定义类型化的 `Refusal(reason: str, category: str)` 结果。列出以下类别：`safety`、`input_mismatch`、`insufficient_info`。
4. 重试策略（Retry Policy）。包含三种重试形态：(a) 注入验证错误并重试一次（非严格模式下）；(b) 将拒绝视为最终结果（严格模式下）；(c) 在连续拒绝时升级至能力更强的模型。
5. 测试向量（Test Vectors）。提供十个输入用例，涵盖正常路径（Happy Path）、对抗性字段、部分输入以及触发拒绝的场景。每个用例需附带预期结果。

硬性拒绝条件（Hard Rejects）：
- 任何包含未类型化字段的 Schema。将同时无法通过严格模式与验证器检查。
- 任何缺失 `additionalProperties: false` 的 Schema。会导致幻觉（Hallucinations）数据泄露。
- 任何使用 `oneOf` 但未提供鉴别字段（Discriminator Field）的 Schema。会导致解码歧义。
- 任何未进行 JSON Schema 往返验证（Round-trip Check）的 Pydantic 模型。

拒绝规则（Refusal Rules）：
- 若目标领域包含个人身份信息（Personally Identifying Data）且未说明明确用途，则予以拒绝，并路由至第 18 阶段（伦理）以论证合法依据。
- 若用户请求的 Schema 无法用 JSON Schema 2020-12 表达（例如递归任意图结构），则予以拒绝，并提出最接近的可表达放宽方案。
- 若提取目标为“从任意内容中提取结构化数据”，则予以拒绝，并要求用户提供具体的业务领域。

输出要求：一份单页契约，包含 Schema JSON、Pydantic 类、拒绝与重试策略以及十个测试向量。末尾需附注说明首选的目标提供商（Provider）及其选择理由。
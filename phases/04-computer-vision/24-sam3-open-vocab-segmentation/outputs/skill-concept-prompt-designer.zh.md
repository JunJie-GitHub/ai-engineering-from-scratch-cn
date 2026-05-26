---
name: skill-concept-prompt-designer
description: 将用户输入转化为格式规范的 SAM 3 概念提示词（Concept Prompt），支持拆分、消歧（Disambiguation）与回退（Fallback）机制
version: 1.0.0
phase: 4
lesson: 24
tags: [sam3, open-vocab, prompt-engineering, segmentation]
---

# 概念提示词设计器（Concept Prompt Designer）

SAM 3 的准确率在很大程度上取决于概念提示词（Concept Prompt）的表述方式。本技能旨在将自由格式的用户输入规范化为 SAM 3 能够高效处理的提示词。

## 适用场景

- 构建支持自然语言对象查询的用户界面（User Interface, UI）。
- 通过应用程序接口（Application Programming Interface, API）暴露 SAM 3 服务，且上游调用方发送的是完整句子。
- 调试 SAM 3 匹配效果不佳的问题——通常是因为提示词格式不规范，而非模型本身的问题。

## 输入参数

- `utterance`：原始用户输入字符串。
- `context`：可选的领域提示（例如 "surveillance"（监控）、"medical"（医疗）、"retail"（零售））。
- `max_concepts`：单次输入最多提取的概念数量；默认值为 5。

## SAM 3 偏好的提示词规则

- **使用简短的名词短语，而非完整句子。** `"cat"` 优于 `"there is a cat"`。
- **使用具体名词。** `"skateboard"` 优于 `"thing to ride on"`。
- **修饰语紧贴名词前置。** `"red car"` 优于 `"car that is red"`。
- **全小写。** SAM 3 虽具备较强的鲁棒性（Robustness），但经验表明全小写输入的效果略优。
- **单数或复数均可。** 两者皆有效；当预期存在多个实例时，使用复数形式更有帮助。

## 处理步骤

1. **按常见分隔符进行分词（Tokenise）**——逗号、分号、"and"、"or"、"&"。
2. **剔除填充前缀**——"find"、"show me"、"segment"、"detect"、"locate"、"a"、"an"、"the"。
3. **仅保留视觉相关的介词修饰语**——例如 `"striped red umbrella"` 保留，而 `"umbrella from yesterday"` 不保留（因为 `"from yesterday"` 不属于图像内容）。
4. **利用可选的 `context` 进行消歧（Disambiguate）**：
   - 监控场景下的 `"window"` -> 转换为 `"building window"`。
   - 医疗场景下的 `"window"` -> 通常会导致错误；建议用户明确具体指代。
5. **回退（Fallback）机制**：若拆分后未提取到任何概念，*且* 输入中至少包含一个具体名词，则回退至原始字符串。若无法提取任何具体名词，则不生成概念——仅返回警告信息并要求用户澄清（参见规则部分）。
6. **限制为 `max_concepts` 上限**。若提取的概念数量超过调用方请求的数量，则按输入顺序保留前 `max_concepts` 个概念，其余概念放入 `dropped` 字段并标注原因为 `"exceeded max_concepts"`。此举可在用户粘贴长列表时有效控制延迟（Latency）。

## 输出格式

[designed prompts]
  utterance:    <original>
  concepts:     ["concept_1", "concept_2", ...]
  dropped:      ["filler_1", ...]
  warnings:     ["concept too abstract", "may match many classes", ...]

[sam3 calls]
  For each concept run: sam3.detect(image, concept)
  Merge outputs with distinct concept tags per detection.

## 示例

in:  "can you find me a cat or two dogs?"
out: ["cat", "dogs"]
dropped: ["can you find me", "a", "or two", "?"]
note: "dogs" kept plural because the utterance says "two dogs" — plural hint preserved.

in:  "segment the big red truck and the blue sedan"
out: ["big red truck", "blue sedan"]
dropped: ["segment", "the", "and"]

in:  "thing near the door"
out: ["door"]
warnings: ["'thing' is too abstract for SAM 3; fell back to 'door'"]

in:  "striped red umbrella, green hat, pink balloon"
out: ["striped red umbrella", "green hat", "pink balloon"]

## 规则

- 切勿将超过 8 个单词的句子直接传入 SAM 3——超过此长度会导致准确率下降。
- 当输入中不包含可提取的具体名词时，不要调用 SAM 3；应返回警告信息并要求用户澄清。
- 不要拆分引号内的标点符号；若 `"black and white cat"` 被引号包裹，则应将其保留为单个概念。
- 始终记录原始输入与派生出的概念，以便于生产环境调试。
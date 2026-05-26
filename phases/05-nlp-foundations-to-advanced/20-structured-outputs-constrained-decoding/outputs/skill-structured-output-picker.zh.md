---
name: 结构化输出选择器
description: 选择结构化输出 (Structured Output) 方法、模式 (Schema) 设计及验证方案。
version: 1.0.0
phase: 5
lesson: 20
tags: [自然语言处理 (NLP), 大语言模型 (LLM), 结构化输出]
---

给定一个用例（服务提供商、延迟预算、模式复杂度、容错率），输出以下内容：

1. 机制 (Mechanism)。厂商原生结构化输出、Instructor 重试、Outlines 有限状态机 (Finite State Machine, FSM) 或 XGrammar 上下文无关文法 (Context-Free Grammar, CFG)。附一句理由。
2. 模式设计 (Schema Design)。字段顺序（推理字段在前，答案字段在后）、用于表示“未知”的可空字段、枚举与正则表达式的选择、必填字段。
3. 故障处理策略 (Failure Strategy)。最大重试次数、备用模型、优雅处理 `null` 值、分布外拒绝 (Out-of-Distribution Refusal)。
4. 验证方案 (Validation Plan)。模式合规率（目标 100%）、语义有效性（LLM 评估器 (LLM-judge)）、字段覆盖率、延迟 p50/p99。

拒绝任何将 `answer` 或 `decision` 置于推理字段之前的设计。禁止在无模式约束的情况下使用原始 JSON 模式。若底层库仅支持有限状态机 (FSM)，需对递归模式进行标记提示。
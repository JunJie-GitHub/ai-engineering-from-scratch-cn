# 结构化输出 (Structured Outputs) 与约束解码 (Constrained Decoding)

> 向大语言模型 (LLM) 请求 JSON 输出。大多数时候你能如愿以偿。但在生产环境中，“大多数时候”恰恰是隐患所在。约束解码通过在采样前修改 logits，将“大多数时候”变为“始终”。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 17（聊天机器人 (Chatbots)），第 5 阶段 · 19（子词分词 (Subword Tokenization)）
**耗时：** 约 60 分钟

## 问题所在

分类器向大语言模型发送提示：“请返回 {positive, negative, neutral} 中的一个。”模型却回复：“情感倾向为正面——该评论极为正面，因为客户明确表示他们……”。你的解析器直接崩溃，分类器的 F1 分数 (F1 Score) 跌至 0.0。

自由文本生成并非契约，而仅仅是建议。生产级系统需要的是契约。

目前主要存在三种实现层级。

1. **提示词引导 (Prompting)。** 礼貌地提出要求：“仅返回 JSON 对象。”在前沿模型上成功率约为 80%，在较小模型上则更低。
2. **原生结构化输出 API (Native Structured Output APIs)。** 例如 OpenAI 的 `response_format`、Anthropic 的工具调用 (Tool Use) 以及 Gemini 的 JSON 模式。在支持的 Schema 上表现可靠，但存在厂商锁定 (Vendor Lock-in) 问题。
3. **约束解码 (Constrained Decoding)。** 在每一步生成过程中修改 logits，使模型*无法*输出无效 Token。从构造上保证 100% 有效。适用于任何本地部署的模型。

本课程将帮助你建立对这三种方案的直观理解，并明确在不同场景下应如何选择。

## 核心概念

![约束解码在每一步屏蔽无效 token](../assets/constrained-decoding.svg)

**约束解码（Constrained Decoding）的工作原理。** 在每一步生成过程中，大语言模型（LLM）会针对完整词表（约 10 万个 token）输出一个 logit 向量（logit vector）。一个 *logit 处理器（logit processor）* 位于模型与采样器（sampler）之间。它会根据目标语法（如 JSON Schema、正则表达式（regex）或上下文无关文法（context-free grammar））的当前位置，计算出哪些 token 是有效的，并将所有无效 token 的 logit 值设为负无穷。对剩余 logit 进行 softmax 操作后，概率质量将仅分配给有效的后续 token。

2026 年的主流实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译为有限状态机（finite-state machine, FSM）。每个 token 均可实现 O(1) 复杂度的有效下一 token 查找。基于 FSM 架构，因此递归型 Schema 需要进行扁平化处理。
- **XGrammar / llguidance。** 上下文无关文法引擎。支持处理递归型 JSON Schema。解码开销接近于零。OpenAI 在其 2025 年的结构化输出（structured output）实现中致谢了 llguidance。
- **vLLM 引导解码（guided decoding）。** 通过 Outlines、XGrammar 或 lm-format-enforcer 后端，内置了 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar` 等功能。
- **Instructor。** 基于 Pydantic 的任意 LLM 封装库。在验证失败时会自动重试。支持跨提供商使用，但不会修改 logit 值——它依赖于重试机制与具备结构化输出感知能力的提示词（prompts）。

### 反直觉的结果

约束解码通常比无约束生成（unconstrained generation）*更快*。原因有二。首先，它缩小了下一 token 的搜索空间。其次，巧妙的实现会完全跳过强制 token 的生成过程（例如 `{"name": "` 这类脚手架结构（scaffolding）——每个字节都是预先确定的）。

### 代价高昂的陷阱

字段顺序至关重要。如果将 `answer` 放在 `reasoning` 之前，模型会在思考之前就锁定答案。生成的 JSON 格式合法，但答案是错的。任何验证机制都无法捕获此类错误。

// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}

Schema 中的字段顺序代表的是逻辑，而非单纯的格式排版。

## 动手构建

### 步骤 1：从零开始实现正则约束生成（regex-constrained generation）

独立有限状态机（Finite State Machine, FSM）的实现请参考 `code/main.py`。其核心思想仅需 30 行代码：

def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)

FSM 会追踪我们目前已满足的语法规则部分。`valid_tokens(state, tokenizer)` 用于计算词表中的哪些词元（token）能够推动 FSM 向前转移，且不会偏离可接受路径。

### 步骤 2：使用 Outlines 处理 JSON 模式（JSON Schema）

from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')

验证错误率为零。永远如此。FSM 使得无效输出在逻辑上变得不可达。

### 步骤 3：使用 Instructor 实现与提供商无关（provider-agnostic）的 Pydantic 集成

import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)

机制不同。Instructor 不会干预对数几率（logits）。它将模式（schema）格式化后嵌入提示词（prompt），解析模型输出，并在验证失败时重试（默认 3 次）。适用于任何服务提供商。重试会增加延迟和成本，但其核心卖点在于跨提供商的可移植性。

### 步骤 4：原生厂商 API

from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)

服务端约束解码（constrained decoding）。对于支持的模式（schema），其可靠性与 Outlines 相当。无需本地模型管理，但会将你绑定至特定厂商。

## 常见陷阱

- **递归模式 (Recursive schemas)。** Outlines 会将递归结构展平为固定深度。树状结构输出（如嵌套注释、抽象语法树 (AST)）需使用 XGrammar 或 llguidance（基于上下文无关文法 (CFG)）。
- **庞大枚举 (Huge enums)。** 包含 10,000 个选项的枚举编译缓慢或会超时。建议改用检索器 (Retriever)：先预测 Top-K 候选项，再将输出限制在这些候选项内。
- **约束过于严格 (Grammar too strict)。** 若强制使用 `date: "YYYY-MM-DD"` 正则表达式 (Regex)，模型将无法为缺失日期输出 `"unknown"`。模型会通过编造日期来进行补偿。应允许 `null` 或设置哨兵值 (Sentinel)。
- **过早锁定 (Premature commitment)。** 参见上文关于字段顺序的陷阱。务必将推理过程放在首位。
- **未提供模式的供应商 JSON 模式 (Vendor JSON mode without schema)。** 纯 JSON 模式仅能保证输出合法的 JSON 格式，无法保证符合*你的具体用例*。务必提供完整的模式 (Schema)。

## 选型指南

2026 年技术栈选型：

| 场景 | 推荐方案 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单模式 | 原生供应商结构化输出 (Native vendor structured output) |
| 任意提供商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需 100% 有效性，扁平模式 (Flat schema) | Outlines（有限状态机 (FSM)） |
| 本地模型，递归模式 | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导解码 (Guided decoding) |
| 批量处理，可接受重试 | Instructor + 成本最低的模型 |

## 交付指南

保存为 `outputs/skill-structured-output-picker.md`：

---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.

## 练习

1. **简单。** 提示一个小型开源权重模型（例如 Llama-3.2-3B），在不使用约束解码 (Constrained decoding) 的情况下生成 `Review(sentiment, confidence, evidence_span)`。在 100 条评论中，测量能解析为有效 JSON 的比例。
2. **中等。** 使用相同语料库，但改用 Outlines 的 JSON 模式。对比合规率 (Compliance rate)、延迟和语义准确性 (Semantic accuracy)。
3. **困难。** 从零开始实现一个用于电话号码（`\d{3}-\d{3}-\d{4}`）的正则约束解码器 (Regex-constrained decoder)。在 1000 个样本上验证无效输出为 0。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 约束解码 (Constrained Decoding) | 强制输出有效内容 | 在每一步生成时，屏蔽无效 token 的 logits。 |
| Logit 处理器 (Logit Processor) | 实现约束的组件 | 函数：`(logits, state) -> masked_logits`。 |
| 有限状态机 (Finite-State Machine, FSM) | 有限状态机 | 编译后的语法表示形式；支持 O(1) 复杂度的有效下一 token 查找。 |
| 上下文无关文法 (Context-Free Grammar, CFG) | 上下文无关文法 | 支持递归处理的语法；速度较慢但表达能力强于 FSM。 |
| Schema 字段顺序 (Schema Field Order) | 重要吗？ | 重要——首个字段会优先提交；务必将推理过程置于答案之前。 |
| 引导解码 (Guided Decoding) | vLLM 中的叫法 | 概念相同，已集成至推理服务器中。 |
| JSON 模式 (JSON Mode) | OpenAI 的早期版本 | 仅保证 JSON 语法正确；不保证符合预定义的 Schema。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 库的奠基论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 基于 CFG 的快速约束解码方案。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成方案。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考文档及常见陷阱。
- [Instructor library](https://python.useinstructor.com/) — 基于 Pydantic 的跨提供商重试封装库。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 针对 6 种约束解码框架的基准测试。
# 对话状态跟踪 (Dialogue State Tracking)

> “我想要一家北区的便宜餐厅……其实改成中等价位的吧……再加个意大利菜。”三轮对话，三次状态更新。DST 保持槽值字典同步，从而确保预订流程顺利执行。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 17（聊天机器人），第 5 阶段 · 20（结构化输出）
**耗时：** 约 75 分钟

## 核心问题

在任务型对话系统 (task-oriented dialogue system) 中，用户的目标被编码为一组槽值对 (slot-value pairs)：`{cuisine: italian, area: north, price: moderate}`。用户的每一轮交互都可能增加、修改或删除某个槽位 (slot)。系统必须读取完整的对话历史，并准确输出当前状态。

只要有一个槽位出错，系统就可能订错餐厅、排错航班或刷错银行卡。DST 是连接用户意图与后端执行逻辑的关键枢纽。

尽管大语言模型 (Large Language Model, LLM) 已广泛应用，但 DST 在 2026 年依然至关重要，原因如下：

- 合规敏感型领域（如银行、医疗、机票预订）需要确定性的槽值，而非自由文本生成。
- 工具调用智能体 (tool-use agents) 在调用 API 前仍需完成槽位解析。
- 多轮对话修正比表面看起来更复杂：“其实不用了，改成周四吧。”

现代处理流水线：经典 DST 概念 + LLM 信息抽取器 + 结构化输出护栏 (structured-output guardrails)。

## 核心概念

![DST：对话历史 → 槽位-值状态](../assets/dst.svg)

**任务结构。** 模式（schema）定义了领域（domain，如餐厅、酒店、出租车）及其对应的槽位（slot，如菜系、区域、价格、人数）。每个槽位可以为空，也可以填入来自封闭词表（closed vocabulary）的值（例如价格：{便宜、中等、昂贵}），或是自由格式（free-form）的值（例如名称："The Copper Kettle"）。

**对话状态追踪（Dialogue State Tracking, DST）的两种建模方式。**

- **分类（Classification）。** 针对每个（槽位，候选值）对，预测是/否。适用于封闭词表槽位。2020 年之前的标准做法。
- **生成（Generation）。** 给定对话内容，以自由文本形式生成槽位值。适用于开放词表（open vocabulary）槽位。当前的现代默认方案。

**评估指标。** 联合目标准确率（Joint Goal Accuracy, JGA）——指*所有*槽位均预测正确的对话轮次（turn）所占的比例。采用全对或全错（all-or-nothing）原则。截至 2026 年，MultiWOZ 2.4 排行榜的最高分约为 83%。

**架构。**

1. **基于规则（Rule-based，槽位正则表达式 + 关键词）。** 在窄领域（narrow domain）中表现强劲的基线模型。具备良好的可调试性。
2. **TripPy / BERT-DST。** 基于 BERT 编码的复制式生成（copy-based generation）。大语言模型（LLM）出现前的标准方案。
3. **LDST（LLaMA + LoRA）。** 结合领域-槽位提示（domain-slot prompting）的指令微调（instruction-tuned）大语言模型。在 MultiWOZ 2.4 上达到 ChatGPT 级别的质量。
4. **无本体（Ontology-free，2024–26）。** 跳过预定义模式，直接生成槽位名称和值。适用于开放领域（open domain）。
5. **提示词 + 结构化输出（Prompt + structured output，2024–26）。** 结合 Pydantic 模式（Pydantic schema）与约束解码（constrained decoding）的大语言模型。仅需 5 行代码，即可达到生产就绪（production-ready）状态。

### 典型失效场景

- **跨轮次指代（Cross-turn coreference）。** “我们就选第一个方案吧。”需要解析具体指代的是哪个方案。
- **覆盖与追加（Overwrite vs append）。** 用户说“再加个意大利菜”。你是替换原有的菜系槽位，还是追加新值？
- **隐式确认（Implicit confirmation）。** “好的，没问题”——这是否表示接受了系统提供的预订？
- **修正（Correction）。** “其实改成晚上 7 点吧。”必须更新时间槽位，同时保留其他槽位不变。
- **对系统上一轮话语的指代（Coreference to previous system utterance）。** “对，就那个。”这里的“那个”具体指什么？

## 动手构建

### 步骤 1：基于规则的槽位提取器 (Rule-based Slot Extractor)

参见 `code/main.py`。正则表达式 (Regex) 结合同义词词典能够覆盖窄域 (Narrow Domain) 中 70% 的标准话语 (Canonical Utterances)：

CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None

在标准词汇表之外的场景下表现较为脆弱。该方法适用于确定性的槽位确认 (Deterministic Slot Confirmations) 任务。

### 步骤 2：状态更新循环 (State Update Loop)

def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state

需遵循三个不变性原则 (Invariants)：

- 绝不重置用户未提及的槽位。
- 明确的否定表达（如“不用管菜系了”）必须清空对应槽位。
- 用户的修正表达（如“其实……”）必须覆盖旧值，而非追加。

### 步骤 3：大语言模型驱动的对话状态跟踪 (LLM-driven Dialogue State Tracking, DST) 与结构化输出 (Structured Output)

from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)

Instructor 库与 Pydantic 结合可保证生成合法的状态对象。无需编写正则表达式，避免了模式不匹配问题，且不会产生幻觉槽位 (Hallucinated Slots)。

### 步骤 4：联合目标准确率 (Joint Goal Accuracy, JGA) 评估

def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)

评估基准：系统在多大比例的对话轮次 (Turns) 中能够完全正确地预测所有槽位？以 MultiWOZ 2.4 数据集为例，2026 年顶尖系统的得分在 80%~83% 之间。你的领域内系统 (In-domain System) 在特定词汇范围内的表现理应超越该基准，否则大语言模型基线 (LLM Baseline) 的表现将优于你的系统。

### 步骤 5：处理用户修正 (Handling Correction)

CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)

当检测到修正意图时，应覆盖最近更新的槽位值，而非进行追加。在不借助大语言模型的情况下，很难准确实现这一逻辑。当前的主流做法是：始终让大语言模型根据完整对话历史重新生成整个状态，而非采用增量更新 (Incremental Update) 方式——这种方式能够天然地妥善处理用户修正。

## 常见陷阱 (Pitfalls)

- **完整历史重建成本（Full-history regeneration cost）。** 让大语言模型（Large Language Model, LLM）在每一轮对话中重新生成状态，总共会消耗 O(n²) 的词元（token）。应限制历史记录长度或对较早的对话轮次进行摘要。
- **模式漂移（Schema drift）。** 事后添加新槽位（slot）会破坏旧的训练数据。请为你的模式（schema）添加版本控制。
- **大小写敏感性（Case sensitivity）。** "Italian" 与 "italian" 或 "ITALIAN" 的区别——需在所有环节进行标准化处理。
- **隐式继承（Implicit inheritance）。** 如果用户之前已指定“4人用餐”，后续更改时间的请求不应清空人数信息。务必始终传递完整的历史记录。
- **自由文本与封闭集合（Free-form vs closed-set）。** 姓名、时间和地址需要自由文本槽位；菜系和区域属于封闭集合。应在模式中混合使用两者。

## 实际应用

2026 年技术栈：

| 场景 | 方案 |
|-----------|----------|
| 窄领域（一两个意图） | 基于规则 + 正则表达式 |
| 宽领域，有标注数据可用 | LDST（基于 MultiWOZ 风格数据微调 LLaMA + 低秩自适应 (LoRA)） |
| 宽领域，无标注数据，生产就绪 | LLM + Instructor + Pydantic 模式 |
| 语音/口语 | 自动语音识别 (ASR) + 标准化器 + LLM-对话状态追踪 (DST) |
| 多领域预订流程 | 基于模式引导的 LLM，配合各领域独立的 Pydantic 模型 |
| 合规敏感型 | 基于规则为主，LLM 为辅并附带确认流程 |

## 交付上线

保存为 `outputs/skill-dst-designer.md`：

---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.

## 练习

1. **简单。** 在 `code/main.py` 中构建基于规则的状态追踪器，包含 3 个槽位（菜系、区域、价格）。使用 10 段手工编写的对话进行测试。测量联合目标准确率（Joint Goal Accuracy, JGA）。
2. **中等。** 使用相同的数据集，结合 Instructor + Pydantic + 小型 LLM。对比 JGA。检查最困难的对话轮次。
3. **困难。** 同时实现两者并设置路由：以基于规则为主，当基于规则的方法以置信度输出少于 2 个槽位时，回退至 LLM。测量综合 JGA 及每轮推理成本。

## 关键术语

| 术语 | 常见表述 | 实际含义 |
|------|-----------------|-----------------------|
| DST (Dialogue State Tracking) | 对话状态跟踪 | 在多轮对话中持续维护槽位-值字典 (slot-value dict)。 |
| Slot | 用户意图单元 | 后端系统所需的具名参数（如菜系、日期）。 |
| Domain | 任务领域 | 餐厅、酒店、出租车等业务场景——即特定槽位的集合。 |
| JGA (Joint Goal Accuracy) | 联合目标准确率 | 所有槽位均预测正确的对话轮次比例。采用“全对或全错”的评估标准。 |
| MultiWOZ | 基准数据集 | 多领域 Wizard-of-Oz (WOZ) 数据集；DST 的标准评估基准。 |
| Ontology-free DST | 无预定义模式 (No schema) | 直接生成槽位名称及其对应值，无需依赖固定的槽位列表。 |
| Correction | “其实/实际上……” | 用于覆盖或修改先前已填充槽位的对话轮次。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) —— 该领域的权威基准 (canonical benchmark)。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) —— 针对 DST 的 LLaMA 与 LoRA 指令微调 (instruction tuning) 方法。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) —— 基于复制机制 (copy-based) 的 DST 主力模型。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) —— 基于期望最大化 (EM) 算法的无监督任务型对话 (Task-Oriented Dialogue, TOD) 方法。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) —— 权威的 DST 评测结果。
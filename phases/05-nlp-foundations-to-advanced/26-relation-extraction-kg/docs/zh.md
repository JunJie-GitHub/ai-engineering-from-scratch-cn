# 关系抽取 (Relation Extraction) 与知识图谱构建 (Knowledge Graph Construction)

> 命名实体识别 (Named Entity Recognition, NER) 发现了实体，实体链接 (Entity Linking) 将它们锚定，而关系抽取则找出它们之间的边。知识图谱是节点、边及其溯源信息 (provenance) 的总和。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 06（NER），第 5 阶段 · 25（实体链接）
**耗时：** 约 60 分钟

## 核心问题

分析师读到这样一句话：“蒂姆·库克于 2011 年成为苹果公司的首席执行官。”其中包含四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取 (Relation Extraction, RE) 将自由文本转化为结构化的三元组 `(subject, relation, object)`。在语料库上进行聚合，你就能得到一个知识图谱。进一步聚合与查询，便能为检索增强生成 (Retrieval-Augmented Generation, RAG)、数据分析或合规审计提供推理基础。

2026 年面临的挑战：大语言模型 (Large Language Model, LLM) 在抽取关系时过于“积极”。它们会凭空捏造（幻觉）源文本并不支持的三元组。如果没有溯源信息，你将无法区分真实的三元组与看似合理的虚构内容。2026 年的解决方案是采用 AEVS 风格的“锚定-验证”流水线。

## 核心概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系可以来自封闭本体 (closed ontology)（如 Wikidata 属性、FIBO、UMLS），也可以来自开放集合（类似开放信息抽取 (Open Information Extraction, Open IE) 风格，任意短语均可）。

**三种抽取方法。**

1. **基于规则/模式。** 赫斯特模式 (Hearst patterns)：例如 "X such as Y" → `(Y, isA, X)`。辅以手工编写的正则表达式。脆弱但精确，且具备可解释性。
2. **监督分类器。** 给定句子中的两个实体指称，从固定集合中预测其关系。通常在 TACRED、ACE、KBP 等数据集上训练。这是 2015–2022 年间的标准做法。
3. **生成式大语言模型。** 通过提示词让模型直接输出三元组。开箱即用。但必须配合溯源机制，否则会产生看似合理实则无用的幻觉内容。

**AEVS（锚定-抽取-验证-补充，2026）。** 当前主流的幻觉缓解 (hallucination-mitigation) 框架：

- **锚定 (Anchor)。** 精确识别每个实体片段和关系短语片段的位置。
- **抽取 (Extract)。** 生成与锚定片段相关联的三元组。
- **验证 (Verify)。** 将每个三元组元素回溯匹配至源文本；剔除任何缺乏文本支持的内容。
- **补充 (Supplement)。** 通过覆盖度检查，确保没有任何锚定片段被遗漏。

幻觉现象大幅减少。虽然需要更多计算资源，但具备可审计性。

**开放与封闭的权衡。**

- **封闭本体。** 固定的属性列表（例如 Wikidata 的 11,000+ 个属性）。结果可预测、易于查询，但难以扩展新关系。
- **开放信息抽取 (Open IE)。** 任何动词短语都可作为关系。召回率高，但精确率低，查询时结构混乱。

生产环境的知识图谱通常采用混合策略：先使用开放 IE 进行关系发现，随后将关系规范化映射到封闭本体上，最后再合并至主图谱中。

## 动手实践

### 步骤 1：基于模式的提取 (pattern-based extraction)

PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]

完整的简易提取器代码请参见 `code/main.py`。Hearst 模式 (Hearst patterns) 因其具备良好的可调试性，至今仍被广泛应用于特定领域的处理流水线中。

### 步骤 2：监督式关系分类 (supervised relation classification)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)

REBEL 是一种序列到序列 (seq2seq) 关系提取器 (relation extractor)：输入文本，输出三元组 (triples)，且结果已直接对应 Wikidata 属性 ID (Wikidata property IDs)。该模型在远程监督 (distant supervision) 数据上进行了微调，属于标准的开源权重基线模型 (open-weights baseline)。

### 步骤 3：结合锚点定位的大语言模型提示词提取 (LLM-prompted extraction with anchoring)

prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""

需将模型返回的每个字符跨度 (span) 与原文进行严格比对。凡是不满足 `text[start:end] != triple_entity` 的结果均应剔除。这便是 AEVS “验证”步骤的最简实现形式。

### 步骤 4：向封闭本体进行规范化映射 (canonicalize onto a closed ontology)

RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review

规范化处理 (canonicalization) 通常占据整个工程工作量的 60% 到 80%。请务必为此预留充足的资源与时间。

### 步骤 5：构建小型图并进行查询 (build a small graph and query)

triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]

这是所有基于知识图谱的检索增强生成 (RAG-over-KG) 系统的核心基础组件。可通过 RDF 三元组存储 (RDF triple stores)（如 Blazegraph、Virtuoso）、属性图 (property graphs)（如 Neo4j）或向量增强型图存储 (vector-augmented graph stores) 对其进行规模化扩展。

## 常见陷阱 (Pitfalls)

- **关系抽取（Relation Extraction, RE）前先进行指代消解（Coreference Resolution）。** “他创立了苹果”——关系抽取需要明确“他”指代的对象。优先运行指代消解模块（第24课）。
- **实体规范化（Entity Canonicalization）。** “Apple Inc”和“Apple”必须映射到同一个节点。优先进行实体链接（Entity Linking）（第25课）。
- **幻觉三元组（Hallucinated Triples）。** 大语言模型（Large Language Models, LLMs）会生成原文本不支持的三元组。必须强制执行跨度验证（Span Verification）。
- **关系规范化漂移（Relation Canonicalization Drift）。** 开放信息抽取（Open Information Extraction, Open IE）生成的关系表述不一致（如“出生于”、“来自”、“是……本地人”）。必须将其归一化为规范标识符（Canonical IDs），否则图谱将无法查询。
- **时序错误（Temporal Errors）。** “蒂姆·库克是苹果CEO”——现在为真，但在2005年为假。许多关系具有时效性。请使用限定词（Qualifiers）（如维基数据（Wikidata）中的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配（Domain Mismatch）。** REBEL 模型基于维基百科训练。法律、医疗和科学文本通常需要经过领域微调（Domain-Fine-Tuned）的关系抽取模型。

## 实践应用

2026 技术栈：

| 场景 | 推荐方案 |
|-----------|------|
| 快速投产，通用领域 | 结合维基数据规范化的 REBEL 或 LlamaPred |
| 特定领域（生物医学、法律） | 类 SciREX 的领域微调 + 自定义本体（Ontology） |
| 基于大语言模型提示词，需审计输出 | AEVS 流水线：锚定 → 抽取 → 验证 → 补充 |
| 海量新闻信息抽取（Information Extraction, IE） | 基于规则 + 监督学习的混合方案 |
| 从零构建知识图谱（Knowledge Graph, KG） | 开放信息抽取 + 人工规范化处理 |
| 时序知识图谱 | 带限定词抽取（开始/结束时间、时间点） |

集成模式：命名实体识别（Named Entity Recognition, NER） → 指代消解 → 实体链接 → 关系抽取 → 本体映射（Ontology Mapping） → 图谱加载（Graph Load）。每个阶段都是潜在的质量关卡（Quality Gate）。

## 交付落地

保存为 `outputs/skill-re-designer.md`：

---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).

## 练习

1. **简单。** 在 `code/main.py` 中运行模式提取器（pattern extractor），对 5 条新闻句子进行处理。人工核对精确率（precision）。
2. **中等。** 对相同的句子使用 REBEL（或小型大语言模型（LLM））。对比提取出的三元组（triples）。哪个提取器的精确率更高？召回率（recall）更高？
3. **困难。** 构建 AEVS 流水线（pipeline）：使用大语言模型进行提取，并对照源文本验证文本跨度（spans）。在 50 条维基百科风格的句子上，测量执行验证步骤前后的幻觉率（hallucination rate）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 三元组（Triple） | 主-谓-宾结构 | `(s, r, o)` 元组，是知识图谱（KG）的原子单元。 |
| 开放信息抽取（Open IE） | 抽取任意内容 | 开放词汇关系短语；高召回率，低精确率。 |
| 封闭本体（Closed ontology） | 固定模式 | 有限的关系类型集合（如 Wikidata、UMLS、FIBO）。 |
| 规范化（Canonicalization） | 统一标准化 | 将表层名称/关系映射到规范标识符。 |
| AEVS | 可溯源抽取（Grounded extraction） | 锚定-抽取-验证-补充流水线（2026）。 |
| 溯源信息（Provenance） | 真实来源链接 | 每个三元组均携带文档 ID 及指向其来源的字符跨度（char-span）。 |
| 远程监督（Distant supervision） | 廉价标签 | 将文本与现有知识图谱对齐以构建训练数据。 |

## 延伸阅读

- [Mintz 等人 (2009). 无标注数据关系抽取的远程监督](https://www.aclweb.org/anthology/P09-1113.pdf) —— 远程监督（distant supervision）领域的开创性论文。
- [Huguet Cabot, Navigli (2021). REBEL：基于端到端语言生成的关系抽取](https://aclanthology.org/2021.findings-emnlp.204.pdf) —— 序列到序列（seq2seq）关系抽取（RE）的主力模型。
- [Wadden 等人 (2019). 基于上下文跨度表示的实体、关系与事件抽取 (DyGIE++)](https://arxiv.org/abs/1909.03546) —— 联合信息抽取（joint IE）方法。
- [AEVS —— 锚定-抽取-验证-补充框架](https://www.mdpi.com/2073-431X/15/3/178) —— 2026 年提出的幻觉缓解（hallucination-mitigation）设计方案。
- [Wikidata SPARQL 教程](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) —— 标准图查询指南。
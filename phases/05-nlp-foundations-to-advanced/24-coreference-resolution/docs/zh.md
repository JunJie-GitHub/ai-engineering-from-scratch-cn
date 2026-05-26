# 指代消解 (Coreference Resolution)

> “她给他打了电话。他没有接听。医生正在吃午饭。”三处指代涉及两个人，但均未直呼其名。指代消解的任务就是弄清楚谁是谁。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段 · 06（命名实体识别 (NER)），第5阶段 · 07（词性标注 (POS) 与句法分析 (Parsing)）
**预计耗时：** 约60分钟

## 问题定义

从一篇300词的文章中提取所有提及“苹果公司（Apple Inc.）”的内容。当文章直接写“Apple”时很简单；但当它使用“该公司”、“他们”、“库珀蒂诺的科技巨头”或“乔布斯的公司”时，难度就大大增加。如果不将这些指代消解为同一个实体，你的命名实体识别流水线将漏掉60%到80%的提及。

指代消解将所有指向同一现实世界实体的表达归入同一个簇。它是连接表层自然语言处理 (NLP)（如命名实体识别、句法分析）与下游语义任务（如信息抽取 (IE)、问答系统 (QA)、文本摘要 (Summarization)、知识图谱 (KG)）的关键纽带。

为何在2026年它依然至关重要：

- 文本摘要：“首席执行官宣布……”与“蒂姆·库克宣布……”——摘要中应明确写出该首席执行官的姓名。
- 问答系统：“她给谁打了电话？”这一问题需要先消解“她”的指代对象。
- 信息抽取：如果知识图谱将“PER1创立了苹果”和“乔布斯创立了苹果”作为两个独立条目存储，那就是错误的。
- 跨文档信息抽取：将多篇报道同一事件的文章中的指代进行合并，属于跨文档指代消解。

## 核心概念

![共指聚类（Coreference clustering）：指称项（mentions） → 实体（entities）](../assets/coref.svg)

**任务。** 输入：一篇文档。输出：指称项（mentions）的聚类（clustering）结果，其中每个聚类对应一个实体（entity）。

**指称项类型。**

- **命名实体（Named entity）。** "Tim Cook"
- **名词性指称（Nominal）。** "the CEO", "the company"
- **代词性指称（Pronominal）。** "he", "she", "they", "it"
- **同位语（Appositive）。** "Tim Cook, Apple's CEO,"

**模型架构。**

1. **基于规则的方法（Rule-based, Hobbs, 1978）。** 基于句法树（syntactic tree）和语法规则进行代词消解。作为基线（baseline）表现优异，且在代词消解任务上出人意料地难以被超越。
2. **指称对分类器（Mention-pair classifier）。** 对每一对指称项 $(m_i, m_j)$ 预测其是否共指（corefer），并通过传递闭包（transitive closure）进行聚类。这是 2016 年之前的标准方法。
3. **指称排序（Mention-ranking）。** 为每个指称项对候选先行词（candidate antecedents，含“无先行词”选项）进行排序，并选取排名最高者。
4. **基于跨度的端到端模型（Span-based end-to-end, Lee et al., 2017）。** 采用 Transformer 编码器。枚举所有不超过预设长度上限的候选跨度（spans）。预测指称项得分，并为每个跨度预测其作为先行词的概率。最后采用贪心策略（greedy）进行聚类。这是当前的现代默认架构。
5. **生成式方法（Generative, 2024+）。** 通过提示词（prompt）引导大语言模型（LLM）：“列出文本中的每个代词及其对应的先行词。”在简单场景下效果良好，但在处理长文档和罕见指代对象（referents）时表现欠佳。

**评估指标。** 采用五种标准指标（MUC、B³、CEAF、BLANC、LEA），因为单一指标无法全面衡量聚类质量。通常将前三项指标的平均值报告为 CoNLL F1 分数。截至 2026 年，在 CoNLL-2012 数据集上的最先进水平（State-of-the-art）约为 83 F1。

**已知的困难案例。**

- 定指描述（definite descriptions）指代前文数页才引入的实体。
- 桥接回指（bridging anaphora）（例如 “the wheels” 指代前文提及的汽车）。
- 中文和日语等语言中的零形回指（zero anaphora）。
- 前指（cataphora，代词出现在指代对象之前）：“When **she** walked in, Mary smiled.”

## 动手构建

### 步骤 1：预训练神经共指消解（Coreference Resolution）（AllenNLP / spaCy-experimental）

import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])

在较长的文档上，你会得到类似以下的结果：
- 聚类簇 1：[Apple, The company, they]
- 聚类簇 2：[new products]

### 步骤 2：基于规则的代词解析器（Rule-based Pronoun Resolver）（教学示例）

仅使用标准库的实现请参考 `code/main.py`：

1. 提取指称项（Mentions）：命名实体（大写片段）、代词（字典查找）、定指描述（如 "the X"）。
2. 针对每个代词，查看前 K 个指称项，并根据以下规则进行打分：
   - 性数一致（启发式规则）
   - 邻近度（距离越近得分越高）
   - 句法角色（优先选择主语）
3. 将得分最高的先行词（Antecedent）进行链接。

其性能无法与神经模型竞争，但它清晰地展示了搜索空间以及端到端模型必须做出的决策。

### 步骤 3：使用大语言模型（LLM）进行共指消解

prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""

需要警惕两种失败模式。首先，大语言模型容易过度合并（例如将分别指代两个不同人的 "him" 和 "her" 合并为一类）。其次，在处理长文档时，大语言模型可能会静默遗漏部分指称项。务必通过跨度偏移量（Span-offset）检查进行验证。

### 步骤 4：评估

标准的 CoNLL-2012 评估脚本会计算 MUC、B³ 和 CEAF-φ4 指标，并报告其平均值。对于内部评估，建议先在标注好的测试集上计算跨度级（Span-level）的精确率（Precision）和召回率（Recall），然后再加入指称项链接的 F1 分数。

## 常见陷阱

- **单例爆炸（Singleton Explosion）。** 某些系统会将每个指称项都报告为独立的聚类簇。B³ 指标对此较为宽容，而 MUC 指标会予以惩罚。务必同时检查这三项指标。
- **长上下文中的代词。** 当文档超过 2,000 个词元（Token）时，性能会下降约 15 个 F1 分数。需谨慎进行文本分块（Chunking）。
- **性别假设。** 硬编码的性别规则在处理非二元性别指代对象、组织机构或动物时会失效。建议使用学习到的模型或中性打分机制。
- **长文档中的大语言模型漂移。** 单次 API 调用无法可靠地对跨越 50 多个段落的指称项进行聚类。建议采用滑动窗口（Sliding-window）结合合并策略。

## 实际应用

2026 年技术栈推荐：

| 场景 | 推荐方案 |
|-----------|------|
| 英文单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经共指消解模型 |
| 多语言 | 基于 OntoNotes 或多语言 CoNLL 数据集训练的 SpanBERT / XLM-R |
| 跨文档事件共指消解 | 专用端到端模型（2025–2026 年 SOTA） |
| 快速大语言模型基线 | 配合结构化输出共指提示词的 GPT-4o / Claude |
| 生产级对话系统 | 规则回退机制 + 神经模型为主 + 关键槽位人工复核 |

2026 年落地的集成模式为：先运行命名实体识别（NER），再运行共指消解，最后将共指聚类簇合并至 NER 实体中。下游任务将看到每个聚类簇对应一个实体，而非每个指称项对应一个实体。

## 部署上线

保存为 `outputs/skill-coref-picker.md`：

---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.

## 练习

1. **简单。** 在 5 段手工编写的段落上运行 `code/main.py` 中的基于规则的解析器（rule-based resolver）。对照真实标签（ground truth）测量指代链接（mention-link）准确率。
2. **中等。** 在一篇新闻文章上使用预训练神经指代消解模型（pretrained neural coreference model）。将生成的簇（clusters）与你自己的手动标注进行对比。它在哪些地方失败了？
3. **困难。** 构建一个经指代消解（coreference resolution）增强的命名实体识别（named entity recognition, NER）流水线：先进行 NER，然后通过指代消解簇进行合并。在 100 篇文章上测量相较于仅使用 NER 的实体覆盖率（entity-coverage）提升。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 提及项 (Mention) | 一个引用 | 指向某个实体的文本片段（名称、代词、名词短语）。 |
| 先行词 (Antecedent) | “它”所指的对象 | 后续指代项所共同指向的较早出现的提及项。 |
| 簇 (Cluster) | 实体的所有提及 | 所有指向同一现实世界实体的提及项集合。 |
| 回指 (Anaphora) | 向后指代 | 后出现的提及项指向前文（“他” → “约翰”）。 |
| 下指 (Cataphora) | 向前指代 | 先出现的提及项指向后文（“当他到达时，约翰……”）。 |
| 桥接 (Bridging) | 隐式指代 | “我买了一辆车。轮子很糟糕。”（指*那辆*车的轮子。） |
| CoNLL F1 | 排行榜上的分数 | MUC、B³ 和 CEAF-φ4 F1 分数的平均值。 |

## 扩展阅读

- [Jurafsky & Martin, SLP3 第 26 章 — 指代消解与实体链接](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教材章节。
- [Lee 等人 (2017). 端到端神经指代消解](https://arxiv.org/abs/1707.07045) — 基于跨度（span-based）的端到端方法。
- [Joshi 等人 (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 提升指代消解效果的预训练模型。
- [Pradhan 等人 (2012). CoNLL-2012 共享任务](https://aclanthology.org/W12-4501/) — 该领域的基准测试（benchmark）。
- [Hobbs (1978). 解析代词指代](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典之作。
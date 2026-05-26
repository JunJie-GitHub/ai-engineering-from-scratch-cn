# 实体链接与消歧 (Entity Linking & Disambiguation)

> 命名实体识别 (Named Entity Recognition, NER) 发现了“Paris”。实体链接 (Entity Linking, EL) 负责判定：是法国巴黎？帕丽斯·希尔顿？得克萨斯州巴黎？还是特洛伊王子帕里斯？如果没有链接，你的知识图谱 (Knowledge Graph) 将始终存在歧义。

**类型：** 构建
**语言：** Python
**前置条件：** 第5阶段 · 06（命名实体识别），第5阶段 · 24（指代消解）
**耗时：** 约60分钟

## 问题描述

假设有一句话：“Jordan beat the press.” 你的 NER 将“Jordan”标记为 PERSON（人物）。很好。但*究竟是哪个* Jordan？

- 迈克尔·乔丹（篮球运动员）？
- 迈克尔·B·乔丹（演员）？
- 迈克尔·I·乔丹（加州大学伯克利分校机器学习教授——是的，这种混淆在机器学习论文中真实存在）？
- 约旦（国家）？
- Jordan（希伯来语名字）？

实体链接将每个提及 (mention) 解析到知识库 (Knowledge Base, KB) 中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域知识库。它包含两个子任务：

1. **候选生成 (Candidate Generation)。** 给定“Jordan”，哪些知识库条目是合理的候选？
2. **消歧 (Disambiguation)。** 结合上下文，哪个候选才是正确的？

这两个步骤均可通过模型学习。两者均有标准基准测试。整个组合流水线 (Pipeline) 在过去十年中保持稳定——真正发生变化的是消歧器的质量。

## 核心概念

![实体链接流水线：提及 → 候选 → 消歧后的实体](../assets/entity-linking.svg)

**候选生成。** 给定提及的表面形式 (Surface Form，如“Jordan”)，在别名索引 (Alias Index) 中查找候选。Wikipedia 别名词典覆盖了大多数命名实体：“JFK” → 约翰·F·肯尼迪、杰奎琳·肯尼迪、肯尼迪国际机场、电影《JFK》。典型的索引每次提及会返回 10-30 个候选。

**消歧：三种主流方法。**

1. **先验概率 + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果良好、速度快，且无需训练。
2. **基于嵌入的方法 (Embedding-based，如 ESS / REL / Blink)。** 对提及及其上下文进行编码。对每个候选的描述进行编码。选择余弦相似度 (Cosine Similarity) 最高的项。这是 2020-2024 年间的默认方案。
3. **生成式方法 (Generative，如 GENRE, 2021；基于大语言模型的方法, 2023+)。** 逐词元 (Token) 解码实体的规范名称。通过有效实体名称的前缀树 (Trie) 进行约束，从而保证输出必定是有效的知识库 ID。

**端到端 (End-to-end) 与流水线架构。** 现代模型（如 ELQ、BLINK、ExtEnD、GENRE）可在单次前向传播中完成 NER + 候选生成 + 消歧。但在生产环境中，流水线系统仍占主导地位，因为你可以灵活替换其中的组件。

### 两项核心评估指标

- **提及召回率 (Mention Recall，候选生成阶段)。** 在真实提及 (Gold Mentions) 中，正确知识库条目出现在候选列表中的比例。它决定了整个流水线的性能下限。
- **消歧准确率 / F1 分数 (Disambiguation Accuracy / F1)。** 在候选正确的情况下，排名第一 (Top-1) 的候选被正确选中的频率。

务必同时报告这两项指标。一个在 80% 候选召回率下达到 99% 消歧准确率的系统，其整体流水线性能仅为 80%。

## 动手实践

### 步骤 1：基于维基百科重定向 (Wikipedia Redirects) 构建别名索引 (Alias Index)

alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}

维基百科别名数据包含约 1800 万个（别名，实体）对。可从维基数据 (Wikidata) 转储文件中下载，并存储为倒排索引 (Inverted Index)。

### 步骤 2：基于上下文的消歧 (Context-based Disambiguation)

def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score

此处的杰卡德重叠系数 (Jaccard Overlap) 仅为示例。实际应用中应替换为基于嵌入向量 (Embeddings) 的余弦相似度 (Cosine Similarity)（基于 Transformer 的版本请参见 `code/main.py` 中的 step-2）。

### 步骤 3：基于嵌入向量的方法（BLINK 风格）

from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]

在索引构建阶段，对知识库 (Knowledge Base, KB) 中的每个实体进行一次向量化编码。在查询阶段，对提及词 (Mention) 及其上下文进行一次编码，与候选池进行点积 (Dot-product) 运算，并选取最大值。

### 步骤 4：生成式实体链接 (Generative Entity Linking)（概念）

GENRE 模型逐字符解码实体的维基百科标题。约束解码 (Constrained Decoding)（参见第 20 课）确保仅输出有效的标题。该模型与基于知识库的前缀树 (Trie) 紧密集成。其现代演进版本包括 REL-GEN 以及结合结构化输出的大语言模型 (Large Language Model, LLM) 提示式实体链接 (Entity Linking, EL)。

prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""

结合白名单机制（如 Outlines 库的 `choice` 功能），这将是 2026 年最易于部署的实体链接流水线 (Entity Linking Pipeline)。

### 步骤 5：在 AIDA-CoNLL 数据集上进行评估

AIDA-CoNLL 是实体链接的标准基准测试集 (Entity Linking Benchmark)：包含 1,393 篇路透社文章、3.4 万个提及词及维基百科实体。需报告库内准确率 (`P@1`) 以及库外 NIL 检测率 (NIL-detection Rate)。

## 常见陷阱

- **NIL（未登录实体）处理。** 部分指称（mention）不在知识库（KB）中（如新兴实体、冷门人物）。系统必须将其预测为 NIL，而非错误匹配其他实体。该指标需单独评估。
- **指称边界错误。** 上游命名实体识别（NER）遗漏了部分文本片段（例如将“Bank of America”仅标注为“Bank”），导致实体链接（EL）召回率下降。
- **流行度偏差。** 训练后的系统倾向于过度预测高频实体。例如，机器学习论文中提及的“Michael I. Jordan”常被错误链接到篮球运动员乔丹。
- **跨语言实体链接（EL）。** 将中文文本中的指称映射至英文维基百科实体。该任务需要多语言编码器或引入翻译步骤。
- **知识库陈旧问题。** 新公司、新事件及新人物未包含在去年的维基百科数据转储（dump）中。生产环境流水线需建立定期更新机制。

## 选型与应用

2026 年技术栈选型：

| 场景 | 推荐方案 |
|-----------|------|
| 通用英文 + 维基百科 | BLINK 或 REL |
| 跨语言场景，知识库 = 维基百科 | mGENRE |
| 适配大语言模型（LLM），每日指称量较少 | 提供候选列表 + 约束 JSON 格式提示 Claude/GPT-4 |
| 垂直领域知识库（医疗、法律） | 结合知识库感知检索的定制 BERT + 在领域 AIDA 风格数据集上微调 |
| 极低延迟要求 | 仅使用精确匹配先验（Milne-Witten 基线） |
| 学术研究前沿（SOTA） | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年投产的标准流水线模式：命名实体识别（NER）→ 指代消解（coref）→ 对每个指称进行实体链接（EL）→ 将聚类结果合并为每个聚类一个标准实体。输出格式：文档中每个实体对应一个知识库 ID，而非每个指称对应一个 ID。

## 交付与部署

保存为 `outputs/skill-entity-linker.md`：

---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.

## 练习

1. **简单。** 在 `code/main.py` 中实现先验+上下文消歧器（prior+context disambiguator），针对 10 个歧义指称（ambiguous mentions）（如 Paris、Jordan、Apple）进行测试。人工标注正确实体，并计算准确率。
2. **中等。** 使用句子转换器（sentence transformer）对 50 个歧义指称进行编码。为每个候选实体的描述生成嵌入向量（embedding）。对比基于嵌入的消歧方法与 Jaccard 上下文重叠度（Jaccard context overlap）方法的效果。
3. **困难。** 构建一个包含 1000 个实体的领域知识库（domain KB）（例如你公司的员工与产品）。实现端到端的命名实体识别（NER）与实体链接（EL）流程。在 100 条预留测试集句子（held-out sentences）上评估精确率（precision）与召回率（recall）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（Entity Linking, EL） | 链接到维基百科 | 将指称映射到知识库中的唯一条目。 |
| 候选生成（Candidate Generation） | 可能是谁？ | 为指称返回一组合理的知识库候选条目。 |
| 消歧（Disambiguation） | 挑出正确的那个 | 利用上下文对候选实体打分，选出最优结果。 |
| 别名索引（Alias Index） | 查找表 | 建立从表面形式（surface form）到候选实体的映射。 |
| NIL | 不在知识库中 | 明确预测没有任何知识库条目与之匹配。 |
| 知识库（Knowledge Base, KB） | 知识库 | 维基数据（Wikidata）、维基百科、DBpedia 或你的领域知识库。 |
| AIDA-CoNLL | 基准数据集 | 包含 1,393 篇路透社文章及标准实体链接标注的数据集。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) —— 奠定先验+上下文方法基础的经典文献。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) —— 基于嵌入方法的主力模型。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) —— 采用受限解码（constrained decoding）的生成式实体链接方法。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) —— 该领域的基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) —— 开源的生产级技术栈。
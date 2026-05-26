# 文本摘要 (Text Summarization)

> 抽取式（Extractive）系统告诉你文档说了什么。抽象式（Abstractive）系统告诉你作者想表达什么。任务不同，陷阱各异。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 02（词袋模型 (BoW) + TF-IDF），第 5 阶段 · 11（机器翻译 (Machine Translation)）
**预计时间：** 约 75 分钟

## 问题描述

一篇 2000 字的新闻文章出现在你的信息流中。你需要用 120 个字概括其核心内容。你可以从文章中挑选最重要的三句话（抽取式），也可以用自己的话重新表述内容（抽象式）。这两者都称为摘要生成，但它们是完全不同的问题。

抽取式摘要（Extractive Summarization）本质上是一个排序问题。对每个句子进行打分，返回得分最高的前 `k` 个句子。由于输出是原文逐字提取的，因此语法总是正确的。其风险在于可能会遗漏分散在文章各处的关键信息。

抽象式摘要（Abstractive Summarization）则是一个生成问题。Transformer 模型会根据输入条件生成新文本。输出通常流畅且高度凝练，但可能会“幻觉”出原文中不存在的事实。其风险在于模型会自信地编造内容。

本节将带你动手实现这两种方法，并深入剖析它们各自典型的失败模式。

## 核心概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**抽取式（Extractive）。** 将文章视为一个图，其中节点代表句子，边代表句子间的相似度。在图上运行 PageRank（或类似算法），根据句子与其他内容的连接程度对其进行打分。得分最高的句子即构成摘要。该方法的经典实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**抽象式（Abstractive）。** 在“文档-摘要”配对数据上微调 Transformer 编码器-解码器（Transformer Encoder-Decoder）架构（如 BART、T5、Pegasus）。在推理阶段，模型阅读文档并通过交叉注意力机制（Cross-Attention）逐词元（Token）生成摘要。特别是 Pegasus 模型，它采用了“留空句子”（Gap-Sentence）的预训练目标，使其无需大量微调即可在摘要任务上表现出色。

评估通常使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation，面向召回的摘要评估替代指标）。ROUGE-1 和 ROUGE-2 分别计算一元词（Unigram）和二元词（Bigram）的重叠度。ROUGE-L 计算最长公共子序列（Longest Common Subsequence）。分数越高越好，通常 ROUGE-L 达到 40 算“良好”，达到 50 算“优异”。每篇相关论文都会报告这三项指标。请使用 `rouge-score` 软件包进行计算。

## 动手实践

### 步骤 1：TextRank（抽取式）

import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]

有两点值得说明。`similarity` 函数采用了对数归一化词重叠度（log-normalized word overlap），这是原始 TextRank 的变体。使用 TF-IDF 向量的余弦相似度（cosine similarity）同样可行。阻尼系数（damping factor）0.85 和迭代次数（iteration count）均沿用了 PageRank 的默认设置。

### 步骤 2：基于 BART 的生成式摘要

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])

`BART-large-CNN` 模型已在 CNN/DailyMail 语料库（corpus）上完成微调（fine-tuned）。它能够开箱即用地生成新闻风格的摘要。针对其他领域（如科学论文、对话或法律文本），请使用对应的 Pegasus 检查点（checkpoint），或在你的目标数据上进行微调。

### 步骤 3：ROUGE 评估

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})

务必启用词干提取（stemming）。若不启用，“running”和“run”会被视为不同的词，从而导致 ROUGE 评分偏低。

### 超越 ROUGE（2026 年摘要评估）

ROUGE 作为主导的摘要评估指标已长达二十年，但在 2026 年，仅靠它已远远不够。一项针对自然语言生成（Natural Language Generation, NLG）论文的大规模元分析表明：

- **BERTScore**（上下文嵌入相似度，contextual embedding similarity）在 2023 年前后逐渐普及，如今在大多数摘要论文中已与 ROUGE 并列报告。
- **BARTScore** 将评估视为生成任务：根据预训练 BART 模型在给定源文本条件下生成该摘要的概率来进行打分。
- **MoverScore**（基于上下文嵌入的推土机距离，Earth Mover's Distance over contextual embeddings）在 2025 年的摘要基准测试中登顶，因为它比 ROUGE 能更好地捕捉语义重叠。
- **FactCC** 和**基于问答的忠实度评估**（QA-based faithfulness）在 2021-2023 年间较为常见，如今通常已被 **G-Eval** 取代（一种基于 GPT-4 的提示链，通过思维链推理（chain-of-thought reasoning）对连贯性、一致性、流畅性和相关性进行打分）。
- 当评分标准设计合理时，**G-Eval** 及类似的大语言模型裁判（LLM-judge）方法与人类判断的一致性可达约 80%。

生产环境建议：报告 ROUGE-L 用于历史对比，使用 BERTScore 衡量语义重叠，采用 G-Eval 评估连贯性与事实性。建议以 50-100 份人工标注的摘要作为校准基准。

### 步骤 4：事实性问题

生成式摘要容易产生幻觉（hallucination）。抽取式摘要的幻觉风险要低得多，因为其输出是逐字从源文本中提取的；但如果源句子脱离上下文、已过时或顺序错乱，仍可能产生误导。这也是生产系统在涉及合规相关内容时，依然首选抽取式方法的最主要原因。

常见的幻觉类型包括：

- **实体替换（Entity swap）**：源文本为“John Smith”，摘要却写成“John Brown”。
- **数值漂移（Number drift）**：源文本为“25,000”，摘要却写成“2500 万”。
- **极性反转（Polarity flip）**：源文本为“拒绝了该提议”，摘要却写成“接受了该提议”。
- **事实捏造（Fact invention）**：源文本未提及 CEO，摘要却声称 CEO 已批准。

有效的评估方法包括：

- **FactCC**：一种基于源句子与摘要句子之间蕴含关系（entailment）训练的二分类器，用于预测内容是否符合事实。
- **基于问答的事实性评估（QA-based factuality）**：向问答模型提出答案存在于源文本中的问题。如果摘要支持不同的答案，则标记为异常。
- **实体级 F1 分数（Entity-level F1）**：对比源文本与摘要中的命名实体。仅出现在摘要中的实体值得怀疑。

对于任何面向用户且对事实性要求较高的场景（如新闻、医疗、法律、金融），抽取式是更安全的默认选择。若使用生成式方法，则必须在流程中引入事实性检查环节。

## 使用方法

2026 年技术栈（Tech Stack）推荐：

| 使用场景 | 推荐方案 |
|---------|-------------|
| 新闻，3-5 句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或微调后的 T5 |
| 多文档、长文本 | 任意具备 32k+ 上下文窗口的大语言模型（Large Language Model, LLM），配合提示词使用 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式（Extractive），架构上天然具备低幻觉（Hallucination）风险 | TextRank 或 `sumy` 的 LSA / LexRank |

在算力不受限的情况下，2026 年具备长上下文的 LLM 通常能超越专用模型。其权衡在于成本与可复现性；专用模型则能提供更稳定的输出。

## 交付部署

保存为 `outputs/skill-summary-picker.md`：

---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank。将排名前 3 的句子与参考摘要进行对比。计算 ROUGE-L 分数。在 CNN/DailyMail 风格的文章上，你应该能看到 30-45 的 ROUGE-L 分数。
2. **中等。** 实现实体级事实性（Factuality）检查：从原文和摘要中提取命名实体（使用 spaCy），计算摘要中原文实体的召回率（Recall），以及原文中摘要实体的精确率（Precision）。高精确率与低召回率意味着结果安全但简略；低精确率则意味着出现了幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上，对比 BART-large-CNN 与 LLM（如 Claude 或 GPT-4）。报告 ROUGE-L 分数、事实性（通过实体 F1 分数衡量）以及单篇摘要的成本。记录各自胜出的场景。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式（Extractive） | 挑选句子 | 逐字返回原文中的句子。绝不会产生幻觉。 |
| 生成式/抽象式（Abstractive） | 重写 | 基于原文生成新文本。可能会产生幻觉。 |
| ROUGE | 摘要评估指标 | 系统输出与参考摘要之间的 N-gram / 最长公共子序列（LCS）重叠度。 |
| TextRank | 基于图的抽取式方法 | 在句子相似度图上运行 PageRank 算法。 |
| 事实性（Factuality） | 是否正确 | 摘要中的主张是否得到原文支持。 |
| 幻觉（Hallucination） | 编造的内容 | 摘要中存在但原文并未支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) —— 抽取式（Extractive）摘要的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) —— BART 模型的原始论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) —— PEGASUS 模型及其间隙句子目标（Gap-sentence Objective）。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) —— ROUGE 评估指标的原始论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) —— 探讨生成式摘要（Abstractive Summarization）中忠实度（Faithfulness）与事实性（Factuality）研究现状的论文。
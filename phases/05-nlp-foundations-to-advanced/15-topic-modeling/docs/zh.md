# 主题建模（Topic Modeling）— LDA 与 BERTopic

> LDA：文档是主题的混合体，主题是词汇的概率分布。BERTopic：文档在嵌入空间（embedding space）中聚类，聚类结果即为主题。目标一致，但底层机制不同。

**类型：** 学习
**语言：** Python
**前置知识：** 第 5 阶段 · 02（词袋模型（BoW）与 TF-IDF）、第 5 阶段 · 03（Word2Vec）
**预计耗时：** 约 45 分钟

## 问题背景

假设你手头有 1 万张客服工单、5 万篇新闻报道或 20 万条推文。你需要在不逐篇阅读的情况下，了解这批数据的核心内容。你没有带标签的分类体系，甚至根本不知道应该分成多少类。

主题建模（Topic Modeling）能够在无监督（unsupervised）的情况下解决这一问题。只需输入一个语料库（corpus），它就能返回一组语义连贯的主题，并为每篇文档输出其在这些主题上的概率分布。

目前主流的方法主要分为两大算法家族。LDA（2003 年提出）将每篇文档视为潜在主题（latent topics）的混合体，并将每个主题视为词汇的概率分布。其推理过程基于贝叶斯（Bayesian）理论。在需要文档多主题归属（mixed-membership）以及可解释的词级概率分布的生产环境中，LDA 依然被广泛部署。

BERTopic（2020 年提出）使用 BERT 对文档进行编码，通过 UMAP 进行降维，利用 HDBSCAN 进行聚类，并借助基于类别的 TF-IDF（class-based TF-IDF）提取主题词。它在短文本、社交媒体数据以及语义相似度比词汇重叠更重要的场景中表现更优。其局限在于每篇文档仅分配一个主题，这对长篇幅内容不太友好。

本节课程将帮助你建立对这两种方法的直观理解，并指导你如何根据具体的语料库选择最合适的算法。

## 核心概念

![LDA 混合模型与 BERTopic 聚类对比](../assets/topic-modeling.svg)

**LDA 的生成过程（generative story）。** 每个主题是词汇的概率分布，每篇文档是主题的混合体。要生成文档中的一个词，首先从文档的主题混合中采样一个主题，再从该主题的词汇分布中采样一个词。推理过程则相反：基于观测到的词汇，反推每篇文档的主题分布以及每个主题的词汇分布。这一数学过程通常通过折叠吉布斯采样（Collapsed Gibbs Sampling）或变分贝叶斯（Variational Bayes）来完成。

LDA 的核心输出：

- `doc_topic`：形状为 `(n_docs, n_topics)` 的矩阵，每行求和为 1（表示文档的主题混合比例）。
- `topic_word`：形状为 `(n_topics, vocab_size)` 的矩阵，每行求和为 1（表示主题的词汇分布）。

**BERTopic 处理流程（pipeline）。**

1. 使用句子转换器（sentence transformer，例如 `all-MiniLM-L6-v2`）对每篇文档进行编码，生成 384 维向量。
2. 使用 UMAP 将维度降至约 5 维。BERT 的嵌入向量（embeddings）维度过高，不适合直接聚类。
3. 使用 HDBSCAN 进行聚类。该算法基于密度，能够生成大小不一的簇，并标记“离群点（outlier）”。
4. 针对每个簇，基于簇内文档计算基于类别的 TF-IDF，以提取核心主题词。

最终输出为每篇文档分配一个主题（离群点标记为 -1）。此外，也可通过 HDBSCAN 的概率向量获取软归属（soft membership）结果。

## 动手实践

### 步骤 1：通过 scikit-learn 实现 LDA（Latent Dirichlet Allocation，潜在狄利克雷分配）

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")

注意：已移除停用词，`min_df` 和 `max_df` 用于过滤罕见词和过于常见的词项；此处使用 `CountVectorizer`（而非 `TfidfVectorizer`），因为 LDA 期望输入原始词频计数。

### 步骤 2：BERTopic（生产环境应用）

from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")

对 `Topic != -1` 的过滤会剔除 BERTopic 的离群值桶（即 HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小聚类大小；BERTopic 库的默认值为 10。本示例根据当前教程的数据规模，显式将其设置为 15。对于超过 10,000 篇文档的语料库，建议将该值提高至 50 或 100。

### 步骤 3：评估

两种方法都会输出主题词。关键在于这些词是否具有语义连贯性。

- **主题连贯性（Topic coherence, c_v）**。结合滑动窗口上下文中高频词对的归一化逐点互信息（Normalized Pointwise Mutual Information, NPMI），将得分聚合为主题向量，并通过余弦相似度（cosine similarity）对这些向量进行比较。得分越高越好。可使用 `gensim.models.CoherenceModel` 并设置 `coherence="c_v"`。
- **主题多样性（Topic diversity）**。所有主题的高频词中唯一词所占的比例。比例越高越好（表明主题之间重叠度低）。
- **定性检查（Qualitative inspection）**。阅读每个主题的高频词，判断它们是否指向实际存在的事物或概念。人工判断仍然是最终的把关环节。

## 如何选择合适的方法

| 场景 | 选择 |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 包含主题混合的长文档 | LDA |
| 无 GPU / 算力有限 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| 集成大语言模型（LLM）进行主题标注 | BERTopic（直接支持） |
| 资源受限的边缘端部署 | LDA |
| 最大化语义连贯性（semantic coherence） | BERTopic |

实际应用中最大的考量因素是文档长度。BERT 嵌入（BERT embeddings）会进行截断；而 LDA 的词频统计（LDA counts）适用于任意长度的文本。对于超过嵌入模型上下文窗口（context window）的文档，要么采用分块后聚合（chunk + aggregate）的策略，要么直接使用 LDA。

## 使用指南
2026 年技术栈推荐：
- **BERTopic。** 短文本及任何对语义要求较高的场景的默认选择。
- **`gensim.models.LdaModel`。** 经典的 LDA 实现，适用于生产环境，成熟且经过充分验证。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 适合实验的快速 LDA 实现。
- **NMF。** 非负矩阵分解（Non-negative Matrix Factorization）。LDA 的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 设计理念与 BERTopic 相似。社区规模较小，但在部分基准测试中表现优异。
- **FASTopic。** 较新的模型，在超大规模语料库上比 BERTopic 更快。
- **基于大语言模型（LLM）的标注。** 先运行任意聚类算法，再通过提示词（prompt）让模型为每个聚类命名。

## 部署交付
保存为 `outputs/skill-topic-picker.md`：

---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.

## 练习

1. **简单。** 在 20 Newsgroups 数据集上拟合潜在狄利克雷分配（Latent Dirichlet Allocation, LDA）模型。输出每个主题的前 10 个关键词。手动为每个主题命名。该算法是否成功识别出了真实的类别？
2. **中等。** 在相同的 20 Newsgroups 子集上拟合 BERTopic 模型。对比两者发现的主题数量、高频词以及主题连贯性（coherence）的定性表现。哪种方法能更清晰地揭示出真实的类别？
3. **困难。** 在你的语料库上分别计算 LDA 和 BERTopic 的 c_v 连贯性（c_v coherence）指标。分别设置 5、10、20、50 个主题运行模型。绘制连贯性随主题数量变化的曲线图。报告哪种方法在不同主题数量下表现更稳定。

## 核心术语

| 术语 | 通俗理解 | 实际含义 |
|------|----------|----------|
| 主题（Topic） | 语料库所讨论的事物 | 词的概率分布（LDA）或相似文档的聚类（BERTopic）。 |
| 混合隶属度（Mixed membership） | 一篇文档属于多个主题 | LDA 为每篇文档分配一个涵盖所有主题的概率分布。 |
| UMAP | 降维技术 | 一种保留局部结构的流形学习算法；在 BERTopic 中使用。 |
| HDBSCAN | 密度聚类算法 | 能够发现大小不一的聚类；会为离群点生成“噪声”标签（-1）。 |
| c_v 连贯性（c_v coherence） | 主题质量评估指标 | 滑动窗口内主题核心词的平均逐点互信息（pointwise mutual information）。 |

## 扩展阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 的奠基论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 的原始论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 提出 c_v 连贯性及其相关指标的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产环境实战参考。包含大量优质示例。
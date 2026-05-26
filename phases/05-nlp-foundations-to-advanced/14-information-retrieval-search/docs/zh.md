# 信息检索与搜索

> BM25 精确但脆弱。稠密检索（Dense Retrieval）覆盖面广但会遗漏关键词。混合检索（Hybrid Retrieval）是 2026 年的默认方案。其余的都是调优工作。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 02（词袋模型 + TF-IDF），第 5 阶段 · 04（GloVe、FastText、子词）
**耗时：** 约 75 分钟

## 核心问题

用户输入“如果有人为了骗钱撒谎会怎样”，期望找到实际涵盖该情况的法条：“《印度刑法典》第 420 条”。关键词搜索会完全错过它（因为没有共享词汇）。如果词嵌入（Embeddings）未在法律文书上训练过，语义搜索也会错过它。真实的搜索系统必须同时兼顾两者。

信息检索（Information Retrieval, IR）是每一个检索增强生成（Retrieval-Augmented Generation, RAG）系统、每个搜索栏、每个文档站点模糊查找背后的底层流水线。2026 年能在生产环境中稳定运行的架构并非单一方法，而是一系列互补方法的链路，每一环都在弥补上一环的不足。

本课程将逐一构建这些组件，并明确每个组件具体解决了哪些失败场景。

## 核心概念

![混合检索：BM25 + 稠密检索 + RRF + 交叉编码器重排序](../assets/retrieval.svg)

四个层级。按需选用。

1. **稀疏检索（Sparse Retrieval，BM25）。** 速度快，精确匹配表现优异，但语义理解能力极差。基于倒排索引（Inverted Index）运行。在数百万文档中，单次查询耗时低于 10 毫秒。能准确命中法条引用、产品代码、错误信息和命名实体。
2. **稠密检索（Dense Retrieval）。** 将查询和文档编码为向量。执行最近邻搜索（Nearest Neighbor Search）。能够捕捉同义改写和语义相似性。但会遗漏仅差一个字符的精确关键词匹配。使用 FAISS 或向量数据库时，单次查询耗时约 50-200 毫秒。
3. **结果融合（Fusion）。** 合并稀疏检索和稠密检索的排序列表。倒数排名融合（Reciprocal Rank Fusion, RRF）是简单易用的默认方案，因为它忽略原始分数（这些分数处于不同量纲），仅依赖排名位置。当你明确知道某一信号在你的业务领域占主导地位时，加权融合（Weighted Fusion）是一个可选方案。
4. **交叉编码器重排序（Cross-encoder Rerank）。** 从融合结果中取出前 30 条。运行交叉编码器（将查询与文档拼接输入，对每一对进行打分）。保留前 5 条。交叉编码器处理每一对的耗时虽高于双塔编码器（Bi-encoders），但准确率远超后者。通过仅在前 30 条上运行，可以摊薄计算成本。

三路检索（BM25 + 稠密检索 + 学习型稀疏检索如 SPLADE）在 2026 年的基准测试中优于双路检索，但需要为学习型稀疏索引搭建基础设施。对大多数团队而言，双路检索结合交叉编码器重排序是最佳平衡点。

## 动手实践

### 步骤 1：从零实现 BM25

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]

有两个关键参数值得了解。`k1=1.5` 控制词频饱和度（term-frequency saturation）；值越高，词项重复的权重越大。`b=0.75` 控制长度归一化（length normalization）；0 表示忽略文档长度，1 表示完全归一化。默认值源自原始论文中 Robertson 的建议，通常无需调整。

### 步骤 2：使用双编码器（bi-encoder）进行稠密检索（dense retrieval）

from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]

对嵌入向量（embeddings）进行 L2 归一化，可使点积等于余弦相似度。`all-MiniLM-L6-v2` 为 384 维，推理速度快，且足以应对大多数英文检索任务。若需处理多语言场景，请使用 `paraphrase-multilingual-MiniLM-L12-v2`。若追求最高精度，可选择 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤 3：倒数排名融合（Reciprocal Rank Fusion）

def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]

常数 `k=60` 源自 RRF 的原始论文。较高的 `k` 会削弱排名差异带来的影响；较低的 `k` 则会让靠前的排名占据主导。60 是论文发布的默认值，通常无需调整。

### 步骤 4：混合检索（hybrid search）与重排序（rerank）

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]

该流程由三个阶段组成。BM25 负责查找词汇匹配（lexical matches），稠密检索负责查找语义匹配（semantic matches）。RRF 将两者的排名结果进行融合，且无需进行分数校准（score calibration）。交叉编码器（Cross-encoder）会结合查询-文档对（query-document pairs）对前 30 个结果重新打分，从而捕捉双编码器遗漏的细粒度相关性（fine-grained relevance）。最终保留前 5 个结果。

### 步骤 5：评估（evaluation）

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在存在正确文档的查询中，该文档出现在前 k 个结果中的频率是多少？ |
| MRR（平均倒数排名） | 首个相关文档排名的倒数的平均值。 |
| nDCG@k | 考虑相关性的程度分级，而非仅仅是二元的相关/不相关。 |

针对检索增强生成（RAG）场景而言，检索器（retriever）的 **Recall@k** 是最关键的指标。如果正确的文本片段未被包含在检索结果集中，下游模型将无法给出正确答案。

调试建议：针对失败的查询，对比稀疏检索与稠密检索的排名结果。如果其中一种方法找到了正确文档而另一种没有，说明存在词汇不匹配（vocabulary mismatch）问题（解决方法：补充缺失的检索方式）或语义歧义（semantic ambiguity）问题（解决方法：使用更优的嵌入模型或引入重排序器）。

## 实际应用

2026 年技术栈推荐：

| 规模 | 技术栈 |
|-------|-------|
| 1k-100k 篇文档 | 内存级 BM25（Best Matching 25） + `all-MiniLM-L6-v2` 嵌入向量（Embeddings） + RRF（Reciprocal Rank Fusion）。无需独立数据库。 |
| 100k-1000 万篇文档 | 使用 FAISS 或 pgvector 处理稠密向量（Dense Vectors），搭配 Elasticsearch / OpenSearch 处理 BM25。并行运行。 |
| 1000 万篇以上文档 | 采用支持混合检索的 Qdrant / Weaviate / Vespa / Milvus。对 Top-30 结果使用交叉编码器（Cross-encoder）进行重排序（Rerank）。 |
| 追求极致质量的前沿方案 | 三路检索（BM25 + 稠密向量 + SPLADE（学习型稀疏检索）） + ColBERT 晚期交互（Late-interaction）重排序 |

无论选择哪种方案，都必须为评估环节预留资源。在评估端到端 RAG（检索增强生成）准确率之前，先对检索召回率（Retrieval Recall）进行基准测试。检索器（Retriever）遗漏的内容，生成模型无法凭空补全。

### 2026 年生产环境 RAG 的实战经验

- **80% 的 RAG 故障根源在于数据摄入（Ingestion）与分块（Chunking），而非模型本身。** 团队往往花费数周时间更换大语言模型（LLM）和调试提示词（Prompt），而检索模块却在每三次查询中就有一次悄悄返回错误的上下文。优先解决分块问题。
- **分块策略比分块大小更重要。** 固定长度切分会破坏表格、代码和嵌套标题。默认应采用基于句子的分块（Sentence-aware Chunking）；对于技术文档和产品手册，采用语义分块或基于 LLM 的分块能带来显著收益。
- **父文档模式（Parent-doc Pattern）。** 检索较小的“子”分块以保证精度。当同一父级段落下的多个子分块同时命中时，替换为完整的父级块以保留上下文。该方法无需重新训练即可稳定提升回答质量。
- **k_rerank=3 通常为最优值。** 超过该数量的额外分块只会增加 Token 成本和生成延迟，却无法提升回答质量。如果你的场景中 k=8 依然优于 k=3，说明重排序器（Reranker）的性能未达预期。
- **HyDE（假设性文档嵌入）/ 查询扩展（Query Expansion）。** 根据查询生成假设性答案，对其进行向量化嵌入后再检索。此举能有效弥合简短提问与长篇文档之间的表述差异。无需训练即可免费提升检索精度。
- **上下文预算控制在 8K Token 以内。** 如果频繁触及该上限，说明重排序器的阈值设置过于宽松。
- **对所有组件进行版本控制。** 包括提示词、分块规则、嵌入模型和重排序器。任何细微的版本漂移都会暗中破坏回答质量。在持续集成（CI）流水线中设置基于忠实度（Faithfulness）、上下文精度（Context Precision）和未回答问题率的卡点，可在用户察觉前拦截性能回退。
- **在 2026 年的基准测试中，三路检索（BM25 + 稠密向量 + 类似 SPLADE 的学习型稀疏向量）的表现优于双路检索**，尤其适用于混合专有名词与语义的查询。当基础设施支持 SPLADE 索引时，即可将其投入生产。

根据 2026 年行业实测数据，合理的检索设计可将幻觉（Hallucinations）减少 70%-90%。RAG 的大部分性能提升源于更优的检索架构，而非模型微调（Fine-tuning）。

## 部署上线

保存为 `outputs/skill-retrieval-picker.md`：

---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.

## 练习

1. **简单。** 在包含 500 篇文档的语料库 (corpus) 上实现上述 `hybrid_search`。测试 20 个查询。对比仅 BM25、仅稠密检索 (dense-only) 以及混合检索 (hybrid) 在 Top-5 召回率 (recall at 5) 上的差异。
2. **中等。** 添加 MRR (平均倒数排名) 计算。对于每个已知正确答案的测试查询，找出正确文档在 BM25、稠密检索和混合检索排序中的具体排名。分别报告各自的 MRR 值。
3. **困难。** 使用 `MultipleNegativesRankingLoss`（Sentence Transformers）在你的领域数据上微调 (fine-tune) 一个稠密编码器 (dense encoder)。从 500 个查询-文档对 (query-document pairs) 中构建训练集。对比微调前后的召回率表现。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25 算法。根据词频 (term frequency)、逆文档频率 (IDF) 和文档长度对文档进行打分。 |
| 稠密检索 (Dense retrieval) | 向量搜索 | 将查询与文档编码为向量，并查找最近邻 (nearest neighbors)。 |
| 双编码器 (Bi-encoder) | 嵌入模型 (Embedding model) | 独立编码查询和文档。查询阶段速度快。 |
| 交叉编码器 (Cross-encoder) | 重排序模型 (Reranker model) | 将查询与文档拼接后共同编码。速度较慢但精度高。 |
| RRF (倒数排名融合) | 排名融合 (Rank fusion) | 通过累加 `1/(k + rank)` 来合并两个排序列表。 |
| Recall@k (Top-k 召回率) | 检索指标 (Retrieval metric) | 相关文档出现在 Top-k 结果中的查询所占比例。 |

## 扩展阅读

- [Robertson 与 Zaragoza（2009）。概率相关性框架：BM25 及其延伸](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) —— BM25 的权威论述。
- [Karpukhin 等人（2020）。面向开放域问答的稠密段落检索](https://arxiv.org/abs/2004.04906) —— 稠密段落检索（Dense Passage Retrieval, DPR），双编码器（bi-encoder）架构的典范。
- [Formal 等人（2021）。SPLADE：稀疏词法与扩展模型](https://arxiv.org/abs/2107.05720) —— 学习型稀疏检索器（learned-sparse retriever），有效弥合了与稠密检索（dense retrieval）模型之间的性能差距。
- [Cormack、Clarke 与 Büttcher（2009）。倒数排名融合优于孔多塞及独立排序学习方法](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) —— 倒数排名融合（Reciprocal Rank Fusion, RRF）的原始论文。
- [Khattab 与 Zaharia（2020）。ColBERT：高效且有效的段落检索](https://arxiv.org/abs/2004.12832) —— 晚期交互检索（late-interaction retrieval）模型。
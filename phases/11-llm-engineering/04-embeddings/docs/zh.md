# 嵌入（Embeddings）与向量表示（Vector Representations）

> 文本是离散的，数学是连续的。每当要求大语言模型（LLM）查找“相似”文档、比较语义或进行超越关键词的搜索时，你都在依赖连接这两个世界的桥梁。这座桥梁就是嵌入（Embedding）。如果不理解嵌入，你就没有真正理解现代人工智能，仅仅是在使用它而已。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第11阶段，第01课（提示词工程（Prompt Engineering））
**Time:** 约75分钟
**Related:** 第5阶段·第22课（嵌入模型深度解析（Embedding Models Deep Dive））涵盖了稠密（Dense）与稀疏（Sparse）及多向量（Multi-vector）模型、套娃截断（Matryoshka Truncation）以及按轴模型选择（Per-axis Model Selection）。本课侧重于生产流水线（向量数据库（Vector DBs）、HNSW、相似度数学计算）。在选择模型前，请先阅读第5阶段·第22课。

## 学习目标

- 使用API提供商和开源模型生成文本嵌入，并计算它们之间的余弦相似度（Cosine Similarity）
- 解释为何嵌入能够解决关键词搜索无法处理的词汇不匹配（Vocabulary Mismatch）问题
- 构建语义搜索（Semantic Search）索引，实现基于语义而非精确关键词匹配的文档检索
- 使用检索基准测试（如精确率@k（Precision@k）、召回率（Recall））评估嵌入质量，并为你的任务选择合适的嵌入模型

## 核心问题

假设你有10,000张客服工单。一位客户留言：“我的付款没有成功。”你需要找出历史上相似的工单。关键词搜索只能找到包含“payment”和“didn't go through”的工单，而会漏掉“transaction failed”（交易失败）、“charge was declined”（扣款被拒）以及“billing error”（账单错误）。这些工单描述的是完全相同的问题，只是用了截然不同的措辞。

这就是词汇不匹配问题。人类语言有几十种方式来表达同一个意思。关键词搜索将每个词视为没有语义的独立符号。它无法理解“declined”和“didn't go through”指向的是同一个概念。

你需要一种文本表示方法，让语义（而非拼写）来决定相似度。你需要一种方法，能够在某个数学空间中将“我的付款没成功”和“交易被拒绝”放置得很近，同时尽管“我的付款准时到账”也包含“付款”一词，却要将它推得很远。

这种表示方法就是嵌入。

## 核心概念

### 什么是嵌入（Embedding）？

嵌入（Embedding）是一个由浮点数组成的稠密向量，用于表示文本的语义。“稠密”一词至关重要——与大多数维度为零的稀疏表示（如词袋模型（Bag-of-Words）、TF-IDF）不同，嵌入向量的每个维度都承载着信息。

“The cat sat on the mat”会被转换为类似 `[0.023, -0.041, 0.087, ..., 0.012]` 的形式——具体包含 768 到 3072 个数字，取决于所使用的模型。这些数字编码了语义。你永远不会直接去查看它们，而是对它们进行比较。

### Word2Vec 的突破

2013 年，Google 的 Tomas Mikolov 及其同事发表了 Word2Vec。其核心思想是：训练一个神经网络，使其能够根据上下文邻居词预测目标词（或反之），此时隐藏层的权重就会转化为具有语义意义的向量表示。

著名的结果如下：

king - man + woman = queen

词嵌入（Word Embedding）上的向量运算能够捕捉语义关系。从“man”到“woman”的向量方向，大致等同于从“king”到“queen”的方向。正是这一刻，该领域意识到几何空间可以编码语义。

Word2Vec 生成的是 300 维向量。无论上下文如何，每个词都对应唯一的向量。“river bank”（河岸）和“bank account”（银行账户）中的“bank”拥有相同的嵌入表示。这一局限性推动了随后十年的研究发展。

### 从词到句子

词嵌入仅表示单个词元（Token）。而生产级系统需要对整个句子、段落或文档进行嵌入。由此衍生出四种主要方法：

**平均法（Averaging）**：计算句子中所有词向量的均值。计算成本低，有信息损失，但在短文本上效果出奇地好。完全丢失了词序信息——“dog bites man”和“man bites dog”会得到完全相同的嵌入。

**CLS 词元（CLS Token）**：Transformer 模型（如 2018 年的 BERT）会输出一个特殊的 `[CLS]` 词元嵌入，用于表示整个输入序列。效果优于平均法，但 `[CLS]` 词元最初是为下一句预测（Next-Sentence Prediction）任务训练的，而非专门用于相似度计算。

**对比学习（Contrastive Learning）**：显式地训练模型，使相似样本对的向量彼此靠近，不相似的彼此远离。Sentence-BERT（Reimers & Gurevych, 2019）采用了该方法，并成为现代嵌入模型的基础。例如，给定“How do I reset my password?”和“I need to change my password”，模型会学习到它们应具有几乎相同的向量。

**指令微调嵌入（Instruction-Tuned Embeddings）**：最新的方法。E5 和 GTE 等模型支持接收任务前缀（如 `"search_query:"`、`"search_document:"`），以指示模型生成何种类型的嵌入。这使得单一模型能够服务于多种任务。

graph LR
    subgraph "2013: Word2Vec"
        W1["king"] --> V1["[0.2, -0.1, ...]"]
        W2["queen"] --> V2["[0.3, -0.2, ...]"]
    end

    subgraph "2019: Sentence-BERT"
        S1["How do I reset my password?"] --> E1["[0.04, 0.12, ...]"]
        S2["I need to change my password"] --> E2["[0.05, 0.11, ...]"]
    end

    subgraph "2024: Instruction-Tuned"
        I1["search_query: password reset"] --> T1["[0.08, 0.09, ...]"]
        I2["search_document: To reset your password, click..."] --> T2["[0.07, 0.10, ...]"]
    end

### 现代嵌入模型

市场目前已收敛为少数几个生产级选项（截至 2026 年初的 MTEB 分数，MTEB v2 版本）：

| 模型 | 提供商 | 维度 | MTEB 分数 | 上下文窗口 | 成本 / 100 万词元 |
|-------|----------|-----------|------|---------|------------------|
| Gemini Embedding 2 | Google | 3072（Matryoshka） | 67.7（检索） | 8192 | $0.15 |
| embed-v4 | Cohere | 1024（Matryoshka） | 65.2 | 128K | $0.12 |
| voyage-4 | Voyage AI | 1024/2048（Matryoshka） | 66.8 | 32K | $0.12 |
| text-embedding-3-large | OpenAI | 3072（Matryoshka） | 64.6 | 8192 | $0.13 |
| text-embedding-3-small | OpenAI | 1536（Matryoshka） | 62.3 | 8192 | $0.02 |
| BGE-M3 | BAAI | 1024（稠密+稀疏+ColBERT） | 63.0（多语言） | 8192 | 开放权重 |
| Qwen3-Embedding | Alibaba | 4096（Matryoshka） | 66.9 | 32K | 开放权重 |
| Nomic-embed-v2 | Nomic | 768（Matryoshka） | 63.1 | 8192 | 开放权重 |

MTEB（大规模文本嵌入基准测试，Massive Text Embedding Benchmark）v2 涵盖检索、分类、聚类、重排和摘要等 100 多项任务。分数越高越好。到 2026 年，开放权重模型（如 Qwen3-Embedding、BGE-M3）在大多数指标上已匹敌甚至超越闭源托管模型。Gemini Embedding 2 在纯检索任务中领先；Voyage/Cohere 在特定垂直领域（金融、法律、代码）表现突出。在正式采用前，务必使用你自己的查询集进行基准测试。

### 相似度度量指标

给定两个嵌入向量，有三种常用的相似度度量方法：

**余弦相似度（Cosine Similarity）**：两个向量夹角的余弦值。取值范围为 -1（方向完全相反）到 1（方向完全相同）。它忽略向量的模长——如果方向一致，10 个词的句子和 500 个词的文档得分均可达到 1.0。这是 90% 应用场景的默认选择。

cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)

**点积（Dot Product）**：两个向量的原始内积。当向量已归一化（单位长度）时，其结果与余弦相似度完全一致。计算速度更快。OpenAI 的嵌入向量已进行归一化处理，因此点积和余弦相似度会给出相同的排序结果。

dot(a, b) = sum(a_i * b_i)

**欧几里得距离（L2 距离，Euclidean Distance）**：向量空间中的直线距离。值越小表示越相似。对模长差异敏感。当空间中的绝对位置（而不仅仅是方向）具有重要意义时使用。

L2(a, b) = sqrt(sum((a_i - b_i)^2))

如何选择：

| 指标 | 适用场景 | 避免场景 |
|--------|----------|------------|
| 余弦相似度 | 比较不同长度的文本；大多数检索任务 | 模长本身携带重要信息时 |
| 点积 | 嵌入向量已归一化；追求极致计算速度 | 向量模长差异较大时 |
| 欧几里得距离 | 聚类任务；空间最近邻问题 | 比较长度差异极大的文档时 |

### 向量数据库与 HNSW

暴力相似度搜索会将查询向量与数据库中存储的每一个向量逐一比对。对于 100 万个 1536 维的向量，每次查询需要进行 15 亿次乘加运算。速度太慢。

向量数据库通过近似最近邻（Approximate Nearest Neighbor, ANN）算法解决这一问题。目前主流的算法是 HNSW（分层可导航小世界图，Hierarchical Navigable Small World）：

1. 构建向量的多层图结构
2. 顶层较为稀疏——连接距离较远的簇
3. 底层较为稠密——建立邻近向量间的细粒度连接
4. 搜索从顶层开始，贪婪地逐层向下细化
5. 以 O(log n) 的时间复杂度返回近似的 Top-K 结果，而非 O(n)

HNSW 以微小的精度损失（通常召回率在 95%-99%）换取了巨大的速度提升。面对 1000 万向量，暴力搜索需要数秒，而 HNSW 仅需几毫秒。

graph TD
    subgraph "HNSW Layers"
        L2["Layer 2 (sparse)"] -->|"long jumps"| L1["Layer 1 (medium)"]
        L1 -->|"shorter jumps"| L0["Layer 0 (dense, all vectors)"]
    end

    Q["Query vector"] -->|"enter at top"| L2
    L0 -->|"nearest neighbors"| R["Top-k results"]

生产级选项：

| 数据库 | 类型 | 最佳适用场景 | 最大规模 |
|----------|------|----------|-----------|
| Pinecone | 托管 SaaS | 零运维生产环境 | 数十亿级 |
| Weaviate | 开源 | 自托管、混合搜索 | 1 亿+ |
| Qdrant | 开源 | 高性能、支持过滤 | 1 亿+ |
| ChromaDB | 嵌入式 | 原型开发、本地开发 | 100 万级 |
| pgvector | Postgres 扩展 | 已在使用 Postgres 的场景 | 1000 万级 |
| FAISS | 库（Library） | 进程内调用、学术研究 | 10 亿+ |

### 分块策略

文档通常过长，无法直接作为单个向量进行嵌入。一份 50 页的 PDF 可能涵盖数十个主题——其嵌入向量会变成所有内容的平均值，导致与任何具体主题都不够相似。因此，你需要将文档切分为多个块（Chunk），并分别对每个块进行嵌入。

**固定大小分块（Fixed-Size Chunking）**：每 N 个词元切分一次，并保留 M 个词元的重叠。简单且可预测。适用于结构不清晰的文档。例如，512 词元块、50 词元重叠：块 1 包含词元 0-511，块 2 包含词元 462-973。

**基于句子的分块（Sentence-Based Chunking）**：在句子边界处切分，将句子组合直到达到词元限制。每个块至少包含一个完整句子。优于固定大小分块，因为它永远不会将一个完整的语义从中切断。

**递归分块（Recursive Chunking）**：优先尝试在最大边界处切分（如章节标题）。如果仍然过大，则尝试段落边界，接着是句子边界，最后是字符限制。这正是 LangChain 的 `RecursiveCharacterTextSplitter` 的工作原理，非常适合混合格式的语料库。

**语义分块（Semantic Chunking）**：对每个句子进行嵌入，然后将嵌入相似的连续句子归为一组。当嵌入相似度低于设定阈值时，开启新块。计算成本较高（需要单独嵌入每个句子），但能生成语义最连贯的块。

| 策略 | 复杂度 | 质量 | 最佳适用场景 |
|----------|-----------|---------|----------|
| 固定大小 | 低 | 尚可 | 非结构化文本、日志 |
| 基于句子 | 低 | 良好 | 文章、电子邮件 |
| 递归 | 中 | 良好 | Markdown、HTML、混合文档 |
| 语义 | 高 | 最佳 | 对检索质量要求极高的场景 |

大多数系统的最佳实践：256-512 词元的块大小，搭配 50 词元的重叠。

### 双编码器与交叉编码器

双编码器（Bi-Encoder）独立地对查询和文档进行嵌入，然后比较向量。速度快——只需对查询嵌入一次，即可与预计算的文档嵌入进行比对。这正是检索阶段所使用的架构。

交叉编码器（Cross-Encoder）将查询和文档作为单一输入，直接输出相关性得分。速度慢——需要将每个查询-文档对完整输入模型进行计算。但准确率高得多，因为它能够同时关注查询和文档词元之间的交互。

生产环境的标准模式：双编码器检索出 Top-100 候选结果，交叉编码器对其进行重排，筛选出 Top-10。这就是“检索-重排”（Retrieve-then-Rerank）流水线。

graph LR
    Q["Query"] --> BE["Bi-Encoder: embed query"]
    BE --> VS["Vector search: top 100"]
    VS --> CE["Cross-Encoder: rerank"]
    CE --> R["Top 10 results"]

重排模型：Cohere Rerank 3.5（每 1000 次查询 2 美元）、BGE-reranker-v2（免费、开源）、Jina Reranker v2（免费、开源）。

### 套娃嵌入（Matryoshka Embeddings）

传统嵌入是“全有或全无”的。一个 1536 维向量需要存储 1536 个浮点数。如果不重新训练，你无法将其截断为 256 维。

套娃表示学习（Matryoshka Representation Learning, Kusupati 等人, 2022）解决了这一问题。该模型经过专门训练，使得前 N 个维度能够捕获最重要的信息，就像俄罗斯套娃一样。将 1536 维的套娃嵌入截断至 256 维会损失部分精度，但仍可正常使用。

OpenAI 的 `text-embedding-3-small` 和 `text-embedding-3-large` 通过 `dimensions` 参数支持套娃截断。请求 256 维而非 1536 维可将存储空间减少 6 倍，在 MTEB 基准测试上的精度损失仅约 3%-5%。

### 二值量化

一个以 float32 存储的 1536 维嵌入占用 6,144 字节。乘以 1000 万份文档：仅向量部分就需要 61 GB 存储空间。

二值量化（Binary Quantization）将每个浮点数转换为单个比特：正值变为 1，负值变为 0。存储空间从 6,144 字节骤降至 192 字节——缩减了 32 倍。相似度通过汉明距离（Hamming Distance，统计不同比特的数量）计算，CPU 仅需一条指令即可完成。

检索召回率的精度损失约为 5%-10%。常见的应用模式是：在数百万向量上进行首轮搜索时使用二值量化，随后用全精度向量对 Top-1000 结果重新打分。这能以 1/32 的内存占用，换取全精度 95% 以上的准确度。

## 构建

我们将从零开始构建一个语义搜索引擎（Semantic Search Engine）。无需向量数据库（Vector Database），也无需外部嵌入 API（Embedding API）。全程仅使用纯 Python 配合 numpy 进行数学运算。

### 步骤 1：文本分块（Text Chunking）

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_by_sentences(text, max_chunk_tokens=200):
    sentences = text.replace("\n", " ").split(".")
    sentences = [s.strip() + "." for s in sentences if s.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

### 步骤 2：从零构建嵌入向量（Embeddings）

我们使用 TF-IDF（词频-逆文档频率）结合 L2 归一化（L2 Normalization）实现了一种简单的稠密嵌入（Dense Embedding）。这并非基于神经网络的嵌入（Neural Embedding），但它遵循相同的接口约定：输入文本，输出固定维度的向量，且语义相似的文本会生成相似的向量。

import math
import numpy as np
from collections import Counter

class SimpleEmbedder:
    def __init__(self):
        self.vocab = []
        self.idf = []
        self.word_to_idx = {}

    def fit(self, documents):
        vocab_set = set()
        for doc in documents:
            vocab_set.update(doc.lower().split())
        self.vocab = sorted(vocab_set)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        n = len(documents)
        self.idf = np.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            doc_count = sum(1 for doc in documents if word in doc.lower().split())
            self.idf[i] = math.log((n + 1) / (doc_count + 1)) + 1

    def embed(self, text):
        words = text.lower().split()
        count = Counter(words)
        total = len(words) if words else 1
        vec = np.zeros(len(self.vocab))
        for word, freq in count.items():
            if word in self.word_to_idx:
                tf = freq / total
                vec[self.word_to_idx[word]] = tf * self.idf[self.word_to_idx[word]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

### 步骤 3：相似度函数（Similarity Functions）

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def dot_product(a, b):
    return float(np.dot(a, b))


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

### 步骤 4：基于暴力搜索（Brute-Force Search）的向量索引（Vector Index）

class VectorIndex:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add(self, vector, text, meta=None):
        self.vectors.append(vector)
        self.texts.append(text)
        self.metadata.append(meta or {})

    def search(self, query_vector, top_k=5, metric="cosine"):
        scores = []
        for i, vec in enumerate(self.vectors):
            if metric == "cosine":
                score = cosine_similarity(query_vector, vec)
            elif metric == "dot":
                score = dot_product(query_vector, vec)
            elif metric == "euclidean":
                score = -euclidean_distance(query_vector, vec)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            results.append({
                "text": self.texts[idx],
                "score": score,
                "metadata": self.metadata[idx],
                "index": idx
            })
        return results

    def size(self):
        return len(self.vectors)

### 步骤 5：语义搜索引擎（Semantic Search Engine）

class SemanticSearchEngine:
    def __init__(self, chunk_size=200, overlap=50):
        self.embedder = SimpleEmbedder()
        self.index = VectorIndex()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def index_documents(self, documents, source_names=None):
        all_chunks = []
        all_sources = []
        for i, doc in enumerate(documents):
            chunks = chunk_text(doc, self.chunk_size, self.overlap)
            all_chunks.extend(chunks)
            name = source_names[i] if source_names else f"doc_{i}"
            all_sources.extend([name] * len(chunks))
        self.embedder.fit(all_chunks)
        for chunk, source in zip(all_chunks, all_sources):
            vec = self.embedder.embed(chunk)
            self.index.add(vec, chunk, {"source": source})
        return len(all_chunks)

    def search(self, query, top_k=5, metric="cosine"):
        query_vec = self.embedder.embed(query)
        return self.index.search(query_vec, top_k, metric)

    def search_with_scores(self, query, top_k=5):
        results = self.search(query, top_k)
        return [
            {
                "text": r["text"][:200],
                "source": r["metadata"].get("source", "unknown"),
                "score": round(r["score"], 4)
            }
            for r in results
        ]

### 步骤 6：对比相似度度量指标（Similarity Metrics）

def compare_metrics(engine, query, top_k=3):
    results = {}
    for metric in ["cosine", "dot", "euclidean"]:
        hits = engine.search(query, top_k=top_k, metric=metric)
        results[metric] = [
            {"score": round(h["score"], 4), "preview": h["text"][:80]}
            for h in hits
        ]
    return results


## 投入使用

在生产环境中使用嵌入 API（Embedding API）时，整体架构保持不变，仅需替换嵌入器（Embedder）：

from openai import OpenAI

client = OpenAI()

def openai_embed(texts, model="text-embedding-3-small", dimensions=None):
    kwargs = {"model": model, "input": texts}
    if dimensions:
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]

使用 OpenAI 进行马特里奥什卡截断（Matryoshka truncation）——同一模型，维度更少，存储更低：

full = openai_embed(["semantic search query"], dimensions=1536)
compact = openai_embed(["semantic search query"], dimensions=256)

256 维向量的存储空间减少了 6 倍。对于 1000 万份文档，存储占用从 61 GB 降至 10 GB。在标准基准测试中，准确率损失约为 3% 到 5%。

使用 Cohere 进行重排序（Reranking）：

import cohere

co = cohere.ClientV2()

results = co.rerank(
    model="rerank-v3.5",
    query="What is the refund policy?",
    documents=["Full refund within 30 days...", "No refunds after 90 days..."],
    top_n=3
)

无需依赖 API 的本地嵌入（Local Embeddings）方案：

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(["semantic search query", "another document"])

我们构建的 `VectorIndex` 类可与上述任意方案无缝配合。只需替换嵌入函数，即可保留原有的搜索逻辑。

## 交付成果

本章节将生成以下文件：
- `outputs/prompt-embedding-advisor.md` —— 用于针对特定用例选择嵌入模型与策略的提示词（Prompt）
- `outputs/skill-embedding-patterns.md` —— 一项技能文件，用于指导智能体（Agent）如何在生产环境中高效使用嵌入技术

## 练习

1. **指标对比**：使用余弦相似度（Cosine Similarity）、点积（Dot Product）和欧氏距离（Euclidean Distance）对相同的 5 个查询在示例文档上运行。分别记录每种指标的前 3 个结果。哪些查询在不同指标下的结果不一致？原因是什么？

2. **分块大小实验**：分别使用 50、100、200 和 500 个单词的分块大小（Chunk Size）对示例文档建立索引。针对每种分块大小运行 5 个查询，并记录最高相似度得分。绘制分块大小与检索质量之间的关系图。找出分块过大开始导致性能下降的临界点。

3. **马特里奥什卡模拟**：构建一个生成 500 维向量的 `SimpleEmbedder`。将其分别截断至 50、100、200 和 500 维。测量每次截断对检索召回率（Retrieval Recall）的影响。此实验无需真实的训练技巧即可模拟马特里奥什卡（Matryoshka）的行为特征。

4. **二值量化（Binary Quantization）**：提取搜索引擎中的嵌入向量，将其转换为二值形式（正数为 1，负数为 0），并实现汉明距离（Hamming Distance）搜索。将前 10 个结果与全精度（Full-Precision）余弦相似度的结果进行对比，计算重叠百分比。

5. **基于句子的分块**：使用 `chunk_by_sentences` 替换固定大小的分块方法。运行相同的查询并对比检索得分。遵循句子边界是否能提升检索效果？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 嵌入 (Embedding) | “将文本转换为数字” | 一种稠密向量，其空间邻近度编码了语义相似性 |
| 词向量模型 (Word2Vec) | “初代嵌入模型” | 2013年提出的模型，通过预测上下文词来学习词向量；证明了向量运算能够编码语义 |
| 余弦相似度 (Cosine Similarity) | “两个向量有多相似” | 向量间夹角的余弦值；1 表示方向完全相同，0 表示正交（无关），-1 表示方向完全相反 |
| 分层可导航小世界图 (HNSW) | “快速向量搜索” | 一种分层图结构，通过多层设计实现 O(log n) 复杂度的近似最近邻搜索 |
| 双编码器 (Bi-encoder) | “分别嵌入，快速比对” | 将查询和文档独立编码为向量；支持预计算与快速检索 |
| 交叉编码器 (Cross-encoder) | “慢但精准的重排序器” | 通过完整模型联合处理查询-文档对；精度更高，但无法预计算 |
| 套娃嵌入 (Matryoshka Embeddings) | “可截断的向量” | 经过特殊训练的嵌入向量，其前 N 个维度已捕获最关键信息，从而支持可变长度的存储 |
| 二值量化 (Binary Quantization) | “1位嵌入” | 将浮点向量转换为仅保留符号位的二进制形式，可将存储空间缩减至 1/32，并支持汉明距离搜索 |
| 文本分块 (Chunking) | “拆分文档以便嵌入” | 将文档切分为 256-512 个词元 (token) 的片段，使每个片段均可独立进行嵌入与检索 |
| 向量数据库 (Vector Database) | “嵌入的搜索引擎” | 专为存储向量及大规模执行近似最近邻搜索而优化的数据存储系统 |
| 对比学习 (Contrastive Learning) | “通过对比进行训练” | 一种训练方法，旨在拉近相似样本对的嵌入距离，同时推远不相似样本对的嵌入距离 |
| 大规模文本嵌入基准 (MTEB) | “嵌入模型基准测试” | 包含 8 项任务共 56 个数据集的基准测试；已成为对比评估嵌入模型的行业标准 |

## 延伸阅读

- Mikolov 等人，《Efficient Estimation of Word Representations in Vector Space》（2013）—— 提出 Word2Vec 的开创性论文，通过“国王-王后”类比开启了词嵌入（Word Embedding）技术革命
- Reimers 与 Gurevych，《Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks》（2019）—— 阐述如何训练双编码器（Bi-encoder）以计算句子级相似度，奠定了现代嵌入模型（Embedding Models）的基础
- Kusupati 等人，《Matryoshka Representation Learning》（2022）—— 提出套娃表示学习（Matryoshka Representation Learning）技术，实现了可变维度嵌入（Variable-dimension Embeddings），该技术已被 OpenAI 应用于 text-embedding-3 模型
- Malkov 与 Yashunin，《Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs》（2018）—— HNSW（Hierarchical Navigable Small World）算法的原始论文，是目前大多数生产环境向量检索（Vector Search）背后的核心算法
- OpenAI 嵌入指南 (platform.openai.com/docs/guides/embeddings) —— 针对 text-embedding-3 模型的实用参考文档，涵盖套娃降维（Matryoshka Dimension Reduction）等关键技术
- MTEB（Massive Text Embedding Benchmark）排行榜 (huggingface.co/spaces/mteb/leaderboard) —— 实时基准测试平台，用于跨任务与多语言对比各类嵌入模型的性能
- [Muennighoff 等人，《MTEB: Massive Text Embedding Benchmark》（EACL 2023）](https://arxiv.org/abs/2210.07316) —— 该基准测试定义了排行榜所报告的 8 大任务类别（分类、聚类、配对分类、重排序（Reranking）、检索、语义文本相似度（Semantic Textual Similarity, STS）、摘要、平行语料挖掘（Bitext Mining））；在采信任何单一 MTEB 分数前，建议优先阅读此文。
- [Sentence Transformers 官方文档](https://www.sbert.net/) —— 权威参考资料，详细对比双编码器与交叉编码器（Cross-encoder）、介绍池化策略（Pooling Strategies），并涵盖本课程所实现的“摄取-切分-嵌入-存储”检索增强生成（Retrieval-Augmented Generation, RAG）流水线。
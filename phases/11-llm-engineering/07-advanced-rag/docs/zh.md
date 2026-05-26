# 高级 RAG（分块 Chunking、重排序 Reranking、混合搜索 Hybrid Search）

> 基础 RAG（Retrieval-Augmented Generation）会检索相似度最高的前 k 个文本块（chunks）。这在处理简单问题时很有效，但在面对多跳推理（multi-hop reasoning）、模糊查询和大规模语料库时就会失效。高级 RAG 正是“仅能在 10 份文档上跑通的演示”与“能支撑 1000 万份文档的生产级系统”之间的关键区别。

**类型：** 构建实践
**编程语言：** Python
**前置要求：** 第 11 阶段，第 06 课（RAG）
**预计耗时：** 约 90 分钟
**相关课程：** 第 5 阶段 · 第 23 课（RAG 的分块策略）涵盖了全部六种分块算法——递归分块（recursive chunking）、语义分块（semantic chunking）、句子分块（sentence chunking）、父文档分块（parent-document chunking）、延迟分块（late chunking）和上下文检索（contextual retrieval）——并附有 Vectara/Anthropic 的基准测试数据。本课将在此基础上进一步深入：混合搜索（hybrid search）、重排序（reranking）和查询转换（query transformation）。

## 学习目标

- 实现能够保留文档结构与上下文的高级分块策略（语义分块、递归分块、父子分块）
- 构建混合搜索流水线，将 BM25 关键词匹配与语义向量搜索及交叉编码器重排序器（cross-encoder reranker）相结合
- 应用查询转换技术（HyDE、多路查询、后退一步查询）以提升对模糊或复杂问题的检索效果
- 诊断并修复常见的 RAG 故障：检索到错误的文本块、答案不在上下文中、多跳推理失效

## 问题背景

你在第 06 课中构建了一个基础 RAG 流水线。它在小规模语料库上处理直白的问题时表现良好。但现在请尝试以下场景：

**模糊查询**：“上季度的营收是多少？”语义搜索会返回关于营收策略、营收预测以及 CFO 对营收增长看法的文本块。这些内容在语义上都与“营收”一词相似，但都不包含具体数字。正确的文本块写的是“2025 年第三季度营收为 4720 万美元”，但使用的是“earnings（收益）”而非“revenue（营收）”。嵌入模型（embedding model）会认为“营收策略”比“第三季度收益为 4720 万美元”与查询更相似。

**多跳问题**：“哪个团队的客户满意度评分提升幅度最大？”这需要找出每个团队的满意度评分，进行比较，并找出最大值。没有任何单个文本块包含完整答案，相关信息分散在各团队的报告中。

**大规模语料库问题**：你拥有 200 万个文本块。正确答案位于第 1,847,293 号文本块中。但你的 Top-5 检索结果拉取的是第 14、89,201、1,200,000、44 和 901,333 号文本块。它们在嵌入空间中距离较近，但都不包含答案。在这种规模下，近似最近邻搜索（approximate nearest neighbor search）引入的误差足以将相关结果挤出 Top-k 范围。

基础 RAG 之所以失效，是因为向量相似度并不等同于相关性。一个文本块可能在语义上与查询相似，但对回答问题毫无帮助。高级 RAG 通过四种技术来解决这一问题：混合搜索（引入关键词匹配）、重排序（更精细地对候选结果打分）、查询转换（在搜索前优化查询语句）以及更优的分块策略（以合适的粒度进行检索）。

## 核心概念

### 混合搜索 (Hybrid Search)：语义 + 关键词

语义搜索 (Semantic Search，基于向量相似度) 擅长理解含义。即使“如何取消我的订阅？”和“终止计划的步骤”没有相同的词汇，它们也能匹配上。但它会遗漏精确匹配。如果嵌入模型 (Embedding Model) 将“错误代码 E-4021”视为噪声，它可能无法匹配到包含“E-4021”的文本块 (Chunk)。

关键词搜索 (Keyword Search) 则恰恰相反。它擅长精确匹配。“E-4021”可以完美匹配。但如果文档写的是“terminate your plan”，搜索“cancel my subscription”将返回零结果。

混合搜索会同时运行这两种方法，然后合并结果。

**BM25** (Best Matching 25) 是标准的关键词搜索算法。自 20 世纪 90 年代以来，它一直是搜索引擎的核心。其公式如下：

BM25(q, d) = sum over terms t in q:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

其中，`tf(t,d)` 表示词项 `t` 在文档 `d` 中的词频 (Term Frequency)，`IDF(t)` 是逆文档频率 (Inverse Document Frequency)，`|d|` 是文档长度，`avgdl` 是平均文档长度，`k1` 控制词频饱和度（默认值为 1.2），`b` 控制长度归一化（默认值为 0.75）。

通俗地说：当文档包含查询词（尤其是罕见词）时，BM25 会给予更高的评分，但重复出现的词项带来的收益会递减。一个包含“revenue”一词 50 次的文档，其相关性并不会是仅包含一次该词的文档的 50 倍。

### 倒数排名融合 (Reciprocal Rank Fusion, RRF)

你会得到两个排序列表：一个来自向量搜索，另一个来自 BM25。如何将它们合并？倒数排名融合是标准做法。

RRF_score(d) = sum over rankings R:
    1 / (k + rank_R(d))

其中，`k` 是一个常数（通常为 60），用于防止排名靠前的结果占据绝对主导地位。

一个在向量搜索中排名第 1、在 BM25 中排名第 5 的文档得分为：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

一个在向量搜索中排名第 3、在 BM25 中排名第 2 的文档得分为：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

RRF 能够自然地平衡这两种信号。在两个列表中都排名靠前的文档将获得最高分。在一个列表中排名第 1 但在另一个列表中未出现的文档将获得中等分数。这种方法非常稳健，因为它使用的是排名而非原始分数，因此两个系统之间分数分布的差异不会影响结果。

### 重排序 (Reranking)

检索 (Retrieval，无论是向量、关键词还是混合检索) 速度快但精度有限。它使用双编码器 (Bi-encoders)：查询和每个文档被独立嵌入，然后进行比较。嵌入向量只需计算一次并缓存。这使得系统能够扩展到数百万份文档。

重排序使用交叉编码器 (Cross-encoders)：将查询和候选文档一起输入模型，输出相关性分数。模型同时看到两段文本，能够捕捉它们之间细粒度的交互。即使双编码器错过了关联，交叉编码器也能理解“第三季度收益是多少？”与包含“第三季度 4720 万美元”的文本块高度相关。

权衡之处在于：由于交叉编码器需要联合处理查询-文档对，其速度比双编码器慢 100 到 1000 倍。你无法预先计算一百万份文档的交叉编码器分数。解决方案是：先检索出更大的候选集（例如混合搜索的前 50 名），然后使用交叉编码器进行重排序，最终得到前 5 名。

graph LR
    Q["Query"] --> H["Hybrid Search"]
    H --> C50["Top 50 candidates"]
    C50 --> RR["Cross-Encoder Reranker"]
    RR --> C5["Top 5 final results"]
    C5 --> P["Build prompt"]
    P --> LLM["Generate answer"]

常见的重排序模型（2026 年阵容）：
- Cohere Rerank 3.5：托管 API，支持多语言，在混合语料库上召回率提升最佳
- Voyage rerank-2.5：托管 API，在托管选项中延迟最低
- Jina-Reranker-v2 Multilingual：开放权重，支持 100 多种语言
- bge-reranker-v2-m3：开放权重，表现强劲的基线模型
- cross-encoder/ms-marco-MiniLM-L-6-v2：开放权重，可在 CPU 上运行以进行原型开发
- ColBERTv2 / Jina-ColBERT-v2：晚期交互 (Late-interaction) 多向量重排序器——在评分时的时间复杂度为 O(tokens) 而非 O(docs)

### 查询转换 (Query Transformation)

有时问题不在于检索系统，而在于查询本身。“关于新政策变更的那件事是什么？”是一个糟糕的搜索查询。它不包含任何具体术语，生成的嵌入向量也很模糊。没有任何检索系统能凭此找到正确的文档。

**查询重写 (Query Rewriting)**：将用户的查询重新表述为更优的搜索查询。大语言模型 (LLM) 可以完成此任务：

User: "What was that thing about the new policy change?"
Rewritten: "Recent policy changes and updates"

**HyDE (Hypothetical Document Embeddings，假设性文档嵌入)**：不直接使用查询进行搜索，而是先生成一个假设性答案，将其嵌入，然后搜索与之相似的真实文档。

Query: "What is the refund policy for enterprise?"
Hypothetical answer: "Enterprise customers are eligible for a full refund
within 60 days of purchase. Refunds are pro-rated based on the remaining
subscription period and processed within 5-7 business days."

对假设性答案进行嵌入，并搜索与之相似的真实文档。其核心直觉是：在嵌入空间中，假设性答案比原始问题更接近真实答案。问题和答案具有不同的语言结构。通过生成假设性答案，你可以在嵌入空间中弥合“问题空间”与“答案空间”之间的差距。

HyDE 会在检索前增加一次 LLM 调用。这会使延迟增加 500-2000 毫秒。当原始查询的检索质量较差时，这种权衡是值得的。

### 父子分块 (Parent-Child Chunking)

传统的分块方法迫使你在两者间权衡：小块用于精确检索，大块用于提供充足上下文。父子分块消除了这种权衡。

为检索建立小块（128 个 token）的索引。当检索到某个小块时，将其父块（512 个 token）返回给提示词 (Prompt)。小块能精确匹配查询，而父块则为 LLM 生成优质答案提供充足的上下文。

graph TD
    P["Parent chunk (512 tokens)<br/>Full section about refund policy"]
    C1["Child chunk (128 tokens)<br/>Standard plan: 30-day refund"]
    C2["Child chunk (128 tokens)<br/>Enterprise: 60-day pro-rated"]
    C3["Child chunk (128 tokens)<br/>Processing time: 5-7 days"]
    C4["Child chunk (128 tokens)<br/>How to submit a request"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["Query: enterprise refund?"] -.->|"matches child"| C2
    C2 -.->|"return parent"| P

查询“enterprise refund?”精确匹配了子块 C2。但提示词接收到的是完整的父块 P，其中包含了关于处理时间和提交流程的周边上下文。

### 元数据过滤 (Metadata Filtering)

在执行向量搜索之前，先按元数据（如日期、来源、类别、作者、语言）过滤语料库。这能缩小搜索空间并防止返回不相关的结果。

“上个月安全政策有什么变化？”应仅搜索安全类别中过去 30 天内的文档。如果没有元数据过滤，你将搜索整个语料库，并可能检索到一份恰好语义相似但已有两年历史的安全文档。

生产环境的检索增强生成 (Retrieval-Augmented Generation, RAG) 系统会将元数据与每个文本块一起存储：源文档、创建日期、类别、作者、版本。向量数据库支持在相似度搜索前按元数据进行预过滤，这对于大规模系统的性能至关重要。

### 评估 (Evaluation)

你构建了一个 RAG 系统。如何判断它是否有效？主要看三个指标：

**检索相关性 (Recall@k，召回率)**：对于一组已知相关文档的测试问题，相关文档出现在前 k 个结果中的比例是多少？如果某个问题的答案位于第 47 号文本块中，该文本块是否出现在前 5 名中？

**忠实度 (Faithfulness)**：生成的答案是否基于检索到的文档？如果检索到的文本块写明“60 天退款期”，而模型却说“90 天退款期”，这就是忠实度失败。尽管模型拥有正确的上下文，但仍产生了幻觉 (Hallucination)。

**答案正确性 (Answer Correctness)**：生成的答案是否与预期答案匹配？这是端到端的指标，综合了检索质量和生成质量。

一个简单的忠实度检查方法：提取生成答案中的每个主张，并验证其（实质上）是否出现在检索到的文本块中。如果答案包含任何检索文本块中未提及的事实，则很可能是幻觉。

graph TD
    subgraph "Evaluation Framework"
        Q["Test questions<br/>+ expected answers<br/>+ relevant doc IDs"]
        Q --> Ret["Retrieval evaluation<br/>Recall@k: are right<br/>docs retrieved?"]
        Q --> Faith["Faithfulness evaluation<br/>Is answer grounded<br/>in retrieved docs?"]
        Q --> Correct["Correctness evaluation<br/>Does answer match<br/>expected answer?"]
    end


## 开始构建

### 步骤 1：BM25 实现

import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

### 步骤 2：倒数排名融合（Reciprocal Rank Fusion）

def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused

### 步骤 3：混合搜索流水线（Hybrid Search Pipeline）

def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]

### 步骤 4：简单重排序器（Simple Reranker）

在生产环境中，通常会使用交叉编码器（Cross-Encoder）模型。此处我们构建了一个重排序器（Reranker），它通过词重叠度、词项重要性以及短语匹配来对查询与文档的相关性进行打分。

def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

### 步骤 5：HyDE（假设性文档嵌入，Hypothetical Document Embeddings）

def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis

### 步骤 6：父子分块（Parent-Child Chunking）

def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent

### 步骤 7：忠实度评估（Faithfulness Evaluation）

def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results


## 使用方法

使用真实的交叉编码器（Cross-Encoder）进行重排序（Reranking）：

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

使用 Cohere 的托管重排序服务：

import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]

结合真实的大语言模型（Large Language Model, LLM）使用假设性文档嵌入（Hypothetical Document Embeddings, HyDE）：

import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text

在 Weaviate 中实现生产级混合搜索（Hybrid Search）：

import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)

`alpha` 参数用于控制权重平衡：0.0 表示纯关键词搜索（BM25），1.0 表示纯向量搜索，0.5 表示两者权重相等。大多数生产环境系统会将 `alpha` 设置在 0.3 到 0.7 之间。

## 交付上线

本章节将生成以下文件：
- `outputs/prompt-advanced-rag-debugger.md` -- 用于诊断和修复检索增强生成（Retrieval-Augmented Generation, RAG）质量问题的提示词（Prompt）
- `outputs/skill-advanced-rag.md` -- 用于构建具备混合搜索与重排序功能的生产级 RAG 的实战指南

## 练习

1. 在示例文档上对比 BM25、向量搜索（Vector Search）与混合搜索（Hybrid Search）。针对 5 个测试查询，分别记录哪种方法能在结果列表的第 1 位返回最相关的文本块（Chunk）。混合搜索应在至少 3 个查询中表现最佳。

2. 实现元数据过滤（Metadata Filtering）。为每个文档添加一个 "category" 字段（取值包括 security、billing、api、product）。在执行向量搜索前，先将文本块过滤至仅保留相关类别。使用查询“使用了什么加密技术？”进行测试，并验证系统是否仅检索了 security 类别的文本块。

3. 使用第 06 课中的简单 generate 函数构建完整的 HyDE 流水线。在所有 5 个测试查询上，对比直接查询搜索与 HyDE 搜索的检索质量（Top-3 相关性）。对于模糊查询，HyDE 应能带来检索效果的提升。

4. 在示例文档上实现父子分块（Parent-Child Chunking）策略。使用 child_size=30 和 parent_size=100。使用子文本块进行检索，但在提示词（Prompt）中返回父文本块。将生成的答案与采用 chunk_size=50 的标准分块策略进行对比。

5. 构建评估数据集：包含 10 个问题及其对应的已知答案文本块。针对以下四种配置分别测量 Recall@3、Recall@5 和 Recall@10：(a) 仅向量搜索，(b) 仅 BM25，(c) 混合搜索，(d) 混合搜索 + 重排序（Reranking）。绘制结果图表，并分析重排序在哪些场景下带来的提升最为显著。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| BM25 | “关键词搜索” | 一种概率排序算法，通过词频（Term Frequency）、逆文档频率（Inverse Document Frequency）和文档长度归一化对文档进行打分 |
| 混合搜索（Hybrid Search） | “两全其美” | 并行运行语义（向量）搜索与关键词（BM25）搜索，然后通过排序融合（Rank Fusion）合并结果 |
| 倒数排名融合（Reciprocal Rank Fusion） | “合并排序列表” | 通过对每个文档在所有列表中的排名计算 1/(k + rank) 并求和，从而合并多个排序列表 |
| 重排序（Reranking） | “二次打分” | 使用计算成本更高的交叉编码器（Cross-Encoder）模型，对初始检索得到的候选集进行重新打分 |
| 交叉编码器（Cross-Encoder） | “查询-文档联合模型” | 将查询和文档作为单一输入并输出相关性得分的模型；精度高于双编码器（Bi-Encoder），但速度过慢，不适用于全库检索 |
| 双编码器（Bi-Encoder） | “独立嵌入模型” | 独立对查询和文档进行嵌入（Embedding）的模型；由于嵌入向量可预先计算，速度较快，但精度不及交叉编码器 |
| HyDE | “用虚构答案进行搜索” | 针对查询生成一个假设性答案，对其进行嵌入，然后检索与该假设答案相似的真实文档 |
| 父子分块（Parent-Child Chunking） | “小粒度检索，大上下文” | 索引较小的文本块以实现精准检索，但返回较大的父文本块以提供充足的上下文 |
| 元数据过滤（Metadata Filtering） | “先缩小范围再搜索” | 在执行向量搜索前，根据属性（日期、来源、类别等）过滤文档，以缩小搜索空间 |
| 忠实度（Faithfulness） | “回答是否基于事实” | 生成的答案是否由检索到的文档支撑，而非模型基于训练数据产生的幻觉（Hallucination） |

## 延伸阅读

- Robertson 与 Zaragoza，《概率相关性框架：BM25 及其延伸》（2009）—— BM25 的权威参考文献，详细阐述了该公式背后的概率论基础。
- Cormack 等人，《倒数排名融合优于孔多塞与独立排名学习方法》（2009）—— 倒数排名融合（Reciprocal Rank Fusion, RRF）的原始论文，证明其效果优于更复杂的融合方法。
- Gao 等人，《无需相关性标签的精准零样本稠密检索》（2022）—— 假设文档嵌入（Hypothetical Document Embeddings, HyDE）论文，证明了无需任何训练数据，假设文档嵌入即可提升检索效果。
- Nogueira 与 Cho，《基于 BERT 的段落重排序》（2019）—— 证明了在 BM25 基础上使用交叉编码器（cross-encoder）进行重排序可显著提升检索质量。
- [Khattab 等人，《DSPy：将声明式语言模型调用编译为自改进流水线》（2023）](https://arxiv.org/abs/2310.03714) —— 将提示词构建与权重选择视为针对检索流水线的优化问题；推荐阅读此文以了解如何“编程式调用大语言模型（program LLMs）”而非仅仅“提示大语言模型（prompt LLMs）”。
- [Edge 等人，《从局部到全局：一种面向查询摘要的图 RAG 方法》（Microsoft Research 2024）](https://arxiv.org/abs/2404.16130) —— GraphRAG 论文：结合实体关系提取与 Leiden 社区发现（Leiden community detection）算法实现面向查询的摘要生成；阐述了全局检索与局部检索的区别。
- [Asai 等人，《Self-RAG：通过自我反思学习检索、生成与批判》（ICLR 2024）](https://arxiv.org/abs/2310.11511) —— 引入反思令牌（reflection tokens）实现具备自我评估能力的 RAG；代表了超越静态“先检索后生成”范式的智能体（agentic）前沿方向。
- [LangChain 查询构建博客](https://blog.langchain.dev/query-construction/) —— 介绍如何在检索前步骤中将自然语言查询转换为结构化数据库查询（如 Text-to-SQL、Cypher）。
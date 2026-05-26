# RAG（检索增强生成，Retrieval-Augmented Generation）

> 你的大语言模型（Large Language Model, LLM）对其训练截止日期之前的知识无所不知，但对你公司的文档、代码库或上周的会议记录却一无所知。RAG 通过检索相关文档并将其填入提示词（Prompt）来解决这一问题。它是生产级 AI 中部署最广泛的架构模式。如果你要从本课程中只完成一个项目，那就构建一个 RAG 流水线（Pipeline）。

**类型：** 构建
**编程语言：** Python
**前置要求：** 第 10 阶段（从零构建 LLM），第 11 阶段第 01-05 课
**预计耗时：** 约 90 分钟
**相关课程：** 第 5 阶段 · 23（RAG 的分块策略）了解六种分块算法及其适用场景。第 5 阶段 · 22（嵌入模型深入解析）了解如何选择嵌入模型。第 11 阶段 · 07（高级 RAG）了解混合检索、重排序和查询转换。

## 学习目标

- 构建完整的 RAG 流水线：涵盖文档加载、文本分块（Chunking）、向量化嵌入（Embedding）、向量存储、检索与生成
- 使用向量数据库（Vector Database）（如 ChromaDB、FAISS 或 Pinecone）并配合正确的索引机制实现语义搜索（Semantic Search）
- 解释为何在知识增强型应用中，RAG 优于模型微调（Fine-tuning）（成本、数据时效性、来源可追溯性）
- 使用检索评估指标（精确率 Precision、召回率 Recall）和生成评估指标（忠实度 Faithfulness、相关性 Relevance）来评估 RAG 质量

## 问题背景

你为公司开发了一款聊天机器人。一位客户询问：“企业版套餐的退款政策是什么？”大语言模型回复了一个关于典型 SaaS 退款政策的通用答案。而实际政策隐藏在一份 200 页的内部 Wiki 文档中，规定企业客户享有 60 天的退款期，并按比例退款。该模型从未见过这份文档，它自然无法知道训练数据中不存在的内容。

模型微调是一种解决方案。将大语言模型基于你的内部文档进行训练，然后部署更新后的模型。这确实可行，但存在严重缺陷。微调需要耗费数千美元的计算成本。一旦文档发生变更，模型知识就会立刻过时。你无法得知模型的回答具体引用了哪些来源。此外，如果公司下个月收购了另一条产品线，你又得重新进行微调。

RAG 是另一种解决方案。保持模型参数不变。当收到问题时，先在文档库中搜索相关段落，将这些段落粘贴到提示词中问题的前面，让模型基于这些上下文进行回答。文档库可以在几分钟内完成更新。你可以清楚地看到模型检索了哪些文档。模型本身始终保持不变。这正是 RAG 成为生产环境主流架构的原因：它成本更低、知识更新更及时、过程更可审计，并且能够适配任何大语言模型。

## 核心概念

### RAG 模式 (Retrieval-Augmented Generation)

整个模式可归纳为四个步骤：

graph LR
    Q["User Query"] --> R["Retrieve"]
    R --> A["Augment Prompt"]
    A --> G["Generate"]
    G --> Ans["Answer"]

    subgraph "Retrieve"
        R --> Embed["Embed query"]
        Embed --> Search["Search vector store"]
        Search --> TopK["Return top-k chunks"]
    end

    subgraph "Augment"
        TopK --> Format["Format chunks into prompt"]
        Format --> Combine["Combine with user question"]
    end

    subgraph "Generate"
        Combine --> LLM["LLM generates answer"]
        LLM --> Cite["Answer grounded in retrieved docs"]
    end

查询 (Query) -> 检索 (Retrieve) -> 增强提示词 (Augment Prompt) -> 生成 (Generate)。每个 RAG 系统都遵循这一模式。生产级 RAG 系统之间的差异在于每个步骤的具体实现细节：如何进行文本分块 (Chunking)、如何生成嵌入向量 (Embedding)、如何执行搜索，以及如何构建提示词 (Prompt)。

### 为什么 RAG 优于微调 (Fine-Tuning)

| 关注点 | 微调 (Fine-Tuning) | RAG |
|---------|------------|-----|
| 成本 | 每次训练运行 $1,000-$100,000+ | 每次查询 $0.01-$0.10（嵌入 + 大语言模型） |
| 时效性 | 重新训练前数据已过时 | 通过重新索引文档，几分钟内即可更新 |
| 可审计性 | 无法将答案追溯至来源 | 可展示精确检索到的原文段落 |
| 幻觉问题 | 仍会自由产生幻觉 | 答案基于检索到的文档，有据可依 |
| 数据隐私 | 训练数据已固化至模型权重中 | 文档保留在您自己的向量数据库中 |

微调会永久改变模型的权重，而 RAG 只是临时改变模型的上下文。对于大多数应用场景而言，临时上下文正是你所需要的。

微调唯一胜出的场景是：你需要模型采用某种特定的风格、语气或推理模式，且仅靠提示词无法实现时。而对于事实性知识检索，RAG 总是赢家。

### 嵌入模型 (Embedding Models)

嵌入模型将文本转换为稠密向量 (Dense Vector)。语义相似的文本在该高维空间中生成的向量距离很近。尽管“How do I reset my password?”和“I need to change my password”这两个句子共用的词汇很少，但它们生成的向量几乎完全相同。而“The cat sat on the mat”生成的向量则截然不同。

主流嵌入模型（2026 年阵容——完整分析请参阅 Phase 5 · 22）：

| 模型 | 维度 | 提供商 | 备注 |
|-------|-----------|----------|-------|
| text-embedding-3-small | 1536 (Matryoshka) | OpenAI | 大多数用例中性价比最佳 |
| text-embedding-3-large | 3072 (Matryoshka) | OpenAI | 精度更高，可截断至 256/512/1024 |
| Gemini Embedding 2 | 3072 (Matryoshka) | Google | MTEB 检索任务表现顶尖；支持 8K 上下文 |
| voyage-4 | 1024/2048 (Matryoshka) | Voyage AI | 提供领域变体（代码、金融、法律） |
| Cohere embed-v4 | 1024 (Matryoshka) | Cohere | 多语言能力强，支持 128K 上下文 |
| BGE-M3 | 1024 (dense + sparse + ColBERT) | BAAI（开放权重） | 单一模型提供三种视图 |
| Qwen3-Embedding | 4096 (Matryoshka) | 阿里巴巴（开放权重） | 开放权重模型中检索得分最高 |
| all-MiniLM-L6-v2 | 384 | 开放权重 (Sentence Transformers) | 原型开发基准模型 |

在本课程中，我们将使用 TF-IDF 构建一个简单的嵌入模型。这并非因为生产系统会使用 TF-IDF，而是因为它能让概念更加具体直观：输入文本，输出向量，相似的文本会产生相似的向量。

### 向量相似度 (Vector Similarity)

给定两个向量，如何衡量它们的相似度？有三种常用方法：

**余弦相似度 (Cosine Similarity)**：两个向量之间夹角的余弦值。取值范围为 -1（方向相反）到 1（完全相同）。它忽略向量的模长，仅关注方向。这是 RAG 的默认选择。

cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)

**点积 (Dot Product)**：原始的内积运算。模长较大的向量会获得更高的分数。当模长本身携带信息时（例如较长的文档可能更相关），此方法非常有用。

dot(a, b) = sum(a_i * b_i)

**L2（欧几里得）距离 (L2/Euclidean Distance)**：向量空间中的直线距离。距离越小 = 相似度越高。对模长差异较为敏感。

L2(a, b) = sqrt(sum((a_i - b_i)^2))

余弦相似度是行业标准。由于它通过模长进行了归一化处理，因此能优雅地处理不同长度的文档。当人们提到“向量搜索 (Vector Search)”时，几乎总是指基于余弦相似度的搜索。

### 文本分块策略 (Chunking Strategies)

文档通常过长，无法直接作为单个向量进行嵌入。一份 50 页的 PDF 可能会生成质量极差的嵌入向量，因为它包含数十个不同的主题。因此，你需要将文档拆分为多个文本块 (Chunks)，并分别对每个块进行嵌入。

**固定大小分块 (Fixed-Size Chunking)**：每 N 个词元 (Token) 进行一次切分。简单且可预测。例如，512 个词元的块设置 50 个词元的重叠 (Overlap)，意味着块 1 包含词元 0-511，块 2 包含词元 462-973，依此类推。重叠机制能确保句子不会在尴尬的边界处被切断。

**语义分块 (Semantic Chunking)**：在自然边界处进行切分，如段落、章节或 Markdown 标题。每个文本块都是一个语义连贯的单元。实现起来更复杂，但能带来更好的检索效果。

**递归分块 (Recursive Chunking)**：优先尝试在最大边界处切分（如章节标题）。如果某个章节仍然过大，则退而求其次在段落边界处切分。如果段落依然过大，则在句子边界处切分。这是 LangChain 中 `RecursiveCharacterTextSplitter` 的实现思路，在实际应用中效果良好。

文本块的大小比人们想象的更重要：

- **过小（64-128 个词元）**：每个块缺乏上下文。如果不知道“它”指代什么，“它上季度增长了 15%”这句话就毫无意义。
- **过大（2048+ 个词元）**：每个块涵盖多个主题，稀释了相关性。当你搜索营收数据时，返回的文本块可能只有 10% 与营收相关，其余 90% 都在讲员工人数。
- **最佳区间（256-512 个词元）**：上下文足够完整以自成一体，同时足够聚焦以保证相关性。

大多数生产级 RAG 系统采用 256-512 个词元的文本块，并设置 50 个词元的重叠。Anthropic 的 RAG 指南也推荐这一范围。

### 向量数据库 (Vector Databases)

生成嵌入向量后，你需要一个地方来存储和检索它们。常见选项如下：

| 数据库 | 类型 | 适用场景 |
|----------|------|----------|
| FAISS | 库（进程内） | 原型开发、中小型数据集 |
| Chroma | 轻量级数据库 | 本地开发、小型部署 |
| Pinecone | 托管服务 | 生产环境，无需运维负担 |
| Weaviate | 开源数据库 | 自托管生产环境 |
| pgvector | Postgres 扩展 | 已在使用 Postgres 的场景 |
| Qdrant | 开源数据库 | 高性能自托管环境 |

在本课程中，我们将构建一个简单的内存向量存储 (In-Memory Vector Store)。它将向量保存在列表中，并执行暴力 (Brute-Force) 余弦相似度搜索。这等同于使用扁平索引 (Flat Index) 的 FAISS。在性能下降前，它大约能扩展到 10 万个向量。生产系统则使用近似最近邻 (Approximate Nearest Neighbor, ANN) 算法（如 HNSW），以便在毫秒级时间内检索数百万个向量。

### 完整流水线 (Full Pipeline)

graph TD
    subgraph "Indexing (offline)"
        D["Documents"] --> C["Chunk"]
        C --> E["Embed each chunk"]
        E --> S["Store vectors + text"]
    end

    subgraph "Querying (online)"
        Q["User query"] --> QE["Embed query"]
        QE --> VS["Vector search (top-k)"]
        VS --> P["Build prompt with chunks"]
        P --> LLM["LLM generates answer"]
    end

    S -.->|"same vector space"| VS

索引阶段针对每个文档运行一次（或在文档更新时运行）。查询阶段则在每次用户请求时触发。在生产环境中，索引阶段可能需要数小时来处理数百万份文档。而查询阶段必须在不到一秒的时间内返回响应。

### 实际参数参考 (Real Numbers)

大多数生产级 RAG 系统采用以下参数配置：

- **k = 5 到 10**：每次查询检索的文本块数量
- **文本块大小 = 256 到 512 个词元**，并设置 50 个词元的重叠
- **上下文预算**：每次查询使用 2,500-5,000 个词元的检索内容
- **总提示词长度**：约 8,000-16,000 个词元（系统提示词 + 检索到的文本块 + 对话历史 + 用户查询）
- **嵌入维度**：384-3072，具体取决于所选模型
- **索引吞吐量**：使用 API 生成嵌入时，每秒处理 100-1,000 份文档
- **查询延迟**：检索阶段 50-200 毫秒，生成阶段 500-3000 毫秒

## 构建

### 步骤 1：文档分块

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

### 步骤 2：TF-IDF 嵌入

我们构建一个简单的嵌入 (Embedding) 函数。TF-IDF（词频-逆文档频率，Term Frequency-Inverse Document Frequency）并非神经网络嵌入，但它能以捕捉词语重要性的方式将文本转换为向量。文档中频繁出现的词会获得较高的 TF 值。在整个语料库 (Corpus) 中罕见的词会获得较高的 IDF 值。两者的乘积构成一个向量，其中重要且具有区分度的词语会具有较高的数值。

import math
from collections import Counter

def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    return sorted(vocab)

def compute_tf(text, vocab):
    words = text.lower().split()
    count = Counter(words)
    total = len(words)
    return [count.get(word, 0) / total for word in vocab]

def compute_idf(documents, vocab):
    n = len(documents)
    idf = []
    for word in vocab:
        doc_count = sum(1 for doc in documents if word in doc.lower().split())
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf

def tfidf_embed(text, vocab, idf):
    tf = compute_tf(text, vocab)
    return [t * i for t, i in zip(tf, idf)]

### 步骤 3：余弦相似度搜索

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def search(query_embedding, stored_embeddings, top_k=5):
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

### 步骤 4：提示词构建

这正是 RAG（检索增强生成，Retrieval-Augmented Generation）中“增强”环节发挥作用的地方。我们将检索到的文本块格式化为提示词 (Prompt)，并要求大语言模型 (Large Language Model, LLM) 基于提供的上下文进行回答。

def build_rag_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    return f"""Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

### 步骤 5：完整的 RAG 流水线

class RAGPipeline:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.vocab = []
        self.idf = []

    def index(self, documents):
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))
        self.chunks = all_chunks
        self.vocab = build_vocabulary(all_chunks)
        self.idf = compute_idf(all_chunks, self.vocab)
        self.embeddings = [
            tfidf_embed(chunk, self.vocab, self.idf)
            for chunk in all_chunks
        ]

    def query(self, question, top_k=5):
        query_emb = tfidf_embed(question, self.vocab, self.idf)
        results = search(query_emb, self.embeddings, top_k)
        retrieved = [(self.chunks[i], score) for i, score in results]
        prompt = build_rag_prompt(
            question, [chunk for chunk, _ in retrieved]
        )
        return prompt, retrieved

### 步骤 6：生成（模拟）

在实际生产环境中，此处将调用大语言模型 (LLM) 的 API。在本教程中，我们通过从检索到的上下文中提取最相关的句子来模拟生成过程。

def simple_generate(prompt, retrieved_chunks):
    query_words = set(prompt.lower().split("question:")[-1].split())
    best_sentence = ""
    best_score = 0
    for chunk in retrieved_chunks:
        for sentence in chunk.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = set(sentence.lower().split())
            overlap = len(query_words & words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
    return best_sentence if best_sentence else "I don't have enough information."


## 实际应用

使用真实的嵌入模型（Embedding Model）和大语言模型（LLM）时，代码几乎不需要改动：

from openai import OpenAI

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

或者使用 Anthropic：

import anthropic

client = anthropic.Anthropic()

def generate(prompt):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

整体流水线（Pipeline）保持不变。只需替换嵌入函数和生成函数。检索逻辑、文本分块（Chunking）以及提示词构建（Prompt Construction）——无论你使用哪种模型，这些部分都完全一致。

若要实现大规模向量存储，请将暴力搜索替换为专业的向量数据库（Vector Database）：

import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

results = collection.query(
    query_texts=["What is the refund policy?"],
    n_results=5
)

Chroma 会在内部自动处理嵌入过程（默认使用 all-MiniLM-L6-v2 模型），并将向量存储在本地数据库中。模式相同，只是底层实现不同。

## 交付成果

本课时将生成以下文件：
- `outputs/prompt-rag-architect.md` -- 用于针对特定用例设计检索增强生成（RAG）系统的提示词
- `outputs/skill-rag-pipeline.md` -- 一项技能文件，用于指导智能体（Agent）如何构建和调试 RAG 流水线

## 练习

1. 将 TF-IDF 嵌入表示替换为简单的词袋模型（Bag-of-Words）方法（二值化：词出现记为 1，未出现记为 0）。在示例文档上对比两者的检索质量。TF-IDF 的表现应更优，因为它对罕见词赋予了更高的权重。

2. 尝试不同的分块大小：在同一文档集上分别测试 50、100、200 和 500 个词的分块效果。针对每种大小，运行相同的 5 个查询，并统计有多少次在前 3 个结果中返回了相关分块。找出检索质量达到峰值的最佳平衡点。

3. 为每个分块添加元数据（Metadata）（如源文档名称、分块位置）。修改提示词模板以包含来源标注，从而使大语言模型能够引用其信息来源。

4. 实现一个简单的评估流程：给定 10 组问答对，将每个问题输入 RAG 流水线，并测量检索到的分块中包含正确答案的比例。这即为 Top-k 检索召回率（Retrieval Recall at k）。

5. 构建具备对话感知能力的 RAG 流水线：保留最近 3 轮对话的历史记录，并将其与检索到的分块一同加入提示词中。使用后续追问进行测试，例如在询问价格后接着问“企业版呢？”。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 检索增强生成（RAG） | “能阅读你文档的AI” | 检索相关文档，将其注入提示词（Prompt），并基于这些文档生成有据可依的答案 |
| 嵌入（Embedding） | “将文本转换为数字” | 文本的稠密向量表示，语义相近的文本会生成相似的向量 |
| 向量数据库（Vector Database） | “AI的搜索引擎” | 专为存储向量而优化的数据存储系统，可通过相似度计算查找最近邻 |
| 分块（Chunking） | “将文档拆分成片段” | 将文档切分为较小的片段（通常为 256-512 个词元（token）），以便每个片段都能独立进行嵌入和检索 |
| 余弦相似度（Cosine Similarity） | “两个向量的相似程度” | 两个向量夹角的余弦值；1 表示方向完全一致，0 表示正交（无关），-1 表示方向完全相反 |
| Top-k 检索（Top-k Retrieval） | “获取 k 个最佳匹配” | 从向量存储中返回与查询最相似的 k 个文本块 |
| 上下文窗口（Context Window） | “大语言模型能处理的文本量” | 大语言模型（LLM）在单次请求中可处理的最大词元（token）数量；检索到的文本块必须控制在此范围内 |
| 增强生成（Augmented Generation） | “基于给定上下文作答” | 利用检索到的文档作为上下文来生成回复，而非仅依赖模型预训练的知识 |
| TF-IDF（词频-逆文档频率） | “词语重要性评分” | 词频乘以逆文档频率；根据词语在语料库中的区分度对其进行加权 |
| 索引构建（Indexing） | “为搜索准备文档” | 离线处理流程，包含文档分块、向量化嵌入与存储，以便在查询阶段进行快速检索 |

## 延伸阅读

- Lewis 等人，《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》（2020）—— Facebook AI Research 发表的 RAG 奠基性论文，正式确立了“先检索后生成（retrieve-then-generate）”模式
- Anthropic 的 RAG 文档 (docs.anthropic.com) —— 关于分块大小、提示词（Prompt）构建与评估的实用指南
- Pinecone 学习中心，《What is RAG?》—— 结合生产环境考量，对 RAG 流程进行清晰的可视化解析
- Sentence-BERT：Reimers & Gurevych（2019）—— all-MiniLM 嵌入模型背后的核心论文，展示了如何训练双编码器（Bi-encoder）以实现语义相似度计算
- [Karpukhin 等人，《Dense Passage Retrieval for Open-Domain Question Answering》（EMNLP 2020）](https://arxiv.org/abs/2004.04906) —— DPR 论文，证明了稠密双编码器检索在开放域问答中优于 BM25，并为现代 RAG 检索器奠定了范式。
- [LlamaIndex 高层概念](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html) —— 构建 RAG 管道需掌握的核心概念：数据加载器、节点解析器、索引、检索器与响应合成器。
- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/) —— 风格迥异的编排框架；以可运行对象链（chain-of-runnables）的视角呈现相同的“先检索后生成”模式。
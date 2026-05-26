# 嵌入模型（Embedding Models）—— 2026 深度解析

> Word2Vec 为每个词生成一个向量。现代嵌入模型则为每个文本片段（Passage）生成一个向量，支持跨语言，并提供稀疏（Sparse）、稠密（Dense）和多向量（Multi-vector）等多种视图，且尺寸经过优化以适配你的索引（Index）。选错模型，你的检索增强生成（RAG）系统就会检索到错误的内容。

**类型：** 学习
**编程语言：** Python
**前置知识：** 第 5 阶段 · 03（Word2Vec），第 5 阶段 · 14（信息检索）
**预计耗时：** 约 60 分钟

## 问题所在

你的检索增强生成（RAG）系统有 40% 的时间会检索到错误的文本片段。罪魁祸首通常不是向量数据库（Vector Database）或提示词（Prompt），而是嵌入模型。

在 2026 年选择嵌入模型，意味着需要在五个维度上进行权衡：

1. **稠密（Dense） vs 稀疏（Sparse） vs 多向量（Multi-vector）。** 每个文本片段对应一个向量，还是每个词元（Token）对应一个向量，亦或是采用稀疏加权词袋（Bag of Words）表示。
2. **语言覆盖范围（Language Coverage）。** 单语英语模型在纯英语任务上依然表现最佳。而当语料库（Corpora）包含多种语言时，多语言模型更具优势。
3. **上下文长度（Context Length）。** 512 个词元 vs 8,192 个词元 vs 32,768 个词元——而实际有效容量通常仅为标称最大值的 60% 到 70%。
4. **维度预算（Dimension Budget）。** 全精度下的 3,072 个浮点数（Floats） = 每个向量 12 KB。在 1 亿个向量的规模下，存储成本约为每月 1,300 美元。采用套娃截断（Matryoshka Truncation）技术可将该开销缩减至原来的 1/4。
5. **开源（Open） vs 托管（Hosted）。** 开放权重（Open-weight）意味着你可以完全掌控技术栈（Tech Stack）和数据。托管服务则意味着你以牺牲控制权为代价，换取始终使用最新版本。

本课程将明确列出这些权衡取舍，助你基于客观证据做出选择，而非盲目追随上一季度的流行趋势。

## 核心概念

![密集、稀疏和多向量嵌入](../assets/embedding-modes.svg)

**密集嵌入（Dense embeddings）。** 每个文本片段对应一个向量（通常为 384 到 3,072 维）。通过余弦相似度（Cosine similarity）根据语义接近程度对片段进行排序。代表模型包括 OpenAI `text-embedding-3-large`、BGE-M3 的密集模式以及 Voyage-3。这是默认的首选方案。

**稀疏嵌入（Sparse embeddings）。** 采用 SPLADE 风格。Transformer 模型会为词汇表中的每个词元（Token）预测一个权重，随后将大部分权重置零。最终生成一个大小为 |vocab| 的稀疏向量。它能够捕捉词汇匹配（类似 BM25），但权重是通过学习得到的。在处理富含关键词的查询时表现优异。

**多向量嵌入（Multi-vector embeddings，延迟交互（Late interaction））。** 代表模型有 ColBERTv2 和 Jina-ColBERT。每个词元对应一个向量。采用 MaxSim 进行打分：针对查询中的每个词元，找出文档中最相似的词元，并将这些最高相似度分数相加。存储和计算成本较高，但在处理长查询和特定领域语料库时效果最佳。

**BGE-M3：三者合一。** 单个模型可同时输出密集、稀疏和多向量表示。每种表示均可独立查询，最终得分通过加权求和进行融合。当你希望仅凭一个模型检查点（Checkpoint）就能获得灵活性时，它是 2026 年的默认选择。

**套娃表示学习（Matryoshka Representation Learning）。** 训练方式使得向量的前 N 个维度即可构成一个独立且有效的嵌入。例如，将 1,536 维的向量截断至 256 维，仅需牺牲约 1% 的准确率，即可节省 6 倍的存储空间。目前受 OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2 以及 Nomic v1.5+ 等模型支持。

### MTEB 排行榜仅反映部分事实

大规模文本嵌入基准测试（Massive Text Embedding Benchmark, MTEB）—— 发布之初（2022 年）涵盖 8 种任务类型的 56 项任务，在 MTEB v2 中已扩展至 100 多项任务。截至 2026 年初，Gemini Embedding 2 在检索任务中位居榜首（MTEB-R 得分 67.71）。Cohere embed-v4 在通用任务中领先（MTEB 得分 65.2）。BGE-M3 则在开放权重（Open-weight）多语言任务中排名第一（得分 63.0）。排行榜必要但不充分——务必在你的特定业务领域进行基准测试。

### 三层模式

| 使用场景 | 模式 |
|----------|---------|
| 快速初筛 | 密集双编码器（Dense bi-encoder，如 BGE-M3、text-3-small） |
| 提升召回率 | 稀疏嵌入（Sparse，如 SPLADE、BGE-M3 稀疏模式）+ 倒数排名融合（Reciprocal Rank Fusion, RRF） |
| 优化 Top-50 精确率 | 多向量嵌入（Multi-vector，如 ColBERTv2）或交叉编码器重排序器（Cross-encoder reranker） |

大多数生产环境的技术栈会同时采用这三种模式。

## 动手构建

### 步骤 1：基线模型 — 使用 Sentence-BERT 的稠密嵌入（Dense Embeddings）

from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))

设置 `normalize_embeddings=True` 可使点积（Dot Product）等于余弦相似度（Cosine Similarity）。请务必始终启用该参数。

### 步骤 2：套娃截断（Matryoshka Truncation）

def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)

截断后需重新进行归一化（Normalization）。Nomic v1.5、OpenAI text-3 和 Voyage-4 等模型在训练时已针对此特性优化，因此在截断前几个维度时可实现无损。非套娃架构模型（如原始 Sentence-BERT）在截断后性能会急剧下降。

### 步骤 3：BGE-M3 的多功能性

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays

一次推理调用即可生成三种索引。分数融合（Score Fusion）方法如下：

dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score

请根据您的具体业务领域调整权重参数。

### 步骤 4：在自定义任务上进行 MTEB 评估（MTEB Evaluation）

from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")

请在具有*代表性*的数据子集上运行候选模型。切勿仅依赖排行榜（Leaderboard）排名——您的具体业务领域才是关键。

### 步骤 5：从零手写余弦相似度计算

详见 `code/main.py`。该示例使用仅依赖标准库的平均哈希技巧嵌入（Averaged Hashing Trick Embeddings）。其性能虽无法与 Transformer 嵌入（Transformer Embeddings）相媲美，但清晰展示了核心流程：分词（Tokenize）→ 向量化（Vectorize）→ 归一化（Normalize）→ 点积（Dot Product）。

## 常见陷阱（Pitfalls）

- **查询与文档使用相同模型。** 部分模型（如 Voyage、Jina-ColBERT）采用非对称编码（asymmetric encoding）——查询和文档会经过不同的处理路径。务必查阅模型卡片（model card）。
- **遗漏前缀。** `bge-*` 系列模型需要在查询语句前添加 `"Represent this sentence for searching relevant passages: "`。若忘记添加，召回率（recall）会下降 3-5 个百分点。
- **过度裁剪马特里奥什卡（Matryoshka）维度。** 从 1,536 维降至 256 维通常是安全的，但降至 64 维则不然。请在你的评估集（eval set）上进行验证。
- **上下文截断（context truncation）。** 大多数模型会静默截断超出其最大长度的输入。长文档需要进行分块处理（chunking）（参见第 23 课）。
- **忽略长尾延迟（latency tail）。** MTEB（大规模文本嵌入基准）分数会掩盖 p99 延迟。一个 6 亿参数的模型可能比 3.35 亿参数的模型高出 2 分，但单次查询成本却高出 3 倍。

## 使用指南

2026 年技术栈推荐：

| 适用场景 | 推荐选择 |
|-----------|------|
| 仅限英文、快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重（open-weight）、英文 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（1.37 亿参数，MoE（混合专家模型）架构） |
| 存储受限 | 马特里奥什卡（Matryoshka）截断 + int8 量化（quantization） |
| 关键词密集型查询 | 添加 SPLADE 稀疏向量，使用 RRF（倒数排名融合）与稠密向量融合 |

2026 年选型模式：从 BGE-M3 或 text-3-large 起步，使用 MTEB 在你的业务领域进行评估；若特定领域模型得分高出 3 分以上，则进行替换。

## 交付与部署

保存为 `outputs/skill-embedding-picker.md`：

---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.

## 练习

1. **简单。** 使用 `bge-small-en-v1.5` 对 100 个句子进行编码，先使用完整维度（384），再使用马特里奥什卡（Matryoshka）128 维。测量在 10 个查询上的 MRR（平均倒数排名）下降幅度。
2. **中等。** 在你的业务领域选取 500 个段落，对比 BGE-M3 的稠密（dense）、稀疏（sparse）和 ColBERT 模式。哪种模式在 recall@10 上表现最佳？RRF（倒数排名融合）是否优于单一最佳模式？
3. **困难。** 针对你的前两大领域任务，在三个候选模型上运行 MTEB 评估。报告 MTEB 得分、100 个查询批次的 p99 延迟以及每百万次查询的成本（$/1M queries）。选择帕累托最优（Pareto-optimal）的模型。

## 关键术语

| 术语 | 业界俗称 | 实际含义 |
|------|----------|----------|
| 稠密嵌入 (Dense Embedding) | “那个向量” | 每段文本对应一个固定维度的向量。通过余弦相似度进行排序。 |
| 稀疏嵌入 (Sparse Embedding) | “可学习的 BM25” | 词汇表中的每个词元对应一个权重；矩阵大部分为零；支持端到端训练。 |
| 多向量 (Multi-vector) | “ColBERT 风格” | 每个词元对应一个向量；采用 MaxSim 评分机制；索引体积更大，但召回率更高。 |
| 套娃表示 (Matryoshka) | “俄罗斯套娃技巧” | 向量的前 N 个维度本身即可独立构成一个有效的低维嵌入。 |
| MTEB | “那个基准测试” | 大规模文本嵌入基准测试 (Massive Text Embedding Benchmark)——发布初期包含 56 个任务，v2 版本已扩展至 100 多个。 |
| BEIR | “那个检索基准” | 包含 18 个零样本 (Zero-shot) 检索任务；常因其出色的跨领域鲁棒性而被广泛引用。 |
| 非对称编码 (Asymmetric Encoding) | “查询与文档路径不同” | 模型为查询和文档分别使用不同的投影层进行编码。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双编码器 (Bi-encoder) 架构的奠基论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 该排行榜的官方论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 统一三种模式的模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 提出维度阶梯式训练目标的论文。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 探讨晚期交互 (Late Interaction) 在生产环境中的应用。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 实时更新的排行榜。
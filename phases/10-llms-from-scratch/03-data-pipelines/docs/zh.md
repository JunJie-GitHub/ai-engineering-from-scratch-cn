# 预训练数据流水线 (Data Pipelines for Pre-Training)

> 模型是一面镜子。你喂给它什么数据，它就反射什么。喂给它垃圾，它就能以极其流畅的方式反射出垃圾。

**类型：** 构建
**语言：** Python
**前置要求：** 第 10 阶段，课程 01-02（分词器 (Tokenizers)、构建分词器）
**预计时间：** 约 90 分钟

## 学习目标

- 构建一个流式数据流水线 (Streaming Data Pipeline)，能够在不将所有数据加载到内存的情况下，对 TB 级文本进行分词 (Tokenization)、分块 (Chunking)、打乱 (Shuffling) 和批处理 (Batching)
- 实现真实预训练流水线中使用的数据质量过滤器（去重 (Deduplication)、语言检测 (Language Detection)、内容过滤 (Content Filtering)）
- 创建固定长度的训练序列 (Training Sequences)，并正确处理注意力掩码 (Attention Masks) 和文档边界 (Document Boundaries)
- 分析流水线吞吐量 (Throughput)，确保数据加载器 (Dataloader) 能够跟上 GPU 的训练速度

## 问题所在

你已经有了分词器。现在你需要数据。

不是一个小数据集，也不是一个 CSV 文件。而是 TB 级的文本——需要经过清洗、去重、质量过滤、分词为固定长度序列，并以足够快的速度提供随机批次，确保你的 8 卡 GPU 集群永远不会因为等待下一个批次而空闲。

大多数人认为训练大语言模型 (LLM) 的关键在于模型架构。事实并非如此。Llama 3 使用了 15.6 万亿个词元 (Tokens)，GPT-3 使用了 3000 亿个，DeepSeek-V2 使用了 8.1 万亿个。这三者的架构大致相同：都是堆叠的 Transformer 模块，包含注意力层和前馈层。输出质量的巨大差异，绝大多数源于数据。

DeepMind 的 Chinchilla 论文对此给出了精确的量化结论。在固定的算力预算下，模型参数量与训练词元数之间存在一个最优比例。Chinchilla 研究表明，2022 年的大多数模型都存在严重的欠训练 (Undertrained) 问题——相对于它们所接触的数据量，它们的参数量过大了。一个拥有 700 亿参数、在 1.4 万亿词元上训练的模型（符合 Chinchilla 最优比例），其表现超越了在 3000 亿词元上训练的 2800 亿参数模型（Gopher）。

你的数据流水线决定了模型学到的是真正的语言规律，还是无意义的噪声。

## 核心概念

### 数据来源

每个大语言模型（Large Language Model）都是在多种数据源的混合数据上进行训练的。确切的数据构成对大多数实验室来说都是严格保密的，但我们已掌握足够的信息来了解其主要类别。

| 数据源 | 数据量 | 质量 | 使用方/代表模型 |
|--------|------|---------|---------|
| Common Crawl | 约 250 TB 原始数据 | 低（需大量过滤） | GPT-3、Llama 及大多数开源模型 |
| 维基百科（Wikipedia） | 约 20 GB | 高 | 所有主流大语言模型 |
| GitHub 代码库 | 1 TB 以上 | 中（包含大量重复代码和废弃代码） | StarCoder、CodeLlama、DeepSeek-Coder |
| 书籍（BookCorpus、Pile） | 约 100 GB | 高 | GPT-2、GPT-3 及早期模型 |
| 学术论文（arXiv、S2ORC） | 约 100 GB | 理工科（STEM）领域质量高 | Llama、Galactica |
| StackOverflow、Reddit | 约 100 GB | 中 | Llama、Falcon |
| 精选网页数据（C4、RefinedWeb） | 约 5 TB | 中高（已预过滤） | T5、Falcon |

Llama 3 公开了其数据配比：约 50% 的网页数据、25% 的代码、13% 的书籍与学术论文、8% 的数学数据，以及 4% 的多语言网页数据。总计使用了 15.6 万亿个词元（token），源自超过 5 TB 的原始文本。

数据配比与数据总量同等重要。网页数据过多，模型就会变成只会鹦鹉学舌的“Reddit 复读机”。代码数据过少，模型就无法编程。数学数据不足，模型的推理能力就会大打折扣。找到最佳的数据配比是训练大语言模型最困难的环节之一，且没有固定公式可循——必须依靠反复的实验与评估。

### 数据清洗

原始网页数据通常非常“脏乱”。典型的 Common Crawl 数据转储通常包含：

- HTML 标签与 JavaScript 代码
- 样板文本（boilerplate），如页眉、页脚和导航菜单
- 重复页面（完全重复与近似重复）
- 机器生成的垃圾内容
- 个人身份信息（Personally Identifiable Information, PII）
- 低质量文本（关键词堆砌、SEO 垃圾内容）
- 以文本形式编码的非文本内容

数据清洗绝非可选项。它决定了模型是能够生成连贯的段落，还是只会输出夹杂着商品列表的 HTML 标签。

graph TD
    A[Raw Text] --> B[HTML Strip]
    B --> C[Language Detection]
    C --> D[Quality Filter]
    D --> E[Deduplication]
    E --> F[PII Removal]
    F --> G[Clean Text]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff

每个步骤旨在消除特定类型的噪声：

**HTML 剥离（HTML stripping）：** 移除所有标记语言，仅保留可见的文本内容。使用 `trafilatura` 或 `readability` 等库可以提取文章正文，同时过滤掉导航栏、广告和样板文本。

**语言检测（Language detection）：** 使用 fastText 的语言识别模型（`lid.176.bin`）对每篇文档进行分类，并筛选出目标语言。若某文档被判定为英语但置信度低于 0.8，则其很可能并非纯净的英文文本。

**质量过滤（Quality filtering）：** 这一步尤为关键。RefinedWeb（Falcon 模型背后的数据集）采用基于困惑度（perplexity）的过滤策略：先在维基百科上训练一个小型语言模型，然后对每篇文档进行打分。困惑度越高，说明该文档与维基百科的文本风格差异越大——很可能是垃圾内容、关键词列表或机器生成内容。超过设定困惑度阈值的文档将被剔除。

**去重（Deduplication）：** 影响最为显著的清洗步骤。Common Crawl 包含海量重复页面——如法律声明、Cookie 提示和服务条款。在重复数据上训练会浪费算力，并可能导致模型死记硬背，逐字复述特定段落。

**PII 移除（PII removal）：** 针对姓名、电子邮件地址、电话号码、社会安全号码等。结构化 PII 采用基于正则表达式（Regex）的检测方法，上下文中的姓名则使用命名实体识别（Named Entity Recognition, NER）模型进行处理。

### 基于 MinHash 的去重

精确去重很简单：对每篇文档计算哈希值并移除重复项即可。但真正的难题在于近似重复（near-duplicates）。同一篇新闻报道的两个副本，仅因周围广告略有不同，就构成了近似重复。其内容 95% 相同，但逐字节对比却存在差异。

MinHash 结合局部敏感哈希（Locality-Sensitive Hashing, LSH）能高效解决这一问题。

graph LR
    A[Document] --> B[Shingling]
    B --> C[MinHash Signature]
    C --> D[LSH Buckets]
    D --> E[Candidate Pairs]
    E --> F[Jaccard Similarity]
    F --> G[Deduplicated Set]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff

核心思路如下：

1. **分片（Shingling）：** 将每篇文档转换为一组 n-gram（例如 5 个词或字符的序列）。以 3 词分片为例，"the quick brown fox" 会变为 {"the quick brown", "quick brown fox"}。

2. **MinHash：** 针对每篇文档的分片集合，计算 k 个哈希值。每个哈希值是在不同哈希函数下，所有分片哈希值中的最小值。这会生成一个固定大小的“签名（signature）”，用于近似估算任意两篇文档之间的杰卡德相似度（Jaccard similarity）。

3. **LSH：** 根据 MinHash 签名的分带（bands）将文档划分到不同的桶（buckets）中。落入同一桶的文档即为候选近似重复项。这种方法避免了全量两两比对——只需比较候选对即可。

4. **验证（Verify）：** 对每个候选对计算精确的杰卡德相似度。若相似度超过设定阈值（通常为 0.8），则移除其中一份副本。

Llama 团队报告称，通过去重操作移除了约 38% 的网页数据。这绝非一个小数目。Common Crawl 中超过三分之一的内容都是重复或近似重复的。

### 序列打包（Sequence Packing）

模型期望接收固定长度的输入序列，但文档的长度却是可变的。有的仅 50 个词元，有的则长达 50,000 个词元。

朴素做法：将每篇文档填充（padding）至最大序列长度。这会在对学习毫无贡献的填充词元上浪费大量算力。

更优做法：将多篇文档打包进单个序列中，并使用序列结束符（end-of-sequence token）进行分隔。一个 2048 词元的序列可能包含三篇短文档，它们之间通过 `[EOS]` 词元连接。

graph TD
    subgraph Naive Packing
        A1["Doc A (200 tokens)"] --> P1["[PAD] x 1848"]
        A2["Doc B (500 tokens)"] --> P2["[PAD] x 1548"]
        A3["Doc C (100 tokens)"] --> P3["[PAD] x 1948"]
    end

    subgraph Efficient Packing
        B1["Doc A (200) | Doc B (500) | Doc C (100) | Doc D (400) | Doc E (848)"]
    end

    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P1 fill:#333,stroke:#666,color:#999
    style P2 fill:#333,stroke:#666,color:#999
    style P3 fill:#333,stroke:#666,color:#999
    style B1 fill:#1a1a2e,stroke:#16c784,color:#fff

必须正确设置注意力掩码（attention mask）。在同一打包序列中，文档 A 的词元不应关注（attend to）文档 B 的词元。这需要使用块对角注意力掩码（block-diagonal attention mask）。

长文档会在序列边界处被截断或切分为多个块（chunks）。切分点的选择至关重要：在句子中间切分会迫使模型处理不完整的语义。部分数据处理流水线会尽可能将切分点对齐到段落或句子边界。

### Chinchilla 缩放定律（Chinchilla Scaling Law）

在固定的计算预算 C（以浮点运算次数 FLOPs 衡量）下，最优模型参数量 N 与数据集规模 D 遵循以下关系：

N_opt ~ C^0.5
D_opt ~ C^0.5

在实践中，这意味着模型规模与数据集规模应大致按比例同步扩展。参数量增加 10 倍的模型，大约需要 10 倍的训练词元才能达到相同的损失值（loss）。

| 模型 | 参数量 | 训练词元数 | 是否符合 Chinchilla 最优？ |
|-------|-----------|----------------|-------------------|
| GPT-3 | 1750 亿 | 3000 亿 | 否（训练不足 3-4 倍） |
| Chinchilla | 700 亿 | 1.4 万亿 | 是（按设计执行） |
| Llama 2 | 700 亿 | 2 万亿 | 过训练（有意为之） |
| Llama 3 | 700 亿 | 15 万亿 | 严重过训练 |

Llama 3 有意打破了 Chinchilla 定律。Meta 发现，使用远超计算最优比例的数据进行过训练（overtraining），能够产出更适合推理的模型。额外的训练成本只需支付一次，但较小的模型在后续部署服务中将永久保持更低的成本。这种方法有时被称为“推理最优（inference-optimal）”缩放策略，并自 2024 年起已成为行业标准。

## 构建

### 步骤 1：文本清洗

去除 HTML 标签、规范化空白字符，并移除非文本内容。我们将使用公有领域文本（Project Gutenberg）作为我们的小型语料库（corpus）。

import re

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True

质量过滤器（quality filter）能够捕获 SEO 垃圾内容（全大写）、机器生成的噪声（特殊字符比例过高）以及存根页面（内容过短）。仅凭这三项检查，就能从网络爬虫数据中剔除大量垃圾信息。

### 步骤 2：最小哈希（MinHash）去重

从零开始实现最小哈希（MinHash）算法。无需外部库，仅需使用 `hashlib`。

import hashlib
from collections import defaultdict

def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets

def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)

`num_hashes=128` 和 `bands=16` 参数用于控制精确率-召回率权衡（precision-recall tradeoff）。哈希函数数量越多，相似度估计越准确。分桶（bands）数量越多，召回率（recall）越高（能捕获更多重复项），但代价是误报（false positives）也会上升。这些参数值对于典型的网络文本效果良好。

### 步骤 3：分词与序列打包

获取经过清洗和去重的文本，对其进行分词（tokenize），并打包为固定长度的序列（sequence）以供训练使用。

def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens

def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks

### 步骤 4：训练数据加载器

随机生成打包序列的批次（batch）。这正是训练循环（training loop）所消耗的数据格式。

import random

class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks

### 步骤 5：数据集统计

计算关键指标：总词元数（total tokens）、唯一词元数、压缩比（compression ratio）以及文档长度分布。

from collections import Counter

def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / total_tokens

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }
    return stats

压缩比（compression ratio）反映了分词器（tokenizer）在当前语料库上的效率。英文文本通常压缩至每个词元（token）约 3-4 个字符。如果该值约为 1.5，说明你的分词器切分过于激进；如果达到 8 以上，则说明它学习到了非常特定领域的合并规则。

序列利用率（sequence utilization）反映了打包序列中真实数据与填充（padding）内容的比例。若低于 90%，则说明打包效率低下——你正在将计算资源浪费在填充词元上。

## 实践应用

### 与 HuggingFace Datasets 对比

通过 HuggingFace 的 datasets 库加载相同的语料库，并对比流水线 (pipeline) 的处理速度。

from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

import time

start = time.time()
tokenized = ds.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
    batched=True,
    num_proc=4,
)
hf_time = time.time() - start
total_tokens = sum(len(t) for t in tokenized["input_ids"])
print(f"HuggingFace: {total_tokens:,} tokens in {hf_time:.2f}s ({total_tokens/hf_time:,.0f} tokens/sec)")

HuggingFace 的流水线在底层使用了 Rust 编写的分词器 (tokenizer)，并在 4 个核心上进行并行处理。你使用纯 Python 实现的流水线速度会慢 10 到 50 倍。正是这种性能差距，使得生产环境中的团队普遍采用编译型分词器。底层算法是相同的，差异仅在于实现语言。

## 交付上线

本课时将生成一个提示词 (prompt)，用于验证和调试大语言模型 (LLM) 训练流水线中的数据质量。请参阅 `outputs/prompt-data-quality-checker.md`。

## 练习

1. **简单：** 使用简单的启发式方法 (heuristic)（字符集分析）在清洗流水线中添加语言检测功能。仅过滤出英文文档，并统计被移除的文档数量。
2. **中等：** 结合 MinHash 近似去重 (near-deduplication)，使用 SHA-256 哈希值实现精确去重 (exact deduplication)。在网络爬取的语料库上，对比两种方法捕获的重复文档数量。
3. **困难：** 构建一个基于困惑度 (perplexity) 的质量过滤器。在维基百科文本上训练一个小型二元语法 (bigram) 语言模型，使用困惑度对每篇文档进行评分，并剔除排名后 20% 的文档。对比使用过滤后数据与未过滤数据训练时，模型输出质量的差异。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| Common Crawl | “互联网” | 一个每月抓取网页的非营利组织——原始数据约 250TB，是大多数大语言模型（Large Language Model, LLM）训练数据的起点 |
| MinHash | “某种哈希技巧” | 一种使用固定大小签名来估计集合间杰卡德相似度（Jaccard Similarity）的技术——支持大规模近似重复检测 |
| LSH | “局部敏感哈希” | 一种将相似项分组到同一哈希桶（Hash Bucket）中的方法——将成对比较的复杂度从 O(n^2) 降低至接近线性 |
| Sequence packing | “拼接文档” | 将多个文档组合成固定长度的序列并配合正确的注意力掩码（Attention Mask）——消除填充（Padding）带来的计算浪费 |
| Chinchilla scaling | “用更多数据训练” | 在固定计算预算下，要达到最优性能，模型参数量与训练词元（Token）数量需大致按比例同步扩展 |
| Fertility | “每个词的词元数” | 每个单词的平均词元（Token）数量——GPT-4 中英语约为 1.3，非拉丁语系语言则更高 |
| Data mixing | “选择训练数据” | 代码、文本、数学与多语言数据之间的混合比例——没有固定公式，需通过实验确定 |
| Perplexity filter | “质量评分” | 使用小型语言模型对文档进行打分——困惑度（Perplexity）越高，说明文本与干净的参考数据差异越大 |
| Deduplication | “删除副本” | 剔除完全重复和近似重复的文档——通常会移除 30%~40% 的原始网页数据 |
| Attention mask | “关注哪些词元” | 一种二值掩码，用于防止打包序列中跨文档边界的注意力计算 |

## 延伸阅读

- [Hoffmann et al., 2022 -- Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) —— 彻底改变我们对数据规模认知的论文
- [Penedo et al., 2023 -- The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) —— 如何将 Common Crawl 过滤为高质量数据集
- [Touvron et al., 2023 -- Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) —— Llama 2 的数据处理流水线（Data Pipeline）细节
- [Lee et al., 2022 -- Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) —— 为什么数据去重（Deduplication）的重要性远超预期
- [Broder, 1997 -- On the Resemblance and Containment of Documents](https://ieeexplore.ieee.org/document/666900) —— MinHash 算法的原始论文
- [Meta, 2024 -- Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) —— 15.6T 词元（Token）、数据混合比例与过滤流水线
# 词袋模型 (Bag of Words)、TF-IDF 与文本表示

> 先计数，后思考。在 2026 年，针对定义明确的任务，TF-IDF 依然优于嵌入模型 (Embedding Model)。

**类型:** 构建
**语言:** Python
**前置知识:** 第 5 阶段 · 01（文本处理），第 2 阶段 · 02（从零实现线性回归）
**耗时:** 约 75 分钟

## 核心问题

模型需要数值，而你手头只有字符串。

每个自然语言处理 (NLP) 流水线都必须回答同一个问题：如何将长度可变的词元 (Token) 流转换为分类器可处理的固定大小向量。该领域最初找到的答案是最简单粗暴却行之有效的方法：统计词频，构建向量。

在工业级 NLP 应用中，这种向量承载的落地场景比任何嵌入模型都要多。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 出现之前）、第一波情感分析，以及学术界 NLP 基准测试的第一个十年。在 2026 年，面对细分分类任务时，从业者依然会首选它。它速度快、可解释性强，并且在那些“词语是否出现”起决定性作用的任务上，其表现往往与拥有 4 亿参数的嵌入模型难分伯仲。

本节将从零开始实现词袋模型，接着实现 TF-IDF。随后展示如何用 scikit-learn 仅用三行代码完成相同操作。最后，我们将指出导致你不得不转向嵌入模型的失效场景。

## 核心概念

**词袋模型 (Bag of Words, BoW)** 会忽略词序。对于每篇文档，统计词汇表中每个词出现的次数。向量长度等于词汇表大小，位置 `i` 的值即为词 `i` 的出现次数。

**TF-IDF** 对 BoW 进行重新加权。如果一个词在所有文档中都出现，它提供的信息量极低，因此需要降低其权重。如果一个词在整个语料库中罕见，但在某篇文档中频繁出现，则说明它包含重要信号，因此需要提高其权重。

TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))

其中，`TF` 表示词在文档中的词频 (Term Frequency)，`df` 表示文档频率 (Document Frequency，即包含该词的文档数量)，`N` 为文档总数。取对数 (`log`) 是为了让那些无处不在的常见词的权重保持在合理范围内。

关键特性：两者都会生成具有可解释维度的稀疏向量 (Sparse Vector)。你可以直接查看训练好的分类器权重，从而判断是哪些词将文档推向特定类别。而使用 768 维的 BERT 嵌入 (BERT Embedding) 则无法做到这一点。

## 动手实现

### 步骤 1：构建词汇表 (Vocabulary)

def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

输入：已分词 (Tokenized) 的文档列表（任何词级分词器 (Word-level tokenizer) 均可；本课程的 `code/main.py` 使用了简化的小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着词汇索引 0 对应第一个文档中首次出现的词。不同框架的惯例有所不同；例如 scikit-learn 会按字母顺序排序。

### 步骤 2：词袋模型 (Bag of Words)

def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix

>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]

行对应文档，列对应词汇表索引。矩阵元素 `[i][j]` 表示“词 `j` 在文档 `i` 中的出现次数”。文档 1 中 `cat` 出现了两次，因为原文确实如此；文档 0 中 `ran` 出现了零次，因为原文并未出现该词。

### 步骤 3：词频 (Term Frequency) 与文档频率 (Document Frequency)

import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]

这里有两个值得提及的平滑 (Smoothing) 技巧。`(n+1)/(d+1)` 的写法避免了 `log(x/0)` 的除零错误。末尾的 `+1` 确保了即使某个词出现在所有文档中，其逆文档频率 (IDF) 仍为 1（而非 0），这与 scikit-learn 的默认行为一致。其他实现可能直接使用原始的 `log(N/df)`。两种方法均可行，但平滑版本在数值计算上更稳定友好。

### 步骤 4：TF-IDF (Term Frequency-Inverse Document Frequency)

def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out

>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)

共三个文档，五个词汇表单词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有三个文档中，因此其 IDF 值较低。`dog` 仅出现在一个文档中，因此其 IDF 值较高。生成的向量是稀疏的 (Sparse)（大多数元素值较小），而具有区分度 (Discriminative) 的词汇则会凸显出来。

### 步骤 5：对行进行 L2 归一化 (L2 Normalization)

def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out

若不进行归一化，较长的文档会生成模长更大的向量，从而在相似度计算中占据主导地位。L2 归一化会将每个文档向量映射到单位超球面 (Unit Hypersphere) 上。此时，行向量之间的余弦相似度 (Cosine Similarity) 计算就简化为点积 (Dot Product) 运算。

## Use It

`scikit-learn` 提供了生产环境版本。

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))

`CountVectorizer` 在一次调用中即可完成分词（tokenization）、构建词表（vocabulary）和生成词袋模型（Bag-of-Words, BoW）。`TfidfVectorizer` 在此基础上增加了逆文档频率（Inverse Document Frequency, IDF）加权与 L2 归一化（L2 normalization）。两者均返回稀疏矩阵（sparse matrix）。处理 10 万份文档时，稠密表示（dense representation）会耗尽内存；在分类器（classifier）强制要求稠密输入前，请始终保持稀疏格式。

决定模型表现的关键参数：

| 参数 | 作用 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含二元语法（bigram）。通常能提升分类效果。 |
| `min_df=2` | 剔除在少于 2 篇文档中出现的词。可在噪声数据中精简词表。 |
| `max_df=0.95` | 剔除在超过 95% 文档中出现的词。无需硬编码停用词表即可近似实现停用词（stopword）过滤。 |
| `stop_words="english"` | 使用 `scikit-learn` 内置的英文停用词表。效果因任务而异——情感分析任务中*不应*剔除否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 替代原始词频（`tf`）。当某个词在单篇文档中频繁重复时效果显著。 |

### TF-IDF 依然胜出的场景（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。此时词项是否出现才是关键，语义细微差别并不重要。
- 低数据量场景（仅数百个标注样本）。TF-IDF 结合逻辑回归（logistic regression）无需预训练成本。
- 对延迟敏感的任何场景。TF-IDF 结合线性模型可在微秒级返回结果。而通过 Transformer 对文档进行嵌入（embedding）则需要 10-100 毫秒。
- 必须提供预测解释的系统。直接检查分类器的权重系数即可。排名靠前的正向词即为判断依据。

### TF-IDF 失效的场景

语义盲区（semantic blindness）缺陷。请考虑以下两篇文档：

- “这部电影一点也不好。”
- “这部电影非常棒。”

前者是负面评价，后者是正面评价。它们的 TF-IDF 重叠词恰好只有 `{the, movie, was}`。词袋模型分类器必须死记硬背：当 `not` 出现在 `good` 附近时，标签会发生翻转。只要有足够的数据，它确实能学会这一点，但永远无法像理解句法（syntax）的模型那样优雅高效。

另一大缺陷：推理阶段的未登录词（out-of-vocabulary, OOV）问题。在 IMDb 影评上训练的 BoW 模型，如果训练集中从未出现过 `Zoomer-approved` 这个词元（token），它将完全不知所措。子词嵌入（subword embedding，见第 04 课）可以解决此问题，而 TF-IDF 则无能为力。

### 混合架构：TF-IDF 加权嵌入

2026 年中等数据量分类任务的务实默认方案：将 TF-IDF 权重作为注意力机制（attention）应用于词嵌入（word embedding）。

def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]

该方案既能从嵌入中获取语义表征能力，又能借助 TF-IDF 突出罕见词的重要性。分类器将基于池化向量（pooled vector）进行训练。在标注样本少于约 5 万条的情感分析、主题分类和意图识别任务中，该方法的性能优于单独使用其中任何一种技术。

## 交付上线

保存为 `outputs/prompt-vectorization-picker.md`：

---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.

## 练习

1. **简单。** 在 L2 归一化（L2-normalized）的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档的得分为 1.0，词汇完全不重叠的文档得分为 0.0。
2. **中等。** 为 `bag_of_words` 添加 n 元语法（n-gram）支持。参数 `n` 用于生成 n 元语法的计数。测试在 `["the", "cat", "sat"]` 上设置 `n=2` 时，是否能正确生成 `["the cat", "cat sat"]` 的二元语法（bigram）计数。
3. **困难。** 使用 GloVe 100d 向量（下载一次并缓存）构建上述的 TF-IDF 加权嵌入（TF-IDF-weighted-embedding）混合模型。在 20 Newsgroups 数据集上，将其分类准确率与纯 TF-IDF 和纯均值池化嵌入（mean-pooled embeddings）进行对比。报告各自在哪些场景下表现更优。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 词袋模型（BoW） | 词频向量 | 统计单个文档中词汇的出现次数。忽略词序。 |
| 词频（TF） | 词频 | 文档中某个词的出现次数，可选择按文档长度进行归一化。 |
| 文档频率（DF） | 文档频率 | 包含该词至少一次的文档数量。 |
| 逆文档频率（IDF） | 逆文档频率 | 经过平滑处理的 `log(N / df)`。降低在大量文档中普遍出现的词的权重。 |
| 稀疏向量（Sparse vector） | 大部分为零 | 词表通常包含 1 万到 10 万个词；对于任意给定文档，其中绝大多数词都不会出现。 |
| 余弦相似度（Cosine similarity） | 向量夹角 | L2 归一化向量的点积。1 表示完全相同，0 表示正交（无关）。 |

## 延伸阅读

- [scikit-learn — 文本特征提取 (Text Feature Extraction)](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— 权威的 API 参考文档，并附有每个可调参数的详细说明。
- [Salton, G., & Buckley, C. (1988). 自动文本检索中的词项加权方法](https://www.sciencedirect.com/science/article/pii/0306457388900210) —— 这篇论文确立了词频-逆文档频率 (TF-IDF) 长达十年的默认标准地位。
- [“为何 TF-IDF 依然优于嵌入 (Embeddings)” — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) —— 2026 年的最新观点，探讨了传统方法在何种场景下依然胜出及其原因。
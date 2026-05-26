# 词嵌入（Word Embeddings）—— 从零实现 Word2Vec

> 观其伴，知其义。基于这一理念训练浅层网络，几何结构便会自然涌现。

**类型：** 实战构建
**语言：** Python
**前置知识：** 第 5 阶段 · 02（词袋模型（BoW）与 TF-IDF）、第 3 阶段 · 03（从零实现反向传播（Backpropagation））
**预计耗时：** 约 75 分钟

## 核心问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词，但它不知道它们的含义几乎相同。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论中。你可以通过列举同义词来勉强掩盖这一缺陷，但这种方法在面对罕见词、领域术语以及你未曾预料到的语言时就会失效。

你希望得到一种表示方法，让 `dog` 和 `puppy` 在空间中彼此靠近。让 `king - man + woman` 的计算结果落在 `queen` 附近。让在 `dog` 上训练的模型能够“免费”将部分信号迁移到 `puppy` 上。

Word2Vec 为我们提供了这样的空间。一个两层神经网络，在万亿级词元（token）上进行训练，于 2013 年发表。其架构极其简单，但其成果却重塑了自然语言处理（Natural Language Processing, NLP）领域长达十年。

## 核心概念

**分布式假设（Distributional Hypothesis）**（Firth, 1957）：“观其伴，知其义。”如果两个词出现在相似的上下文中，它们的含义很可能也相近。

Word2Vec 有两种变体，均基于这一思想。

- **跳字模型（Skip-gram）**。给定中心词，预测其周围的词。例如窗口大小为 2 时：`cat -> (the, sat, on)`。
- **连续词袋模型（Continuous Bag of Words, CBOW）**。给定周围的词，预测中心词。例如：`(the, sat, on) -> cat`。

跳字模型（Skip-gram）的训练速度较慢，但对罕见词的处理效果更好。它也因此成为了默认选择。

该网络包含一个隐藏层，且不使用非线性激活函数。输入是词汇表上的独热向量（one-hot vector）。输出是词汇表上的 Softmax 分布。训练完成后，你可以丢弃输出层。隐藏层的权重即为词嵌入（embeddings）。

one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding

关键技巧在于：对 10 万个词进行 Softmax 计算的成本极其高昂。Word2Vec 采用**负采样（Negative Sampling）**将其转化为二分类任务。即预测“该上下文词是否出现在该中心词附近（是或否）”。对于每个训练样本对，仅采样少量负例（未共同出现的词），而无需在整个词汇表上计算 Softmax。

## 动手实现

### 步骤 1：从语料库 (corpus) 中构建训练对 (training pairs)

def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs

>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]

窗口内的每个（中心词 (center word)，上下文词 (context word)）对都是一个正样本 (positive training example)。

### 步骤 2：嵌入表 (embedding tables)

两个矩阵。`W` 是中心词嵌入表（即最终保留的表）。`W'` 是上下文词表（通常会被丢弃，有时也会与 `W` 取平均）。

import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime

采用小幅度的随机初始化 (random initialization)。词表大小 (vocabulary size) 设为 10k、维度 (dimension) 设为 100 是实际应用中常见的配置；在教学演示中，50 个词 × 16 维足以观察到向量空间的几何结构。

### 步骤 3：负采样目标 (negative sampling objective)

对于每个正样本对 `(center, context)`，从词表中随机采样 `k` 个词作为负样本 (negative samples)。训练模型的目标是：使正样本的点积 `W[center] · W'[context]` 尽可能高，而负样本的点积尽可能低。

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center

核心公式：正样本对的逻辑损失 (logistic loss)（希望 sigmoid 输出接近 1）加上负样本对的逻辑损失（希望 sigmoid 输出接近 0）。梯度会同时更新两张表。完整的数学推导见原始论文；如果想彻底掌握，建议用纸笔手动推导一遍。

### 步骤 4：在玩具语料库 (toy corpus) 上进行训练

def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W

在大规模语料库上经过足够多的训练轮次 (epochs) 后，共享相似上下文的词会拥有相似的中心词嵌入 (center embeddings)。在玩具语料库上，这种效果较为微弱；而在数十亿词元 (tokens) 的数据上训练时，效果会非常显著。

### 步骤 5：类比推理技巧 (analogy trick)

def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})

在预训练的 300 维 Google News 词向量 (pre-trained vectors) 上：

>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]

`king - man + woman = queen`。这并不是因为模型理解“王室”的概念，而是因为向量 `(king - man)` 捕捉到了类似“王室/君主”的语义特征，将其加到 `woman` 上后，结果会落在表示“王室女性”的向量区域附近。

## 上手使用

从零开始编写词向量模型（Word2Vec）是为了教学目的。在生产环境的自然语言处理（Natural Language Processing, NLP）中，通常直接使用 `gensim`。

from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))

在实际工作中，你几乎不需要自己训练 Word2Vec。直接下载预训练词向量（Pre-trained Vectors）即可。

- **GloVe** —— 斯坦福大学基于共现矩阵分解（Co-occurrence Matrix Factorization）的方法。提供 50 维、100 维、200 维和 300 维的检查点（Checkpoints）。通用覆盖范围良好。第 04 课专门讲解 GloVe。
- **fastText** —— Facebook 对 Word2Vec 的扩展，嵌入了字符级 n-gram（Character N-grams）。通过组合子词（Subwords）来处理未登录词（Out-of-Vocabulary Words）。见第 04 课。
- **Google News 预训练 Word2Vec** —— 300 维，包含 300 万词汇量，发布于 2013 年。至今仍有大量每日下载量。

### 2026 年 Word2Vec 依然胜出的场景

- 轻量级领域特定检索（Domain-specific Retrieval）。在笔记本电脑上花一小时训练医学摘要，即可获得通用模型无法捕捉的专用向量。
- 类比式特征工程（Feature Engineering）。`gender_vector = mean(man - woman pairs)`。将其从其他词向量中减去，即可得到性别中立轴。该方法仍用于公平性研究。
- 可解释性。100 维的向量足够小，可以通过主成分分析（Principal Component Analysis, PCA）或 t-SNE 进行可视化，并能直观地看到聚类形成。
- 任何需要在无 GPU 的设备端（On-device）进行推理（Inference）的场景。Word2Vec 的查找仅需单次行数据读取。

### Word2Vec 的局限性

一词多义（Polysemy）瓶颈。`bank` 只有一个向量。`river bank`（河岸）和 `financial bank`（银行）共享该向量。`table`（表格与家具）也共享同一向量。下游分类器无法从该向量中区分不同的词义。

上下文嵌入（Contextual Embeddings，如 ELMo、BERT 及之后的所有 Transformer 架构）通过根据周围上下文为每个词的出现生成不同的向量，解决了这一问题。这就是从 Word2Vec 到 BERT 的跨越：从静态（Static）到上下文感知（Contextual）。第 7 阶段将涵盖 Transformer 部分。

未登录词问题（Out-of-Vocabulary Problem）是另一个缺陷。如果 `Zoomer-approved` 未出现在训练数据中，Word2Vec 就从未见过它，且没有任何回退机制（Fallback）。fastText 通过子词组合（Subword Composition）解决了这一问题（见第 04 课）。

## 部署上线

保存为 `outputs/skill-embedding-probe.md`：

---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.

你通过探测（probe）已训练的词嵌入（word embeddings）来验证其是否正常工作。给定一个 `gensim.models.KeyedVectors` 对象和一个词汇表（vocabulary），请运行以下操作：

1. 三项经典类比测试（analogy tests）。`king : man :: queen : woman`、`paris : france :: tokyo : japan`、`walking : walked :: swimming : ?`。报告排名第一（top-1）的结果及其余弦相似度（cosine）。
2. 针对用户提供的领域特定词汇（domain-specific words）进行五项最近邻测试（nearest-neighbor tests）。打印余弦值最高的前 5 个邻居词。
3. 一项对称性检查（symmetry check）。验证 `similarity(a, b) == similarity(b, a)` 是否在浮点数精度（float precision）范围内成立。
4. 一项退化检查（degenerate check）。如果任何嵌入向量的范数（norm）低于 0.01 或高于 100，则说明模型存在训练缺陷（training bug）。请将其标记出来。

切勿仅凭类比准确率（analogy accuracy）就断言模型表现良好。类比基准测试（analogy benchmarks）容易被针对性优化（gameable），且无法有效迁移至下游任务（downstream tasks）。建议结合内在评估（intrinsic evaluation）与下游评估（downstream evaluation）进行综合判断。

## 练习

1. **简单。** 在微型语料库（tiny corpus，包含 20 句关于猫和狗的句子）上运行训练循环（training loop）。经过 200 个训练轮次（epochs）后，验证 `nearest(vocab, W, W[vocab["cat"]])` 返回的结果中前 3 名是否包含 `dog`。如果没有，请增加训练轮次或扩大词汇表。
2. **中等。** 添加高频词下采样（subsampling of frequent words）。对于频率高于 `10^-5` 的词汇，按与其频率成正比的概率从训练对中剔除。测量其对稀有词相似度（rare-word similarity）的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练模型。计算两个偏见轴（bias axes）：`he - she` 和 `doctor - nurse`。将职业词汇投影到这两个轴上。报告哪些职业的偏见差距（bias gap）最大。这正是公平性研究人员（fairness researchers）常用的探测方法。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 词嵌入 (Word embedding) | 将词表示为向量 | 从上下文中学习到的稠密低维（通常为 100-300 维）表示。 |
| Skip-gram | Word2Vec 的技巧 | 根据中心词预测上下文词。比 CBOW 慢，但对稀有词效果更好。 |
| 负采样 (Negative sampling) | 训练捷径 | 用针对 `k` 个随机词的二分类任务，替换全词汇表上的 softmax 计算。 |
| 静态嵌入 (Static embedding) | 每个词一个向量 | 无论上下文如何，向量保持不变。无法处理一词多义（polysemy）。 |
| 上下文嵌入 (Contextual embedding) | 上下文敏感向量 | 根据周围词汇为每次出现生成不同的向量。Transformer 模型输出的正是此类向量。 |
| OOV | 词汇表外词 | 训练时未见过的词。Word2Vec 无法为这些词生成向量。 |

## 延伸阅读

- [Mikolov 等人 (2013). 词与短语的分布式表示及其组合性](https://arxiv.org/abs/1310.4546) —— 提出负采样（negative sampling）方法的论文。篇幅简短，易于阅读。
- [Rong, X. (2014). word2vec 参数学习详解](https://arxiv.org/abs/1411.2738) —— 梯度（gradient）推导最为清晰，若你觉得原论文的数学内容过于晦涩，这篇是极佳的选择。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html) —— 真正适用于生产环境（production environment）的训练参数配置。
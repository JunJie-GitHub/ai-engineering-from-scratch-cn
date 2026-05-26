# GloVe、FastText 与子词嵌入（Subword Embeddings）

> Word2Vec 为每个词训练一个嵌入向量。GloVe 对共现矩阵（co-occurrence matrix）进行分解。FastText 对词的组成部分进行嵌入。BPE 则架起了通往 Transformer 的桥梁。

**Type:** 实战构建
**Languages:** Python
**Prerequisites:** 第 5 阶段 · 03（从零实现 Word2Vec）
**Time:** 约 45 分钟

## 核心问题

Word2Vec 留下了两个开放性问题。

首先，存在一条并行的研究路线，它直接对共现矩阵进行分解（如 LSA、HAL），而不是进行在线的 Skip-gram 更新。Word2Vec 的迭代方法在本质上更优吗？还是说这种差异仅仅是两种方法处理词频计数方式不同所导致的假象？**GloVe** 给出了答案：采用精心设计的损失函数进行矩阵分解，其效果不逊于甚至优于 Word2Vec，且训练成本更低。

其次，这两种方法都无法处理从未见过的词。`Zoomer-approved`、`dogecoin`、上周刚造出的任何专有名词，以及罕见词根的所有屈折变化形式。**FastText** 通过嵌入字符级 n-gram（character n-gram）解决了这一问题：一个词是其各个组成部分（包括词素）的向量之和，因此即使是词表外（out-of-vocabulary）的词也能获得合理的向量表示。

第三，随着 Transformer 的出现，问题再次发生了转变。词级词表的容量上限通常在一百万左右；而真实语言远比这更加开放。**字节对编码（Byte-pair encoding, BPE）** 及其衍生算法通过学习一个覆盖所有内容的常用子词单元词表，解决了这一难题。如今所有现代大语言模型（Large Language Model, LLM）使用的分词器（tokenizer）均为子词分词器。

本节将逐一讲解这三种方法，并说明在不同场景下该如何选择。

## 核心概念

**GloVe（全局向量，Global Vectors）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 出现在词 `i` 上下文中的频次。训练向量，使其满足 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失函数进行加权，以避免高频词对占据主导地位。完成。

**FastText。** 一个词的向量等于其所有字符级 n-gram 向量与该词本身向量之和。`where` 会被拆解为 `<wh, whe, her, ere, re>` 以及 `<where>`。该词的最终向量即为这些组成部分向量的总和。训练方式与 Word2Vec 相同。优势：未见过的词（如 `whereupon`）可由已知的 n-gram 组合而成。

**BPE（字节对编码，Byte-Pair Encoding）。** 从由单个字节（或字符）组成的初始词表开始。统计语料库中所有相邻的字符对。将频次最高的字符合并为一个新词元（token）。重复此过程 `k` 次迭代。结果：得到一个包含 `k + 256` 个词元的词表，其中高频序列（如 `ing`、`tion`、`the`）成为独立词元，而罕见词则被拆分为熟悉的片段。任何句子都能被成功分词。

## 动手实践

### GloVe：分解共现矩阵 (Co-occurrence Matrix)

import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde

有两个关键机制值得特别说明。权重函数 `f(x) = (x/x_max)^alpha` 会降低极高频词对（如 `(the, and)`）的权重，防止它们在损失函数中占据主导地位。最终的嵌入 (Embedding) 是中心词表 `W` 与上下文词表 `W_tilde` 之和。将两者相加是一种文献中提出的技巧，其表现通常优于仅使用单一词表。

### FastText：感知子词的嵌入 (Subword-aware Embeddings)

def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams

>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}

每个词由其 n-gram (N-gram) 集合表示（通常包含 3 到 6 个字符）。该词的嵌入是其所有 n-gram 嵌入的总和。在进行跳字模型 (Skip-gram) 训练时，可直接用此方法替换 Word2Vec 中原本使用的单一向量。

def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)

对于未登录词 (Out-of-Vocabulary, OOV)，只要其包含的部分 n-gram 已知，依然能生成对应的向量。例如，`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where` 等子词片段，因此它们在向量空间中的位置会彼此靠近。

### BPE (Byte Pair Encoding)：学习子词词表

def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']

首次迭代会合并出现频率最高的相邻字符对。经过足够多次迭代后，高频子串（如 `low`、`est`、`tion`）会合并为独立的词元 (Token)，而罕见词则会被清晰、合理地切分。

在实际应用中，GPT、BERT 和 T5 等模型的分词器 (Tokenizer) 通常会学习 3 万到 10 万次合并操作。其结果是：任何文本都能被切分为长度有限且由已知 ID 组成的序列，从而彻底消除未登录词 (OOV) 问题。

## 实际应用

在实际工程中，你很少需要自己从头训练这些模型。通常直接加载预训练检查点（pre-trained checkpoints）。

import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)

在 Transformer 时代，针对 BPE（Byte Pair Encoding，字节对编码）风格的子词分词（subword tokenization）：

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))

['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']

`Ġ` 前缀用于标记词边界（这是 GPT-2 的惯例）。现代的分词器（tokenizer）无一例外都是 BPE 的变体、WordPiece（用于 BERT）或 SentencePiece（用于 T5、LLaMA）。

### 如何选择

| 场景 | 推荐方案 |
|-----------|------|
| 预训练的通用词向量（word vectors），无需处理未登录词（out-of-vocabulary, OOV） | GloVe 300d |
| 预训练的通用词向量，必须处理拼写错误/新词/形态丰富的语言 | FastText |
| 任何输入到 Transformer（训练或推理）的内容 | 使用模型自带的分词器。切勿替换。 |
| 从零开始训练自己的语言模型 | 先在你的语料库上训练 BPE 或 SentencePiece 分词器 |
| 使用线性模型进行生产环境的文本分类 | 依然使用 TF-IDF。参见第 02 课。 |

## 交付使用

保存为 `outputs/skill-tokenizer-picker.md`：

---
name: tokenizer-picker
description: 为新的语言模型或文本处理流水线选择分词方案。
version: 1.0.0
phase: 5
lesson: 04
tags: [自然语言处理, 分词, 词嵌入]
---

给定任务和数据集描述后，你需要输出：

1. 分词策略（词级别、BPE、WordPiece、SentencePiece、字节级别）。附一句理由。
2. 词表大小（vocabulary size）目标（例如：纯英语语言模型设为 32k，多语言模型设为 64k-100k）。
3. 包含确切训练命令的库调用。指明库名称。引用参数。
4. 一个可复现性陷阱。分词器与模型不匹配是生产环境中最常见的隐蔽 Bug；明确指出必须配套使用的组合。

当用户微调预训练大语言模型（large language model, LLM）时，拒绝推荐训练自定义分词器。对于面向生产推理的任何模型，拒绝推荐词级别分词。将非英语/多文字语料库标记为需要使用带字节回退（byte fallback）的 SentencePiece。

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算这两个 n-gram（n-gram）集合的 Jaccard 重叠度（Jaccard overlap）。你应该会观察到大量共享的片段（`pla`、`lay`、`play`），这也是 FastText（FastText）能够在不同形态变体（morphological variants）之间实现良好迁移的原因。
2. **中等。** 扩展 `learn_bpe` 以跟踪词表（vocabulary）的增长。绘制“每语料字符对应的 token 数（tokens-per-corpus-character）”随合并次数（merges）变化的曲线。你应该会看到初期压缩速度很快，随后逐渐趋于平稳，最终渐近于每个词元（token）约 2-3 个字符的水平。
3. **困难。** 在莎士比亚全集中训练一个包含 1000 次合并的字节对编码（Byte-Pair Encoding, BPE）模型。对比常见词与罕见专有名词的分词（tokenization）结果。测量分词前后每个单词的平均词元数量。写下令你意外的发现。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 共现矩阵（Co-occurrence matrix） | 词-词频率表 | `X[i][j]` = 词 `j` 出现在词 `i` 上下文窗口中的频次。 |
| 子词（Subword） | 词的片段 | 字符 n-gram（FastText）或学习得到的词元（BPE/WordPiece/SentencePiece）。 |
| BPE（Byte-Pair Encoding） | 字节对编码 | 迭代合并出现频率最高的相邻字符对，直至词表达到目标规模。 |
| OOV（Out of Vocabulary） | 未登录词 | 模型从未见过的词。Word2Vec/GloVe 无法处理，而 FastText 和 BPE 可以应对。 |
| 字节级 BPE（Byte-level BPE） | 基于原始字节的 BPE | GPT-2 采用的方案。词表初始包含 256 个字节，因此永远不会出现 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，仅七页，至今仍是推导损失函数（loss）最清晰的文献。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText 论文。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代自然语言处理（Natural Language Processing, NLP）的开创性论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — 详解 BPE、WordPiece 和 SentencePiece 在实际应用中的具体差异。
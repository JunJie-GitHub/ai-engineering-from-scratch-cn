# Transformer 出现之前的文本生成——N-gram 语言模型 (N-gram Language Model)

> 如果一个词的出现令人意外，说明模型效果不佳。困惑度 (Perplexity) 将这种意外程度量化为数值。平滑处理 (Smoothing) 则确保该数值保持有限。

**类型：** 构建
**语言：** Python
**前置知识：** 第 5 阶段 · 01（文本处理），第 2 阶段 · 14（朴素贝叶斯）
**耗时：** 约 45 分钟

## 问题描述

在 Transformer (Transformer)、循环神经网络 (RNN) 和词嵌入 (Word Embedding) 问世之前，语言模型 (Language Model) 通过统计前 `n-1` 个词之后紧跟某个词的频次来预测下一个词。例如，统计到 "the cat" 后接 "sat" 出现 47 次，后接 "jumped" 出现 12 次，后接 "refrigerator" 出现 0 次。随后进行归一化处理，即可得到概率分布 (Probability Distribution)。

这就是 N-gram 语言模型。从 1980 年到 2015 年，它驱动了所有的语音识别器、拼写检查器以及基于短语的机器翻译系统。如今，当你需要低成本的端侧语言建模时，它依然发挥着作用。

真正棘手的问题在于如何处理未观测到的 N-gram (Unseen N-gram)。基于原始计数的模型会为任何未出现过的序列分配零概率，这在长句中是灾难性的，因为几乎每个长句都至少包含一个未见过的词序列。五十年的平滑处理研究解决了这一难题。克内泽-内平滑 (Kneser-Ney Smoothing) 便是其集大成者，而现代深度学习 (Deep Learning) 也继承了这一实证传统。

## 核心概念

![N-gram 模型（N-gram model）：计数、平滑、生成](../assets/ngram.svg)

**N-gram 概率（N-gram probability）：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n` 的值（三元组通常为 3，四元组通常为 4）。通过计数计算：

P(w | context) = count(context, w) / count(context)

**零计数问题（Zero-count problem）。** 任何在训练集中未出现过的 n-gram 都会被赋予零概率。2007 年一项针对布朗语料库（Brown corpus）的研究发现，即使是四元组模型，其预留测试集（held-out test set）中也有 30% 的四元组在训练时未曾见过。如果不进行平滑处理，你将无法在任何真实文本上进行评估。

**平滑方法（Smoothing approaches），按复杂程度递增排列：**

1. **拉普拉斯平滑（Laplace/add-one）。** 为每个计数加 1。方法简单，但对罕见事件的处理效果极差。
2. **古德-图灵估计（Good-Turing）。** 基于频率的频率（frequency-of-frequencies），将高频事件的概率质量（probability mass）重新分配给未见事件。
3. **插值法（Interpolation）。** 使用可调权重将 n-gram、(n-1)-gram 等模型的估计值进行组合。
4. **回退法（Backoff）。** 如果 n-gram 的计数为零，则回退到 (n-1)-gram。Katz 回退法对此进行了归一化处理。
5. **绝对折扣法（Absolute discounting）。** 从所有计数中减去一个固定的折扣值 `D`，并将这部分概率重新分配给未见事件。
6. **克内泽-内平滑（Kneser-Ney）。** 在绝对折扣法的基础上，为低阶模型做出了巧妙选择：使用*延续概率（continuation probability）*（即一个词出现在多少种不同上下文中）而非原始频率。

克内泽-内平滑的洞察十分深刻。“San Francisco”是一个常见的二元组。一元词“Francisco”大多出现在“San”之后。朴素的绝对折扣法会赋予“Francisco”较高的一元概率（因为其总计数很高）。而克内泽-内平滑注意到“Francisco”仅出现在一种上下文中，因此相应地降低了它的延续概率。结果是：以“Francisco”结尾的陌生二元组会被赋予合理的低概率。

**评估指标：困惑度（Perplexity）。** 它是预留测试集上每个词的平均负对数似然值的指数。数值越低越好。困惑度为 100 意味着模型的困惑程度相当于从 100 个词中均匀随机选择一个。

perplexity = exp(- (1/N) * Σ log P(w_i | context_i))

## 动手构建

### 步骤 1：三元语法（trigram）计数

from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]

输入为分词（tokenized）后的句子列表。输出为 n元语法（n-gram）计数和上下文（context）计数。`<s>` 和 `</s>` 表示句子边界。

### 步骤 2：拉普拉斯平滑（Laplace smoothing）

def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator

为每个计数加 1。该方法能实现平滑，但会将过多的概率质量（probability mass）分配给未见事件（unseen events），从而损害已知但罕见事件（rare-known events）的概率估计。

### 步骤 3：Kneser-Ney 平滑（二元语法 bigram 插值版）

def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob

包含三个核心部分。`continuation_prob` 用于捕捉“该词出现在多少种不同的上下文中？”（这是 Kneser-Ney 平滑的创新之处）。`lambda_prev` 表示通过折扣（discount）释放出的概率质量，用于对回退（backoff）项进行加权。最终概率由折扣后的主项与加权后的延续项相加得出。

### 步骤 4：通过采样生成文本

import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens

按照概率比例进行采样（sampling）。每次使用不同的随机种子（seed）都会产生不同的输出。若需要类似束搜索（beam search）的输出效果，可在每一步选择概率最大的词（贪心策略/greedy），并引入一个微小的随机性调节参数（温度/temperature）。

### 步骤 5：困惑度（perplexity）

import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)

数值越低越好。在布朗语料库（Brown corpus）上，经过精细调优的 4-gram Kneser-Ney 模型困惑度约为 140。而在相同的测试集上，Transformer 语言模型（Transformer LM）的困惑度仅为 15-30。两者差距约为 10 倍。正是这一巨大差距促使该领域转向了新的技术路线。

## 使用场景

- **经典自然语言处理（Natural Language Processing, NLP）教学。** 这是理解平滑（Smoothing）、最大似然估计（Maximum Likelihood Estimation, MLE）和困惑度（Perplexity）最清晰的途径。
- **KenLM。** 生产级 n-gram 语言模型库。在对延迟要求严格的语音识别和机器翻译（Machine Translation, MT）系统中常用作重打分器（Rescorer）。
- **设备端自动补全。** 键盘中的三元语法（Trigram）模型。至今仍在广泛使用。
- **基线模型（Baseline）。** 在宣称你的神经语言模型（Neural Language Model, Neural LM）表现优异之前，务必先计算 n-gram 语言模型的困惑度。如果你的 Transformer 模型没有大幅超越 Kneser-Ney（KN）平滑算法，那说明模型存在问题。

## 交付指南

保存为 `outputs/prompt-lm-baseline.md`：

---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.

## 练习

1. **简单。** 在包含 1000 句的莎士比亚语料库上训练一个三元语言模型。生成 20 个句子。这些句子在局部看起来合理，但全局缺乏连贯性。这是该领域的经典演示。
2. **中等。** 在莎士比亚语料库的预留测试集（Held-out set）上为你的 KN 模型实现困惑度计算。与拉普拉斯平滑（Laplace Smoothing）进行对比。你应该会观察到 KN 的困惑度降低了 30% 到 50%。
3. **困难。** 构建一个三元拼写纠错器：给定一个拼写错误的单词及其上下文，生成候选纠正词，并根据语言模型下的上下文概率进行排序。在公开的 Birkbeck 拼写语料库上进行评估。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| N元语法（N-gram） | 词序列 | 连续 `n` 个标记（Token）的序列。 |
| 平滑（Smoothing） | 避免零概率 | 重新分配概率质量，使未出现的事件获得非零概率。 |
| 困惑度（Perplexity） | 语言模型质量指标 | 在预留数据上计算的 `exp(-平均对数概率)`。数值越低越好。 |
| 回退（Backoff） | 回退到较短上下文 | 如果三元语法计数为零，则使用二元语法。Katz 回退算法对此进行了形式化。 |
| Kneser-Ney 平滑（KN） | N-gram 的最佳平滑算法 | 绝对折扣法 + 低阶模型的延续概率（Continuation Probability）。 |
| 延续概率（Continuation Probability） | KN 算法特有 | 根据词 `w` 出现的上下文数量进行加权的 `P(w)`，而非原始词频。 |

## 延伸阅读

- [Jurafsky 与 Martin ——《语音与语言处理》第 3 章（2026 年草案）](https://web.stanford.edu/~jurafsky/slp3/3.pdf) —— 关于 n-gram 语言模型（n-gram Language Models）与平滑技术（Smoothing）的权威参考。
- [Chen 与 Goodman（1998 年）.《语言建模平滑技术的实证研究》](https://dash.harvard.edu/handle/1/25104739) —— 该论文确立了 Kneser-Ney 平滑法（Kneser-Ney Smoothing）为最佳的 n-gram 平滑器（n-gram Smoother）。
- [Kneser 与 Ney（1995 年）.《改进的 M-gram 语言模型回退机制》](https://ieeexplore.ieee.org/document/479394) —— Kneser-Ney 算法的原始论文。
- [KenLM](https://kheafield.com/code/kenlm/) —— 一款高效的生产级（Production）n-gram 语言模型工具，截至 2026 年仍被广泛应用于对延迟敏感（Latency-sensitive）的场景中。
# 词性标注（POS Tagging）与句法分析（Syntactic Parsing）

> 语法曾一度失宠。但随着每个大语言模型（LLM）流水线都需要验证结构化提取（structured extraction），它又重新回到了舞台中央。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 01（文本处理），第 2 阶段 · 14（朴素贝叶斯）
**耗时：** 约 45 分钟

## 问题背景

第 01 课曾指出，词形还原（lemmatization）需要依赖词性标签（part-of-speech tag）。若无法识别 `running` 为动词，词形还原器便无法将其还原为 `run`；同理，若不知 `better` 为形容词，也无法将其还原为 `good`。

这一承诺背后隐藏着一个完整的子领域。词性标注负责分配语法类别，而句法分析则用于恢复句子的树状结构：明确哪个词修饰哪个词，哪个动词支配哪些论元。经典自然语言处理（NLP）曾花费二十年时间不断优化这两项技术。随后，深度学习将它们统一为基于预训练 Transformer（transformer）的词元分类（token-classification）任务，学术界也随之转向了其他方向。

但工业界并未放弃。每个结构化提取流水线底层依然在使用词性标注和依存树（dependency trees）。大语言模型生成的 JSON 数据仍需根据语法约束进行验证。问答系统利用依存分析（dependency parsing）来拆解查询语句。机器翻译质量评估器则会检查解析树（parse trees）的对齐情况。

这些知识值得掌握。本课将介绍标签集（tagsets）、基线模型，以及何时该停止从零实现，转而直接调用 spaCy。

## 核心概念

**词性标注** 为每个词元（token）分配一个语法类别。**宾州树库（Penn Treebank, PTB）** 标签集是英语任务的默认标准。它包含 36 个标签，区分细致到让普通读者觉得繁琐：`NN` 表示单数名词，`NNS` 表示复数名词，`NNP` 表示单数专有名词，`VBD` 表示动词过去式，`VBZ` 表示动词第三人称单数现在时，等等。**通用依存（Universal Dependencies, UD）** 标签集则更为粗粒度（17 个标签）且与语言无关，已成为跨语言工作的默认标准。

The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT

**句法分析** 会生成一棵树状结构。主要有两种风格：

- **成分句法分析（Constituency parsing）。** 名词短语、动词短语、介词短语相互嵌套。输出是一棵以非终结符类别（NP、VP、PP）为节点、以单词为叶子的树。
- **依存句法分析（Dependency parsing）。** 每个词都有一个它所依赖的单一中心词（head word），并带有语法关系标签。输出是一棵树，其中每条边都是一个（中心词，依存词，关系）三元组。

依存句法分析在 2010 年代胜出，因为它能干净利落地跨语言泛化，尤其适用于自由语序语言。

running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at

## 动手构建

### 步骤 1：最常见标签基线 (most-frequent-tag baseline)

这是一个能跑通的最简单的词性标注器 (POS tagger)。对于每个单词，预测它在训练集中出现频率最高的词性标签。

from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]

在布朗语料库 (Brown corpus) 上，该基线模型的准确率约为 85%。虽然不算高，但这是一个底线，任何严肃的模型都不应低于此水平。

### 步骤 2：二元隐马尔可夫模型标注器 (bigram HMM tagger)

对序列的联合概率进行建模：

P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)

需要维护两张表：转移概率 (transition probabilities，给定前一个标签预测当前标签) 和发射概率 (emission probabilities，给定标签预测当前单词)。使用拉普拉斯平滑 (Laplace smoothing) 基于频次统计来估计这两张表。使用维特比算法 (Viterbi algorithm) 进行解码（在标签网格上进行动态规划）。

import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]

在布朗语料库上，二元隐马尔可夫模型 (bigram HMM) 的准确率约为 93%。从 85% 提升到 93% 主要归功于转移概率——模型学到了 `DET NOUN`（限定词后接名词）很常见，而 `NOUN DET` 则很罕见。

### 步骤 3：为何现代标注器能超越此方法

转移概率与发射概率本质上是局部的。它们无法捕捉到 `saw` 在 "I bought a saw" 中是名词，而在 "I saw the movie" 中是动词的上下文差异。引入任意特征（如后缀、词形、前后词、单词本身）的条件随机场 (CRF) 能达到约 97% 的准确率。而双向长短期记忆网络结合条件随机场 (BiLSTM-CRF) 或 Transformer 架构则能突破 98%。

该任务的性能上限由标注者之间的分歧决定。在宾州树库 (Penn Treebank) 上，人工标注者的一致性约为 97%。准确率超过 98% 的模型很可能是在测试集上过拟合了。

### 步骤 4：依存句法分析 (dependency parsing) 概览

从零开始实现完整的依存句法分析超出了本文范围；该主题的标准教材讲解可参考 Jurafsky 和 Martin 的著作。需要了解的两个经典流派如下：

- **基于转移的 (Transition-based)** 解析器（如 arc-eager、arc-standard）的工作方式类似于移进-归约解析器 (shift-reduce parser)：它们读取词元 (token)，将其移入栈中，并执行生成依存弧的归约操作。贪婪解码 (greedy decoding) 速度很快。经典实现是 MaltParser。现代神经网络版本为 Chen 和 Manning 提出的基于转移的解析器。
- **基于图的 (Graph-based)** 解析器（如 Eisner 算法、Dozat-Manning 双仿射模型）会对所有可能的中心词-依存词 (head-dependent) 边进行打分，并选取最大生成树 (maximum spanning tree)。速度较慢但精度更高。

对于大多数实际应用，直接调用 spaCy 即可：

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")

The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running

从下往上阅读 `dep`（依存关系）列，句子的语法结构便一目了然。

## 实际应用

每个生产级自然语言处理（NLP）库都会将词性标注（POS）和依存句法分析（Dependency Parsing）解析器作为标准流水线（Pipeline）的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。速度快、精度高，与分词（Tokenization）+ 命名实体识别（NER）+ 词形还原（Lemmatization）深度集成。提供 `token.tag_`（Penn Treebank 标签集）、`token.pos_`（通用依存 Universal Dependencies, UD 标签集）、`token.dep_`（依存关系）。
- **Stanford NLP (stanza)**。斯坦福大学 CoreNLP 的继任者。在 60 多种语言上达到当前最优（State-of-the-Art）水平。
- **trankit**。基于 Transformer 架构，在 UD 标签上具有较高的准确率。
- **NLTK**。提供 `pos_tag` 函数。功能可用但速度较慢、架构较老，适合教学使用。

### 在 2026 年为何依然重要

- **词形还原（Lemmatization）。** 第 01 课中，必须依赖 POS 才能正确进行词形还原。始终如此。
- **大语言模型（LLM）输出的结构化提取。** 验证生成的句子是否符合语法约束（例如主谓一致、必需的修饰语）。
- **基于方面的情感分析（Aspect-Based Sentiment）。** 依存句法分析能够明确指出哪个形容词修饰哪个名词。
- **查询理解（Query Understanding）。** 通过句法分析，可以将“movies directed by Wes Anderson starring Bill Murray”这类查询分解为结构化约束条件。
- **跨语言迁移（Cross-Lingual Transfer）。** UD 标签和依存关系具有语言无关性，能够支持对新语言进行零样本（Zero-Shot）结构化分析。
- **低算力流水线。** 如果无法部署 Transformer 模型，仅依靠 POS + 依存句法分析 + 地名词典（Gazetteer）也能取得出乎意料的优异效果。

## 交付指南

保存为 `outputs/skill-grammar-pipeline.md`：

---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.

## 练习

1. **简单。** 在小型已标注语料库（例如 NLTK 的 Brown 子集）上使用最高频标签基线（Most-Frequent-Tag Baseline），在留出集（Held-Out）句子上测量准确率。验证其是否达到约 85% 的结果。
2. **中等。** 训练上述二元隐马尔可夫模型（Bigram HMM），并报告每个标签的精确率（Precision）与召回率（Recall）。HMM 最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存句法分析器，从 1000 句样本中提取主谓宾（Subject-Verb-Object）三元组。在 50 个手动标注的三元组上进行评估。记录提取失败的情况（通常出现在被动语态、并列结构和省略主语的句子中）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 词性标签 (POS Tag) | 词的类型 | 语法类别。PTB 包含 36 类；UD 包含 17 类。 |
| 宾州树库 (Penn Treebank) | 标准标签集 | 专为英语设计。包含细粒度的动词时态和名词单复数。 |
| 通用依存 (Universal Dependencies) | 多语言标签集 | 粒度比 PTB 更粗；语言中立；跨语言任务的默认选择。 |
| 依存句法分析 (Dependency Parse) | 句子树 | 每个词仅有一个中心词，每条边代表一种语法关系。 |
| 维特比算法 (Viterbi) | 动态规划 | 在给定发射概率与转移概率的条件下，寻找概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky 与 Martin —《语音与语言处理》第 8 章与第 18 章](https://web.stanford.edu/~jurafsky/slp3/) — 关于词性标注与句法分析的经典权威教材。
- [通用依存 (Universal Dependencies) 项目](https://universaldependencies.org/) — 所有多语言句法分析器均采用的跨语言标签集与树库集合。
- [spaCy 语言特性指南](https://spacy.io/usage/linguistic-features) — 针对 `Token` 对象所暴露的每个属性的实用参考文档。
- [Chen 与 Manning (2014). 《基于神经网络的高效精准依存句法分析器》](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 推动神经网络句法分析器走向主流的开山之作。
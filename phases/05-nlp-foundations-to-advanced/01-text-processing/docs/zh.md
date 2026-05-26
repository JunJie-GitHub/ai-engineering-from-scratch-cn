# 文本处理——分词（Tokenization）、词干提取（Stemming）与词形还原（Lemmatization）

> 语言是连续的，模型是离散的。预处理是连接两者的桥梁。

**类型：** 构建
**语言：** Python
**前置条件：** 第二阶段 · 14（朴素贝叶斯）
**耗时：** 约45分钟

## 问题背景

模型无法直接阅读“The cats were running.”这样的句子，它只能处理整数。

每个自然语言处理（NLP）系统都始于三个核心问题：词语的边界在哪里？词根是什么？在何种情况下应将“run”、“running”、“ran”视为同一概念，又在何种情况下需将它们区分对待？

分词（Tokenization）一旦出错，模型就会从垃圾数据中学习。如果你的分词器将 `don't` 视为一个词元（Token），却将 `do n't` 拆分为两个，训练数据的分布就会发生分裂。如果你的词干提取器（Stemmer）将 `organization` 和 `organ` 错误地归为同一词干，主题建模（Topic Modeling）就会失效。如果你的词形还原器（Lemmatizer）需要词性（Part-of-Speech）上下文但你未提供，动词就会被误当作名词处理。

本节将从零开始构建这三种预处理原语，随后展示 NLTK 和 spaCy 如何实现相同的功能，以便你直观了解其中的权衡取舍。

## 核心概念

三种操作，各司其职，也各有其失效模式。

**分词**将字符串切分为词元。“词元”一词故意保持模糊，因为合适的粒度取决于具体任务。传统自然语言处理通常采用词级（Word-level）；Transformer 模型采用子词级（Subword）；而无空格分隔的语言则采用字符级（Character）。

**词干提取**通过规则截断后缀。速度快、力度大、但缺乏语义理解。`running -> run`。`organization -> organ`。第二个例子正是其典型的失效模式。

**词形还原**利用语法知识将单词还原为词典形式。速度较慢、精度高，需要依赖查找表或形态分析器。`ran -> run`（需识别“ran”是“run”的过去式）。`better -> good`（需识别比较级形式）。

经验法则：当速度优先且可容忍一定噪声时（如搜索索引、粗略分类），使用词干提取；当语义准确性至关重要时（如问答系统、语义搜索或任何面向用户展示的场景），使用词形还原。

## 动手实现

### 步骤 1：正则表达式词元分析器（Regex Word Tokenizer）

最基础且实用的词元分析器（Tokenizer）会按非字母数字字符进行分割，同时将标点符号保留为独立的词元。它并不完美，也不是最终方案，但只需一行代码即可运行。

import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)

按优先级排列的三个匹配模式：包含可选内部撇号的单词（如 `don't`、`it's`）；纯数字；以及任何单个非空白、非字母数字字符作为独立词元（即标点符号）。

>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']

需要注意的失效情况（Failure Modes）。`3pm` 会被分割为 `['3', 'pm']`，因为我们的正则交替匹配了字母序列和数字序列。这对大多数任务来说已经足够。但 URL、电子邮件和话题标签（Hashtags）都会解析失败。在生产环境中，应在通用模式之前添加特定的匹配模式。

### 步骤 2：Porter 词干提取器（Porter Stemmer）（仅步骤 1a）

完整的 Porter 算法包含五个阶段的规则。仅步骤 1a 就涵盖了最常见的英语后缀，并能很好地演示该模式。

def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word

>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']

从上到下阅读这些规则。`ies -> i` 的规则解释了为什么 `ponies` 会变成 `poni` 而不是 `pony`。完整的 Porter 算法包含步骤 1b 来修正此问题。规则之间存在竞争，优先匹配的规则胜出。因此，规则的顺序比任何单条规则本身更重要。

### 步骤 3：基于查找表的词形还原器（Lemmatizer）

严格的词形还原（Lemmatization）需要依赖形态学（Morphology）分析。一个易于教学实现的版本会使用一个小型词元表（Lemma Table）并配合回退机制（Fallback）。

LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()

>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'

最后一个案例是教学的关键点。`watched` 不在我们的表中，且回退机制仅处理 `ing` 结尾。真正的词形还原会覆盖 `ed` 结尾、不规则动词、比较级形容词以及发生音变的复数形式（如 `children -> child`）。这也是为什么生产级系统会使用 WordNet、spaCy 的形态分析器（Morphologizer）或完整的形态学分析器。

### 步骤 4：将它们串联成处理管道（Pipeline）

def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}

缺失的组件是词性标注器（POS Tagger）。第 5 阶段 · 07（词性标注）将构建一个。目前，我们默认将所有词性标记为 `NOUN`（名词），并明确这一局限性。

## 使用方法

NLTK 和 spaCy 均提供可用于生产环境的版本。各自只需几行代码即可运行。

### NLTK

import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]

`word_tokenize` 能够处理缩写、Unicode 字符以及正则表达式容易遗漏的边界情况。`PorterStemmer` 会完整执行全部五个阶段。`WordNetLemmatizer` 需要将词性标注（Part-of-Speech Tagging）从 NLTK 的宾州树库（Penn Treebank）方案转换为 WordNet 的缩写集。上述的转换逻辑正是大多数教程所忽略的关键部分。

### spaCy

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)

The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT

spaCy 将整个处理流程封装在 `nlp(text)` 调用背后。分词（Tokenization）、词性标注（Part-of-Speech Tagging）和词形还原（Lemmatization）会自动执行。在大规模数据处理时，其速度优于 NLTK，且开箱即用的准确率更高。代价在于，你无法轻易替换其中的独立组件。

### 如何选择

| 场景 | 推荐选择 |
|-----------|------|
| 教学、研究、需要灵活替换组件 | NLTK |
| 生产环境、多语言支持、对速度要求高 | spaCy |
| Transformer 流水线（反正你会使用模型自带的分词器） | 使用 `tokenizers` / `transformers` 并跳过传统预处理 |

### 鲜有人提醒的两种失败模式

大多数教程只讲解算法便戛然而止。但在真实的预处理流水线中，有两个问题极易引发故障，却几乎从未被提及。

**可复现性漂移（Reproducibility Drift）。** NLTK 和 spaCy 在不同版本间会改变分词和词形还原器的行为。在 spaCy 2.x 中输出 `['do', "n't"]` 的文本，在 3.x 中可能会输出 `["don't"]`。你的模型是基于某一数据分布训练的，而推理阶段却运行在另一种分布上。模型准确率会悄然下降，且无人知晓原因。请在 `requirements.txt` 中锁定库的版本。编写一个预处理回归测试，固定 20 个样本句子的预期分词结果，并在每次升级时运行该测试。

**训练与推理不匹配（Training/Inference Mismatch）。** 训练时使用激进的预处理（如全小写转换、停用词移除、词干提取），部署时却直接处理原始用户输入，最终导致性能断崖式下跌。这是生产环境自然语言处理（Natural Language Processing, NLP）中最常见的单一故障点。如果在训练阶段进行了预处理，推理阶段就必须执行完全相同的函数。应将预处理逻辑作为函数打包进模型包中，而不是留作一段供服务团队重新编写的 Notebook 代码单元。

## 发布上线

一个可复用的提示词（Prompt），帮助工程师无需通读三本教科书即可选定预处理策略（Preprocessing Strategy）。

保存为 `outputs/prompt-preprocessing-advisor.md`：

---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.

## 练习

1. **简单。** 扩展 `tokenize` 函数，使 URL 作为单个词元（Token）保留。测试：`tokenize("Visit https://example.com today.")` 应输出一个 URL 词元。
2. **中等。** 实现 Porter 算法的第 1b 步。若单词包含元音且以 `ed` 或 `ing` 结尾，则将其移除。需处理双辅音规则（`hopping -> hop`，而非 `hopp`）。
3. **困难。** 构建一个词形还原器（Lemmatizer），以 WordNet 作为查找表，当 WordNet 无对应条目时回退至你实现的 Porter 词干提取器（Stemmer）。在已标注语料库上评估其准确率，并与单独使用 WordNet 或 Porter 的结果进行对比。

## 关键术语

| 术语 | 通俗理解 | 实际含义 |
|------|-----------------|-----------------------|
| Token（词元） | 一个单词 | 模型实际处理的最小单元。可以是单词、子词（Subword）、字符或字节。 |
| Stem（词干） | 词根 | 基于规则剥离后缀后的结果。不一定是真实存在的单词。 |
| Lemma（词典原形） | 词典形式 | 词典中收录的标准形式。需结合语法上下文才能准确计算。 |
| POS tag（词性标签） | 词性 | 如名词（NOUN）、动词（VERB）、形容词（ADJ）等类别。准确进行词形还原所必需。 |
| Morphology（词法形态） | 词形变化规则 | 单词如何根据时态、数、格等发生形态变化。词形还原依赖于该规则。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，仅五页，至今仍是解释最清晰的文献。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实处理流水线（Pipeline）的搭建方式。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你尚未考虑到的词元化（Tokenization）边界情况。
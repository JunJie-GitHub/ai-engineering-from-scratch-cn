# 子词分词（Subword Tokenization）—— BPE、WordPiece、Unigram、SentencePiece

> 词级分词器（Word Tokenizer）遇到未登录词就会失效。字符级分词器（Character Tokenizer）会导致序列长度急剧膨胀。子词分词器（Subword Tokenizer）则巧妙地折中了两者。如今，所有现代大语言模型（LLM）都默认搭载其中一种方案。

**类型：** 学习
**编程语言：** Python
**前置知识：** 第 5 阶段 · 01（文本处理），第 5 阶段 · 04（GloVe / FastText / 子词）
**预计耗时：** 约 60 分钟

## 问题背景

假设你的词表（Vocabulary）包含 50,000 个词。用户输入了 "untokenizable"，你的分词器（Tokenizer）返回了 `[UNK]`。此时模型对该词没有任何语义信号。更糟的是：语料库中第 90 百分位的文档包含 40 个罕见词，这意味着每篇文档都会丢失 40 个词元的信息。

子词分词（Subword Tokenization）正是为了解决这一问题。常见词保持为单个词元（Token），罕见词则被拆解为有意义的片段：`untokenizable` → `un`, `token`, `izable`。由于任何字符串最终都可以表示为字节序列，因此训练数据能够覆盖所有可能的输入。

到 2026 年，所有前沿大语言模型（LLM）都基于三种算法之一（BPE、Unigram、WordPiece）构建，并封装在三大主流库之一（tiktoken、SentencePiece、HF Tokenizers）中。发布语言模型时，你必须从中做出选择。

## 核心概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE（字节对编码，Byte-Pair Encoding）。** 从字符级词表开始，统计所有相邻字符对的出现频率。将频率最高的字符对合并为一个新词元（Token）。重复此过程，直到达到目标词表大小。主流应用算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE（Byte-level BPE）。** 算法逻辑相同，但操作对象是原始字节（256 个基础词元）而非 Unicode 字符。这保证了零 `[UNK]` 词元——任何字节序列均可编码。GPT-2 使用了 50,257 个词元（256 个字节 + 50,000 次合并 + 1 个特殊词元）。

**Unigram（一元模型）。** 从一个庞大的初始词表开始，为每个词元分配一元概率（Unigram Probability）。通过迭代剪枝，移除那些对语料库对数似然（Log-Likelihood）影响最小的词元。推理阶段具有概率性：可以对分词结果进行采样（通过子词正则化（Subword Regularization）进行数据增强时非常有用）。代表模型：T5、mBART、ALBERT、XLNet、Gemma。

**WordPiece。** 合并那些能最大化训练语料库似然值（Likelihood）的字符对，而非单纯依据原始频率。代表模型：BERT、DistilBERT、ELECTRA。

**SentencePiece 与 tiktoken 对比。** SentencePiece 是一个直接在原始 Unicode 文本上*训练*词表（支持 BPE 或 Unigram）的库，并将空白字符编码为 `▁`。tiktoken 是 OpenAI 推出的针对预构建词表的高速*编码器*，它本身不具备训练功能。

经验法则：

- **训练新词表：** 使用 SentencePiece（支持多语言，无需预分词）或 HF Tokenizers。
- **针对 GPT 词表进行快速推理：** 使用 tiktoken（如 `cl100k_base`、`o200k_base`）。
- **两者兼顾：** 使用 HF Tokenizers —— 一个库同时搞定训练与服务部署。

## 动手实践

### 步骤 1：从零实现字节对编码（Byte Pair Encoding）

参见 `code/main.py`。其核心循环如下：

def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges

该算法体现了三个关键事实。`</w>` 用于标记词尾，从而确保 "low"（后缀）和 "lower"（前缀）保持区分。频率加权机制使得高频词对优先合并。合并列表是有序的——推理（inference）阶段会严格按照训练时的顺序应用这些合并规则。

### 步骤 2：使用学习到的合并规则进行编码

def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols

朴素实现的时间复杂度为 O(n·|merges|)。生产环境中的实现（如 tiktoken、HF Tokenizers）采用基于优先队列的合并优先级查找（merge-rank lookup），可在接近线性的时间内完成。

### 步骤 3：SentencePiece 的实际应用

import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']

注意：无需进行预分词（pre-tokenization），空格被编码为 `▁`，`character_coverage` 参数用于控制保留罕见字符的激进程度，以及将其映射为 `<unk>`（未知标记）的比例。

### 步骤 4：使用 tiktoken 处理兼容 OpenAI 的词表

import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4

仅支持编码。速度极快（基于 Rust 后端）。与 GPT-4/5 的分词（tokenization）结果完全一致，适用于字节计数、成本估算以及上下文窗口预算规划。

## 2026 年依然存在的常见陷阱

- **分词器漂移（Tokenizer drift）。** 在词表 A 上训练，却在词表 B 上部署。词元 ID（Token ID）不一致会导致模型输出乱码。需在持续集成（CI）流水线中检查 `tokenizer.json` 的哈希值。
- **空白字符歧义（Whitespace ambiguity）。** 字节对编码（BPE）算法对 `"hello"` 和 `" hello"` 会生成不同的词元。务必显式指定 `add_special_tokens` 和 `add_prefix_space` 参数。
- **多语言训练不足（Multilingual undertraining）。** 偏向英语的语料库生成的词表会将非拉丁语系文本切分成多出 5-10 倍的词元。在 GPT-3.5 上，相同的提示词（Prompt）在日语/阿拉伯语中的词元消耗会高出 5-10 倍。`o200k_base` 词表部分解决了该问题。
- **表情符号切分（Emoji splits）。** 单个表情符号可能占用多达 5 个词元。在规划上下文（Context）预算时，需重点评估表情符号的处理方式。

## 使用指南

2026 年技术栈选型：

| 场景 | 推荐方案 |
|-----------|------|
| 从零训练单语言模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供兼容 OpenAI 的 API 服务 | tiktoken（GPT-4+ 使用 `o200k_base`） |
| 领域专用词表（代码、数学、蛋白质） | 在领域语料上训练自定义 BPE，并与基础词表合并 |
| 边缘端推理、小型模型 | Unigram（较小的词表表现更佳） |

词表大小（Vocabulary size）是一项扩展性决策，而非固定常量。经验法则参考：参数量小于 10 亿（<1B）时建议 32k；10 亿至 100 亿（1-10B）时建议 50k-100k；多语言或前沿大模型建议 200k 以上。

## 交付规范

保存为 `outputs/skill-tokenizer-picker.md`：

---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.

## 练习

1. **简单。** 在 `code/main.py` 的微型语料上训练一个合并次数为 500 的 BPE 模型。对三个预留测试词（Held-out words）进行编码。统计其中恰好生成 1 个词元与生成多于 1 个词元的数量各是多少？
2. **中等。** 对比 `cl100k_base`、`o200k_base` 以及你使用 vocab=32k 训练的 SentencePiece BPE 在 100 句英文维基百科句子上的词元数量。报告各自的压缩率（Compression ratio）。
3. **困难。** 使用 BPE、Unigram 和 WordPiece 分别在同一语料上进行训练。将每种分词结果应用于小型情感分类器（Sentiment classifier），测量其下游准确率（Downstream accuracy）。该选择是否能使 F1 分数（F1 Score）产生超过 1 个点的显著提升？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 字节对编码 (BPE) | 字节对编码 | 贪婪合并最高频的字符对，直至达到目标词表 (vocab) 大小。 |
| 字节级 BPE (Byte-level BPE) | 永远不会出现未知词元 | 在原始 256 个字节上执行 BPE；GPT-2 / Llama 均采用此方案。 |
| 一元模型分词 (Unigram) | 概率型分词器 (tokenizer) | 基于对数似然 (log-likelihood) 从庞大的候选集中进行剪枝；T5、Gemma 等模型使用该方法。 |
| SentencePiece | 专门处理空格的方案 | 支持在原始文本上训练 BPE/Unigram 的库；空格字符被编码为 `▁`。 |
| tiktoken | 速度最快的方案 | OpenAI 基于 Rust 实现的 BPE 编码器，专为预构建词表设计。无需训练过程。 |
| 合并列表 (Merge list) | 魔法数字 | 有序的 `(a, b) → ab` 合并规则列表；推理 (inference) 阶段按顺序依次应用。 |
| 字符覆盖率 (Character coverage) | 多罕见才算太罕见？ | 分词器必须覆盖的训练语料库字符比例；典型值约为 0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— BPE 的原始论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) —— Unigram 的原始论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) —— 该分词器库的原始论文。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) —— 简明参考指南。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) —— 使用手册与编码列表。
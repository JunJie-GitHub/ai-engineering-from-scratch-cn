# 从零构建分词器 (Tokenizer)

> 第 01 课给了你一件玩具。这一课将给你一件武器。

**类型：** 构建
**语言：** Python
**前置要求：** 第 10 阶段，第 01 课（分词器：BPE、WordPiece、SentencePiece）
**时长：** 约 90 分钟

## 学习目标

- 构建一个生产级 BPE 分词器 (BPE tokenizer)，能够处理 Unicode、空白字符规范化 (whitespace normalization) 和特殊标记 (special tokens)
- 实现字节级回退 (byte-level fallback) 机制，使分词器能够对任意输入（包括表情符号 (emoji)、中日韩字符 (CJK) 和代码）进行编码，且不会产生未知标记 (unknown tokens)
- 添加预分词 (pre-tokenization) 正则表达式模式 (regex patterns)，在应用 BPE 合并 (BPE merges) 之前按词边界 (word boundaries) 拆分文本
- 在语料库 (corpus) 上训练自定义分词器，并在多语言文本 (multilingual text) 上评估其压缩率 (compression ratio)，与 tiktoken 进行对比

## 问题

你在第 01 课中构建的 BPE 分词器 (BPE tokenizer) 适用于英文文本。现在，试着把日文丢给它。或者表情符号 (emoji)。又或者混合了制表符和空格的 Python 代码。

它会崩溃。

这并不是因为 BPE 算法本身有误——而是因为实现尚不完整。一个生产级的分词器 (tokenizer) 需要处理任意编码 (encoding) 下的原始字节 (raw bytes)，在切分前进行 Unicode 规范化 (Unicode normalization)，管理永远不会被合并的特殊标记 (special tokens)，并将预分词 (pre-tokenization) 与子词切分 (subword splitting) 串联起来，而且所有这些操作都必须足够快，以免成为处理 15 万亿词元 (tokens) 的训练流水线 (training pipeline) 的瓶颈。

GPT-2 的分词器包含 50,257 个词元。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些可不是玩具级别的数字。支撑这些词表 (vocabularies) 的合并表 (merge tables) 是在数百 GB 的文本上训练得出的，而围绕它的整套机制——规范化 (normalization)、预分词、特殊标记注入、聊天模板格式化 (chat template formatting)——正是区分一个只能处理“hello world”的分词器与一个能处理整个互联网内容的分词器的关键所在。

你将亲手构建这套机制。

## 核心概念

### The Full Pipeline

A production tokenizer is not one algorithm. It is a pipeline of five stages, each solving a different problem.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Each stage has a specific job:

| Stage | What It Does | Why It Matters |
|-------|-------------|----------------|
| Normalize | NFKC Unicode, lowercase optional, strip accents optional | "fi" ligature (U+FB01) becomes "fi" (two chars). Without this, same word gets different tokens. |
| Pre-Tokenize | Split text into chunks before BPE | Prevents BPE from merging across word boundaries. "the cat" should never produce a token "e c". |
| BPE Merge | Apply learned merge rules to byte sequences | The core compression. Turns raw bytes into subword tokens. |
| Special Tokens | Inject [BOS], [EOS], [PAD], chat template markers | These tokens have fixed IDs. They never participate in BPE merges. The model needs them for structure. |
| ID Mapping | Convert token strings to integer IDs | The model sees integers, not strings. |

### Byte-Level BPE

Lesson 01's tokenizer operated on UTF-8 bytes. That was the right call. But we skipped something important: what happens when those bytes are not valid UTF-8?

Byte-level BPE solves this by treating every possible byte value (0-255) as a valid token. Your base vocabulary is exactly 256 entries. Any file -- text, binary, corrupted -- can be tokenized without producing an unknown token.

GPT-2 added a trick: map each byte to a printable Unicode character so the vocabulary stays human-readable. Byte 0x20 (space) becomes the character "G" in their mapping. This is purely cosmetic. The algorithm does not care.

The real power: byte-level BPE handles every language on earth. Chinese characters are 3 UTF-8 bytes each. Japanese can be 3-4 bytes. Arabic, Devanagari, emoji -- all just byte sequences. The BPE algorithm finds patterns in these byte sequences exactly the same way it finds patterns in English ASCII bytes.

### Pre-Tokenization

Before BPE touches your text, you need to split it into chunks. This prevents the merge algorithm from creating tokens that span word boundaries.

GPT-2 uses a regex pattern to split text:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

This pattern splits on contractions ("don't" becomes "don" + "'t"), words with optional leading spaces, numbers, punctuation, and whitespace. The leading space is kept attached to the word -- so "the cat" becomes [" the", " cat"], not ["the", " ", "cat"].

Llama uses SentencePiece, which skips regex entirely. It treats the raw byte stream as one long sequence and lets the BPE algorithm figure out the boundaries. This is simpler but gives BPE more freedom to create cross-word tokens.

The choice matters. GPT-2's regex prevents the tokenizer from learning that "the" at the end of one word and "the" at the start of the next should merge. SentencePiece allows it, which sometimes produces more efficient compression but less interpretable tokens.

### Special Tokens

Every production tokenizer reserves token IDs for structural markers:

| Token | Purpose | Used By |
|-------|---------|---------|
| `[BOS]` / `<s>` | Beginning of sequence | Llama 3, GPT |
| `[EOS]` / `</s>` | End of sequence | All models |
| `[PAD]` | Padding for batch alignment | BERT, T5 |
| `[UNK]` | Unknown token (byte-level BPE eliminates this) | BERT, WordPiece |
| `<\|im_start\|>` | Chat message boundary start | ChatGPT, Qwen |
| `<\|im_end\|>` | Chat message boundary end | ChatGPT, Qwen |
| `<\|user\|>` | User turn marker | Llama 3 |
| `<\|assistant\|>` | Assistant turn marker | Llama 3 |

Special tokens are never split by BPE. They are matched exactly before the merge algorithm runs, replaced with their fixed ID, and the surrounding text is tokenized normally.

### Chat Templates

This is where most people get confused and most implementations break.

When you send messages to a chat model, the API accepts a list of messages:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

The model does not see JSON. It sees a flat token sequence. The chat template converts messages into that flat sequence using special tokens. Every model does this differently:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Get the template wrong and the model produces garbage. It was trained on one exact format. Any deviation -- a missing newline, a swapped token, an extra space -- puts the input outside the training distribution.

### Speed

Python is too slow for production tokenization.

tiktoken (OpenAI) is written in Rust with Python bindings. HuggingFace tokenizers is also Rust. SentencePiece is C++. These achieve 10-100x speedups over pure Python.

For perspective: tokenizing 15 trillion tokens for Llama 3 pre-training at 1 million tokens per second (fast Python) would take 174 days. At 100 million tokens per second (Rust), it takes 1.7 days.

You are building in Python to understand the algorithm. In production, you would use a compiled implementation and only touch the Python wrapper.

## 构建项目

### 步骤 1：字节级编码 (Byte-Level Encoding)

这是基础。将任意字符串转换为字节序列，将每个字节映射为可打印字符以便显示，并支持反向转换过程。

def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")

在多语言文本上进行测试，查看字节数：

texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")

"hello" 占 5 个字节。"你好" 占 6 个字节（每个字符 3 个字节）。火焰表情符号占 4 个字节。字节级分词器 (Byte-Level Tokenizer) 并不关心文本使用的是什么语言。字节就是字节。

### 步骤 2：基于正则表达式的预分词器 (Pre-Tokenizer)

使用 GPT-2 的正则表达式模式将文本切分为多个片段。每个片段将由字节对编码 (Byte Pair Encoding, BPE) 独立进行分词 (Tokenization)。

import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持此功能，因此我们回退到使用 ASCII 字符类。对于生产环境的多语言分词器，建议安装 `regex`。

尝试运行：

print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']

前导空格会保留在单词前面。缩写词会在撇号处拆分。标点符号会成为独立的片段。BPE 永远不会跨越这些边界合并词元 (Token)。

### 步骤 3：在字节序列上应用 BPE

这是第 01 课中的核心算法，但现在改为独立处理预分词后的各个片段。

from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged

### 步骤 4：特殊词元处理 (Special Token Handling)

特殊词元 (Special Token) 需要精确匹配和固定的 ID。它们会完全绕过 BPE 算法。

class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts

### 步骤 5：完整的分词器类 (Full Tokenizer Class)

将所有步骤串联起来：文本规范化 (Normalization)、按特殊词元拆分、预分词、BPE 合并、映射为 ID。

import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)

### 步骤 6：多语言测试 (Multilingual Test)

真正的考验来了。将英文、中文、表情符号和代码输入其中进行测试。

corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()

每个中文字符生成 3 个字节。表情符号生成 4 个字节。这些输入都不会导致分词器崩溃，也不会产生未知词元。这就是字节级 BPE 的强大之处。

## 使用方法

### 比较真实分词器 (Tokenizers)

加载 Llama 3、GPT-4 和 Mistral 的实际分词器。观察它们如何处理同一段多语言文本。

import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")

from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")

你会发现同一段文本的词元 (Token) 数量各不相同。拥有 128K 词表 (Vocabulary) 的 Llama 3 在合并常见模式时更为激进。拥有 100K 词表的 GPT-4 处于中间水平。而拥有 32K 词表的 Mistral 会生成更多的词元，但其嵌入层 (Embedding Layer) 更小。

其中的权衡 (Tradeoff) 始终如一：更大的词表意味着更短的序列 (Sequence)，但也会带来更多的参数 (Parameter)。

## 发布上线

本课将生成一个用于构建和调试生产级分词器（Tokenizer）的提示词（Prompt）。请参阅 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加一个 `get_token_bytes(id)` 方法，用于显示任意词元 ID（token ID）的原始字节（raw bytes）。使用该方法来检查你最常用的合并词元（merged tokens）实际代表的内容。
2. **中等：** 实现一种 Llama 风格的预分词器（pre-tokenizer），该分词器按空白字符和数字进行分割，但保留前导空格。在同一语料库（corpus）上，将其词表（vocabulary）与 GPT-2 的正则表达式（regex）方法进行比较。
3. **困难：** 添加一个聊天模板（chat template）方法，该方法接收一个包含 `{"role": ..., "content": ...}` 消息的列表，并生成符合 Llama 3 聊天格式的正确词元序列（token sequence）。将其与 HuggingFace 的实现进行对比测试。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|----------------|----------------------|
| 字节级 BPE (Byte-level BPE) | “基于字节的分词器” | 基础词表包含 256 个字节值的 BPE（字节对编码）—— 能够处理任意输入，且不会产生未知 token |
| 预分词 (Pre-tokenization) | “在 BPE 之前进行切分” | 基于正则表达式或规则的切分，用于防止 BPE 跨越单词边界进行合并 |
| NFKC 规范化 (NFKC normalization) | “Unicode 清理” | 先进行标准分解，再进行兼容组合 —— 例如将连字“ﬁ”转换为“fi”，将全角“A”转换为半角“A” |
| 对话模板 (Chat template) | “消息如何转换为 token” | 将包含角色/内容的消息列表转换为扁平 token 序列的精确格式 —— 具有模型特异性，且必须与训练时的格式保持一致 |
| 特殊标记 (Special tokens) | “控制 token” | 绕过 BPE 的保留 token ID —— 如 [BOS]、[EOS]、[PAD]、对话标记等 —— 在合并前进行精确匹配 |
| 词元产出率 (Fertility) | “每个单词对应的 token 数” | 输出 token 数量与输入单词数量的比值 —— GPT-4 处理英文时约为 1.3，韩文为 2-3；该值越高，意味着上下文窗口浪费越严重 |
| tiktoken | “OpenAI 的分词器” | 带有 Python 绑定的 Rust 版 BPE 实现 —— 速度比纯 Python 实现快 10 到 100 倍 |
| 合并表 (Merge table) | “词表” | 训练过程中学习到的字节对合并有序列表 —— 这**就是**分词器所掌握的核心知识 |

## 延伸阅读

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken) -- GPT-3.5/4 所使用的 Rust 语言字节对编码 (Byte Pair Encoding) 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- 支持字节对编码 (Byte Pair Encoding)、WordPiece 和 Unigram 的 Rust 分词器 (Tokenizer) 库
- [Llama 3 论文 (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 详细介绍 128K 词表 (Vocabulary) 规模及分词器训练方法
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- 与语言无关的分词 (Tokenization) 方案
- [GPT-2 分词器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 最初的字节到 Unicode 映射 (Byte-to-Unicode Mapping) 实现
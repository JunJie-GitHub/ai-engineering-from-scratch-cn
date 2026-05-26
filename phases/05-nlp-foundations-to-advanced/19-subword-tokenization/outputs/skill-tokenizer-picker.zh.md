---
name: 分词器选择器
description: 为给定的语料库（corpus）和部署目标（deployment target）选择分词器算法、词表大小及对应库。
version: 1.0.0
phase: 5
lesson: 19
tags: [自然语言处理, 分词]
---

给定语料库（规模、语言、领域）和部署目标（从头训练 / 微调 / 兼容 API 的推理），输出以下内容：

1. 算法（Algorithm）。选择 BPE、Unigram 或 WordPiece。附一句理由。
2. 库（Library）。选择 SentencePiece、HF Tokenizers 或 tiktoken。附理由。
3. 词表大小（vocab size）。四舍五入至最接近的 1k。理由需与模型规模及语言覆盖率挂钩。
4. 覆盖率设置（coverage settings）。包括 `character_coverage`、`byte_fallback` 及特殊词元（special token）列表。
5. 验证计划（validation plan）。在保留集（held-out set）上的平均词元/词比（tokens-per-word）、未登录词率（OOV rate）、压缩比（compression ratio）以及往返解码一致性（round-trip decode equality）。

拒绝在包含稀有文字内容的语料库上训练字符覆盖率（character-coverage）低于 0.995 的分词器（tokenizer）。拒绝发布未经持续集成（CI）中固定 `tokenizer.json` 哈希校验的词表。将任何词表大小低于 16k 的单语分词器标记为可能规格不足（under-spec）。
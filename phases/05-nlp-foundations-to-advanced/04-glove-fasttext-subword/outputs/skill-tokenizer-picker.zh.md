---
name: 分词器选择器
description: 为新的语言模型或文本流水线选择分词方法。
version: 1.0.0
phase: 5
lesson: 04
tags: [自然语言处理, 分词, 嵌入]
---

给定任务和数据集描述后，请输出以下内容：

1. 分词策略（词级分词 (word-level)、字节对编码 (BPE)、WordPiece、SentencePiece、字节级 BPE (byte-level BPE)）。附一句理由。
2. 词表大小 (Vocabulary size) 目标。仅英语语言模型 (Language Model, LM)：32k。多语言：64k-100k。代码：50k-100k。
3. 包含确切训练命令的库调用。指明所用库（Hugging Face `tokenizers` 或 `sentencepiece`）。引用参数。
4. 一个可复现性 (Reproducibility) 陷阱。分词器与模型不匹配是生产环境中最常见的静默错误。明确指出哪个分词器应与哪个预训练检查点 (checkpoint) 搭配使用，并警告切勿随意替换。

当用户微调 (fine-tuning) 预训练大语言模型 (Large Language Model, LLM) 时，拒绝推荐训练自定义分词器（微调必须使用预训练分词器）。拒绝为任何生产推理路径 (inference path) 推荐词级分词。将非英语或多书写系统 (multi-script) 语料库 (corpora) 标记为需要使用带字节回退 (byte fallback) 机制的 SentencePiece。
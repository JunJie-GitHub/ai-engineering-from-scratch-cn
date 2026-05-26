---
name: 多语言选择器
description: 为多语言自然语言处理（Natural Language Processing, NLP）任务选择源语言、目标模型及评估方案。
version: 1.0.0
phase: 5
lesson: 18
tags: [自然语言处理, 多语言, 跨语言]
---

根据给定需求（目标语言、任务类型、各语言可用的标注数据），输出以下内容：

1. 微调（Fine-tuning）源语言。默认使用英语；若目标语言存在类型学相近的高资源语言（High-resource Language），请查阅 LANGRANK 或 qWALS。
2. 基座模型（Base Model）。XLM-R（分类）、mT5（生成）、NLLB（翻译）、Aya-23（生成式大语言模型）。
3. 少样本（Few-shot）预算。若有可用数据，初始阶段使用 100-500 条目标语言样本。仅在数据标注不可行时采用零样本（Zero-shot）策略。
4. 评估方案。各语言独立准确率（而非聚合指标 Aggregate Metrics）、跨语言一致性、非拉丁语系文字上的实体级 F1 分数（F1 Score）。

拒绝在未进行分语言评估的情况下部署多语言模型——聚合指标会掩盖长尾（Long-tail）失效问题。标记分词覆盖率（Tokenization Coverage）较低的文字系统（如阿姆哈拉语、提格里尼亚语及众多非洲语言），并指出其需要使用支持字节回退（Byte-fallback）的模型（例如配置 byte_fallback=True 的 SentencePiece，或类似 GPT-2 的字节级分词器 Byte-level Tokenizer）。
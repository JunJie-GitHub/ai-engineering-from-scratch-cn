---
name: embedding-picker
description: 为给定的语料库 (Corpus) 和部署环境选择嵌入模型 (Embedding Model)、维度及检索模式。
version: 1.0.0
phase: 5
lesson: 22
tags: [自然语言处理 (NLP), 嵌入 (Embeddings), 检索 (Retrieval)]
---

给定语料库（规模、语言、领域、平均长度）、部署目标（云端 / 边缘端 / 本地部署）、延迟预算 (Latency Budget) 和存储预算 (Storage Budget)，输出以下内容：

1. 模型。指定检查点 (Checkpoint) 名称或 API。提供一句理由。
2. 维度。完整维度 / Matryoshka 截断 / int8 量化 (INT8 Quantization)。理由需与存储预算挂钩。
3. 模式。稠密 (Dense) / 稀疏 (Sparse) / 多向量 (Multi-vector) / 混合 (Hybrid)。提供理由。
4. 查询前缀 / 模板（若模型卡片 (Model Card) 要求）。
5. 评估计划。与领域相关的 MTEB (Massive Text Embedding Benchmark) 任务 + 使用 nDCG@10 指标在保留领域 (Held-out Domain) 上进行评估。

拒绝在未进行领域验证的情况下将 Matryoshka 截断至 <64 维的建议。对于少于 1 万段落 (Passages) 的语料库，拒绝推荐 ColBERTv2（开销不合理）。标记将长文档语料库（>8k 词元 (Tokens)）路由至仅支持 512 词元上下文窗口 (Context Window) 的模型的情况。
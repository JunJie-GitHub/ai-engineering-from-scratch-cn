---
name: topic-picker
description: 为语料库（Corpus）选择 LDA 或 BERTopic。指定依赖库、调参项（Knobs）与评估指标。
version: 1.0.0
phase: 5
lesson: 15
tags: [自然语言处理, 主题建模]
---

给定语料库描述（文档数量、平均长度、领域、语言、计算预算），输出：

1. 算法（Algorithm）。LDA / NMF / BERTopic / Top2Vec / FASTopic。用一句话说明理由。
2. 配置（Configuration）。主题数量（初始值建议设为 ~sqrt(n_docs)）、`min_df` / `max_df` 过滤阈值，以及用于神经网络方法的嵌入模型（Embedding Model）。
3. 评估（Evaluation）。通过 `gensim.models.CoherenceModel` 计算主题一致性（Topic Coherence, c_v）与主题多样性（Topic Diversity），并辅以 20 个样本的人工审阅。
4. 需排查的故障模式（Failure Mode）。对于 LDA，需关注吸收停用词和高频词的“垃圾主题”；对于 BERTopic，需关注吞没模糊文档的 -1 离群簇（Outlier Cluster）。

若文档长度超过嵌入模型的上下文窗口（Context Window）且未采用分块策略（Chunking Strategy），则拒绝使用 BERTopic。若文本极短（如推文或不足 10 个词元（Token）的评论），则拒绝使用 LDA，因为其一致性会急剧下降。将任何低于 5 或高于 200 的 `n_topics` 选择标记为在实际数据中可能错误。
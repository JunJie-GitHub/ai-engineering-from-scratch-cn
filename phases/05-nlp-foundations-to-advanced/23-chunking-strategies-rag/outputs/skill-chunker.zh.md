---
name: 分块器 (Chunker)
description: 针对给定的语料库和查询分布，选择合适的分块策略、大小及重叠率。
version: 1.0.0
phase: 5
lesson: 23
tags: [自然语言处理 (NLP), 检索增强生成 (RAG), 文本分块 (Chunking)]
---

给定语料库（Corpus，包含文档类型、平均长度、领域）与查询分布（Query Distribution，事实型 Factoid / 分析型 Analytical / 多跳型 Multi-hop），请输出：

1. 策略（Strategy）。递归式（Recursive）/ 句子级（Sentence）/ 语义级（Semantic）/ 父文档（Parent-document）/ 后期分块（Late）/ 上下文分块（Contextual）。需阐述理由。
2. 分块大小（Chunk Size）。以 Token 数量（Token Count）计。理由需与查询类型挂钩。
3. 重叠率（Overlap）。默认值为 0；若大于 0 需提供合理性论证。
4. 最小/最大值强制约束（Min/Max Enforcement）。配置 `min_tokens` 与 `max_tokens` 保护阈值。
5. 评估方案（Evaluation Plan）。在包含 50 个查询的分层评估集（Stratified Evaluation Set，涵盖事实型、分析型、多跳型）上计算 Recall@5。

拒绝任何未实施最小/最大分块大小强制约束的分块策略。若重叠率超过 20% 且未提供消融实验（Ablation Study）证明其增益，则予以拒绝。对未设置最小 Token 下限（Min-token Floor）的语义分块建议进行标记警告。
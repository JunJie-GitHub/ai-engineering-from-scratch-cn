---
name: 情感分析基线
description: 为新数据集设计情感分析基线。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、规模、标签粒度、延迟预算），你需要输出：

1. 特征提取方案（Feature extraction recipe）。指定分词器（tokenizer）、n-gram 范围、停用词策略（stopword policy，通常保留）、否定处理（negation handling，采用作用域前缀或二元组 bigrams）。
2. 分类器（Classifier）。基线模型使用朴素贝叶斯（Naive Bayes），生产环境使用逻辑回归（Logistic Regression），仅当领域需要处理反讽（sarcasm）、基于方面的情感输出（aspect-based output）或跨语言覆盖（cross-lingual coverage）时才使用 Transformer。
3. 评估计划（Evaluation plan）。报告精确率（precision）、召回率（recall）、F1 分数、混淆矩阵（confusion matrix）以及各类别的错误样本。在类别不平衡（imbalanced data）的数据上，切勿仅报告准确率（accuracy）。
4. 部署后需监控的一种故障模式（failure mode）。领域漂移（domain drift）和反讽是排名前两位的风险。建议每周进行一次样本审计（sample audit）。

明确拒绝为情感分析任务推荐移除停用词（stopwords）。在类别不平衡时，拒绝将准确率作为唯一指标。针对富含子词（subword）的语言（如德语、芬兰语、土耳其语），需特别指出其应使用 FastText 或 Transformer 嵌入（embeddings），而非词级 TF-IDF。
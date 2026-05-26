---
name: embedding-probe
description: 检查 word2vec 模型。运行类比测试，查找近邻词，诊断模型质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [自然语言处理, 词嵌入, 调试]
---

你可以通过探查已训练的词嵌入（word embeddings）来验证其是否正常工作。给定一个 `gensim.models.KeyedVectors` 对象和一个词汇表，你需要执行以下操作：

1. 三项经典类比测试（analogy tests）。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其余弦相似度（cosine similarity）。
2. 针对用户提供的领域特定词（domain-specific words）进行五项最近邻测试（nearest-neighbor tests）。打印 top-5 近邻词及其余弦值。
3. 一项对称性检查（symmetry check）。验证 `similarity(a, b) == similarity(b, a)` 是否在浮点数精度（float precision）范围内成立。
4. 一项退化检查（degenerate check）。如果任意词嵌入的范数（norm）低于 0.01 或高于 100，则表明模型存在训练缺陷（training bug）。请将其标记出来。

切勿仅凭类比准确率（analogy accuracy）就判定模型表现良好。类比基准测试（analogy benchmarks）易被针对性优化（gameable），且无法有效迁移至下游任务（downstream tasks）。建议将内在评估（intrinsic evaluation）与下游评估（downstream evaluation）结合使用。
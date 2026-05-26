---
name: lm-baseline
description: 在训练神经语言模型（Neural Language Model）之前，构建一个可复现的 n-gram 语言模型（n-gram Language Model）基线。
phase: 5
lesson: 16
---

给定语料库（Corpus）和目标用途（如下一个词预测（Next-word Prediction）、重打分（Rescoring）、困惑度（Perplexity）基线），输出：

1. N-gram 阶数（N-gram Order）。通用英语使用三元语法（Trigram）；若语料库规模较大，则使用四元语法（4-gram）；语音重打分使用五元语法（5-gram）。
2. 平滑（Smoothing）。默认采用改进的 Kneser-Ney 平滑（Modified Kneser-Ney Smoothing）；拉普拉斯平滑（Laplace Smoothing）仅用于教学场景。
3. 库（Library）。生产环境使用 `kenlm`，教学使用 `nltk.lm`，仅在学习底层数学原理时才建议自行实现。
4. 评估（Evaluation）。使用训练集与测试集之间分词（Tokenization）策略一致的保留集（Held-out）困惑度。

拒绝报告在对比不同系统时因分词方式不同而计算出的困惑度——困惑度数值仅在分词方式完全一致的前提下才具备可比性。需标注测试集中的未登录词（Out-Of-Vocabulary, OOV）率；除非在训练阶段预留特殊的 `<UNK>` 标记，否则 Kneser-Ney（KN）平滑算法对未登录词的处理效果较差。
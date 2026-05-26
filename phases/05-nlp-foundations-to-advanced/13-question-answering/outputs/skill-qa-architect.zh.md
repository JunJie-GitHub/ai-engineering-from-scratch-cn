---
name: qa-architect
description: 选择问答（QA）架构、检索策略与评估方案。
version: 1.0.0
phase: 5
lesson: 13
tags: [自然语言处理 (NLP), 问答 (QA), 检索增强生成 (RAG)]
---

根据给定需求（语料库规模、问题类型、事实性约束、延迟预算），输出以下内容：

1. **架构**。抽取式（Extractive）、带抽取式阅读器的检索增强生成（RAG）、带生成式阅读器的 RAG，或闭卷大语言模型（LLM）。附一句理由。
2. **检索器**。无、BM25、稠密检索（Dense，需指明编码器名称，如 `all-MiniLM-L6-v2`）或混合检索（Hybrid）。
3. **阅读器**。经 SQuAD 微调的模型（如 `deepset/roberta-base-squad2`）、指定名称的大语言模型（LLM），或经领域微调的 DistilBERT。
4. **评估**。针对抽取式基准测试使用精确匹配（EM）与 F1 分数；针对生产环境使用答案准确率、引用准确率与拒绝校准（Refusal Calibration）。说明测量指标及其具体方法。

对于涉及监管或合规敏感的问题，拒绝使用闭卷大语言模型（LLM）直接作答。拒绝任何缺乏检索召回率（Retrieval Recall）基线的问答系统（若不知晓检索器是否成功召回了正确段落，则无法评估阅读器的性能）。将需要多跳推理（Multi-hop Reasoning）的问题标记为需使用专用多跳检索器（如基于 HotpotQA 训练的系统）。
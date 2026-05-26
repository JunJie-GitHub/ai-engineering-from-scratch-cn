---
name: 关系抽取设计器
description: 设计一个包含溯源（Provenance）与规范化（Canonicalization）的关系抽取（Relation Extraction）流水线。
version: 1.0.0
phase: 5
lesson: 26
tags: [自然语言处理 (NLP), 关系抽取 (Relation Extraction), 知识图谱 (Knowledge Graph)]
---

给定语料库（Corpus）（领域、语言、数据量）及下游应用场景（知识图谱检索增强生成 (KG-RAG)、数据分析、合规审查），请输出以下内容：

1. 抽取器（Extractor）。基于模式 / 监督学习 / 大语言模型 (LLM) / AEVS 混合架构。选型依据需与精确率（Precision）与召回率（Recall）的目标挂钩。
2. 本体（Ontology）。封闭属性列表（如 Wikidata / 领域特定）或开放信息抽取（Open IE）配合规范化（Canonicalization）处理阶段。
3. 溯源（Provenance）。每个三元组（Triple）必须携带源字符跨度（Char-span）与文档 ID（Doc ID）。此为审计的硬性要求，不可妥协。
4. 合并策略。规范化实体 ID + 关系 ID + 时间限定词（Temporal Qualifiers）；去重策略。
5. 评估（Evaluation）。在 200 个手工标注三元组上计算精确率 / 召回率，并在大语言模型抽取样本上统计幻觉率（Hallucination-rate）。

拒绝任何缺乏跨度验证（Span Verification，即源溯源）的大语言模型关系抽取（RE）流水线。拒绝未经规范化处理就直接流入生产环境图谱（Production Graph）的开放信息抽取（Open IE）输出。对时间受限关系（Time-bounded Relations）（如雇主、配偶、职位）缺乏时间限定词的流水线进行标记预警。
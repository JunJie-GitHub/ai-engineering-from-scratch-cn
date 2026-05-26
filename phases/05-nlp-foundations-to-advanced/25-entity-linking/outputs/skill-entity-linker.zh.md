---
name: 实体链接器（Entity Linker）
description: 设计实体链接（Entity Linking）流水线——知识库（Knowledge Base, KB）、候选生成器（Candidate Generator）、消歧器（Disambiguator）、评估（Evaluation）。
version: 1.0.0
phase: 5
lesson: 25
tags: [自然语言处理, 实体链接, 知识图谱]
---

给定具体用例（领域知识库、语言、数据规模、延迟预算），输出以下内容：

1. 知识库（Knowledge Base, KB）。Wikidata / Wikipedia / 自定义知识库。版本日期。更新频率。
2. 候选生成器（Candidate Generator）。别名索引（Alias-index）、嵌入（Embedding）或混合架构。目标提及召回率 @ K（Mention Recall @ K）。
3. 消歧器（Disambiguator）。先验概率 + 上下文（Prior + Context）、基于嵌入、生成式（Generative）或大语言模型提示（LLM-prompted）。
4. NIL 策略（NIL Strategy）。最高分阈值、分类器或显式 NIL 候选项。
5. 评估（Evaluation）。提及召回率 @ 30、Top-1 准确率、在预留集（Held-out Set）上的 NIL 检测 F1 分数。

拒绝任何缺乏提及召回率基线（Mention Recall Baseline）的实体链接（Entity Linking, EL）流水线（若无法确认候选生成阶段是否已召回正确实体，则无从评估消歧器的性能）。拒绝任何采用大语言模型提示进行实体链接但未将输出严格约束为有效知识库 ID（Valid KB IDs）的流水线。标记那些存在流行度偏差（Popularity Bias）影响少数实体（例如名称冲突）且未进行领域微调（Domain Fine-tuning）的系统。
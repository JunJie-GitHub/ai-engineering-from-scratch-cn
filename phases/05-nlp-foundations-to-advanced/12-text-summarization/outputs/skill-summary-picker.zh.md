---
name: 摘要选择器
description: 选择抽取式（Extractive）或生成式（Abstractive）摘要，指定使用的库，并添加事实性检查（Factuality Check）。
version: 1.0.0
phase: 5
lesson: 12
tags: [自然语言处理 (NLP), 文本摘要 (Summarization)]
---

给定一项任务（文档类型、合规要求、长度限制、计算预算），输出以下内容：

1. 方法。选择抽取式（Extractive）或生成式（Abstractive）。用一句话解释原因。
2. 初始模型/库。明确指定名称。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed`，或大语言模型（LLM）提示词（Prompt）。
3. 评估方案。ROUGE-1、ROUGE-2、ROUGE-L（使用 `rouge-score` 并启用词干提取（Stemming））。若为生成式摘要，需额外加入事实性检查（Factuality Check）。
4. 需探测的一种失败模式。实体替换（Entity Swap）在生成式新闻摘要中最常见；需标记出源实体未出现在摘要中的样本。

对于医疗、法律、金融或受监管的内容，若未设置事实性校验关卡（Factuality Gate），则拒绝使用生成式摘要。若输入内容超出模型的上下文窗口（Context Window），应标记为需要采用分块映射-归约（Chunked Map-Reduce）摘要策略，而非直接截断。
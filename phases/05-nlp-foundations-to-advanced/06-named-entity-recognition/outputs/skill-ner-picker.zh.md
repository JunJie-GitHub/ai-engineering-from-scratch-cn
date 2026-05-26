---
name: ner-picker
description: 为指定的信息抽取（Information Extraction）任务选择合适的命名实体识别（Named Entity Recognition, NER）方案。
version: 1.0.0
phase: 5
lesson: 06
tags: [自然语言处理（Natural Language Processing, NLP）, 命名实体识别（Named Entity Recognition, NER）, 信息抽取（Information Extraction）]
---

给定任务描述（领域、标签集、语言、延迟要求、数据量），输出以下内容：

1. 方法（Approach）。基于规则（Rule-based）结合实体词典（Gazetteer）、条件随机场（Conditional Random Field, CRF）、双向长短期记忆网络-条件随机场（BiLSTM-CRF）或 Transformer 微调（Transformer Fine-tuning）。
2. 初始模型。请指明具体名称（例如 spaCy 模型 ID `en_core_web_sm` / `en_core_web_trf`，Hugging Face 检查点（Checkpoint）ID `dslim/bert-base-NER`，或“自定义，从零开始训练”）。
3. 标注策略（Labeling Strategy）。BIO、BILOU 或基于跨度（Span-based）。请用一句话说明理由。
4. 评估（Evaluation）。使用 `seqeval`。务必报告实体级（Entity-level）F1 分数，切勿报告词元级（Token-level）分数。

若标注样本不足 500 条，除非用户已拥有预训练（Pretrained）的领域模型（例如用于医疗领域的 BioBERT），否则拒绝推荐对 Transformer 进行微调。若存在嵌套实体（Nested Entities），需标记为必须使用基于跨度（Span-based）或多阶段（Multi-pass）模型。若用户提及“生产级规模（Production Scale）”却仍直接使用开箱即用的 CoNLL-2003 标签，则必须要求进行实体词典（Gazetteer）审查。
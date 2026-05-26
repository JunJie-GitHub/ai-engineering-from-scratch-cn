---
name: coref-picker
description: 选择指代消解（coreference）方法、评估计划与集成策略。
version: 1.0.0
phase: 5
lesson: 24
tags: [自然语言处理, 指代消解, 信息抽取]
---

给定一个用例（单文档/多文档、领域、语言），输出以下内容：

1. 方法。基于规则 / 基于神经网络的跨度（neural span-based）/ 大语言模型提示（LLM-prompted）/ 混合方法。提供一句理由。
2. 模型。若为神经网络模型，需指定具体的检查点（checkpoint）名称。
3. 集成。操作顺序：分词（tokenize）→ 命名实体识别（NER）→ 指代消解（coref）→ 下游任务。
4. 评估。在保留集（held-out set）上计算 CoNLL F1 分数（MUC、B³ 与 CEAF-φ4 的平均值），并人工审查 20 个文档的聚类结果。

对于超过 2,000 个词元（token）的文档，若未采用滑动窗口合并（sliding-window merge）机制，则拒绝仅依赖大语言模型的指代消解方案。拒绝任何在运行指代消解时未提供提及级别精确率-召回率（mention-level precision-recall）报告的流水线（pipeline）。对在人口统计学特征多样化的文本中部署的基于性别启发式（gender-heuristic）的系统进行标记。
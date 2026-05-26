---
name: nli-picker
description: 为分类/忠实度（Faithfulness）/零样本（Zero-Shot）任务选择自然语言推理（Natural Language Inference, NLI）模型、标签模板和评估设置。
version: 1.0.0
phase: 5
lesson: 21
tags: [自然语言处理（NLP）, 自然语言推理（NLI）, 零样本（Zero-Shot）]
---

给定一个用例（忠实度检查（Faithfulness Check）、零样本分类（Zero-Shot Classification）、文档级推理（Document-Level Inference）），输出以下内容：

1. 模型（Model）。指定 NLI 检查点（Checkpoint）名称。选择理由需与具体领域、文本长度及语言相关联。
2. 模板（若为零样本任务）。表述模式（Verbalization Pattern）。示例。
3. 阈值（Threshold）。用于决策规则的蕴涵（Entailment）截断值。理由需基于校准（Calibration）结果。
4. 评估（Evaluation）。在预留标注集（Held-out Labeled Set）上的准确率、仅假设基线（Hypothesis-only Baseline）以及对抗性子集（Adversarial Subset）上的表现。

若未进行包含 100 个样本的标注合理性检查（Sanity Check），则拒绝部署零样本分类任务。拒绝在文档长度的前提（Premise）上使用句子级 NLI 模型。标记任何声称 NLI 能解决幻觉（Hallucination）的说法——它仅能降低幻觉发生率，而无法彻底消除。
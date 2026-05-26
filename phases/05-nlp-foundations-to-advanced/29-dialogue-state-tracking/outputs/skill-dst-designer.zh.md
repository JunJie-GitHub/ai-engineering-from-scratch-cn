---
name: dst-designer
description: 设计对话状态追踪器（Dialogue State Tracker）—— 涵盖模式定义、抽取器、更新策略与评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [自然语言处理, 对话系统, 任务导向型]
---

给定一个用例（包含领域、语言、词表开放程度、合规需求），输出以下内容：

1. 模式（Schema）。领域列表、各领域对应的槽位（Slot）、每个槽位的开放词表与封闭词表。
2. 抽取器（Extractor）。基于规则（Rule-based）/ 序列到序列（Seq2Seq）/ 结合 Pydantic 的大语言模型（LLM）。需说明选择理由。
3. 更新策略（Update Policy）。全量状态重新生成 / 增量更新；纠错处理；否定处理。
4. 评估（Evaluation）。在预留对话集上的联合目标准确率（Joint Goal Accuracy）、槽位级别的精确率/召回率（Precision/Recall），以及最难槽位的混淆情况。
5. 确认流程（Confirmation Flow）。何时需明确要求用户确认（例如执行破坏性操作、低置信度抽取结果）。

对于合规敏感型槽位，若缺乏基于规则的二次校验，则拒绝采用纯大语言模型（LLM-only）的对话状态追踪器（DST）。拒绝任何无法在用户纠错时回滚槽位状态的 DST 方案。对缺少版本标签的模式定义进行标记提示。
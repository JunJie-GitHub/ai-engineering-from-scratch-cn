---
name: 长上下文评估
description: 为指定模型和应用场景设计长上下文评估套件。
version: 1.0.0
phase: 5
lesson: 28
tags: [自然语言处理, 长上下文, 评估]
---

给定目标模型、目标上下文长度（Context Length）和应用场景，输出以下内容：

1. 测试（Tests）。NIAH（Needle In A Haystack）深度 × 长度网格；RULER 多跳（Multi-hop）任务；自定义领域任务。
2. 采样（Sampling）。在每个长度下，设置深度为 0、0.25、0.5、0.75、1.0。
3. 指标（Metrics）。检索通过率；推理通过率；首字延迟（Time-to-First-Token）；单次查询成本（Cost-per-Query）。
4. 阈值（Cutoff）。有效检索长度（通过率 90%）与有效推理长度（通过率 70%）。需同时报告这两项。
5. 回归测试（Regression）。使用固定测试框架（Harness），在每次模型升级时重新运行，并呈现性能差异（Deltas）。

切勿仅凭模型卡片（Model Card）上的上下文窗口（Context Window）声明就予以采信。对于任何多跳（Multi-hop）工作负载，拒绝仅依赖 NIAH 进行评估。切勿将厂商自行报告的长上下文得分作为独立证据。
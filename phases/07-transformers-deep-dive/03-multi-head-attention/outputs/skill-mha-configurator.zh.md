---
name: mha-configurator
description: 为新 Transformer (Transformer) 模型推荐注意力头数量、KV 头数量及投影策略（MHA / MQA / GQA / MLA）。
version: 1.0.0
phase: 7
lesson: 3
tags: [Transformer, 注意力机制, MHA, GQA]
---

给定 Transformer 规格（参数预算、隐藏层维度 `d_model`、目标上下文长度、推理设备内存、训练与推理优先级），输出以下内容：

1. 投影变体 (Projection variant)。从以下选项中选择其一：MHA、GQA、MQA 或 MLA。提供一句与 KV 缓存 (KV-cache) 限制相关的理由。
2. 头几何结构 (Head geometry)。`n_heads`、`n_kv_heads`、`d_head`。取值必须满足 `d_model = n_heads * d_head` 且 `n_heads % n_kv_heads == 0`。
3. KV 缓存估算。在目标上下文长度下，所选变体每层每个 Token 的字节数（fp16 精度）。若单个批次 (batch) 超出目标设备内存，需进行标记。
4. 初始化。Q、K、V、O 矩阵的 Xavier / Kaiming 缩放系数。注明是否包含偏置项 (bias terms)（大多数 2026 年的模型已弃用偏置项）。
5. 可测试性钩子 (Testability hook)。一个单一的综合任务（例如归纳头 (induction-head) 模式 `A B A ? → B`），该配置训练出的双层模型版本在此任务上的准确率应达到 ≥95%。

拒绝推荐 `d_head < 32` 的配置——此时注意力动力学 (attention dynamics) 将失效。对于上下文长度超过 32K 的场景，若未明确计算 KV 缓存开销并建议改用 GQA 或 MLA，则拒绝推荐 `n_heads > 16` 的 MHA 配置。除非用户明确要求进行基准测试 (benchmarking)，否则拒绝为参数量低于 1B 的模型推荐 MLA。
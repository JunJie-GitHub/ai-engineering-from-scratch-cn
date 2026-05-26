---
name: positional-encoding-picker
description: 根据上下文长度（Context Length）和训练预算，选择位置编码（Positional Encoding，如 RoPE、ALiBi、正弦编码）及缩放策略。
version: 1.0.0
phase: 7
lesson: 4
tags: [Transformer, 位置编码, RoPE, ALiBi]
---

给定 Transformer 规格（推理目标上下文长度、训练上下文长度、外推（Extrapolation）需求、以 Token 为单位的微调（Fine-tuning）预算），请输出以下内容：

1. 基础编码（Base Encoding）。从以下选项中选择其一：RoPE、ALiBi、正弦编码（Sinusoidal）、学习型绝对编码（Learned-Absolute）。附一句选择理由。
2. 超参数（Hyperparameters）。若选择 RoPE：需提供 `base` 值，以及 `d_head` 需满足的均分要求。若选择 ALiBi：需提供斜率公式。若选择正弦编码：需提供 `max_len`。
3. 扩展策略（Extension Strategy）。若目标长度大于训练长度：需提供 NTK 感知缩放因子（NTK-aware Scaling Factor）、YaRN 配置、LongRoPE 规范或位置插值比例（Position Interpolation Ratio）。请明确微调的 Token 预算。
4. 测试计划（Test Plan）。在最大上下文长度下的 NIAH（大海捞针测试，Needle-In-A-Haystack）通过率目标，以及困惑度（Perplexity）需控制在训练长度基线（Baseline）的 X 范围内。
5. 回退策略（Fallback）。若长上下文评估失败时的应对措施：使用更大的 `base` 值重新训练、切换至 ALiBi，或限制部署时的上下文长度上限。

拒绝为 2026 年的新模型推荐正弦编码或学习型绝对编码——它们不具备外推能力，且现代技术栈均默认采用 RoPE 或 ALiBi。拒绝在未经过微调阶段的情况下，将 RoPE 缩放至超过训练长度 8 倍的范围。拒绝在未对完整部署长度执行 NIAH 测试的情况下，交付长上下文配置。
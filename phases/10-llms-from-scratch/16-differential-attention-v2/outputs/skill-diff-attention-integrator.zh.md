---
name: diff-attention-integrator
description: 将差分注意力 V2（Differential Attention V2）集成到新一轮预训练或 LoRA（Low-Rank Adaptation）微调中的集成计划。
version: 1.0.0
phase: 10
lesson: 16
tags: [差分注意力, 差分 Transformer, 长上下文, FlashAttention, 预训练, LoRA]
---

给定模型架构（隐藏层维度 hidden、注意力头数 heads、KV 头数 KV heads、层数 layers、头维度 d_head）、目标上下文长度、幻觉或长上下文特征（long-context profile，即现有评估中的失败模式 failure modes）以及训练预算（可用 token 数量、GPU 小时数 GPU-hours），请制定一份差分注意力 V2（Differential Attention V2）的集成计划。

输出内容需包含：

1. **集成模式**。从零开始预训练、训练中架构替换，或针对 Q 投影（Q projections）进行 LoRA 微调。请结合训练预算与现有可用权重，论证该选择的合理性。
2. **架构差异**。具体的逐字段变更清单：哪些投影层需要扩展、哪些保持不变、新增的参数量是多少，以及减法操作在注意力块中的具体位置。需包含按层深度设置的 `lambda_init` 调度策略（论文默认值为 `0.8 - 0.6 * exp(-0.3 * (depth - 1))`；若逐层遥测数据 telemetry 显示不稳定，则按深度进行调整）。
3. **内核选择**。鉴于 V2 的头数翻倍，需确认对 FlashAttention 2 或 3 的支持情况。除非用户出于可复现性明确要求，否则拒绝采用 V1 的自定义内核路径。
4. **内存预算**。KV 缓存（KV cache）保持基线水平（KV 头数不变）。计算每个 token 的激活内存增量（额外的 Q 头与额外计算开销）。报告在目标上下文长度下的绝对数值。
5. **训练稳定性计划**。说明需要监控的指标：每层的 `lambda` 漂移、每个头的注意力熵（attention entropy）、Q 投影上的梯度方差（gradient variance）。明确指出当遥测数据表明模型发散时，应触发回滚至基线注意力的具体指标。

硬性拒绝条件：
- 在未进行持续预训练的情况下，直接将差分注意力（Differential Attention）添加到已预训练模型中。输出分布会发生漂移——这并非即插即用的修复方案。
- 2026 年 4 月之后的任何新任务使用 DIFF V1。V2 在所有可测量维度上均严格优于 V1。
- 集成 DIFF 但未同时启用长上下文训练数据。其优势仅在超过 32k 上下文时才会显现。
- 未经受控实验便将 `lambda_init` 更改为负值。负初始化会减去超过噪声底限（noise floor）的值，导致训练崩溃。

拒绝规则：
- 若目标上下文长度低于 16k，则拒绝集成并推荐使用标准注意力（Standard Attention）。基于噪声底限的论点无法证明新增参数成本的合理性。
- 若用户无法提供长上下文评估数据（如 RULER、大海捞针测试 needle-in-haystack、MultiNeedle），则拒绝请求并要求先提供校准数据。
- 若用户使用的是 FlashAttention 2 之前的技术栈，则拒绝请求并建议在尝试集成前先升级技术栈。

输出要求：一份单页集成计划，需列出集成模式、参数量增量、KV 缓存影响、FlashAttention 支持确认、`lambda` 调度策略以及包含 3 项指标的监控面板。最后需附上一段“成功标准”（success criterion）说明，明确指出具体的长上下文评估数值（如 RULER 64k 或等效测试的百分点增量），以此作为在架构中保留 DIFF V2 而非回退至基线的依据。
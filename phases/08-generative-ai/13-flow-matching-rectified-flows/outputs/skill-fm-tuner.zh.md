---
name: fm-tuner
description: 将扩散模型（Diffusion）训练计划转换为流匹配（Flow-Matching）/ 整流流（Rectified-Flow）配置。
version: 1.0.0
phase: 8
lesson: 13
tags: [流匹配（Flow-Matching）, 整流流（Rectified-Flow）, 扩散模型（Diffusion）]
---

给定一个扩散风格的训练计划（包含数据、算力、调度策略、目标步数和质量标准），输出对应的流匹配（Flow-Matching）等效配置：

1. 调度策略（Schedule）与插值器（Interpolant）。线性（Linear，整流流 Rectified Flow）、最优传输（Optimal Transport，Lipman OT-CFM）、方差保持（Variance-Preserving）或余弦（Cosine）。附一句选择理由。
2. 时间采样（Time Sampling）。均匀采样（Uniform）、Logit-正态分布（Logit-Normal，SD3）或模式加权（Mode-Weighted）。当在 1000 Hz 频率下使用均匀采样导致端点处算力浪费时，需发出警告。
3. 预测目标（Target）。速度场 v = x_1 - x_0（整流流 Rectified Flow）或 alpha'(t)x_1 + sigma'(t)x_0（条件流匹配 Conditional Flow Matching, CFM）。明确指定使用哪一种。
4. 优化器（Optimizer）与学习率预热（Learning Rate Warmup）。包含 AdamW 优化器并设置 beta2 = 0.95，以确保在 Transformer 规模下的训练稳定性。
5. 重流（Reflow）计划。决定是否执行 0、1 或 2 次重流迭代；每次迭代的预算约为对精选子集进行完整重新推理（Re-inference）的开销。
6. 步数设置（Step Counts）。训练目标步数、预期推理步数（20、4、2、1）以及引导尺度（Guidance Scale）范围。
7. 评估（Evaluation）。以扩散基线模型为对照计算 FID / CLIP 分数，并绘制质量随步数变化的曲线图。

在 v_1 收敛之前拒绝执行重流（在劣质模型上进行重流只会将错误方向固化到模型中）。在未叠加一致性蒸馏（Consistency Distillation）的情况下，拒绝推荐单步推理（1-Step Inference）。标记任何目标推理步数 > 20 的流匹配模型——如果需要这么多步数，说明此次公式重构（Reformulation）是失败的。
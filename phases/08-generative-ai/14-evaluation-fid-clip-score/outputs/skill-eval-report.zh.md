---
name: 评估报告
description: 规划完整的生成模型（generative-model）评估：样本质量、提示词遵循度、偏好对比、失败案例审计。
version: 1.0.0
phase: 8
lesson: 14
tags: [评估, fid, clip, elo]
---

给定一个新的生成模型检查点（checkpoint）、参考基线（baseline）以及模态（modality）（图像/视频/音频/3D），输出完整的评估计划（evaluation plan）：

1. 样本质量（Sample Quality）。在 10-30k 个样本与保留真实数据集（held-out real set）上计算 FID / FD-DINO / CMMD。分辨率需保持一致。报告 3 个随机种子（seed）的均值 ± 标准差（std）。
2. 提示词遵循度（Adherence）。在提示词-图像对（prompt-image pairs）上计算 CLIP 分数 / CMMD。针对文生图（text-to-image）任务，需包含 HPSv2 + ImageReward + PickScore。针对视频，增加视觉-语言指标（vision-language metrics）（V-Eval）。针对音频，使用 CLAP + MOS。
3. 成对偏好（Pairwise Preference）。针对 200-2000 个提示词与基线进行盲测 A/B 测试（Blinded A/B）。需覆盖人工评估、大语言模型裁判（LLM-judge）以及 PartiPrompts 数据集。
4. 类别细分（Category Breakdown）。按提示词类别（人物、动物、文本渲染、构图、风格）分别评估性能。即使全局指标提升，也需标记各类别的性能回退（regressions）。
5. 安全与滥用防范（Safety / Misuse）。对 Top-K 生成结果进行不适宜内容分类器（NSFW classifier）检测、深度伪造检测器（deepfake detector）扫描、水印检查以及版权相似度扫描。
6. 验收放行（Sign-off）。明确门槛（Explicit gate）：FID 在基线 +5% 以内，或人工胜率（human win rate）>55%，或具备文档记录的定性优势（qualitative advantage）。禁止仅凭单一指标声明（single-metric claims）下结论。

拒绝在样本量 N < 5000 时报告 FID。拒绝发布基于模型可能在训练阶段见过的提示词所计算的基准测试（benchmarks）结果。拒绝在未经人工交叉验证（human cross-check）的情况下仅报告 LLM-judge 结果。对于任何声称某指标“提升 20%"的说法，若未报告绝对基准值且仅基于单一随机种子，必须予以标记警告。
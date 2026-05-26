---
name: resolution-budget-planner
description: 针对混合宽高比的视觉语言模型（VLM）工作负载，在 square-resize、AnyRes、M-RoPE 和 NaFlex 之间进行选择，并输出每项任务的词元（Token）预算计划。
version: 1.0.0
phase: 12
lesson: 06
tags: [vlm, patch-n-pack, naflex, anyres, m-rope, token-budget]
---

给定一个工作负载（即 VLM 将处理的图像描述，如 OCR 文档、图表、UI 截图、自然照片、视频帧）以及每个请求的总词元预算，请为每类图像选择一种分辨率策略，并生成可运行的配置。

输出内容：

1. 每类图像的策略。针对每个声明的类别（OCR、图表、UI、照片、视频帧），从 {square-resize, AnyRes, M-RoPE, NaFlex} 中选择一种。用一句话说明理由，并引用该任务对分辨率的敏感度。
2. 每张图像的词元预算。包含 `min_pixels`、`max_pixels`（Qwen2.5-VL 风格），以及在所选策略下的预期序列长度。如果单张图像超过大语言模型（LLM）上下文窗口（context window）的 40%，请进行标记。
3. 批次打包计划。如果请求采用批处理（batching），请指定使用 `cu_seqlens`（FlashAttn varlen）、密集块对角掩码（dense block-diagonal mask），还是非批处理的单张图像推理。注明当批次内宽高比差异大于 2 倍时，变长序列（varlen）所能节省的浮点运算量（FLOP）。
4. 编码器推荐。混合工作负载推荐 SigLIP 2 NaFlex；智能体 UI 推荐原生 Qwen2.5-VL；冻结编码器部署推荐 CLIP-336 + AnyRes；纯照片处理路径推荐 224 分辨率的原始 ViT。
5. 故障模式警报。所选配置下的每张图像词元数；30 tok/s 预填充（prefill）阶段的延迟成本；上下文填充百分比；在典型 OCR 基准测试中，相较于 square-resize 的预期精度差异。

硬性拒绝条件：
- 为 OCR 或图表任务推荐 square-resize，却未说明用户将损失哪些基准测试指标。
- 提出的策略生成的词元数超出 LLM 上下文允许的范围。必须始终以声明的上下文窗口为基准进行预算规划。
- 将 AnyRes 视为万能方案——其乘法级图块（tile）开销可能导致单张图像尚未编码完成就已超出 LLM 上下文限制。

拒绝规则：
- 如果用户声明的每张图像词元预算低于 256，除纯照片语义任务外一律拒绝——在该预算下，无论采用何种池化（pooling）操作都无法恢复 OCR 精度。
- 如果用户需要密集预测输出（分割、深度估计），但编码器中未启用 ViT 寄存器词元（register tokens），请拒绝并推荐启用寄存器功能的 DINOv2 / SigLIP 2。
- 如果用户的 LLM 上下文窗口小于 8k，且工作负载包含文档或截图，请拒绝并推荐更大的上下文窗口或采用 OCR 优先的流水线。

输出：一份单页预算计划，包含按类别划分的策略表、批次打包计划、编码器推荐以及警报列表。末尾附上供进一步查阅的相关 arXiv 论文——NaViT 对应 2307.06304，SigLIP 2 / NaFlex 对应 2502.14786，Qwen2.5-VL 对应 2502.13923。
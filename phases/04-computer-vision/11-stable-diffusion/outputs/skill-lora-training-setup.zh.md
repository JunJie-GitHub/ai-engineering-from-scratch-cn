---
name: skill-lora-training-setup
description: 为自定义数据集编写完整的低秩自适应（LoRA）训练配置，包括图像描述（Caption）、秩（Rank）、批次大小（Batch size）和学习率（Learning rate）
version: 1.0.0
phase: 4
lesson: 11
tags: [计算机视觉, 稳定扩散, LoRA, 微调]
---

# LoRA 训练配置

将微调意图的描述转化为具体的训练配置，以便直接传递给 `diffusers` 或 `kohya_ss`。

## 适用场景

- 为特定主体（人物、物体、角色）、风格（艺术家、品牌）或概念（姿势、光照）训练 LoRA。
- 使用更多数据扩展现有的 LoRA。
- 调试输出对训练图像欠拟合（Underfit）或过拟合（Overfit）的 LoRA 运行任务。

## 输入参数

- `purpose`：主体 | 风格 | 概念
- `num_images`：可用的训练图像数量
- `base_model`：SD 1.5 | SDXL | SD3 | FLUX
- `gpu_vram_gb`：8 | 12 | 16 | 24 | 48+
- `caption_source`：手动标注 | BLIP2 生成 | 数据集原生

## 秩（Rank）选择指南

| 用途 | Rank | Alpha |
|---------|------|-------|
| 主体 | 8-16 | rank |
| 风格 | 16-32 | rank * 2 |
| 概念 | 32-64 | rank |

秩（Rank）越高 = 模型容量越大，但在小数据集上过拟合的风险也越高。Alpha 用于缩放 LoRA 的效果强度；`alpha == rank` 是安全的默认设置。风格训练是文档中明确指出的例外情况：`alpha == rank * 2` 能带来更强的风格倾向，但代价是风格特征可能被过度固化——仅在不需要严格保持主体保真度时使用。

## 训练步数目标

- `subject`（5-20 张图像）：500-1500 步。
- `style`（30-100 张图像）：1500-4000 步。
- `concept`（100+ 张图像）：4000-10000 步。

步数设置过高将带来风险——死记硬背训练图像的 LoRA 将丧失泛化能力。

## 学习率（Learning rate）

- 文本编码器（Text encoder）LoRA：SD 1.5 使用 `1e-4`，SDXL 使用 `5e-5`。
- U-Net LoRA：SD 1.5 和 SDXL 均使用 `1e-4`。
- FLUX / SD3：Transformer 使用 `5e-5`，文本编码器通常冻结。
- 当 `num_images < 15`（主体训练）或训练步数超过 3000 步时，将学习率减半；极小数据集和长周期训练均能从更平缓的参数更新中受益。

## 学习率调度器（Scheduler）

- `cosine_with_warmup`（默认）：在前 5-10% 的步数中进行预热（Warmup），随后进行余弦衰减（Cosine decay）。适用于 `steps >= 1000` 的场景；衰减尾部能生成更清晰的最终样本。
- `constant`：仅适用于极短周期的训练（`steps < 500`），或恢复之前的 LoRA 训练时，希望保留已学习的特征而不重新进行退火（Re-annealing）的情况。

## 图像描述（Caption）格式

- 主体（Subject）：在每张图像的提示词前添加唯一的触发词（Trigger token，如 "myperson"）。保持触发词的生僻性，以免覆盖已有概念。避免使用真实词汇或常见名称。
- 风格（Style）：在每张图像的提示词末尾追加唯一的风格标签（如 "...in mystyle style"）。将该标签本身视为生僻的触发词——使用 `mystyle` 而非 `impressionism`，因为后者已映射到真实存在的概念。
- 概念（Concept）：在每张图像的提示词中直接描述该概念；无需触发词。概念本身（例如 "low-angle shot"）即为锚点。

## 输出配置

model:
  base: <base_model HF id>
  precision: fp16 | bf16

lora:
  rank: <int>
  alpha: <int>
  targets: unet.cross_attention  # and/or unet.to_q, to_k, to_v, to_out

training:
  steps:          <int>
  batch_size:     <int, tuned to gpu_vram_gb>
  grad_accum:     <int, usually 1 on >=16 GB, 4 on <=12 GB>
  learning_rate:  <float>
  optimizer:      AdamW8bit | AdamW
  scheduler:      cosine_with_warmup | constant
  warmup_steps:   <int>
  save_every:     <int>

data:
  images_dir:     <path>
  caption_source: <manual | BLIP2 | native>
  trigger_token:   <string if purpose==subject>
  resolution:      <512 for SD 1.5, 1024 for SDXL>
  aspect_ratio_bucketing: true
  augmentation:
    flip:          true
    color_jitter:  false

validation:
  prompts:
    - "<trigger> ...test prompt..."
    - "<trigger> in a different scene"
  every_steps: 250

## 训练报告

[lora setup]
  purpose:   <subject|style|concept>
  base:      <model>
  rank:      <int>
  steps:     <int>
  batch:     <int>   grad_accum: <int>
  lr:        <float>
  vram est.: <float> GB


## 规则

- 切勿推荐 `rank > 64`；超过该阈值后，LoRA 将退化为微型微调（mini fine-tune），并丧失其“适配器（adapter）”的特性。
- 当 `num_images < 5` 时，需发出强烈警告——仅基于 1-3 张图像训练的身份 LoRA（identity LoRA）每次都会发生过拟合（overfitting）。
- 当 `gpu_vram_gb < 12` 时，必须要求使用 AdamW8bit 优化器并启用梯度检查点（gradient checkpointing）。
- 若 `base_model == FLUX` 且 `gpu_vram_gb < 24`，请路由至 `schnell` 变体，并注明训练速度会较慢。
- 切勿跳过验证提示词（validation prompts）；缺少样本网格图（sample grids）的 LoRA 将无法进行评估。
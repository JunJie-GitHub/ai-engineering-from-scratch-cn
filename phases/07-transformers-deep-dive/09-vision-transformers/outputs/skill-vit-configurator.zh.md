---
name: vit-configurator
description: 为新的视觉任务选择 ViT 变体、图像块大小（patch size）和预训练来源。
version: 1.0.0
phase: 7
lesson: 9
tags: [Transformer, ViT, 视觉]
---

给定视觉任务（分类 / 分割 / 检测 / 检索）、图像分辨率、数据集规模（标注数据 + 未标注数据）以及部署目标，输出以下内容：

1. 主干网络（Backbone）。从以下选项中选择其一：DINOv2 ViT-L/14（检索/分类默认选项）、SAM 3 编码器（分割）、SigLIP（视觉-语言模型）、ConvNeXt（延迟敏感场景）。附一句选择理由。
2. 图像块大小（Patch size）。224 分辨率下的标准分类使用 16，DINOv2 使用 14，高分辨率下的密集预测（dense prediction）使用 8。标注序列长度 `(H/P)^2 + 1` 及注意力计算开销 `O(N^2)`。
3. 预训练来源（Pretraining source）。提供检查点（Checkpoint）名称。对于小规模标注数据集（<1万）：冻结 DINOv2 特征 + 线性探测（linear probe）。对于大规模数据集（>10万）：微调（fine-tune）最后几个模块。说明原因。
4. 训练方案（Training recipe）。优化器（AdamW）、学习率（lr）、数据增强（augmentations，如 RandAug、MixUp、随机擦除 Random Erasing）、标签平滑（label smoothing，通常为 0.1）、指数移动平均（EMA）。
5. 风险提示（Risk note）。数据规模风险（Data regime risk，数据量不足以支持全量微调 full fine-tune）、分辨率不匹配（resolution mismatch，预训练分辨率 224 → 部署分辨率 1024 且未进行位置插值 position interpolation）、缺失注册令牌（register-token，可能损害 DINOv2 特征）。

拒绝推荐在少于 100 万张图像的数据集上从头训练（train from scratch）ViT —— 此时卷积神经网络（CNN）基线模型将更具优势。拒绝推荐会导致序列长度 > 4096 的图像块大小，除非明确讨论过 Flash Attention 与分层变体（hierarchical variants，如 Swin）。标记任何在未对位置嵌入（positional embeddings）进行插值的情况下更改输入分辨率的部署方案。
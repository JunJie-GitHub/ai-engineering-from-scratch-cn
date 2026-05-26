---
name: skill-vit-patch-and-pos-embed-inspector
description: 验证视觉变换器（Vision Transformer, ViT）的图像块嵌入（patch embedding）和位置嵌入（positional embedding）形状是否与模型预期的序列长度匹配
version: 1.0.0
phase: 4
lesson: 14
tags: [vision-transformer, debugging, pytorch]
---

# ViT 图像块与位置嵌入检查器

视觉变换器（Vision Transformer, ViT）迁移中最常见的错误：将在 224x224 分辨率下预训练的权重检查点（checkpoint）加载到配置为 384x384 的模型中（反之亦然）。这会导致位置嵌入的序列长度不匹配，模型会静默地输出无意义的结果。

## 适用场景

- 在非默认分辨率下对预训练的 ViT 进行微调（fine-tuning）。
- 排查 ViT-B/16 与 ViT-B/32 之间权重迁移失败的原因；检查器会标记图像块尺寸（patch size）不匹配的问题，以便调用者知晓应更换架构而非强行迁移。
- 调试加载无报错但训练效果不佳的 ViT 模型。

## 输入参数

- `model`：已实例化的 ViT `nn.Module`。
- `expected_image_size`：模型在生产环境中将处理的图像高度（H）与宽度（W）。
- `patch_size`：预期的图像块尺寸。

## 执行步骤

1. 在模型中定位图像块嵌入卷积层（patch embedding conv）。报告其 `kernel_size`、`stride`、`in_channels` 和 `out_channels`。
2. 计算预期的图像块数量。对于正方形图像：`(image_size / patch_size)^2`。对于矩形图像：`(H / patch_size) * (W / patch_size)`。要求 `H % patch_size == 0` 且 `W % patch_size == 0`；若不满足则标记并拒绝执行。
3. 定位可学习的位置嵌入（learned positional embedding）。报告其形状 `(1, N, dim)`。
4. 将 `N` 与 `num_patches + 1`（包含分类标记（CLS token））或 `num_patches`（不包含分类标记）进行对比。若不匹配，则说明该检查点是在不同分辨率或图像块尺寸下预训练的。
5. 检查图像块卷积层的 `out_channels` 是否等于位置嵌入的 `dim`。
6. 若模型需要为新分辨率插值位置嵌入，请验证插值工具是否存在（大多数 `timm` ViT 会通过 `resize_pos_embed` 自动完成此操作）。

## 输出报告

[vit-inspector]
  image_size:         HxW
  patch_size:         <int>
  num_patches (computed): <int>
  patch_conv:         k=<int>  s=<int>  in=<int>  out=<int>
  pos_embed shape:    (1, N, dim)
  has CLS token:      yes | no
  pos_embed N:        <int>    expected: <int>
  verdict:            ok | mismatch

[if mismatch]
  action:  reinitialise pos_embed for new sequence length
  tool:    timm.models.vision_transformer.resize_pos_embed

## 规则

- 切勿在未发出警告的情况下静默执行插值；必须明确提示该操作，以便用户知晓预训练的位置结构可能已发生偏移。
- 若图像块尺寸不匹配，拒绝推荐插值方案——应直接更换为正确的架构。
- 不要尝试就地修复模型；仅负责报告问题并提供建议。
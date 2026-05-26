---
name: skill-segmentation-mask-inspector
description: 报告类别分布、预测掩码统计信息，以及最可能被预测不足或边界模糊的类别
version: 1.0.0
phase: 4
lesson: 7
tags: [计算机视觉, 图像分割, 调试, 评估]
---

# 分割掩码检查器 (Segmentation Mask Inspector)

用于诊断“损失函数（Loss）已下降”与“掩码（Mask）实际效果良好”之间差距的工具。

## 适用场景

- 训练结束后，当平均交并比（mIoU）指标看似良好，但目视检查结果却不尽如人意时。
- 部署前：检查预测结果的类别分布与真实标签（Ground Truth）是否平衡。
- 当大目标的类别交并比（IoU）较高，但小目标的 IoU 较低时。
- 调试边界伪影（Boundary Artefacts）时，这些伪影因像素数量较少而未在 IoU 指标中显现。

## 输入参数

- `preds`：形状为 (N, H, W) 的张量（Tensor），包含预测的类别 ID。
- `targets`：形状为 (N, H, W) 的张量，包含真实标签的类别 ID。
- `num_classes`：整数。
- 可选参数 `class_names`：包含 C 个字符串的列表。

## 执行步骤

1. **类别像素直方图。** 计算 `preds` 和 `targets` 中每个类别的像素占比。标记满足 `|pred% - gt%| / max(gt%, 1e-6) > 0.30`（相对偏差超过 30%）的任意类别。对于真实标签中不存在的类别（`gt% == 0`），直接标记预测占比超过 `0.3` 的类别。

2. **各类别交并比（IoU）** 与 **各类别边界 F1 分数（Boundary F1）**。边界 F1 的计算方式为：将每个掩码膨胀（Dilate）3 个像素，计算交集并进行评分。若某类别的 IoU > 0.7 但边界 F1 < 0.5，则表明其边缘存在模糊现象。

3. **小目标召回率（Recall）。** 将每个真实标签的连通域（Connected Component）按尺寸划分为不同区间（极小 < 100 像素，小 < 1000 像素，中 < 10000 像素，大 >= 10000 像素）。报告每个类别在各尺寸区间的召回率。若小目标召回率低于 0.3 而大目标召回率高于 0.9，则表明存在分辨率或感受野（Receptive Field）问题。

4. **混淆对（Confusion Pairs）。** 针对每个类别，找出其最常混淆的类别（即在真实标签掩码内，被错误预测为其他类别中出现频率最高的那个）。报告排名前 3 的混淆对。

5. **饱和度检查（需传入 `probs` 或 `logits`，而非仅 `preds`）。** 若调用方传入原始的逐像素概率分布 `probs: (N, C, H, W)`，则计算每个类别中满足 `probs.max(dim=1) > 0.99` 的像素比例。高饱和度（某类别超过 0.9 的像素）表明模型过度自信（Overconfidence）——可考虑引入标签平滑（Label Smoothing）或置信度校准（Calibration）。若仅有经过 `argmax` 处理的 `preds`，则跳过此步骤并在报告中注明。

## 报告格式

[mask-inspector]
  classes: C

[class distribution]
  name       gt %    pred %   delta
  ...

[metrics]
  class       IoU     bF1    recall_tiny  recall_small  recall_medium  recall_large
  ...

[confusion pairs]
  class A confused with class B: <N> pixels (most common)
  class B confused with class A: <N> pixels
  ...

[verdict]
  most impactful issue: <one sentence>

## 规则

- 按真实标签像素占比降序排列类别行，使出现频率最高的类别排在最前。
- 将 IoU < 0.4 或边界 F1 < 0.3 的类别标记为 `critical`（严重）。
- 当小目标召回率是主要失败原因时，建议：采用更高分辨率进行训练、在编码器（Encoder）最后阶段使用更小的步长（Stride），或使用特征金字塔解码器（Feature-Pyramid Decoder）。
- 当边界 F1 是主要失败原因时，建议：使用边界感知损失函数（Boundary-Aware Loss，如 Lovasz 或 BoundaryLoss）、结合水平翻转的测试时增强（TTA），以及无步长解码器。
- 切勿仅输出类别索引作为标识符；若提供了 `class_names`，请在每一行中均使用类别名称。
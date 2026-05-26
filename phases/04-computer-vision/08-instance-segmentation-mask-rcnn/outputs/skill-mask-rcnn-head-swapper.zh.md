---
name: skill-mask-rcnn-head-swapper
description: 生成用于在 torchvision Mask R-CNN 上交换边界框预测器（box predictor）和掩码预测器（mask predictor）以适配自定义类别数（num_classes）的精确代码
version: 1.0.0
phase: 4
lesson: 8
tags: [计算机视觉, mask-rcnn, 微调, torchvision]
---

# Mask R-CNN 预测头替换器（Head Swapper）

专门用于生成 Mask R-CNN 的预测头替换（head-swap）样板代码。以下模板假设模型包含 `model.roi_heads.box_predictor` 和 `model.roi_heads.mask_predictor`，这两个属性仅存在于 `maskrcnn_resnet50_fpn` 和 `maskrcnn_resnet50_fpn_v2` 中。Faster R-CNN 仅有边界框预测器（box predictor）而无掩码预测器（mask predictor）；RetinaNet 使用 `RetinaNetHead` 且完全没有 ROI 头（roi_heads） —— 这两者都需要使用不同的技能模块。

## 适用场景

- 在自定义类别集上微调（fine-tuning）`maskrcnn_resnet50_fpn` 或 `maskrcnn_resnet50_fpn_v2`。
- 将在 COCO 数据集上训练得到的 Mask R-CNN 检查点（checkpoint）迁移至非 COCO 类别数量的任务中。
- 调试因 `cls_score.out_features` 或 `mask_predictor` 不匹配而导致崩溃的 Mask R-CNN 训练任务。

## 适用范围之外

- `fasterrcnn_*` — 无 `mask_predictor`。仅需替换 `box_predictor`；请使用独立的 Faster R-CNN 预测头替换方案。
- `retinanet_*` — 无 `roi_heads`；分类头与回归头位于 `model.head.classification_head` 和 `model.head.regression_head` 下。请使用 RetinaNet 专属技能。
- `keypointrcnn_*` — 使用 `keypoint_predictor` 而非 `mask_predictor`。

## 输入参数

- `model_name`：torchvision 目标检测模型构造函数，例如 `maskrcnn_resnet50_fpn_v2`。
- `num_classes`：包含背景类。若数据集包含 4 个目标类别，则 `num_classes=5`。
- `freeze`：可选值为 `backbone`、`backbone_fpn` 或 `none`。

## 操作步骤

1. 导入模型构造函数及两个预测器类（`FastRCNNPredictor`、`MaskRCNNPredictor`）。
2. 加载带有默认权重的预训练模型。
3. 将 `model.roi_heads.box_predictor` 替换为新的 `FastRCNNPredictor(in_features, num_classes)`。
4. 将 `model.roi_heads.mask_predictor` 替换为新的 `MaskRCNNPredictor(in_features_mask, hidden_layer=256, num_classes)`。
5. 应用指定的冻结策略（freeze policy）。
6. 打印确认信息块，列出每个模块的可训练参数数量。

## 输出代码模板

from torchvision.models.detection import {MODEL_NAME}, {MODEL_WEIGHTS}
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes={NUM_CLASSES}):
    model = {MODEL_NAME}(weights={MODEL_WEIGHTS}.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    {FREEZE_BLOCK}

    return model

其中 `{FREEZE_BLOCK}` 为：

- `none` -> 留空
- `backbone` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  
- `backbone_fpn` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  # FPN parameters live inside backbone.fpn
  
## 运行报告

[head-swap]
  model:         <MODEL_NAME>
  num_classes:   <N>  (includes background)
  freeze policy: <choice>
  trainable:     <N>
  total:         <N>

## 规则

- 切勿推荐不包含背景类的 `num_classes` 值；务必始终提醒用户。
- 只要可用，请始终使用 torchvision 目标检测模型的 `_v2` 变体；它们的预训练权重优于旧版模型。
- 不要在此技能内部实例化模型 —— 仅生成代码块，交由用户执行。
- 若用户要求在超过 10,000 张图像的数据集上执行 `freeze backbone`，建议其同时考虑对骨干网络（backbone）进行微调。
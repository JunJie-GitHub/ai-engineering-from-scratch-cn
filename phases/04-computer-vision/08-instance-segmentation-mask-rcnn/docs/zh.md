# 实例分割（Instance Segmentation）— Mask R-CNN

> 在 Faster R-CNN 检测器上添加一个微小的掩码分支，即可实现实例分割。难点在于感兴趣区域对齐（RoIAlign），且其实现比表面看起来更为复杂。

**类型：** 构建与学习
**语言：** Python
**前置条件：** 第 4 阶段第 06 课（YOLO），第 4 阶段第 07 课（U-Net）
**时长：** 约 75 分钟

## 学习目标

- 端到端梳理 Mask R-CNN 架构：主干网络（Backbone）、特征金字塔网络（FPN）、区域提议网络（RPN）、感兴趣区域对齐（RoIAlign）、边界框头（Box Head）与掩码头（Mask Head）
- 从零实现 RoIAlign，并解释为何不再使用感兴趣区域池化（RoIPool）
- 使用 torchvision 的 `maskrcnn_resnet50_fpn_v2` 预训练模型生成生产级实例掩码，并正确解析其输出格式
- 在小规模自定义数据集上微调 Mask R-CNN：替换边界框头与掩码头，同时冻结主干网络

## 核心问题

语义分割（Semantic Segmentation）为每个类别生成单一掩码。而实例分割则为每个独立对象生成掩码，即使多个对象属于同一类别也不例外。个体计数、跨帧目标跟踪以及物理测量（例如墙体中每块砖的边界框、显微图像中每个细胞）均依赖于实例分割。

Mask R-CNN（He 等人，2017）通过将实例分割重新定义为“检测+掩码”任务解决了该问题。其设计极为优雅，以至于随后五年间几乎所有的实例分割研究都基于 Mask R-CNN 的变体展开。该模型的 torchvision 实现至今仍是中小规模数据集在生产环境中的默认方案。

工程上的核心难点在于特征采样：如何从角点未与像素边界对齐的候选框中，精准裁剪出固定大小的特征区域？一旦处理不当，会导致各项评估指标损失零点几个平均精度均值（mAP）分数。感兴趣区域对齐（RoIAlign）正是解决此问题的关键。

## 核心概念

### 架构

flowchart LR
    IMG["Input"] --> BB["ResNet<br/>backbone"]
    BB --> FPN["Feature<br/>Pyramid Network"]
    FPN --> RPN["Region<br/>Proposal<br/>Network"]
    FPN --> RA["RoIAlign"]
    RPN -->|"top-K proposals"| RA
    RA --> BH["Box head<br/>(class + refine)"]
    RA --> MH["Mask head<br/>(14x14 conv)"]
    BH --> NMS["NMS"]
    MH --> NMS
    NMS --> OUT["boxes +<br/>classes + masks"]

    style BB fill:#dbeafe,stroke:#2563eb
    style FPN fill:#fef3c7,stroke:#d97706
    style RPN fill:#fecaca,stroke:#dc2626
    style OUT fill:#dcfce7,stroke:#16a34a

理解该架构的五个关键部分：

1. **主干网络（Backbone）** — 在 ImageNet 上预训练的 ResNet-50 或 ResNet-101。生成步长（stride）为 4、8、16、32 的多层级特征图。
2. **特征金字塔网络（Feature Pyramid Network, FPN）** — 采用自顶向下（top-down）与横向连接（lateral connections），为每一层提供包含丰富语义信息的 C 通道特征。检测模块会根据目标尺寸查询对应的 FPN 层级。
3. **区域提议网络（Region Proposal Network, RPN）** — 一个小型卷积头，在每个锚框（anchor）位置预测“此处是否有目标？”以及“如何调整边界框？”。每张图像生成约 1000 个候选区域（proposal）。
4. **感兴趣区域对齐（RoIAlign）** — 从任意 FPN 层级的任意边界框中采样固定大小（如 7x7）的特征块。采用双线性插值（bilinear interpolation）采样，无量化操作。
5. **检测头（Heads）** — 包含一个两层边界框头，用于微调边界框并预测类别；以及一个小型卷积头，为每个候选区域输出 `28x28` 的二值掩码（mask）。

### 为什么使用感兴趣区域对齐（RoIAlign）而非感兴趣区域池化（RoIPool）

原始的 Fast R-CNN 使用了 RoIPool，它将候选框划分为网格，取每个单元格中的最大特征值，并将所有坐标舍入为整数。这种舍入操作会导致特征图与输入像素坐标之间产生最大达一个特征图像素的错位——在 224x224 的图像上影响尚小，但当特征图步长为 32 时，这种错位将是灾难性的。

RoIPool:
  box (34.7, 51.3, 98.2, 142.9)
  round -> (34, 51, 98, 142)
  split grid -> round each cell boundary
  misalignment accumulates at every step

RoIAlign:
  box (34.7, 51.3, 98.2, 142.9)
  sample at exact float coordinates using bilinear interpolation
  no rounding anywhere

RoIAlign 能在 COCO 数据集上免费提升掩码平均精度（mask Average Precision, mask AP）3-4 个点。如今所有注重定位精度的检测器都采用了它——无论是 YOLOv7 seg、RT-DETR 还是 Mask2Former。

### 一段话理解区域提议网络（RPN）

在特征图的每个位置上放置 K 个不同尺寸和形状的锚框（anchor boxes）。为每个锚框预测一个目标置信度分数（objectness score）以及回归偏移量，从而将锚框调整为更贴合目标的边界框。按分数保留前约 1000 个边界框，在交并比（Intersection over Union, IoU）阈值为 0.7 时应用非极大值抑制（Non-Maximum Suppression, NMS），并将筛选后的结果送入检测头。RPN 使用其独立的微型损失函数进行训练——其结构与第 6 课中的 YOLO 损失相同，仅包含两个类别（目标/非目标）。

### 掩码头（Mask Head）

对于每个候选区域（经过 RoIAlign 后），掩码头是一个微型全卷积网络（Fully Convolutional Network, FCN）：包含四个 3x3 卷积层、一个 2 倍反卷积层，以及一个最终的 1x1 卷积层，在 `28x28` 分辨率下输出 `num_classes` 个通道。仅保留与预测类别对应的通道，其余通道被忽略。这种设计将掩码预测与分类任务解耦。

将 28x28 的掩码上采样至候选区域的原始像素尺寸，即可生成最终的二值掩码。

### 损失函数（Losses）

Mask R-CNN 的损失函数由以下多项相加构成：

L = L_rpn_cls + L_rpn_box + L_box_cls + L_box_reg + L_mask

- `L_rpn_cls`、`L_rpn_box` — RPN 候选区域的目标置信度损失与边界框回归损失。
- `L_box_cls` — 检测头分类器在 (C+1) 个类别（包含背景）上的交叉熵损失。
- `L_box_reg` — 检测头边界框微调的平滑 L1 损失。
- `L_mask` — 28x28 掩码输出上的逐像素二值交叉熵损失。

每项损失都有其默认权重；`torchvision` 的实现将它们作为构造函数参数暴露出来。

### 输出格式

`torchvision.models.detection.maskrcnn_resnet50_fpn_v2` 返回一个字典列表，每张图像对应一个字典：

{
    "boxes":  (N, 4) in (x1, y1, x2, y2) pixel coordinates,
    "labels": (N,) class IDs, 0 = background so indices are 1-based,
    "scores": (N,) confidence scores,
    "masks":  (N, 1, H, W) float masks in [0, 1] — threshold at 0.5 for binary,
}

掩码已恢复至完整图像分辨率。28x28 的检测头输出已在内部完成上采样。

## 构建

### 步骤 1：从零实现感兴趣区域对齐 (RoIAlign)

这是掩码区域卷积神经网络 (Mask R-CNN) 中唯一一个通过代码比通过文字描述更容易理解的组件。

import torch
import torch.nn.functional as F

def roi_align_single(feature, box, output_size=7, spatial_scale=1 / 16.0):
    """
    feature: (C, H, W) single-image feature map
    box: (x1, y1, x2, y2) in original image pixel coordinates
    output_size: side of the output grid (7 for box head, 14 for mask head)
    spatial_scale: reciprocal of the feature map stride
    """
    C, H, W = feature.shape
    x1, y1, x2, y2 = [c * spatial_scale - 0.5 for c in box]
    bin_w = (x2 - x1) / output_size
    bin_h = (y2 - y1) / output_size

    grid_y = torch.linspace(y1 + bin_h / 2, y2 - bin_h / 2, output_size)
    grid_x = torch.linspace(x1 + bin_w / 2, x2 - bin_w / 2, output_size)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    gx = 2 * (xx + 0.5) / W - 1
    gy = 2 * (yy + 0.5) / H - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(feature.unsqueeze(0), grid, mode="bilinear",
                            align_corners=False)
    return sampled.squeeze(0)

每个数值均位于双线性插值 (bilinear interpolation) 采样位置。无舍入、无量化、无梯度丢失。

### 步骤 2：与 torchvision 的 RoIAlign 进行对比

from torchvision.ops import roi_align

feature = torch.randn(1, 16, 50, 50)
boxes = torch.tensor([[0, 10, 20, 100, 90]], dtype=torch.float32)  # (batch_idx, x1, y1, x2, y2)

ours = roi_align_single(feature[0], boxes[0, 1:].tolist(), output_size=7, spatial_scale=1/4)
theirs = roi_align(feature, boxes, output_size=(7, 7), spatial_scale=1/4, sampling_ratio=1, aligned=True)[0]

print(f"shape ours:   {tuple(ours.shape)}")
print(f"shape theirs: {tuple(theirs.shape)}")
print(f"max|diff|:    {(ours - theirs).abs().max().item():.3e}")

当设置 `sampling_ratio=1` 和 `aligned=True` 时，两者的输出差异在 `1e-5` 以内。

### 步骤 3：加载预训练 (pretrained) 的 Mask R-CNN 模型

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
print(f"classes (including background): {len(model.roi_heads.box_predictor.cls_score.out_features * [0])}")

包含 46M 个参数，91 个类别（COCO 数据集）。第一个类别（ID 0）为背景；模型实际检测到的所有目标均从 ID 1 开始。

### 步骤 4：执行推理 (inference)

with torch.no_grad():
    x = torch.randn(3, 400, 600)
    predictions = model([x])
p = predictions[0]
print(f"boxes:  {tuple(p['boxes'].shape)}")
print(f"labels: {tuple(p['labels'].shape)}")
print(f"scores: {tuple(p['scores'].shape)}")
print(f"masks:  {tuple(p['masks'].shape)}")

掩码张量 (mask tensor) 的形状为 `(N, 1, H, W)`。以 0.5 为阈值进行截断，即可获取每个目标的二值掩码：

binary_masks = (p['masks'] > 0.5).squeeze(1)  # (N, H, W) boolean

### 步骤 5：替换预测头以适配自定义类别数量

常见的微调 (fine-tuning) 方案：复用主干网络 (backbone)、特征金字塔网络 (Feature Pyramid Network, FPN) 和区域提议网络 (Region Proposal Network, RPN)；仅替换两个分类预测头 (classifier heads)。

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_custom_maskrcnn(num_classes):
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

custom = build_custom_maskrcnn(num_classes=5)
print(f"custom cls_score.out_features: {custom.roi_heads.box_predictor.cls_score.out_features}")

`num_classes` 必须包含背景类别，因此包含 4 个目标类别的数据集需设置 `num_classes=5`。

### 步骤 6：冻结无需训练的层

在小型数据集上，冻结主干网络和 FPN。仅训练 RPN 的目标置信度 (objectness) 与边界框回归 (regression) 参数，以及两个预测头。

def freeze_backbone_and_fpn(model):
    # torchvision Mask R-CNN packs the FPN inside `model.backbone` (as
    # `model.backbone.fpn`), so iterating `model.backbone.parameters()` covers
    # both the ResNet feature layers and the FPN lateral/output convs.
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model

custom = freeze_backbone_and_fpn(custom)
trainable = sum(p.numel() for p in custom.parameters() if p.requires_grad)
print(f"trainable after freeze: {trainable:,}")

对于仅含 500 张图像的数据集，这一操作直接决定了模型是能够收敛 (convergence) 还是陷入过拟合 (overfitting)。

## 使用它

`torchvision` 中掩码区域卷积神经网络 (Mask R-CNN) 的完整训练循环仅需 40 行代码，且在不同任务间基本无需实质性修改——只需更换数据集即可直接运行。

def train_step(model, images, targets, optimizer):
    model.train()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    return {k: v.item() for k, v in loss_dict.items()}

`targets` 列表必须包含针对每张图像的字典，其中需包含 `boxes`、`labels` 和 `masks`（格式为 `(num_instances, H, W)` 的二值张量）。在训练期间，模型会返回包含四项损失的字典；在评估期间，则返回预测结果列表。具体返回哪种格式由 `model.training` 标志位决定。

`pycocotools` 评估器会同时输出边界框和掩码的 mAP@IoU=0.5:0.95 指标；你需要结合这两个数值来判断性能瓶颈究竟出在边界框预测头还是掩码预测头。

## 交付物

本课时将产出以下内容：

- `outputs/prompt-instance-vs-semantic-router.md` — 一个提示词 (prompt)，通过提出三个问题来引导选择实例分割 (instance segmentation)、语义分割 (semantic segmentation) 或全景分割 (panoptic segmentation)，并指定初始使用的具体模型。
- `outputs/skill-mask-rcnn-head-swapper.md` — 一项技能 (skill)，在给定新的 `num_classes` 参数后，可自动生成用于替换任意 `torchvision` 检测模型预测头的 10 行代码。

## 练习

1. **(简单)** 在 100 个随机生成的边界框上，将你实现的感兴趣区域对齐 (RoIAlign) 与 `torchvision.ops.roi_align` 进行对比验证。报告最大绝对误差。同时运行感兴趣区域池化 (RoIPool)（2017 年之前的实现方式），并展示其在靠近图像边界的框上会产生约 1~2 个特征图 (feature-map) 像素的偏差。
2. **(中等)** 在包含 50 张图像的自定义数据集（任选两类：气球、鱼、坑洼或标志）上微调 `maskrcnn_resnet50_fpn_v2`。冻结骨干网络 (backbone)，训练 20 个轮次 (epoch)，并报告掩码 AP@0.5 指标。
3. **(困难)** 将 Mask R-CNN 的掩码预测头替换为输出 56x56 分辨率（而非默认的 28x28）的版本。分别测量替换前后的 mAP@IoU=0.75 指标。解释性能提升（或无提升）的原因，并说明其如何符合预期的边界精度与内存消耗之间的权衡 (trade-off)。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 掩码区域卷积神经网络 (Mask R-CNN) | “检测加掩码” | 快速区域卷积神经网络 (Faster R-CNN) + 一个小型全卷积网络 (FCN) 预测头，为每个候选区域 (proposal) 的每个类别预测一个 28x28 的掩码 |
| 特征金字塔网络 (FPN) | “特征金字塔” | 自顶向下 (Top-down) 与横向连接 (lateral connections) 相结合的结构，为每个步长层级提供包含丰富语义信息的 C 个通道特征 |
| 区域提议网络 (RPN) | “区域提议器” | 一个小型卷积预测头，为每张图像生成约 1000 个包含/不包含物体的候选区域 |
| 感兴趣区域对齐 (RoIAlign) | “无舍入裁剪” | 从任意浮点坐标的边界框中，通过双线性插值采样固定大小的特征网格 |
| 感兴趣区域池化 (RoIPool) | “2017 年前的裁剪方式” | 目的与 RoIAlign 相同，但会对边界框坐标进行取整舍入；目前已淘汰 |
| 掩码平均精度 (Mask AP) | “实例 mAP” | 使用掩码交并比 (IoU) 而非边界框交并比计算的平均精度；即 COCO 实例分割评估指标 |
| 二值掩码预测头 (Binary mask head) | “逐类掩码” | 为每个候选区域的每个类别预测一个二值掩码；最终仅保留预测类别对应的通道 |
| 背景类 (Background class) | “类别 0” | 兜底的“无物体”类别；真实类别的索引从 1 开始 |

## 延伸阅读

- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870) — 该论文；第 3 节关于感兴趣区域对齐（RoIAlign）的内容为必读部分
- [FPN: Feature Pyramid Networks (Lin et al., 2017)](https://arxiv.org/abs/1612.03144) — 特征金字塔网络（FPN）论文；现代所有目标检测器（Object Detector）均广泛采用该架构
- [torchvision Mask R-CNN 教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) — 微调循环（Fine-tuning Loop）的参考指南
- [Detectron2 模型库](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) — 提供几乎所有目标检测（Object Detection）与图像分割（Image Segmentation）变体的生产级实现及预训练权重（Pre-trained Weights）
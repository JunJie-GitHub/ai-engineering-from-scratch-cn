# 目标检测（Object Detection）—— 从零实现 YOLO

> 目标检测本质上是分类（Classification）与回归（Regression）的结合，在特征图（Feature Map）的每个位置上运行预测，最后通过非极大值抑制（Non-Maximum Suppression）进行结果清理。

**类型：** 构建实践
**编程语言：** Python
**前置知识：** 第4阶段第03课（卷积神经网络/CNNs）、第4阶段第04课（图像分类/Image Classification）、第4阶段第05课（迁移学习/Transfer Learning）
**预计耗时：** 约 75 分钟

## 学习目标

- 解释网格与锚框（Grid-and-Anchor）设计如何将检测任务转化为密集预测（Dense Prediction）问题，并阐明输出张量（Output Tensor）中每个数值的具体含义
- 计算边界框之间的交并比（Intersection-over-Union），并从头实现非极大值抑制（Non-Maximum Suppression）
- 在预训练骨干网络（Pretrained Backbone）之上构建一个最小化的 YOLO 风格检测头（Detection Head），包含分类损失、目标置信度损失以及边界框回归损失
- 解读检测指标数据（如 precision@0.5、recall、mAP@0.5、mAP@0.5:0.95），并据此确定下一步的调优方向

## 问题背景

图像分类（Image Classification）回答的是“这张图里有一只狗”。而目标检测回答的是“在像素坐标 (112, 40, 280, 210) 处有一只狗，在 (400, 180, 560, 310) 处有一只猫，画面中没有其他目标。”这一结构上的转变——从为整张图像预测单一标签，变为预测数量不定的带标签边界框（Bounding Box）——正是所有自动驾驶系统、安防监控产品、文档版面解析工具以及工业视觉产线所依赖的核心技术。

目标检测也是计算机视觉中所有工程权衡（Engineering Trade-off）集中体现的领域。你需要精准的边界框（回归头/Regression Head），需要为每个框分配正确的类别（分类头/Classification Head），需要模型能够判断何时画面中不存在目标（目标置信度/Objectness Score），还需要确保每个真实物体仅对应一次预测（非极大值抑制/Non-Maximum Suppression）。任何一环的缺失都会导致流水线出现漏检、产生幻觉框（Hallucinated Boxes），或者对同一物体在略微不同的位置上重复预测十几次。

YOLO（You Only Look Once，Redmon 等人于 2016 年提出）通过让卷积网络（Convolutional Network）仅执行一次前向传播（Forward Pass）即可实现上述所有功能，从而让实时检测成为可能。相同的结构设计至今仍是现代检测器（如 YOLOv8、YOLOv9、YOLO-NAS、RT-DETR）的基石。只要掌握了核心原理，所有变体模型都不过是相同组件的重新组合。

## 核心概念

### 检测作为密集预测 (Dense prediction)

分类器对每张图像输出 C 个数值。YOLO 风格的检测器对每张图像输出 `(S x S x (5 + C))` 个数值，其中 S 是空间网格 (Spatial grid) 的大小。

flowchart LR
    IMG["Input 416x416 RGB"] --> BB["Backbone<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["Feature map<br/>(C_feat, 13, 13)"]
    FM --> HEAD["Detection head<br/>(1x1 convs)"]
    HEAD --> OUT["Output tensor<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["Decode<br/>(grid + sigmoid + exp)"]
    DEC --> NMS["Non-max suppression"]
    NMS --> RESULT["Final boxes"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a

每个 `S * S` 网格单元 (Grid cell) 预测 `B` 个边界框 (Box)。对于每个边界框：

- 4 个数值描述几何信息：`tx, ty, tw, th`。
- 1 个数值是目标存在性得分 (Objectness score)：“该单元中心是否存在目标？”
- C 个数值是类别概率。

每个单元的总输出：`B * (5 + C)`。以 VOC 数据集为例，当 `S=13, B=2, C=20` 时，每个单元对应 50 个数值。

### 为什么使用网格和锚框 (Grids and Anchors)

直接回归 (Plain regression) 会为每个目标预测绝对坐标 `(x, y, w, h)`。这对卷积网络 (Convolutional network) 而言较为困难，因为图像的平移不应导致所有预测值产生等量的偏移——每个目标在空间上都有其固定的锚点。网格机制通过将每个真实边界框 (Ground-truth box) 分配给其中心点落入的网格单元来解决此问题；仅由该单元负责预测该目标。

锚框 (Anchor) 解决了第二个问题。一个 3x3 卷积核很难从感受野 (Receptive field) 仅为 16 像素的特征单元中直接回归出一个 500 像素宽的边界框。相反，我们为每个单元预定义 `B` 个先验框形状（即锚框），并预测相对于每个锚框的微小偏移量 (Delta)。模型学习的是选择合适的锚框并对其进行微调，而不是从零开始回归。

Anchor box priors (example for 416x416 input):

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

At each grid cell, every anchor emits (tx, ty, tw, th, obj, c_1, ..., c_C).

现代检测器通常使用特征金字塔网络 (Feature Pyramid Network, FPN)，并在不同分辨率层级使用不同的锚框集合——浅层高分辨率特征图使用小锚框，深层低分辨率特征图使用大锚框。核心思想相同，只是增加了尺度。

### 解码预测结果

原始的 `tx, ty, tw, th` 并非边界框坐标；它们是需要在绘制前进行转换的回归目标：

centre x  = (sigmoid(tx) + cell_x) * stride
centre y  = (sigmoid(ty) + cell_y) * stride
width     = anchor_w * exp(tw)
height    = anchor_h * exp(th)

`sigmoid` 函数确保中心偏移量限制在网格单元内部。`exp` 函数允许宽度相对于锚框自由缩放，且不会出现符号翻转。`stride`（步长）将网格坐标重新缩放回像素级别。自 v2 版本以来，每个 YOLO 版本都采用相同的解码步骤。

### 交并比 (Intersection over Union, IoU)

检测任务中衡量两个边界框相似度的通用指标：

IoU(A, B) = area(A intersect B) / area(A union B)

IoU = 1 表示完全重合；IoU = 0 表示无重叠。预测框与真实框之间的 IoU 决定了该预测是否被计为真正例 (True positive)（通常阈值为 IoU >= 0.5）。两个预测框之间的 IoU 则是非极大值抑制 (Non-maximum suppression, NMS) 用于去重的依据。

### 非极大值抑制 (Non-maximum suppression, NMS)

在相邻锚框上训练的卷积网络通常会为同一目标预测出重叠的边界框。NMS 会保留置信度最高的预测框，并删除与已选框 IoU 超过阈值的其他所有预测框。

NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep

典型阈值：目标检测中通常设为 0.45。近期的检测器使用 `soft-NMS`、`DIoU-NMS` 替代标准 NMS，或直接学习抑制过程（如 RT-DETR），但其结构目的保持一致。

### 损失函数 (Loss function)

YOLO 的损失函数由三项加权损失相加构成：

L = lambda_coord * L_box(pred, target, where obj=1)
  + lambda_obj   * L_obj(pred, 1,     where obj=1)
  + lambda_noobj * L_obj(pred, 0,     where obj=0)
  + lambda_cls   * L_cls(pred, target, where obj=1)

仅包含目标的单元才会对边界框回归损失和分类损失产生贡献。不包含目标的单元仅对目标存在性损失 (Objectness loss) 产生贡献（即教导模型在无目标时保持“沉默”）。`lambda_noobj` 通常设置得较小（约 0.5），因为绝大多数单元是空的，否则它们将主导总损失。

现代变体将均方误差 (Mean Squared Error, MSE) 边界框损失替换为 CIoU / DIoU（直接优化 IoU），使用焦点损失 (Focal loss) 处理类别不平衡，并使用质量焦点损失 (Quality focal loss) 平衡目标存在性。三项损失的基本结构保持不变。

### 检测评估指标 (Detection metrics)

准确率 (Accuracy) 不适用于检测任务。以下四个指标才是适用的：

- **精确率 (Precision)@IoU=0.5** — 在被计为正例的预测中，实际正确的比例。
- **召回率 (Recall)@IoU=0.5** — 在所有真实目标中，我们成功找到的比例。
- **平均精度 (Average Precision, AP)@0.5** — IoU 阈值为 0.5 时的精确率-召回率曲线面积；每个类别对应一个数值。
- **平均精度均值 (Mean Average Precision, mAP)@0.5:0.95** — 在 IoU 阈值 0.5, 0.55, ..., 0.95 上计算 AP 的平均值。这是 COCO 数据集的评估指标，最为严格且信息量最大。

应同时报告这四个指标。若检测器在 mAP@0.5 上表现强劲但在 mAP@0.5:0.95 上较弱，说明其定位较为粗略但不够紧密；可通过改进边界框回归损失来修复。若检测器精确率高但召回率低，则说明其过于保守；可降低置信度阈值或提高目标存在性权重。

## 构建

### 步骤 1：交并比 (Intersection over Union, IoU)

本课程的核心函数。用于处理两个格式为 `(x1, y1, x2, y2)` 的边界框 (bounding box) 数组。

import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)

返回一个形状为 `(N_a, N_b)` 的成对 IoU 矩阵。若需与单个真实框 (ground-truth box) 进行计算，只需将其中一个数组的形状设为 `(1, 4)` 即可。

### 步骤 2：非极大值抑制 (Non-Maximum Suppression, NMS)

def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)

该算法是确定性的，排序步骤的时间复杂度为 `O(N log N)`，且在相同输入下，其行为与 `torchvision.ops.nms` 完全一致。

### 步骤 3：边界框编码与解码 (Box Encoding and Decoding)

用于在像素坐标与网络实际回归的目标值 `(tx, ty, tw, th)` 之间进行转换。

def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

测试：对边界框进行编码后再解码——你应该能得到与原始值非常接近的结果（注：当 `tx` 不在 sigmoid 函数的输出范围内时，其反函数并非完全可逆，因此可能存在微小误差）。

### 步骤 4：极简 YOLO 检测头 (YOLO Head)

在特征图 (feature map) 上应用一个 1x1 卷积层，并将其重塑为 `(B, S, S, num_anchors, 5 + C)` 的形状。

import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y

输出形状：`(N, H, W, num_anchors, 5 + C)`。最后一个维度依次存储 `[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`。

### 步骤 5：真实框分配 (Ground-Truth Assignment)

为每个真实框 (ground-truth box) 确定负责预测它的 `(网格单元, 锚框)` 组合。

def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj

锚框 (anchor) 的选择依据是“与真实框的形状 IoU 最大”——这是一种计算开销较低的简易替代方案，与 YOLOv2/v3 的分配策略一致。v5 及后续版本采用了更复杂的策略（如任务对齐匹配 task-aligned matching、动态 k 值 dynamic k），它们都是对这一核心思想的进一步优化。

### 步骤 6：三项损失函数 (Loss Functions)

def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}

这里涉及五个超参数 (hyper-parameters)，在各类 YOLO 教程中通常会被硬编码或进行参数扫描。这些参数的比例至关重要：`lambda_coord=5, lambda_noobj=0.5` 沿用了原始 YOLOv1 论文中的设定，至今仍是一个合理的默认值。

### 步骤 7：推理流程 (Inference Pipeline)

对检测头的原始输出进行解码，应用 sigmoid/exp 激活函数，根据目标置信度 (objectness) 进行阈值过滤，最后执行非极大值抑制 (NMS)。

def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]

这就是完整的评估/推理路径：检测头输出 -> 解码 -> 阈值过滤 -> NMS。

## 上手使用

`torchvision.models.detection` 模块提供了具有相同概念结构的生产级目标检测器（production detectors）。加载预训练模型仅需三行代码。

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")

对于实时推理流水线（real-time inference pipelines），`ultralytics`（YOLOv8/v9）是行业标准：`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`。该模型在内部处理解码和非极大值抑制（Non-Maximum Suppression, NMS），并返回与上文构建的相同的 `boxes / scores / labels` 三元组。

## 交付产出

本课程的产出包括：

- `outputs/prompt-detection-metric-reader.md` — 一个提示词（prompt），可将包含精确率（precision）、召回率（recall）、平均精度（AP）和 mAP@0.5:0.95 的数据行转化为单行诊断结论，并给出最具价值的下一步实验建议。
- `outputs/skill-anchor-designer.md` — 一项技能脚本，在给定包含真实边界框（ground-truth boxes）的数据集后，会对宽高 `(w, h)` 执行 K-Means 聚类算法，并返回每个特征金字塔网络（Feature Pyramid Network, FPN）层级的锚框（anchor）集合，以及用于选择合适锚框数量的覆盖率统计信息。

## 练习

1. **(简单)** 实现 `box_iou` 函数，并在 1,000 对随机边界框上将其与 `torchvision.ops.box_iou` 进行对比测试。验证最大绝对误差是否低于 `1e-6`。
2. **(中等)** 将 `yolo_loss` 移植为使用 `CIoU` 边界框损失（box loss）替代均方误差（Mean Squared Error, MSE）的版本。在一个包含 100 张图像的合成数据集上证明，在相同的训练轮数（epochs）下，`CIoU` 能收敛到比 `MSE` 更优的最终 mAP@0.5:0.95。
3. **(困难)** 实现多尺度推理（multi-scale inference）：将同一张图像以三种分辨率输入模型，合并边界框预测结果，并在最后执行一次非极大值抑制（NMS）。在预留测试集（held-out set）上测量其相较于单尺度推理的 mAP 提升幅度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 锚框 (Anchor) | “先验框” | 在每个网格单元上预定义的框形状，网络基于它预测偏移量而非绝对坐标 |
| 交并比 (IoU) | “重叠度” | 两个框的交集与并集之比；目标检测中通用的相似度度量标准 |
| 非极大值抑制 (NMS) | “去重” | 一种贪心算法，保留得分最高的预测框，并移除与它重叠度超过阈值的其余预测框 |
| 目标存在性 (Objectness) | “这里有没有东西” | 针对每个锚框和网格单元的标量值，用于预测该单元中心是否包含目标 |
| 网格步长 (Grid stride) | “下采样因子” | 每个网格单元对应的像素数；例如 416 像素的输入配合 13 网格的检测头，其步长为 32 |
| 平均精度均值 (mAP) | “平均平均精度” | 精确率-召回率曲线下面积的平均值，通常对所有类别以及（针对 COCO 数据集）不同的 IoU 阈值进行平均 |
| AP@0.5 | “PASCAL VOC AP” | IoU 阈值为 0.5 时的平均精度；该指标较为宽松 |
| mAP@0.5:0.95 | “COCO AP” | 在 IoU 阈值 0.5 到 0.95 之间（步长 0.05）取平均；该指标更为严格，也是当前社区的通用标准 |

## 延伸阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) — 奠基之作；此后所有的 YOLO 版本均是在此架构基础上的改进
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) — 引入了多尺度特征金字塔网络 (FPN) 风格检测头的论文；其架构图至今仍是同类中最清晰的
- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com) — 当前的工业级参考指南；涵盖数据集格式、数据增强与训练策略
- [目标检测图解指南 (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) — 用最通俗的语言全面梳理了各类检测器；对于理解 DETR、RetinaNet、FCOS 与 YOLO 之间的关联极具价值
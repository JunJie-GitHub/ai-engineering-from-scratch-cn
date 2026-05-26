# 关键点检测（Keypoint Detection）与姿态估计（Pose Estimation）

> 姿态（Pose）是一组有序的关键点（Keypoint）。关键点检测器本质上是一个热图回归器（Heatmap Regressor）。其余工作主要是数据记录与流程管理。

**Type:** 构建实践
**Languages:** Python
**Prerequisites:** 第4阶段 第06课（目标检测），第4阶段 第07课（U-Net）
**Time:** 约45分钟

## 学习目标

- 区分自顶向下（Top-down）与自底向上（Bottom-up）的姿态估计方法，并说明各自的适用场景
- 使用以每个关键点为中心的高斯分布（Gaussian）作为目标，回归 K 个关键点的热图（Heatmap），并在推理（Inference）阶段提取关键点坐标
- 解释部件亲和场（Part Affinity Fields, PAFs）的概念，以及自底向上流水线如何将关键点关联为独立实例
- 使用 MediaPipe Pose 或 MMPose 进行生产环境的关键点估计，并理解其输出格式

## 问题背景

关键点任务在不同领域有着不同的名称：人体姿态（17个身体关节）、面部特征点（68或478个点）、手部（21个点）、动物姿态、机器人物体姿态、医学解剖标志点等。但它们都遵循相同的底层结构：在目标上检测 K 个离散点，并输出它们的 (x, y) 坐标。

姿态估计是动作捕捉、健身应用、体育分析、手势控制、动画制作、AR虚拟试穿以及机器人抓取等技术的基石。目前，2D姿态估计技术已相当成熟；而3D姿态估计（从单摄像头图像中推断关节在世界坐标系中的位置）则是当前的研究前沿。

工程层面的核心挑战在于规模。单张图像、单人姿态估计是一个耗时约 20ms 的任务；而在人群场景中以 30 fps 进行多人姿态估计，则是一个截然不同的问题，需要采用不同的网络架构。

## 核心概念

### 自顶向下（Top-down）与自底向上（Bottom-up）

flowchart LR
    subgraph TD["Top-down pipeline"]
        A1["Detect person boxes"] --> A2["Crop each box"]
        A2 --> A3["Per-box keypoint model<br/>(HRNet, ViTPose)"]
    end
    subgraph BU["Bottom-up pipeline"]
        B1["One pass over image"] --> B2["All keypoint heatmaps<br/>+ association field"]
        B2 --> B3["Group keypoints into<br/>instances (greedy matching)"]
    end

    style TD fill:#dbeafe,stroke:#2563eb
    style BU fill:#fef3c7,stroke:#d97706

- **自顶向下**——先检测人体边界框，然后在每个裁剪区域上运行单人关键点模型。精度最高；计算复杂度随人数呈线性增长。
- **自底向上**——单次前向传播预测所有关键点及关联场（Association Field），随后进行分组。无论人群规模大小，推理时间基本恒定。

自顶向下方法（如 HRNet、ViTPose）在精度上处于领先地位；而自底向上方法（如 OpenPose、HigherHRNet）则在拥挤场景下的吞吐量（Throughput）表现更优。

### 热图回归（Heatmap Regression）

不直接回归 `(x, y)` 坐标，而是为每个关键点预测一个 `H x W` 的热图（Heatmap），其中心以真实位置为基准生成高斯斑点（Gaussian Blob）。

target[k, y, x] = exp(-((x - cx_k)^2 + (y - cy_k)^2) / (2 sigma^2))

在推理阶段，每个热图的 `argmax`（最大值索引）即为预测的关键点位置。

热图为何优于直接回归：网络的空间结构（卷积特征图）与空间输出天然对齐。此外，高斯目标具有正则化（Regularisation）作用——微小的定位误差只会产生较小的损失，而非直接归零。

### 亚像素定位（Sub-pixel Localisation）

`argmax` 仅能提供整数坐标。为达到亚像素精度，可通过对 `argmax` 及其邻域拟合抛物线进行细化，或使用经典的偏移量公式 `(dx, dy) = 0.25 * (heatmap[y, x+1] - heatmap[y, x-1], ...)` 进行方向修正。

### 部件亲和场（Part Affinity Fields, PAFs）

OpenPose 用于自底向上关联的核心技巧。对于每一对相连的关键点（例如左肩到左肘），预测一个双通道场，用于编码从一点指向另一点的单位向量。在将肩部与肘部进行关联时，沿候选点对连线对 PAF 进行线积分；积分值最高的点对即为匹配结果。

For each connection (limb):
  PAF channels: 2 (unit vector x, y)
  Line integral: sum over sample points of (PAF . line_direction)
  Higher integral = stronger match

该方法设计优雅，且无需对单人进行裁剪即可扩展至任意规模的人群。

### COCO 关键点（COCO Keypoints）

标准人体姿态数据集：每人包含 17 个关键点，评估指标为 PCK（正确关键点百分比，Percentage of Correct Keypoints）和 OKS（目标关键点相似度，Object Keypoint Similarity）。OKS 是关键点版本的 IoU（交并比，Intersection over Union），也是 COCO mAP@OKS 报告的核心指标。

### 2D 与 3D 姿态

- **2D 姿态**——基于图像坐标；已达到工业级生产标准（如 MediaPipe、HRNet、ViTPose）。
- **3D 姿态**——基于世界/相机坐标；仍是活跃的研究领域。常见方法包括：
  - 使用小型多层感知机（Multilayer Perceptron, MLP）将 2D 预测结果升维至 3D（如 VideoPose3D）。
  - 直接从图像进行 3D 回归（如 PyMAF、MHFormer）。
  - 采用多视角设置（如 CMU Panoptic）获取真实标注（Ground Truth）。

## 动手实践（Build It）

### 步骤 1：高斯热力图目标 (Gaussian Heatmap Target)

import numpy as np
import torch

def gaussian_heatmap(size, cx, cy, sigma=2.0):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)

hm = gaussian_heatmap(64, 32, 32, sigma=2.0)
print(f"peak: {hm.max():.3f} at ({hm.argmax() % 64}, {hm.argmax() // 64})")

将每个关键点的热力图沿通道轴堆叠，即可构建出完整的目标张量 (Target Tensor)。

### 步骤 2：轻量级关键点检测头 (Keypoint Head)

一种采用 U-Net 风格的模型，用于输出 K 个热力图通道。

import torch.nn as nn
import torch.nn.functional as F

class TinyKeypointNet(nn.Module):
    def __init__(self, num_keypoints=4, base=16):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, base, 3, 2, 1), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.up2 = nn.ConvTranspose2d(base, num_keypoints, 2, 2)

    def forward(self, x):
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.mid(h2)
        u1 = self.up1(h3)
        return self.up2(u1)

输入维度为 `(N, 3, H, W)`，输出维度为 `(N, K, H, W)`。损失函数为针对高斯目标的逐像素均方误差 (Mean Squared Error, MSE)。

### 步骤 3：推理阶段——提取关键点坐标

def heatmap_to_coords(heatmaps):
    """
    heatmaps: (N, K, H, W)
    returns:  (N, K, 2) float coordinates in image pixels
    """
    N, K, H, W = heatmaps.shape
    hm = heatmaps.reshape(N, K, -1)
    idx = hm.argmax(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    return torch.stack([xs, ys], dim=-1)

coords = heatmap_to_coords(torch.randn(2, 4, 32, 32))
print(f"coords: {coords.shape}")  # (2, 4, 2)

推理时仅需一行代码。若需亚像素级 (Sub-pixel) 精度优化，可在最大值索引 (Argmax) 邻域内进行插值计算。

### 步骤 4：合成关键点数据集

实现很简单：在白色画布上绘制四个点，并训练模型预测其位置。

def make_synthetic_sample(size=64):
    img = np.ones((3, size, size), dtype=np.float32)
    rng = np.random.default_rng()
    kps = rng.integers(8, size - 8, size=(4, 2))
    for cx, cy in kps:
        img[:, cy - 2:cy + 2, cx - 2:cx + 2] = 0.0
    hms = np.stack([gaussian_heatmap(size, cx, cy) for cx, cy in kps])
    return img, hms, kps

该任务足够简单，轻量级模型通常在一分钟内即可学会。

### 步骤 5：模型训练

model = TinyKeypointNet(num_keypoints=4)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(200):
    batch = [make_synthetic_sample() for _ in range(16)]
    imgs = torch.from_numpy(np.stack([b[0] for b in batch]))
    hms = torch.from_numpy(np.stack([b[1] for b in batch]))
    pred = model(imgs)
    # Upsample pred to full resolution
    pred = F.interpolate(pred, size=hms.shape[-2:], mode="bilinear", align_corners=False)
    loss = F.mse_loss(pred, hms)
    opt.zero_grad(); loss.backward(); opt.step()


## 使用它

- **MediaPipe Pose** — Google 的生产级姿态估计器 (Pose Estimator)；提供 WebGL 与移动端运行时，延迟低于 10 毫秒。
- **MMPose** (OpenMMLab) — 全面的研究代码库；涵盖所有最先进 (State-of-the-Art, SOTA) 架构及预训练权重。
- **YOLOv8-pose** — 通过单次前向传播实现最快的实时多人姿态估计。
- **transformers HumanDPT / PoseAnything** — 较新的视觉-语言 (Vision-Language) 方法，用于开放词汇姿态估计 (Open-Vocabulary Pose)（支持任意对象、任意关键点集）。

## 交付成果

本课程的产出包括：

- `outputs/prompt-pose-stack-picker.md` — 一个提示词 (Prompt)，可根据延迟要求、人群规模以及 2D 与 3D 需求，自动选择 MediaPipe / YOLOv8-pose / HRNet / ViTPose。
- `outputs/skill-heatmap-to-coords.md` — 一个技能 (Skill)，用于编写所有生产级姿态模型均使用的亚像素热图转坐标 (Heatmap-to-Coordinate) 例程。

## 练习

1. **（简单）** 在合成的 4 点数据集上训练微型关键点模型 (Keypoint Model)。报告 200 步训练后预测关键点与真实关键点之间的平均 L2 误差 (L2 Error)。
2. **（中等）** 添加亚像素优化 (Sub-pixel Refinement)：基于最大值索引 (Argmax) 位置，利用相邻像素在 x 和 y 方向上分别拟合一维抛物线。报告相较于整数最大值定位的精度提升。
3. **（困难）** 构建一个双人合成数据集，其中每张图像包含两个 4 关键点模式的实例。训练一个基于部件亲和场 (Part Affinity Fields, PAFs) 的自底向上 (Bottom-up) 流水线，用于预测关键点所属的实例，并评估对象关键点相似度 (Object Keypoint Similarity, OKS)。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 关键点 (Keypoint) | “地标/标志点” | 对象上具有特定顺序的点（如关节、角点、特征点） |
| 姿态 (Pose) | “骨架” | 属于同一实例的一组有序关键点 |
| 自顶向下 (Top-down) | “先检测，后姿态估计” | 两阶段流水线：人体检测器 + 逐裁剪区域关键点模型；精度最高 |
| 自底向上 (Bottom-up) | “先姿态估计，后分组” | 单次前向传播预测所有关键点并进行分组；计算时间不随人群规模增加而变化 |
| 热图 (Heatmap) | “高斯目标” | 每个关键点对应一个 H x W 张量，峰值位于真实位置；首选的回归目标 |
| 部件亲和场 (PAF) | “部件亲和场” | 编码肢体方向的 2 通道单位向量场；用于将关键点分组到对应实例 |
| 对象关键点相似度 (OKS) | “关键点 IoU” | Object Keypoint Similarity；COCO 数据集用于姿态评估的指标 |
| 高分辨率网络 (HRNet) | “高分辨率网络” | 主流的自顶向下关键点架构；在整个网络中保持高分辨率特征 |

## 扩展阅读

- [OpenPose (Cao et al., 2017)](https://arxiv.org/abs/1812.08008) — 基于 PAFs 的自底向上方法；仍是该思路最详尽的阐述
- [HRNet (Sun et al., 2019)](https://arxiv.org/abs/1902.09212) — 自顶向下方法的参考架构
- [ViTPose (Xu et al., 2022)](https://arxiv.org/abs/2204.12484) — 使用纯视觉 Transformer (Vision Transformer, ViT) 作为姿态骨干网络；在多项基准测试中达到当前最先进水平 (SOTA)
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) — 生产级实时姿态估计；2026 年部署最快的技术栈
# 单目深度与几何估计

> 深度图（Depth Map）是一种单通道图像，其中每个像素值代表该点到相机的距离。过去，若不依赖立体视觉（Stereo Vision）或激光雷达（LiDAR），仅凭单张 RGB 帧（RGB Frame）预测深度几乎是不可能的。到了 2026 年，采用冻结的视觉 Transformer 编码器（ViT Encoder）搭配轻量级预测头（Lightweight Head），其预测结果与基准真值（Ground Truth）的误差已控制在百分之几以内。

**类型：** 构建与使用
**编程语言：** Python
**前置知识：** 第 4 阶段 第 14 课（ViT）、第 4 阶段 第 17 课（自监督视觉）、第 4 阶段 第 07 课（U-Net）
**预计耗时：** 约 60 分钟

## 学习目标

- 区分相对深度（Relative Depth）与度量深度（Metric Depth），并说明各主流生产级模型（MiDaS、Marigold、Depth Anything V3、ZoeDepth）分别解决的是哪一种
- 使用 Depth Anything V3（基于 DINOv2 骨干网络 DINOv2 Backbone）对任意单张图像进行深度预测，且无需相机标定（Camera Calibration）
- 解释单目深度估计为何仅凭单张图像即可生效（透视线索 Perspective Cues、纹理梯度 Texture Gradients、学习先验 Learned Priors），以及它无法恢复哪些信息（绝对尺度 Absolute Scale、遮挡几何 Occluded Geometry）
- 结合深度图与针孔相机内参（Pinhole Camera Intrinsics），将 2D 检测目标提升（Lift）为 3D 空间点

## 问题背景

深度是二维计算机视觉（2D Computer Vision）中缺失的维度。给定 RGB 图像，你只能获知物体在图像平面（Image Plane）上的投影位置，却无法判断其实际距离。深度传感器（如立体相机阵列 Stereo Rigs、激光雷达 LiDAR、飞行时间传感器 Time-of-Flight）能够直接解决该问题，但往往成本高昂、易损且探测范围受限。

单目深度估计（Monocular Depth Estimation）——即从单张 RGB 图像预测深度——过去往往只能输出模糊且不可靠的结果。到了 2026 年，大规模预训练编码器（Pretrained Encoders）彻底改变了这一现状：Depth Anything V3 采用冻结的 DINOv2 骨干网络，其生成的深度图能够泛化（Generalise）至室内、室外、医疗及卫星遥感等多个领域。Marigold 将深度预测重新构建为条件扩散（Conditional Diffusion）问题。ZoeDepth 则通过回归（Regression）直接输出真实的度量距离。

深度同样是连接二维目标检测与三维场景理解的桥梁：将检测框的像素坐标与深度值相乘，即可将 2D 目标提升为三维点云（3D Point Cloud）。这正是所有增强现实遮挡系统（AR Occlusion System）、避障流水线（Obstacle-Avoidance Pipeline）以及各类“抓取物体”机器人的核心基础。

## 核心概念

### 相对深度 (Relative Depth) 与度量深度 (Metric Depth)

- **相对深度 (Relative Depth)** —— 具有顺序关系的 `z` 值，但缺乏现实世界中的物理单位。“像素 A 比像素 B 更近，但两者距离的比例并未以米为基准进行标定。”
- **度量深度 (Metric Depth)** —— 距离摄像头的绝对物理距离（以米为单位）。要求模型学习图像线索与实际距离之间的统计关系。

MiDaS 和 Depth Anything V3 输出相对深度。Marigold 同样输出相对深度。ZoeDepth、UniDepth 和 Metric3D 输出度量深度。度量模型对相机内参 (Camera Intrinsics) 敏感，而相对模型则不敏感。

### 编码器-解码器模式 (Encoder-Decoder Pattern)

flowchart LR
    IMG["Image (H x W x 3)"] --> ENC["Frozen ViT encoder<br/>(DINOv2 / DINOv3)"]
    ENC --> FEATS["Dense features<br/>(H/14, W/14, d)"]
    FEATS --> DEC["Depth decoder<br/>(conv upsampler,<br/>DPT-style)"]
    DEC --> DEPTH["Depth map<br/>(H, W, 1)"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style DEC fill:#fef3c7,stroke:#d97706
    style DEPTH fill:#dcfce7,stroke:#16a34a

Depth Anything V3 冻结了编码器，仅训练 DPT 风格 (DPT-style) 的解码器。编码器提供丰富的特征表示；解码器将这些特征上采样插值回原始图像分辨率，并回归出深度值。

### 为何单张图像能够生成深度信息

二维图像包含许多与深度相关的单目线索 (Monocular Cues)：

- **透视 (Perspective)** —— 三维空间中的平行线在二维图像中会汇聚。
- **纹理梯度 (Texture Gradient)** —— 距离较远的表面纹理更小、更密集。
- **遮挡顺序 (Occlusion Order)** —— 较近的物体会遮挡较远的物体。
- **尺寸恒常性 (Size Constancy)** —— 已知物体（如汽车、人类）可提供大致的尺度参考。
- **大气透视 (Atmospheric Perspective)** —— 在户外场景中，远处的物体看起来更模糊且偏蓝。

在数十亿张图像上训练的视觉 Transformer (Vision Transformer, ViT) 能够内化这些线索。凭借充足的数据和强大的骨干网络，单目深度估计 (Monocular Depth Estimation) 无需任何显式的 3D 监督即可达到合理的精度。

### 单目深度估计的局限性

- **绝对度量尺度 (Absolute Metric Scale)**：在缺乏相机内参或场景中已知参照物的情况下无法获取。网络可以预测“杯子距离是勺子的两倍”，但无法判断杯子实际距离是 1 米还是 10 米。
- **被遮挡的几何结构 (Occluded Geometry)**：例如椅子的背面不可见，无法被可靠推断。
- **完全无纹理或高反射表面**：如镜子、玻璃、纯色墙壁。网络会输出看似合理但实际错误的深度值。

### 2026 年的 Depth Anything V3

- 使用原生 DINOv2 ViT-L/14 作为编码器（已冻结）。
- DPT 解码器。
- 使用来自多种来源的带位姿图像对进行训练（除光度一致性外，无需显式的深度监督）。
- 能够从**任意数量的视觉输入中预测空间一致的几何结构，无论是否已知相机位姿**。
- 在单目深度估计、任意视角几何重建、视觉渲染和相机位姿估计任务中均达到最先进水平 (State-of-the-Art, SOTA)。

这是 2026 年需要深度估计时可直接调用的即插即用模型。

### Marigold —— 基于扩散模型的深度估计

Marigold（Ke 等人，CVPR 2024）将深度估计重新构建为条件图像到图像的扩散过程。条件输入：RGB 图像。目标输出：深度图。采用预训练的 Stable Diffusion 2 U-Net 作为骨干网络。输出的深度图在物体边界处异常清晰。代价：推理速度慢于前馈模型 (Feed-Forward Models)（需要 10-50 步去噪）。

### 相机内参与针孔相机模型

要将具有深度 `d` 的像素 `(u, v)` 反投影至相机坐标系下的三维点 `(X, Y, Z)`：

fx, fy, cx, cy = camera intrinsics
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d

内参可来源于 EXIF 元数据、标定板或单目内参估计器（如 Perspective Fields、UniDepth）。若无内参，仍可通过假设 60-70° 视场角 (Field of View, FOV) 和中等分辨率的主点 (Principal Points) 来渲染点云——适用于可视化，但不适用于精确测量。

### 评估指标

两项标准评估指标：

- **AbsRel**（绝对相对误差）：`mean(|d_pred - d_gt| / d_gt)`。数值越低越好。生产级模型通常在 0.05-0.1 之间。
- **delta < 1.25**（阈值准确率）：满足 `max(d_pred/d_gt, d_gt/d_pred) < 1.25` 的像素比例。数值越高越好。最先进 (SOTA) 模型可达 0.9 以上。

对于相对深度模型（如 Depth Anything V3、MiDaS），评估时会使用这两项指标的尺度与平移不变 (Scale-and-Shift Invariant) 版本。

## 构建

### 步骤 1：深度指标 (Depth Metrics)

import torch

def abs_rel_error(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return (torch.abs(pred - target) / target.clamp(min=1e-6)).mean().item()


def delta_accuracy(pred, target, threshold=1.25, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    ratio = torch.maximum(pred / target.clamp(min=1e-6), target / pred.clamp(min=1e-6))
    return (ratio < threshold).float().mean().item()

在进行评估之前，务必对无效深度像素 (Invalid Depth Pixels)（如零值、NaN 或饱和值）应用掩码 (Mask) 处理。

### 步骤 2：尺度与平移对齐 (Scale-and-Shift Alignment)

对于相对深度模型 (Relative-Depth Models)，在计算评估指标前，需将预测结果与真实值 (Ground Truth) 进行对齐。此处采用最小二乘法 (Least-Squares Fit) 拟合 `a * pred + b = target`：

def align_scale_shift(pred, target, mask=None):
    if mask is not None:
        p = pred[mask]
        t = target[mask]
    else:
        p = pred.flatten()
        t = target.flatten()
    A = torch.stack([p, torch.ones_like(p)], dim=1)
    coeffs, *_ = torch.linalg.lstsq(A, t.unsqueeze(-1))
    a, b = coeffs[:2, 0]
    return a * pred + b

在评估 MiDaS / Depth Anything 模型时，请在调用 `abs_rel_error` 之前先运行 `align_scale_shift`。

### 步骤 3：将深度图反投影为点云 (Lift Depth to Point Cloud)

import numpy as np

def depth_to_point_cloud(depth, intrinsics):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


depth = np.random.uniform(0.5, 4.0, (240, 320))
intr = (320.0, 320.0, 160.0, 120.0)
pc = depth_to_point_cloud(depth, intr)
print(f"point cloud shape: {pc.shape}  (H, W, 3)")

仅需一个函数，即可满足所有涉及三维空间转换 (3D-Lifted) 的应用需求。将生成的点云 (Point Cloud) 导出为 `.ply` 格式，便可直接在 MeshLab 或 CloudCompare 中打开查看。

### 步骤 4：使用合成深度场景进行冒烟测试 (Smoke Test)

def synthetic_depth(size=96):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    # Floor: linear gradient from near (top) to far (bottom)
    depth = 1.0 + (yy / size) * 4.0
    # Box in the middle: closer
    mask = (np.abs(xx - size / 2) < size / 6) & (np.abs(yy - size * 0.6) < size / 6)
    depth[mask] = 2.0
    return depth.astype(np.float32)


gt = torch.from_numpy(synthetic_depth(96))
pred = gt + 0.3 * torch.randn_like(gt)  # simulated prediction
aligned = align_scale_shift(pred, gt)
print(f"before align  absRel = {abs_rel_error(pred, gt):.3f}")
print(f"after align   absRel = {abs_rel_error(aligned, gt):.3f}")

### 步骤 5：Depth Anything V3 使用示例（参考）

import torch
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-v2-large")

image = Image.open("street.jpg").convert("RGB")
out = pipe(image)
depth_np = np.array(out["depth"])

仅需三行代码。`out["depth"]` 返回的是 PIL 灰度图像 (PIL Grayscale Image)；若需进行数学运算，请将其转换为 numpy 数组。针对 Depth Anything V3，模型正式发布后只需替换模型标识符 (Model ID) 即可，应用程序接口 (API) 保持不变。

## 使用方法

- **Depth Anything V3**（Meta AI / 字节跳动，2024-2026）—— 相对深度（Relative Depth）估计的默认选择。目前生产环境中基于 ViT-large 主干网络（ViT-large Backbone）速度最快的模型。
- **Marigold**（苏黎世联邦理工学院 ETH，2024）—— 视觉质量最高，但推理速度较慢。
- **UniDepth**（苏黎世联邦理工学院 ETH，2024）—— 支持相机内参（Camera Intrinsics）估计的度量深度（Metric Depth）模型。
- **ZoeDepth**（英特尔 Intel，2023）—— 度量深度模型；版本较旧，但依然稳定可靠。
- **MiDaS v3.1** —— 经典遗留模型，稳定性高；非常适合作为对比基线。

典型集成流程：

1. 接收 RGB 图像帧。
2. 深度模型生成深度图。
3. 目标检测器输出边界框。
4. 结合深度信息将边界框中心点反投影至 3D 空间；若可用，则与点云（Point Cloud）进行融合。
5. 下游应用：增强现实（AR）遮挡处理、路径规划、物体尺寸估计、立体视觉替代方案。

在实时应用场景中，经过 INT8 量化（INT8 Quantization）的 Depth Anything V2 Small 模型在消费级 GPU 上处理 518x518 分辨率图像时，可达到约 30 FPS。

## 交付成果

本章节将产出：

- `outputs/prompt-depth-model-picker.md` —— 根据延迟要求、度量深度与相对深度的需求差异以及场景类型，在 Depth Anything V3、Marigold、UniDepth 和 MiDaS 之间进行模型选择。
- `outputs/skill-depth-to-pointcloud.md` —— 一项技能模块，用于根据深度图构建点云，包含正确的内参处理逻辑，并支持导出为 `.ply` 格式。

## 练习

1. **（简单）** 使用 Depth Anything V2 处理任意 10 张你书桌的照片。将深度结果保存为灰度 PNG 图像并进行检查。找出一个预测深度明显错误的物体，并解释单目视觉线索（Monocular Cues）为何失效。
2. **（中等）** 基于 Depth Anything V2 输出的 RGB 图像与深度图，将其反投影为点云并使用 `open3d` 进行渲染。对比室内与室外两个场景，记录哪一个的视觉效果更符合真实物理规律。
3. **（困难）** 准备五组图像对，每组图像仅存在一个已知物体的位置差异（例如：瓶子向前移动了 30 厘米）。使用 UniDepth 分别预测两组图像的度量深度。报告预测的距离变化量与真实 30 厘米移动量之间的误差。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 单目深度（Monocular Depth） | “单图深度” | 仅从单张 RGB 图像帧进行深度估计，无需双目视觉或激光雷达（LiDAR） |
| 相对深度（Relative Depth） | “有序深度” | 仅表示远近顺序的 Z 轴值，不包含真实物理单位 |
| 度量深度（Metric Depth） | “绝对距离” | 以米为单位的深度值；需要相机标定或使用带度量监督信号训练的模型 |
| AbsRel | “绝对相对误差” | 计算 \|d_pred - d_gt\| / d_gt 的平均值；深度估计的标准评估指标 |
| Delta 准确率（Delta Accuracy） | “delta < 1.25” | 预测值落在真实值（Ground Truth）25% 误差范围内的像素占比 |
| 针孔相机模型（Pinhole Camera） | “fx, fy, cx, cy” | 用于将像素坐标与深度 (u, v, d) 反投影至 3D 空间坐标 (X, Y, Z) 的相机模型 |
| DPT（Dense Prediction Transformer） | “密集预测 Transformer” | 部署在冻结的 ViT 编码器之上、基于卷积的深度解码器架构 |
| DINOv2 主干网络（DINOv2 Backbone） | “模型有效的核心原因” | 通过自监督学习提取的特征，无需深度标签即可跨领域泛化 |

## 扩展阅读

- [Depth Anything V3 论文页面](https://depth-anything.github.io/) — 采用 DINOv2 编码器 (DINOv2 encoder) 的最先进 (SOTA) 单目深度估计 (monocular depth estimation)
- [Marigold (Ke et al., CVPR 2024)](https://marigoldmonodepth.github.io/) — 基于扩散模型 (diffusion model) 的深度估计 (depth estimation)
- [UniDepth (Piccinelli et al., 2024)](https://arxiv.org/abs/2403.18913) — 结合相机内参 (camera intrinsics) 的度量深度 (metric depth)
- [MiDaS v3.1 (Intel ISL)](https://github.com/isl-org/MiDaS) — 经典的相对深度 (relative depth) 基准 (baseline)
- [DINOv3 博客文章 (Meta)](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) — 显著提升深度精度的编码器系列 (encoder family)
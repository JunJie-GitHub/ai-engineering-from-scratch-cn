---
name: 技能-点云加载器
description: 为 .ply / .pcd / .xyz 文件编写 PyTorch 数据集（Dataset），包含正确的归一化（Normalization）、居中（Centering）和点采样（Point Sampling）
version: 1.0.0
phase: 4
lesson: 13
tags: [3D视觉, 点云, 数据加载, PyTorch]
---

# 点云加载器（Point Cloud Loader）

将包含 3D 扫描文件的文件夹转换为可直接用于训练的 PyTorch 数据集（Dataset）。

## 适用场景

- 启动新的点云（Point Cloud）分类或分割项目。
- 在 `.ply`、`.pcd` 和 `.xyz` 格式之间切换。
- 调试训练无报错但收敛效果差的模型；此类问题通常源于数据加载器（Data Loader）的归一化（Normalization）处理不当。

## 输入参数

- `data_root`：包含点云文件的文件夹，以及可选的包含标签的 CSV 文件。
- `file_format`：ply | pcd | xyz | npy。
- `num_points`：固定采样点数，通常为 1024 或 2048。
- `augmentation`：none | rotate | jitter | mixup。

## 归一化策略

每个生产级点云处理流水线（Pipeline）均按以下顺序执行：

1. **居中（Center）**点云：减去质心（Centroid）坐标。
2. **缩放（Scale）**至单位球（Unit Sphere）：除以距中心的最大距离。
3. **采样（Sample）** `num_points` 个点。若点数过多，为精确还原几何形状可使用**最远点采样（Farthest Point Sampling, FPS）**，或为追求速度使用随机采样。若点数不足，则重复现有数据点。
4. **打乱（Shuffle）**点序（模型本身通常对点序不敏感，但打乱可消除潜在的非预期顺序依赖）。

## 输出模板

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

def _read_ply(path):
    if HAS_O3D:
        pc = o3d.io.read_point_cloud(path)
        return np.asarray(pc.points, dtype=np.float32)
    # Fallback: minimal ascii-ply reader
    ...

def _fps(points, k):
    idx = np.zeros(k, dtype=np.int64)
    dist = np.full(len(points), np.inf)
    seed = np.random.randint(len(points))
    idx[0] = seed
    for i in range(1, k):
        dist = np.minimum(dist, ((points - points[idx[i-1]]) ** 2).sum(axis=1))
        idx[i] = int(np.argmax(dist))
    return idx

def normalise(points):
    centre = points.mean(axis=0)
    points = points - centre
    scale = np.max(np.linalg.norm(points, axis=1))
    return points / max(scale, 1e-8)

class PointCloudDataset(Dataset):
    def __init__(self, files, labels, num_points=1024, augment=False):
        self.files = files
        self.labels = labels
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        pts = _read_ply(self.files[i])
        pts = normalise(pts)
        if len(pts) >= self.num_points:
            idx = _fps(pts, self.num_points)
            pts = pts[idx]
        else:
            reps = int(np.ceil(self.num_points / len(pts)))
            pts = np.tile(pts, (reps, 1))[:self.num_points]
        # Shuffle point order to break any accidental dependencies (especially
        # important when tiling repeats points in deterministic order).
        np.random.shuffle(pts)
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
            pts = pts @ R
            pts = pts + np.random.normal(0, 0.02, pts.shape).astype(np.float32)
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        return torch.from_numpy(pts).transpose(0, 1), int(self.labels[i])

## 报告

[dataset]
  files:          <N>
  format:         <ply|pcd|xyz|npy>
  points_per_sample: <int>
  normalise:      centre + unit sphere
  sampling:       FPS | random
  augmentation:   <list>

## 规则

- 始终先进行中心化（centering）再进行缩放（scaling）；交换顺序会改变“单位球（unit sphere）”的含义。
- 对于形状任务，优先使用最远点采样（Farthest Point Sampling, FPS）而非随机采样（random sampling）；对于分割（segmentation）任务，由于每个点都至关重要，使用随机采样即可。
- 评估（evaluation）阶段切勿进行数据增强（data augmentation）；数据增强仅应在训练（training）阶段使用。
- 如果点云（point cloud）文件包含颜色或法向量作为额外通道，请扩展 `Dataset` 以返回 `(3 + C, num_points)` 的张量（tensor），而不仅仅是 xyz 坐标。
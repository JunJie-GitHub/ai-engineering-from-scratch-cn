---
name: 深度转点云
description: 基于深度图构建点云，正确处理相机内参并导出为 .ply 格式
version: 1.0.0
phase: 4
lesson: 26
tags: [深度图, 点云, 3D, 相机内参]
---

# 深度图转点云

将深度图（Depth Map）与彩色图像结合，生成带纹理的点云（Point Cloud），便于可视化或后续 3D 处理。

## 适用场景

- 将深度预测结果可视化为真实的 3D 场景。
- 从单张图像引导生成稀疏的 3D 重建。
- 当运动恢复结构（Structure from Motion, SfM）失败时，为 3D 高斯溅射（3D Gaussian Splatting, 3DGS）训练提供输入数据。
- 将预测深度与激光雷达（LiDAR）真值进行对比。

## 输入参数

- `depth`：形状为 `(H, W)` 的 NumPy 数组，表示深度值，单位需与输出保持一致（推荐使用米）。
- `rgb`：形状为 `(H, W, 3)` 的 NumPy 数组，表示颜色信息（`uint8` 或 `[0, 1]` 范围内的 `float32`）。
- `intrinsics`：相机内参（Camera Intrinsics），格式为 `(fx, fy, cx, cy)`，单位为像素。
- 可选参数 `depth_scale`：缩放系数，用于将预测的深度单位转换为米。

## 处理流程

1. **验证（Validate）** — 计划保留区域的深度值必须为正数且有限。需将无效像素进行掩码（Mask）处理并剔除。
2. **反投影（Lift）** — 逐像素计算：`X = (u - cx) * d / fx`，`Y = (v - cy) * d / fy`，`Z = d`。
3. **颜色配对（Pair）** — 为每个 3D 点分配对应像素的 `(r, g, b)` 颜色三元组。
4. **导出（Export）** — 支持多种格式：PLY（通用）、`.xyz`（轻量级）、`.pcd`（Open3D 原生）、`.las`/`.laz`（地理空间）。

## 实现模板

import numpy as np

def depth_to_point_cloud(depth, intrinsics, depth_scale=1.0, min_depth=0.1, max_depth=100.0):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth.astype(np.float32) * depth_scale
    valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    return points, valid


def write_ply(path, points, colors=None, valid_mask=None):
    p = points.reshape(-1, 3)
    if valid_mask is not None:
        p = p[valid_mask.flatten()]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {p.shape[0]}",
        "property float x", "property float y", "property float z",
    ]
    if colors is not None:
        c = colors.reshape(-1, 3).astype(np.uint8)
        if valid_mask is not None:
            c = c[valid_mask.flatten()]
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        if colors is not None:
            for pt, col in zip(p, c):
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")
        else:
            for pt in p:
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")

## 输出报告

[export]
  input depth shape:  (H, W)
  valid points:       <N> of <H*W>
  output format:      ply | xyz | pcd | las
  coordinate system:  camera (+X right, +Y down, +Z forward)
  scale:              metres | millimetres | normalised

## 注意事项

- 务必对无效深度值（零、NaN、无穷大、饱和值）进行掩码处理；若直接包含这些值，会在坐标原点处生成大量无意义的噪点。
- 若使用相对深度模型（Relative-Depth Model）进行预测，切勿按度量（Metric）单位导出；应在输出文件名前添加 `relative_` 前缀以标明该约定。
- 保持相机坐标系约定一致（OpenCV：+X 向右，+Y 向下，+Z 向前）。若下游工具采用 OpenGL 坐标系（+Y 向上），需相应调整符号。
- 针对密集场景（超过 100 万个点），应提供下采样（Subsample）参数；体积超过 500 MB 的 PLY 文件在多数软件中加载困难。
- 切勿静默截断深度值以生成“看似合理”的结果；必须显式设置截断阈值并给出警告，以便用户知晓被丢弃的数据。
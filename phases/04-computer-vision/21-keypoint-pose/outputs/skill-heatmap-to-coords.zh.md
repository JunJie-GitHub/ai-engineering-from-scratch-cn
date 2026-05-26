---
name: skill-heatmap-to-coords
description: 编写每个生产级姿态估计 (pose estimation) 模型中使用的亚像素 (sub-pixel) 热力图转坐标例程
version: 1.0.0
phase: 4
lesson: 21
tags: [关键点, 姿态估计, 亚像素, 推理]
---

# 热力图转坐标 (Heatmap to Coords)

将原始关键点 (keypoint) 热力图转换为亚像素级精度的坐标。这是每个姿态估计流水线中成本最低、效果最显著的精度提升方案。

## 适用场景

- 部署基于热力图的关键点检测模型。
- 评估姿态估计指标——OKS（Object Keypoint Similarity，对象关键点相似度）对亚像素精度极为敏感。
- 将姿态估计代码从一个框架迁移到另一个框架。

## 输入参数

- `heatmaps`：形状为 `(N, K, H, W)` 的张量 (tensor)，表示模型输出的每个关键点的热力图。
- `confidence_threshold`：置信度阈值，丢弃峰值低于该值的关键点。

## 处理步骤

1. **Argmax（取最大值索引）**：对每个热力图进行操作，找到整数坐标的峰值位置。
2. **一阶差分偏移 (First-difference offset)**：利用相邻像素估算亚像素偏移量。系数 `0.25` 是针对 `sigma >= 1` 的高斯热力图 (Gaussian heatmaps) 校准的经验值；若需更严谨的亚像素恢复，请使用完整的二次拟合 (quadratic fit)（DARK 算法）或高斯拟合。

dx = 0.25 * sign(heatmap[y, x+1] - heatmap[y, x-1])
dy = 0.25 * sign(heatmap[y+1, x] - heatmap[y-1, x])

对于 DARK / 二次拟合变体，可使用局部二次函数进行近似：

dx = -0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
        / (heatmap[y, x+1] - 2 * heatmap[y, x] + heatmap[y, x-1] + eps)

二次拟合在峰值尖锐的热力图上精度更高；当热力图存在噪声时，基于符号的偏移量是更安全的默认选择。

3. **添加偏移量**：将计算出的偏移量加到整数峰值坐标上。
4. **置信度 (Confidence)**：返回每个关键点的峰值；客户端可利用该值过滤低置信度的预测结果。
5. **边界情况 (Boundary case)**：当峰值落在坐标轴的第一个或最后一个像素时，其中一个相邻像素会被截断 (clamp)；此时偏移量会归零，这是最安全的回退策略。

## 输出模板

import torch

def heatmap_to_coords_subpixel(heatmaps, threshold=0.2):
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    conf, idx = flat.max(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()

    ys_int = ys.long()
    xs_int = xs.long()

    x_minus = (xs_int - 1).clamp(min=0)
    x_plus = (xs_int + 1).clamp(max=W - 1)
    y_minus = (ys_int - 1).clamp(min=0)
    y_plus = (ys_int + 1).clamp(max=H - 1)

    batch_idx = torch.arange(N).view(-1, 1).expand(-1, K)
    kp_idx = torch.arange(K).view(1, -1).expand(N, -1)

    dx_raw = (heatmaps[batch_idx, kp_idx, ys_int, x_plus]
              - heatmaps[batch_idx, kp_idx, ys_int, x_minus])
    dy_raw = (heatmaps[batch_idx, kp_idx, y_plus, xs_int]
              - heatmaps[batch_idx, kp_idx, y_minus, xs_int])
    dx = 0.25 * torch.sign(dx_raw)
    dy = 0.25 * torch.sign(dy_raw)

    at_left = xs_int == 0
    at_right = xs_int == (W - 1)
    at_top = ys_int == 0
    at_bottom = ys_int == (H - 1)
    dx = torch.where(at_left | at_right, torch.zeros_like(dx), dx)
    dy = torch.where(at_top | at_bottom, torch.zeros_like(dy), dy)

    refined_x = xs + dx
    refined_y = ys + dy
    coords = torch.stack([refined_x, refined_y], dim=-1)
    mask = conf >= threshold
    return coords, conf, mask

## 报告格式

[subpixel decode]
  keypoints:   K
  threshold:   <float>
  valid_rate:  fraction of keypoints above threshold

## 注意事项

- 始终将相邻像素索引限制在有效范围内；位于边缘的关键点会产生零差分偏移，但不会导致程序崩溃。
- 返回坐标的同时返回置信度，以便客户端屏蔽低置信度的关键点。
- 亚像素优化仅在峰值周围热力图平滑时有效——请确保训练时使用了 `sigma >= 1` 的高斯目标分布。
- 对于分辨率极低的热力图（< 48x48），建议在提取坐标前先将热力图上采样至原始图像尺寸；亚像素偏移量会随步长 (stride) 缩放。
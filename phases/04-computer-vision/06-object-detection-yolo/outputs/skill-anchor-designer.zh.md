---
name: 锚框设计器技能
description: 给定真实框 (ground-truth boxes) 数据集，对 (w, h) 运行 k-means 聚类 (k-means)，并返回每个特征金字塔网络 (Feature Pyramid Network, FPN) 层级的锚框 (anchor) 集合及覆盖率统计信息
version: 1.0.0
phase: 4
lesson: 6
tags: [计算机视觉, 目标检测, 锚框, k-means]
---

# 锚框设计器 (Anchor Designer)

锚框 (anchor) 是基于锚框的检测器 (anchor-based detector) 中最具数据集特异性的超参数 (hyperparameter)。默认的 COCO 锚框在细胞培养图像、卫星瓦片或小目标监控任务中往往表现不佳。本工具旨在生成真正贴合目标数据的锚框。

## 适用场景

- 在新数据集上进行首次训练之前。
- 当模型整体表现良好，但对极小或极大目标的召回率 (recall) 较低时。
- 在数据集大幅扩充且目标框尺寸分布可能发生偏移之后。

## 输入参数

- `boxes`：形状为 (N, 4) 的 numpy 数组，格式为 `(cx, cy, w, h)` 或 `(x1, y1, x2, y2)`；建议至少包含 1000 个正样本框。
- `num_anchors_per_level`：通常为 3。
- `num_fpn_levels`：通常为 3（P3、P4、P5）或 4。
- `input_size`：训练分辨率 HxW。
- 可选参数 `strides`：各层级的步长 (stride)；若省略，则默认取 `[8, 16, 32, 64]` 的前 `num_fpn_levels` 个值。若检测器的 FPN 步长不同，请显式传入更长或更短的数组。

## 操作步骤

1. **归一化目标框**：将框转换为 `input_size` 分辨率下以像素为单位的 `(w, h)` 对。丢弃任何宽度或高度小于 2 像素的框。
2. **运行 k-means 聚类**：对 `(w, h)` 对进行聚类，设置 `k = num_anchors_per_level * num_fpn_levels`。使用 `1 - IoU(box, cluster)` 作为距离函数，而非欧氏距离 (Euclidean distance)——在 `(w, h)` 上使用欧氏距离会导致细长框和方形框被错误地合并。所有框的权重相等（无加权）；如果数据集存在类别不平衡且希望提升大框的召回率，请在输入数组中重复稀有类别的框，而不是传入权重向量。
3. **按面积升序排列聚类中心**：将其划分为 `num_fpn_levels` 组，每组包含 `num_anchors_per_level` 个锚框。面积最小的组分配给最高分辨率层级（最小步长）。
4. **计算各层级的覆盖率统计信息**：
   - 该层级中每个真实框 (ground-truth box) 与其最佳匹配锚框的交并比 (Intersection over Union, IoU) 中位数 (`median IoU`)。
   - `recall@IoU=0.5`（召回率）：最佳匹配锚框 IoU >= 0.5 的框所占百分比。
   - `area coverage`（面积覆盖率）：面积落在该层级 `[anchor_min_area / 4, anchor_max_area * 4]` 范围内的框所占比例。
5. **输出各层级锚框**并标记 `recall@IoU=0.5 < 0.9` 的层级；这些层级的锚框与数据匹配度较差，应重新调优或增加该层级的锚框数量。

## 报告格式

[anchor-designer]
  total boxes:         <N>
  clusters:            <k>
  distance metric:     1 - IoU

[level P3  stride=8]
  anchors (w, h):      [(A, B), (C, D), (E, F)]
  median IoU:          <X>
  recall@IoU=0.5:      <X>
  coverage:            <X>
  flag:                ok | retune

[level P4  stride=16]
  ...

[summary]
  overall recall@IoU=0.5: <X>
  smallest anchor:        <w x h>
  largest anchor:         <w x h>
  recommendation:         <one sentence if any level flagged>

## 规则与注意事项

- 始终使用基于 IoU 的距离度量；基于欧氏距离的 k-means 聚类生成的锚框在视觉上看似合理，但实际效果往往更差。
- 按面积对聚类中心进行排序，并按升序分配至各层级。
- 当 `num_anchors_per_level = 1` 时，完全跳过 k-means 聚类：按面积分位数（例如 3 个层级对应三分位数）将框划分为 `num_fpn_levels` 个区间，并将每个层级的锚框设置为对应区间的中位数 `(w, h)`。在小数据集上，这种方法比运行 `k = num_fpn_levels` 的 k-means 聚类更稳健。
- 绝不输出负数的锚框尺寸；最小值截断为 1。
- 如果数据集包含的框少于 200 个，需警告用户锚框搜索不可靠，并建议使用默认的 COCO 锚框并补充更多训练数据。
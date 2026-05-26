---
name: skill-image-tensor-inspector
description: 检查任意图像形状的张量或数组，并报告数据类型、布局、取值范围，以及其状态是原始、归一化还是标准化
version: 1.0.0
phase: 4
lesson: 1
tags: [计算机视觉, 调试, 预处理, 张量]
---

# 图像张量检查器 (Image Tensor Inspector)

适用于视觉流水线 (vision pipeline) 中任意节点的诊断技能，当你手中持有一个图像形状的数组并需要确切了解其当前状态时使用。

## 适用场景

- 预训练模型 (pretrained model) 返回了错误的预测结果，且你怀疑是预处理 (preprocessing) 环节出了问题。
- 在 OpenCV 和 torchvision 之间迁移流水线时，通道顺序 (channel order) 不明确。
- 堆叠来自多个框架的层时，批次轴 (batch axis) 总是出现在错误的位置。
- 调试训练循环时，损失值 (loss) 卡在 `log(num_classes)` 处。

## 输入参数

- `x`：任意二维、三维或四维类数组对象 (array-like)（NumPy、PyTorch、JAX）。
- 可选参数 `expected`：用于校验的不变量 (invariants) 字典，例如 `{"layout": "CHW", "range": "standardized"}`。

## 执行步骤

1. **确定后端 (backend)** — 检测 `x` 属于 NumPy、Torch 还是 JAX。将其转换为 NumPy 格式进行检查，且不修改原始数据。

2. **判定秩 (rank)**：
   - 秩为 2 -> 单通道图像 (H, W)。
   - 秩为 3 -> 若最后一个轴为 1、3 或 4，且严格小于另外两个轴，则为 `HWC`；否则为 `CHW`。
   - 秩为 4 -> 若轴 1 的值在 {1, 3, 4} 中，**且**轴 2 或轴 3 的值大于 16，则优先判定为 `NCHW`；否则优先判定为 `NHWC`。仅检查轴 1 会导致类似 `(3, 4, 224, 3)` 的小图像 `NHWC` 批次被错误分类。
   - 始终将模棱两可的情况（例如 `(1, 3, 3, 3)`）标记为 `ambiguous`（不明确），而非盲目猜测；要求调用方提供 `expected` 参数。

3. **判定数据类型与取值范围**：
   - `uint8` 且范围在 [0, 255] -> `raw`（原始）。
   - `float*` 且最小值 >= 0、最大值 <= 1.01 -> `normalized`（归一化）。
   - `float*` 且最小值 < 0、|均值| < 0.5、0.5 <= 标准差 (std) <= 1.5 -> `standardized`（标准化）。
   - 其他情况 -> `unusual`（异常），并打印直方图 (histogram)。

4. **逐通道统计 (per-channel stats)** — 报告每个通道的均值和标准差。若数组呈现标准化特征，则与 ImageNet 的均值/标准差进行对比，并输出匹配置信度。

5. **输出报告**，格式严格遵循以下代码块：

[inspector]
  backend:   numpy | torch | jax
  rank:      2 | 3 | 4
  layout:    HW | HWC | CHW | NHWC | NCHW
  dtype:     <dtype>
  shape:     <shape>
  range:     raw | normalized | standardized | unusual
  min/max:   <min> / <max>
  per-channel mean: [ ... ]
  per-channel std:  [ ... ]
  likely source:    camera | PIL | OpenCV | torchvision | random init
  likely target:    display | training | inference

6. 根据 `likely target`（预期用途）**推荐下一步操作**：
   - 针对 `display`（显示）：转置为 HWC 格式，进行截断处理，并转换为 uint8。
   - 针对 `training`（训练）：使用数据集统计信息进行标准化，转置为 CHW 格式，并添加批次轴。
   - 针对 `inference`（推理）：严格匹配模型卡片 (model card) 中规定的不变量。

## 规则

- 绝不修改输入数据。仅打印诊断信息。
- 若提供了 `expected` 参数，则对每一项不匹配之处标记为 `[expected X got Y]`。
- 当布局或通道顺序不明确时，明确指出静默失败 (silent-failure) 的风险。
- 每次仅推荐一项操作，而非提供选项列表。
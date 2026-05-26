---
name: skill-freeze-inspector
description: 报告哪些参数是可训练的，哪些批归一化 (BatchNorm) 层处于评估模式，以及优化器是否实际接收了可训练参数
version: 1.0.0
phase: 4
lesson: 5
tags: [计算机视觉, 迁移学习, 调试, PyTorch]
---

# 冻结检查器 (Freeze Inspector)

迁移学习 (Transfer Learning) 的缺陷通常隐藏在三个地方：本应冻结却未冻结的参数、本应可训练却未可训练的参数，以及在冻结状态更改前就已构建的优化器 (Optimizer)。本技能可在一次遍历中同时暴露这三类问题。

## 使用时机

- 在设置部分参数的 `requires_grad` 属性之后立即使用。
- 在微调 (Fine-tuning) 运行的第一个训练步骤之前。
- 在调用 `freeze_bn_stats` 或任何切换批归一化 (Batch Normalization, BN) 模式的辅助函数之后。
- 当验证集准确率停滞在随机水平，且你怀疑模型实际上并未进行训练时。

## 输入参数

- `model`：一个 PyTorch `nn.Module`。
- `optimizer`：即将用于训练的优化器。
- 可选参数 `expected_frozen_prefixes`：应被冻结的参数名前缀列表（例如 `["conv1", "bn1", "layer1"]`）。

## 执行步骤

1. **遍历参数。** 针对每个 `(name, param)`：
   - 记录 `requires_grad` 状态
   - 记录 `shape`（形状）和 `numel`（元素数量）

2. **遍历模块。** 针对每个模块：
   - 若为批归一化层，记录其是否处于评估模式 (eval mode)，以及其仿射参数 (affine parameters) 是否可训练。

3. **检查优化器。** 针对每个参数组：
   - 将其 `params` 展平为 `id(p)` 集合。
   - 与所有 `requires_grad == True` 的参数的 `id(p)` 集合进行比对。

4. **检测四种故障模式：**
   - `leaked_train`：参数 `requires_grad=True` 但未出现在优化器中（梯度已计算但从未应用）。
   - `ghost_train`：参数出现在优化器中但 `requires_grad=False`（浪费优化器状态；若后续重新启用 `requires_grad` 还可能引发缺陷）。
   - `bn_mismatch`：(a) BN 层处于训练模式（累积运行统计量 running stats）但其仿射参数（`weight`、`bias`）被冻结，或 (b) BN 层处于评估模式（统计量冻结）但其仿射参数可训练。这两种状态均不一致，几乎总是缺陷。
   - `expected_vs_actual`：`expected_frozen_prefixes` 中列出的任意前缀下仍存在可训练参数。

## 报告输出

[freeze-inspector]
  model trainable params: <N>
  model frozen params:    <N>
  batchnorm layers in eval mode: <count>
  batchnorm layers in train mode: <count>

[optimizer coverage]
  trainable params fed to optimizer: <M> of <N>
  leaked_train: <list of names> (trainable but not in optimizer)
  ghost_train:  <list of names> (in optimizer but frozen)

[bn audit]
  mismatched layers: <list of names>

[expectations]
  expected_frozen_prefixes: <...>
  violating params:         <list>

[verdict]
  ok | <one-line summary of the most severe issue>

## 规则

- 仅报告参数名称；切勿打印权重值本身。
- 所有列表均按参数名称的字母顺序排序。
- 若优化器覆盖率为 100% 且无状态不匹配，则返回 `ok` 并终止。
- 针对 `leaked_train`，始终建议在更改冻结状态后重建优化器。
- 针对 `ghost_train`，建议移除该参数组，或若本意是训练该参数，则将其 `requires_grad` 设置为 `True`。
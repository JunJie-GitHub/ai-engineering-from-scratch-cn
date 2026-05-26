---
name: skill-residual-block-reviewer
description: 审查 PyTorch 残差块（Residual Block）的跳跃连接（Skip-connection）正确性、批归一化（Batch Normalization, BN）位置、激活函数顺序（Activation Order）以及形状对齐（Shape Alignment）情况
version: 1.0.0
phase: 4
lesson: 3
tags: [计算机视觉, ResNet, 代码审查, PyTorch]
---

# 残差块审查器（Residual Block Reviewer）

专为任何声称实现残差块的 PyTorch `nn.Module` 设计的专项审查工具。能够精准捕捉导致绝大多数 ResNet 重写失败的四大常见错误。

## 适用场景

- 有人编写了自定义的 `BasicBlock` 或 `Bottleneck`，但损失值（Loss）出现 `NaN` 或准确率停滞不前。
- 正在将某个模块从一个框架迁移到另一个框架，并希望验证其等效性。
- 正在审查修改 ResNet 内部结构（预激活（Pre-activation）、压缩与激励（Squeeze-and-Excitation）、抗混叠（Anti-aliasing））的拉取请求（Pull Request, PR）。
- 模型在 CIFAR 尺寸输入下运行正常，但在 ImageNet 分辨率下崩溃，原因是捷径连接（Shortcut）配置错误。

## 输入要求

- PyTorch 类定义，可以是源代码文本或可导入的路径。
- 可选参数 `variant`：`basic` | `bottleneck` | `preact` | `seblock`。

## 四项检查

### 1. 捷径连接形状对齐（Shortcut Shape Alignment）

对于任何 `stride != 1` 或 `in_channels != out_channels` 的模块，捷径路径**必须**是一个形状匹配的模块——通常为 1x1 卷积（Convolution）加 BN。在此情况下若仅使用裸 `nn.Identity()`，在前向传播（Forward Pass）时必然会导致形状不匹配错误。

诊断输出：
[shortcut]
  detected:  nn.Identity | 1x1 Conv + BN | 1x1 Conv + BN + ReLU | other
  required:  shape-matching Conv if (stride != 1 or in_c != out_c) else Identity
  verdict:   ok | wrong | unnecessarily heavy

### 2. 批归一化相对于加法操作的位置（BN Placement Relative to Addition）

加法操作 `out + shortcut(x)` 必须发生在最终的 ReLU 之前（后激活（Post-activation），原始 ResNet），或者完全省略最终的 ReLU（预激活 ResNet v2）。若模块在主分支应用 ReLU 后再与原始捷径相加，会产生不对称的激活范围，从而损害训练效果。

诊断输出：
[activation order]
  pattern:  post-act (conv-BN-ReLU-conv-BN-add-ReLU) | pre-act (BN-ReLU-conv-BN-ReLU-conv-add) | other
  verdict:  ok | suspect

### 3. 卷积层的偏置（Bias on Conv Layers）

紧随批归一化（BatchNorm）之后的卷积层应设置 `bias=False`。由于 BN 的 beta 参数已对偏置进行了建模，额外的卷积偏置不仅浪费参数，还可能减缓收敛（Convergence）速度。

诊断输出：
[bias]
  convs with BN and bias=True: <count>
  recommended fix: set bias=False on those layers

### 4. 原地 ReLU 与自动微分（In-place ReLU and Autograd）

对即将与捷径相加的张量（Tensor）使用 `nn.ReLU(inplace=True)` 会覆盖残差相加时可能仍需用到的值。标记任何在相加前未跟随生成新张量层的 `inplace=True` 操作。

诊断输出：
[in-place]
  risky inplace ops: <list>
  fix: inplace=False before the residual add

## 报告格式

[block-review]
  variant:       basic | bottleneck | preact | se | other
  shortcut:      ok | wrong | heavy
  activation:    ok | suspect
  bias-bn:       ok | <N> convs need bias=False
  in-place:      ok | <N> risky ops
  summary:       one sentence

## 规则

- 不要重写该模块。仅输出审查报告。
- 若模块正确，则在所有检查项中标记为 `ok` 并终止。不提供额外建议。
- 若存在多处错误，请按上述顺序列出（优先检查捷径连接，因为它是导致崩溃的最常见原因）。
- 当用户已明确指定时，切勿将有意设计的预激活或压缩与激励变体标记为错误。
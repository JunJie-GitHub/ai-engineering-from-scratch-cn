---
name: 卷积形状计算器技能
description: 逐层遍历卷积神经网络 (Convolutional Neural Network, CNN) 规格，并报告每个模块的输出形状 (Output Shape)、感受野 (Receptive Field) 和参数量 (Parameter Count)
version: 1.0.0
phase: 4
lesson: 2
tags: [计算机视觉, 卷积神经网络, 架构, 调试]
---

# 卷积形状计算器 (Conv Shape Calculator)

用于规划或调试卷积神经网络 (CNN) 的确定性辅助工具。给定输入形状和层规格列表，无需运行模型即可追踪形状、感受野和参数量。

## 适用场景

- 设计新 CNN 时，希望验证每次下采样 (Downsampling) 都能得到规整的尺寸。
- 阅读论文并将其架构表转换为代码时。
- 预训练骨干网络 (Backbone) 在分类头 (Classifier Head) 处因形状不匹配而崩溃，需要查明是哪一层改变了空间尺寸。
- 在训练前比较两个骨干网络的参数效率。

## 输入参数

- `input_shape`：`(C, H, W)`。
- `layers`：层字典的有序列表。每个字典支持以下格式：
  - `{type: "conv", c_out, k, s, p, groups=1, bias=true}`
  - `{type: "pool", mode: "max"|"avg", k, s, p=0}`
  - `{type: "adaptive_pool", out_h, out_w}`
  - `{type: "flatten"}`
  - `{type: "linear", out_features, bias=true}`

## 执行步骤

1. **初始化追踪状态**：设置初始形状为 `(C, H, W)`，感受野为 `1`，有效步长 (Effective Stride) 为 `1`，累计参数量 (Cumulative Params) 为 `0`。

2. **遍历每一层**，按以下顺序更新：
   - 计算 `C_out`（卷积/线性层），或保持 `C_in` 不变（池化层）。
   - 使用公式 `(H + 2P - K) / S + 1` 计算卷积和池化层的空间输出；自适应池化层 (Adaptive Pool) 使用 `out_h/out_w`；展平层 (Flatten) 在进入线性层 (Linear Layer) 前的输出形状为 `(C * H * W, 1, 1)`，线性层输出为标量 `1x1`。
   - 更新感受野和有效步长：
     - 卷积/池化层：`RF_new = RF_old + (K - 1) * effective_stride`，`effective_stride *= S`。
     - 自适应池化层：视为有效步长 `S = H_in / out_h`（向下取整）的池化操作。`RF_new = RF_old + (H_in - 1) * effective_stride_old`；`effective_stride *= S`。注意，自适应池化层的感受野等于此前完整的空间范围。
     - 展平/线性层：感受野和有效步长不再具有实际意义；将其冻结为展平前的值，并在后续行中省略。
   - 计算参数量：
     - 卷积层：`C_out * (C_in / groups) * K * K + (C_out if bias else 0)`。
     - 线性层：`out_features * in_features + (out_features if bias else 0)`。
     - 池化层和展平层：0。

3. **检测问题**并标记：
   - 输出尺寸非整数（步长/填充未对齐）。
   - 在网络堆栈结束前出现 `H_out <= 0`。
   - 感受野超过输入尺寸（该点之后可能存在计算浪费）。
   - 单层参数量出现 10 倍级突增，表明通道规划可能存在错误。

4. **以单表形式输出报告**：

idx  layer                C_in  C_out  K  S  P  H_out  W_out  RF    params     cum_params
1    conv 3x3 s=1 p=1     3     32     3  1  1  224    224    3     896        896
2    conv 3x3 s=2 p=1     32    64     3  2  1  112    112    7     18,496     19,392
3    pool max 2x2         64    64     2  2  0  56     56     11    0          19,392
...

5. **汇总行**：最终 `(C, H, W)`、最终感受野、总参数量及警告信息。

## 规则

- 空间尺寸必须始终返回整数。若公式计算结果为非整数，应标记为错误，切勿静默向下取整。
- 当 `groups > 1` 时，需验证 `C_in % groups == 0` 且 `C_out % groups == 0`；否则报错。
- 对于深度卷积 (Depthwise Convolution, `groups == C_in`)，需在 `layer` 列中明确标注，以便读者理解参数量较低的原因。
- 若用户提供了批归一化 (Batch Normalization) 或激活层，在形状计算中忽略它们，但需累加其参数量（每个 BatchNorm 层为 `2 * C`）。
- 绝不猜测缺失字段的默认值。每个卷积层和池化层都必须明确提供 `k`、`s`、`p`。
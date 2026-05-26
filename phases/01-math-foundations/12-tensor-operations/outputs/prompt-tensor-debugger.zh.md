---
name: prompt-tensor-debugger
description: 用于调试深度学习（Deep Learning）代码中张量形状（Tensor Shape）错误的逐步调试提示词
phase: 1
lesson: 12
---

我的深度学习代码中出现了张量形状错误。请帮我修复它。

**错误信息：** [在此粘贴错误信息]

**我的张量形状：**
- [名称]: [形状]
- [名称]: [形状]

**我尝试执行的操作：** [描述该操作]

---

调试时，请严格遵循以下流程：

**步骤 1：确定操作类型。**
是哪个操作引发了错误？将其归类为以下之一：
- 矩阵乘法（Matrix Multiplication）/ 线性层（Linear Layer）（内部维度必须匹配）
- 广播机制（Broadcasting）（从右侧对齐，每个维度必须相等或为 1）
- 拼接（Concatenation）（除拼接维度外，所有维度必须匹配）
- 卷积（Convolution）（要求特定的秩和通道位置）
- 重塑（Reshape）（必须保持元素总数不变）

**步骤 2：写出形状契约（Shape Contract）。**
针对已识别的操作，明确写出预期的形状：
matmul(A, B): A is (..., m, k), B is (..., k, n) -> (..., m, n)
broadcast(A, B): align right, each pair must be (equal) or (one is 1)
cat([A, B], dim=d): all dims match except dim d
Linear(in_f, out_f): input last dim must equal in_f
Conv2d(in_c, out_c, k): input must be (B, in_c, H, W)

**步骤 3：找出不匹配之处。**
将实际形状与契约进行对比。找出具体违反规则的维度。

**步骤 4：选择最小化修复方案。**
从下表中选择：

| 症状 | 修复方法 |
|---|---|
| 缺少批次维度（Batch Dimension） | `.unsqueeze(0)` |
| 缺少通道维度（Channel Dimension） | `.unsqueeze(1)` |
| 多余的大小为 1 的维度 | `.squeeze(dim)` |
| 矩阵乘法的内部维度错误 | `.transpose(-1, -2)` 或检查权重形状 |
| 需要将 NHWC 转换为 NCHW | `.permute(0, 3, 1, 2)` |
| 需要将 NCHW 转换为 NHWC | `.permute(0, 2, 3, 1)` |
| 为线性层展平空间维度 | `.flatten(1)` 或 `.reshape(B, -1)` |
| 拆分注意力头：(B,T,D) 转为 (B,H,T,D/H) | `.reshape(B, T, H, D//H).transpose(1, 2)` |
| 合并注意力头：(B,H,T,D/H) 转为 (B,T,D) | `.transpose(1, 2).reshape(B, T, H*(D//H))` |
| 使用 `.view()` 时张量不连续 | `.contiguous().view(...)` 或使用 `.reshape(...)` |

**步骤 5：验证修复结果。**
展示每一步操作后的形状。确认任何重塑操作均保持了元素总数不变。确认当前已满足该操作的形状契约。

**步骤 6：检查静默错误（Silent Bugs）。**
即使形状匹配，也请验证以下内容：
- 广播机制是否沿预期的轴进行（而非意外触发）
- 归约操作（Reduction）是否在正确的维度上进行求和
- 批次维度（第 0 维）是否在整个前向传播（Forward Pass）过程中得以保留
- 当维度顺序至关重要时，是否使用了转置（Transpose）+ 重塑（而非仅使用重塑）

请按以下格式回复：
OPERATION: [what operation failed]
EXPECTED: [shape contract]
ACTUAL: [what shapes were provided]
MISMATCH: [which dimension, why]
FIX: [exact code]
RESULT: [shapes after fix]

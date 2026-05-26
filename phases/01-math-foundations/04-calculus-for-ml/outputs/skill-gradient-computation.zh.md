---
name: 技能-梯度计算
description: 计算常见机器学习损失函数的梯度并选择合适的求导方法
version: 1.0.0
phase: 1
lesson: 4
tags: [微积分, 梯度, 反向传播]
---

# 机器学习中的梯度计算 (Gradient Computation)

本文档提供了计算神经网络中损失函数 (Loss Functions)、激活函数 (Activation Functions) 和层操作 (Layer Operations) 梯度的实用参考。

## 决策清单

1. 该函数是否由简单初等函数（幂函数、指数函数、对数函数、三角函数）组成？使用解析求导法 (Analytical Derivatives) 和链式法则 (Chain Rule)。
2. 该函数是否为自定义或黑盒操作？使用数值微分法 (Numerical Differentiation)：`(f(x+h) - f(x-h)) / (2h)`，其中 h = 1e-7。
3. 该函数是否由 PyTorch/JAX 中的张量操作构建？交由自动微分引擎 (Autograd) 处理。通过数值检查进行验证。
4. 是否需要计算标量损失函数相对于权重矩阵的梯度？沿计算图 (Computation Graph) 应用链式法则，逐个节点进行。
5. 是否存在不可微操作（如 argmax、取整、采样）？使用直通估计器 (Straight-Through Estimator) 或重参数化技巧 (Reparameterization Trick)。

## 各方法适用场景

| 方法 | 适用场景 | 计算成本 |
|---|---|---|
| 解析法（手动推导） | 简单函数，验证自动微分输出 | 运行时零成本 |
| 数值法（有限差分） | 调试、梯度检查、黑盒函数 | 针对 n 个参数需 2n 次前向传播 |
| 自动微分 (Automatic Differentiation) | 任意可微计算图（默认方法） | 一次反向传播 |
| 符号计算（SymPy, Mathematica） | 为论文推导闭式梯度 | 仅编译/推导期成本 |

## 快速参考：常见导数

| 函数 | f(x) | f'(x) | 机器学习应用场景 |
|---|---|---|---|
| 均方误差损失 (MSE Loss) | (1/n) sum(y_hat - y)^2 | (2/n)(y_hat - y) | 回归任务 |
| 交叉熵损失（二分类） | -(y log(p) + (1-y) log(1-p)) | p - y（经过 Sigmoid 后） | 二分类 |
| 交叉熵损失（多分类） | -log(p_true_class) | p - one_hot(y)（经过 Softmax 后） | 多分类 |
| Sigmoid | 1 / (1 + e^(-x)) | sigma(x) * (1 - sigma(x)) | 输出门控、二分类输出 |
| Tanh | (e^x - e^(-x)) / (e^x + e^(-x)) | 1 - tanh(x)^2 | 隐藏层激活（传统用法） |
| ReLU | max(0, x) | 1 if x > 0, 0 if x < 0 | 默认隐藏层激活函数 |
| Leaky ReLU | max(0.01x, x) | 1 if x > 0, 0.01 if x < 0 | 避免神经元死亡 |
| GELU | x * Phi(x) | Phi(x) + x * phi(x) | Transformer 架构 |
| Softmax_i | e^(x_i) / sum(e^(x_j)) | s_i(1 - s_i) for i=j, -s_i*s_j for i!=j | 输出层（雅可比矩阵） |
| Log-softmax | x_i - log(sum(e^(x_j))) | 1 - softmax(x_i) for the i-th entry | 数值稳定的交叉熵计算 |
| 线性层 (Linear Layer) | y = Wx + b | dL/dW = dL/dy * x^T, dL/db = dL/dy | 所有层 |
| L2 正则化 | lambda * sum(w^2) | 2 * lambda * w | 权重衰减 (Weight Decay) |
| L1 正则化 | lambda * sum(\|w\|) | lambda * sign(w) | 稀疏性约束 |

## 常见错误

- 在批次平均损失（如 MSE、交叉熵）中遗漏 1/n 因子。梯度会随批次大小缩放。
- 将 Softmax 梯度误算为向量，而实际上它是一个雅可比矩阵 (Jacobian Matrix)。对于交叉熵与 Softmax 结合的情况，梯度可简化为 (p - y)，从而避免计算完整的雅可比矩阵。
- 链式法则应用顺序错误。应从损失函数开始反向推导：dL/dW = dL/dy * dy/dW。
- 数值求导时步长 h 设置过大（如 h = 0.1）或过小（如 h = 1e-15）。对于 float64 精度，建议固定使用 h = 1e-7。
- 忽略 ReLU 在 x = 0 处梯度未定义的事实。在实际工程中，通常将其设为 0 或 0.5。

## 梯度检查步骤

For each parameter w:
  numeric_grad = (loss(w + h) - loss(w - h)) / (2h)
  auto_grad = backward pass value
  relative_error = |numeric - auto| / max(|numeric|, |auto|, 1e-8)
  assert relative_error < 1e-5

相对误差大于 1e-3 表示实现存在错误。若介于 1e-5 到 1e-3 之间，需进一步排查。
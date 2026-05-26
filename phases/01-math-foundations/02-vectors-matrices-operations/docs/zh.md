# 向量 (Vectors)、矩阵 (Matrices) 与运算 (Operations)

> 每个神经网络 (Neural Network) 本质上只是多了几个步骤的矩阵乘法 (Matrix Multiplication)。

**类型：** 构建
**语言：** Python, Julia
**先修要求：** 第一阶段，第 01 课（线性代数直观理解）
**时长：** 约 60 分钟

## 学习目标

- 构建一个矩阵类（Matrix class），支持逐元素运算（element-wise operations）、矩阵乘法（matrix multiplication）、转置（transpose）、行列式（determinant）与逆矩阵（inverse）
- 区分逐元素乘法（element-wise multiplication）与矩阵乘法，并解释各自适用的场景
- 仅使用从零实现的矩阵类，实现单个全连接神经网络层（dense neural network layer）（`relu(W @ x + b)`）
- 解释广播规则（broadcasting rules），以及偏置相加（bias addition）在神经网络框架（neural network frameworks）中的工作原理

## 问题

你想构建一个神经网络 (Neural Network)。你阅读代码时会看到这一行：

output = activation(weights @ input + bias)

其中的 `@` 表示矩阵乘法 (Matrix Multiplication)。`weights` 是一个矩阵。`input` 是一个向量 (Vector)。如果你不了解这些运算的作用，这行代码就如同魔法。但如果你了解，它仅仅通过三个操作就完成了一个层 (Layer) 的完整前向传播 (Forward Pass)。

你的模型处理的每一张图像都是一个像素值 (Pixel Values) 矩阵。每一个词嵌入 (Word Embedding) 都是一个向量。每一个神经网络的每一层都是一次矩阵变换 (Matrix Transformation)。正如不理解变量就无法编写代码一样，不精通矩阵运算 (Matrix Operations) 也就无法构建 AI 系统。

本课将从零开始帮你建立这种熟练度。

## 概念

### 向量（Vector）：有序的数字列表

向量（Vector）是一组具有方向和大小的数字。在人工智能中，向量用于表示数据点、特征或参数。

v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector

二维向量 `[3, 4]` 指向平面上的坐标 (3, 4)。其长度（即大小（magnitude））为 5（符合 3-4-5 直角三角形规律）。

### 矩阵（Matrix）：数字网格

矩阵（Matrix）是一个二维网格，由行和列组成。一个 m x n 矩阵包含 m 行和 n 列。

A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |

在神经网络中，权重矩阵（weight matrix）负责将输入向量转换为输出向量。例如，一个具有 784 个输入和 128 个输出的层会使用一个 128x784 的权重矩阵。

### 为什么形状（Shape）很重要

矩阵乘法遵循一条严格规则：`(m x n) @ (n x p) = (m x p)`。其中，内部维度必须匹配。

(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid

如果你在 PyTorch 中遇到形状不匹配（shape mismatch）错误，原因就在于此。

### 运算操作对照表

| 运算操作 | 作用说明 | 神经网络中的应用 |
|-----------|-------------|-------------------|
| 加法（Addition） | 逐元素相加 | 为输出添加偏置（bias） |
| 标量乘法（Scalar multiplication） | 缩放每个元素 | 学习率 * 梯度 |
| 矩阵乘法（Matrix multiplication） | 变换向量 | 层的前向传播（forward pass） |
| 转置（Transpose） | 行列互换 | 反向传播（backpropagation） |
| 行列式（Determinant） | 单个数值摘要 | 检查矩阵是否可逆 |
| 逆矩阵（Inverse） | 撤销变换 | 求解线性方程组 |
| 单位矩阵（Identity matrix） | 恒等变换矩阵 | 初始化、残差连接（residual connections） |

### 逐元素乘法与矩阵乘法

初学者经常在这两者的区别上栽跟头。

逐元素（Element-wise）：对应位置的元素相乘。两个矩阵的形状必须完全相同。

| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |

矩阵乘法：行与列的点积运算。内部维度必须匹配。

| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |

不同的运算，不同的结果，不同的规则。

### 广播机制（Broadcasting）

当你将偏置向量（bias vector）加到输出矩阵时，两者的形状并不匹配。广播机制会自动扩展较小的数组以适配形状。

| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |

所有现代框架都会自动执行此操作。理解广播机制可以避免你在代码能正常运行但形状看似不匹配时感到困惑。

## 动手构建

### 步骤 1：向量类

class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5

### 步骤 2：矩阵类与核心操作

class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])

### 步骤 3：运行验证

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)

### 步骤 4：关联神经网络

import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")

这是一个单密集层（Dense Layer）：`output = relu(W @ x + b)`。每一个神经网络（Neural Network）中的密集层执行的正是这一操作。

## 使用方法

NumPy 仅需更少的代码行数即可完成上述所有操作，且速度要快几个数量级。

import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")

Python 中的 `@` 运算符会调用 `__matmul__` 方法。NumPy 使用由 C 和 Fortran 编写的优化 BLAS 例程（BLAS routines）来实现该功能。数学原理相同，但速度提升了 100 倍。

NumPy 中的广播机制（Broadcasting）：

matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)

NumPy 会自动将一维偏置（1D bias）广播到这两行上。这也是所有神经网络框架（Neural Network Framework）中偏置加法的工作原理。

## 发布

本课将生成一个提示词（prompt），用于通过几何直觉（geometric intuition）教授矩阵运算（matrix operations）。请参阅 `outputs/prompt-matrix-operations.md`。

此处构建的 `Matrix` 类是我们将在第 3 阶段第 10 课中构建的迷你神经网络框架（mini neural network framework）的基础。

## 练习

1. **验证逆矩阵 (Inverse)。** 计算 `A @ A.inverse_2x2()` 的乘积，并确认结果为单位矩阵 (Identity Matrix)。请使用三个不同的 2x2 矩阵进行测试。当行列式 (Determinant) 为零时会发生什么？

2. **实现 3x3 逆矩阵。** 扩展 Matrix 类 (Matrix Class)，使用伴随矩阵法 (Adjugate Method) 计算 3x3 矩阵的逆矩阵。将其与 NumPy 的 `np.linalg.inv` 进行对比测试。

3. **构建两层网络 (Two-layer Network)。** 仅使用你的 Matrix 类（不使用 NumPy），创建一个两层神经网络 (Neural Network)：输入层 (3) -> 隐藏层 (4) -> 输出层 (2)。初始化随机权重，执行一次前向传播 (Forward Pass)，并验证所有形状 (Shapes) 是否正确。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 向量 (Vector) | “一个箭头” | 一组有序的数字。在 AI 中：高维空间中的一个点。 |
| 矩阵 (Matrix) | “一张数字表格” | 一种线性变换。它将向量从一个空间映射到另一个空间。 |
| 矩阵乘法 (Matrix Multiplication) | “就是把数字相乘” | 第一个矩阵的每一行与第二个矩阵的每一列进行点积运算。顺序至关重要。 |
| 转置 (Transpose) | “把它翻转一下” | 交换行和列。将 m x n 矩阵变为 n x m 矩阵。在反向传播中至关重要。 |
| 行列式 (Determinant) | “从矩阵里算出来的某个数” | 衡量矩阵在二维空间中缩放面积（或三维空间中缩放体积）的程度。值为零意味着该变换使某一维度坍缩。 |
| 逆矩阵 (Inverse Matrix) | “撤销矩阵的操作” | 能够逆转该变换的矩阵。仅当行列式不为零时才存在。 |
| 单位矩阵 (Identity Matrix) | “无聊的矩阵” | 矩阵运算中相当于乘以 1 的存在。常用于残差连接（ResNets）。 |
| 广播机制 (Broadcasting) | “神奇的形状修复” | 通过在缺失的维度上重复数据，将较小的数组拉伸以匹配较大的数组。 |
| 逐元素运算 (Element-wise Operation) | “常规乘法” | 对应位置的元素相乘。两个数组必须具有相同的形状（或可广播）。 |

## 延伸阅读

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 为此处涵盖的每种运算提供直观的视觉化理解
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 所遵循的广播机制 (broadcasting) 确切规则
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向机器学习 (machine learning) 的线性代数 (linear algebra) 简明参考资料
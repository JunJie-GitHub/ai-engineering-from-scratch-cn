# 线性系统 (Linear Systems)

> 求解 Ax = b 是数学中最古老的问题之一，至今仍在驱动着你的神经网络 (neural network)。

**类型：** 构建 (Build)
**语言：** Python
**前置要求：** 第一阶段，课程 01（线性代数直觉）、02（向量与矩阵）、03（矩阵变换）
**时长：** 约 120 分钟

## 学习目标

- 使用带部分主元的高斯消元法 (Gaussian elimination with partial pivoting) 和回代法 (back substitution) 求解 Ax = b
- 使用 LU 分解 (LU decomposition)、QR 分解 (QR decomposition) 和乔列斯基分解 (Cholesky decomposition) 对矩阵进行因式分解，并解释各自的适用场景
- 推导最小二乘法 (least squares) 的正规方程 (normal equations)，并将其与线性回归 (linear regression) 和岭回归 (ridge regression) 建立联系
- 使用条件数 (condition number) 诊断病态系统 (ill-conditioned systems)，并应用正则化 (regularization) 来稳定它们

## 问题背景

每次训练线性回归模型时，你都在求解一个线性系统。每次计算最小二乘拟合时，你都在求解一个线性系统。每当神经网络层计算 `y = Wx + b` 时，它实际上就是在评估线性系统的一侧。当你添加正则化项时，你修改了该系统。当你使用高斯过程 (Gaussian processes) 时，你对矩阵进行了分解。当你为计算马哈拉诺比斯距离 (Mahalanobis distance) 而求协方差矩阵 (covariance matrix) 的逆时，你同样在求解一个线性系统。

方程 Ax = b 无处不在。A 是已知系数矩阵。b 是已知输出向量。x 是你希望求解的未知向量。在线性回归中，A 是你的数据矩阵 (data matrix)，b 是你的目标向量 (target vector)，x 是权重向量 (weight vector)。整个模型可归结为：寻找 x，使得 Ax 尽可能接近 b。

本课程将从零开始构建求解该方程的所有主要方法。你将理解为何某些方法速度快而另一些方法稳定性高，为何某些方法仅适用于方阵系统 (square systems) 而另一些能处理超定系统 (overdetermined systems)，以及为何矩阵的条件数决定了你的求解结果是否真正具有意义。

## 核心概念

### Ax = b 的几何意义

线性方程组具有几何解释。每个方程定义了一个超平面（hyperplane）。解就是所有超平面相交的点（或点集）。

2x + y = 5          Two lines in 2D.
x - y  = 1          They intersect at x=2, y=1.

graph LR
    A["2x + y = 5"] --- S["Solution: (2, 1)"]
    B["x - y = 1"] --- S

可能出现三种情况：

graph TD
    subgraph "One Solution"
        A1["Lines intersect at a single point"]
    end
    subgraph "No Solution"
        A2["Lines are parallel — no intersection"]
    end
    subgraph "Infinite Solutions"
        A3["Lines are identical — every point is a solution"]
    end

在矩阵形式中，“唯一解”意味着矩阵 A 是可逆的（invertible）。“无解”意味着方程组是不相容的（inconsistent）。“无穷多解”意味着矩阵 A 存在零空间（null space）。大多数机器学习（ML）问题属于“无精确解”的类别，因为方程的数量（数据点）多于未知数的数量（参数）。这正是最小二乘法（least squares）发挥作用的地方。

### 列视角与行视角

理解 Ax = b 有两种视角。

**行视角（Row picture）。** A 的每一行定义一个方程。每个方程都是一个超平面。解就是它们全部相交的位置。

**列视角（Column picture）。** A 的每一列都是一个向量。问题转化为：A 的列向量的何种线性组合（linear combination）能生成 b？

A = | 2  1 |    b = | 5 |
    | 1 -1 |        | 1 |

Row picture: solve 2x + y = 5 and x - y = 1 simultaneously.

Column picture: find x1, x2 such that:
  x1 * [2, 1] + x2 * [1, -1] = [5, 1]
  2 * [2, 1] + 1 * [1, -1] = [4+1, 2-1] = [5, 1]   check.

列视角更为本质。如果 b 位于 A 的列空间（column space）内，则方程组有解。如果不在，则需要在列空间中找到距离 b 最近的点。这个最近的点就是最小二乘解。

### 高斯消元法（Gaussian elimination）

高斯消元法将 Ax = b 转化为上三角方程组（upper triangular system）Ux = c，然后通过回代（back substitution）求解。这是最直接的方法。

算法步骤如下：

1. For each column k (the pivot column):
   a. Find the largest entry in column k at or below row k (partial pivoting).
   b. Swap that row with row k.
   c. For each row i below k:
      - Compute multiplier m = A[i][k] / A[k][k]
      - Subtract m times row k from row i.
2. Back substitute: solve from the last equation upward.

示例：

Original:
| 2  1  1 | 8 |       R2 = R2 - (2)R1     | 2  1   1 |  8 |
| 4  3  3 |20 |  -->  R3 = R3 - (1)R1 --> | 0  1   1 |  4 |
| 2  3  1 |12 |                            | 0  2   0 |  4 |

                       R3 = R3 - (2)R2     | 2  1   1 |  8 |
                                       --> | 0  1   1 |  4 |
                                           | 0  0  -2 | -4 |

Back substitute:
  -2 * x3 = -4    -->  x3 = 2
  x2 + 2  = 4     -->  x2 = 2
  2*x1 + 2 + 2 = 8 --> x1 = 2

高斯消元法的计算复杂度为 O(n^3)。对于一个 1000x1000 的方程组，这大约需要十亿次浮点运算。速度很快，但如果需要多次求解具有相同矩阵 A 的方程组，还有更优的方案。

### 部分主元法（Partial pivoting）：为何重要

如果不使用主元法，高斯消元法可能会失败或产生错误结果。如果主元（pivot element）为零，会导致除以零。如果主元很小，则会放大舍入误差（rounding errors）。

Bad pivot:                       With partial pivoting:
| 0.001  1 | 1.001 |            Swap rows first:
| 1      1 | 2     |            | 1      1 | 2     |
                                 | 0.001  1 | 1.001 |
m = 1/0.001 = 1000              m = 0.001/1 = 0.001
R2 = R2 - 1000*R1               R2 = R2 - 0.001*R1
| 0.001  1     | 1.001   |      | 1      1     | 2     |
| 0     -999   | -999.0  |      | 0      0.999 | 0.999 |

x2 = 1.000 (correct)            x2 = 1.000 (correct)
x1 = (1.001 - 1)/0.001          x1 = (2 - 1)/1 = 1.000 (correct)
   = 0.001/0.001 = 1.000        Stable because the multiplier is small.

在有限精度的浮点运算中，不使用主元法的版本可能会丢失有效数字。部分主元法始终选择当前可用的最大主元，以最大限度地减少误差放大。

### LU 分解（LU decomposition）

LU 分解将矩阵 A 分解为一个下三角矩阵（lower triangular matrix）L 和一个上三角矩阵 U：A = LU。L 矩阵存储了高斯消元过程中的乘数。U 矩阵是消元后的结果。

A = L @ U

| 2  1  1 |   | 1  0  0 |   | 2  1   1 |
| 4  3  3 | = | 2  1  0 | @ | 0  1   1 |
| 2  3  1 |   | 1  2  1 |   | 0  0  -2 |

为什么要进行分解而不是直接消元？因为一旦得到 L 和 U，对于任意新的 b 求解 Ax = b 的复杂度仅为 O(n^2)：

Ax = b
LUx = b
Let y = Ux:
  Ly = b    (forward substitution, O(n^2))
  Ux = y    (back substitution, O(n^2))

O(n^3) 的计算成本仅在分解阶段支付一次。后续每次求解的复杂度都是 O(n^2)。如果你需要用相同的 A 但不同的 b 向量求解 1000 个方程组，LU 分解能将总工作量减少约 1000/3 倍。

结合部分主元法，你会得到 PA = LU，其中 P 是一个置换矩阵（permutation matrix），用于记录行交换操作。

### QR 分解（QR decomposition）

QR 分解将矩阵 A 分解为一个正交矩阵（orthogonal matrix）Q 和一个上三角矩阵 R：A = QR。

正交矩阵具有性质 Q^T Q = I。它的列向量是标准正交向量（orthonormal vectors）。乘以 Q 会保持向量的长度和夹角不变。

A = Q @ R

Q has orthonormal columns: Q^T Q = I
R is upper triangular

To solve Ax = b:
  QRx = b
  Rx = Q^T b    (just multiply by Q^T, no inversion needed)
  Back substitute to get x.

在求解最小二乘问题时，QR 分解的数值稳定性优于 LU 分解。格拉姆-施密特正交化过程（Gram-Schmidt process）逐列构建 Q：

Given columns a1, a2, ... of A:

q1 = a1 / ||a1||

q2 = a2 - (a2 . q1) * q1        (subtract projection onto q1)
q2 = q2 / ||q2||                (normalize)

q3 = a3 - (a3 . q1) * q1 - (a3 . q2) * q2
q3 = q3 / ||q3||

R[i][j] = qi . aj    for i <= j

每一步都会移除沿之前所有 q 向量方向的分量，仅保留新的正交方向。

### 乔列斯基分解（Cholesky decomposition）

当 A 是对称矩阵（symmetric matrix，A = A^T）且正定（positive definite，所有特征值均为正）时，可以将其分解为 A = L L^T，其中 L 是下三角矩阵。这就是乔列斯基分解。

A = L @ L^T

| 4  2 |   | 2  0 |   | 2  1 |
| 2  5 | = | 1  2 | @ | 0  2 |

L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k < i))
L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]    for i > j

乔列斯基分解的速度是 LU 分解的两倍，且只需一半的存储空间。它仅适用于对称正定矩阵，但这类矩阵在机器学习中非常常见：

- 协方差矩阵（covariance matrices）是对称半正定的（加入正则化后变为正定）。
- 高斯过程（Gaussian processes）中的核矩阵是对称正定的。
- 凸函数在极小值点处的海森矩阵（Hessian matrix）是对称正定的。
- A^T A 始终是对称半正定的。

在高斯过程中，你使用乔列斯基分解对核矩阵 K 进行分解，然后求解 K alpha = y 以获得预测均值。乔列斯基因子还能用于计算边缘似然（marginal likelihood）的对数行列式：log det(K) = 2 * sum(log(diag(L)))。

### 最小二乘法（Least squares）：当 Ax = b 无精确解时

如果 A 是 m x n 矩阵且 m > n（方程数多于未知数），则该方程组是超定的（overdetermined）。此时不存在精确解。取而代之的是最小化平方误差：

minimize ||Ax - b||^2

This is the sum of squared residuals:
  sum((A[i,:] @ x - b[i])^2 for i in range(m))

使误差最小化的解满足正规方程（normal equations）：

A^T A x = A^T b

推导过程：展开 ||Ax - b||^2 = (Ax - b)^T (Ax - b) = x^T A^T A x - 2 x^T A^T b + b^T b。对 x 求梯度并令其为零：2 A^T A x - 2 A^T b = 0。

Original system (overdetermined, 4 equations, 2 unknowns):
| 1  1 |         | 3 |
| 1  2 | x     = | 5 |       No exact x satisfies all 4 equations.
| 1  3 |         | 6 |
| 1  4 |         | 8 |

Normal equations:
A^T A = | 4  10 |    A^T b = | 22 |
        | 10 30 |            | 63 |

Solve: x = [1.5, 1.7]

This is linear regression. x[0] is the intercept, x[1] is the slope.

### 正规方程即线性回归

这种联系是精确的。在线性回归中，数据矩阵 X 的每一行代表一个样本，每一列代表一个特征。目标向量 y 的每个元素对应一个样本。权重向量 w 满足：

X^T X w = X^T y
w = (X^T X)^(-1) X^T y

这就是线性回归的闭式解（closed-form solution）。每次调用 `sklearn.linear_model.LinearRegression.fit()` 时，底层都会计算该公式（或通过 QR 分解或奇异值分解（SVD）计算等价形式）。

在矩阵中加入正则化项 lambda * I，即可得到岭回归（ridge regression）：

(X^T X + lambda * I) w = X^T y
w = (X^T X + lambda * I)^(-1) X^T y

正则化改善了矩阵的条件数（condition number），使其更容易被精确求逆，同时通过将权重向零收缩来防止过拟合。当 lambda > 0 时，矩阵 X^T X + lambda * I 始终是对称正定的，因此可以使用乔列斯基分解进行求解。

### 伪逆（Pseudoinverse / Moore-Penrose）

伪逆 A+ 将矩阵求逆的概念推广到了非方阵和奇异矩阵（singular matrices）。对于任意矩阵 A：

x = A+ b

where A+ = V Sigma+ U^T    (computed via SVD)

Sigma+ 是通过将每个非零奇异值取倒数并对结果进行转置得到的。如果 A = U Sigma V^T，则 A+ = V Sigma+ U^T。

A = U Sigma V^T        (SVD)

Sigma = | 5  0 |       Sigma+ = | 1/5  0  0 |
        | 0  2 |                | 0  1/2  0 |
        | 0  0 |

A+ = V Sigma+ U^T

伪逆给出了最小范数最小二乘解。如果方程组：
- 有唯一解：A+ b 直接给出该解。
- 无解：A+ b 给出最小二乘解。
- 有无穷多解：A+ b 给出范数 ||x|| 最小的那个解。

NumPy 的 `np.linalg.lstsq` 和 `np.linalg.pinv` 在底层均使用奇异值分解（SVD）。

### 条件数（Condition number）

条件数衡量解对输入微小变化的敏感程度。对于矩阵 A，条件数为：

kappa(A) = ||A|| * ||A^(-1)|| = sigma_max / sigma_min

其中 sigma_max 和 sigma_min 分别是最大和最小奇异值。

Well-conditioned (kappa ~ 1):        Ill-conditioned (kappa ~ 10^15):
Small change in b -->                Small change in b -->
small change in x                    huge change in x

| 2  0 |   kappa = 2/1 = 2          | 1   1          |   kappa ~ 10^15
| 0  1 |   safe to solve            | 1   1+10^(-15) |   solution is garbage

经验法则：
- kappa < 100：安全，解是精确的。
- kappa ~ 10^k：浮点运算会损失约 k 位有效数字精度。
- kappa ~ 10^16（针对 float64）：解毫无意义。该矩阵在数值上可视为奇异矩阵。

在机器学习中，当特征近似共线（collinear）时会出现病态（ill-conditioning）问题。正则化（添加 lambda * I）能将条件数从 sigma_max / sigma_min 改善为 (sigma_max + lambda) / (sigma_min + lambda)。

### 迭代法：共轭梯度法（Conjugate gradient）

对于超大规模稀疏方程组（数百万个未知数），LU 或乔列斯基等直接法（direct methods）计算成本过高。迭代法（iterative methods）通过多次迭代逐步优化初始猜测来逼近解。

当 A 为对称正定时，共轭梯度法（CG）可用于求解 Ax = b。在精确算术下，它最多经过 n 次迭代即可找到精确解；但如果 A 的特征值聚集在一起，通常收敛速度会快得多。

Algorithm sketch:
  x0 = initial guess (often zero)
  r0 = b - A x0           (residual)
  p0 = r0                 (search direction)

  For k = 0, 1, 2, ...:
    alpha = (rk . rk) / (pk . A pk)
    x_{k+1} = xk + alpha * pk
    r_{k+1} = rk - alpha * A pk
    beta = (r_{k+1} . r_{k+1}) / (rk . rk)
    p_{k+1} = r_{k+1} + beta * pk
    if ||r_{k+1}|| < tolerance: stop

CG 算法常用于：
- 大规模优化（牛顿-CG 法）
- 偏微分方程（PDE）离散化求解
- 核方法（当核矩阵过大无法进行分解时）
- 作为其他迭代求解器的预处理（preconditioning）

收敛速度取决于条件数。条件数越好的系统收敛越快，这也是正则化能带来帮助的另一个原因。

### 全局概览：何时使用何种方法

| 方法 | 适用条件 | 计算成本 | 使用场景 |
|--------|-------------|------|----------|
| 高斯消元法 | 方阵，非奇异 A | O(n^3) | 一次性求解方阵系统 |
| LU 分解 | 方阵，非奇异 A | O(n^3) 分解 + O(n^2) 求解 | 使用相同 A 多次求解 |
| QR 分解 | 任意 A (m >= n) | O(mn^2) | 最小二乘，数值稳定 |
| 乔列斯基分解 | 对称正定 A | O(n^3/3) | 协方差矩阵，高斯过程，岭回归 |
| 正规方程 | 超定 (m > n) | O(mn^2 + n^3) | 线性回归（n 较小） |
| SVD / 伪逆 | 任意 A | O(mn^2) | 秩亏系统，最小范数解 |
| 共轭梯度法 | 对称正定，稀疏 A | O(n * k * nnz) | 大型稀疏系统，k = 迭代次数 |

### 与机器学习的联系

本节介绍的每种方法都会出现在生产环境的机器学习系统中：

**线性回归。** 闭式解通过求解正规方程 X^T X w = X^T y 获得。具体实现可通过乔列斯基分解（当 n 较小时）、QR 分解（当注重数值稳定性时）或奇异值分解（当矩阵可能秩亏时）。

**岭回归。** 在 X^T X 中加入 lambda * I。正则化后的系统 (X^T X + lambda * I) w = X^T y 始终可通过乔列斯基分解求解，因为当 lambda > 0 时，X^T X + lambda * I 是对称正定的。

**高斯过程。** 预测均值需要求解 K alpha = y，其中 K 是核矩阵。对 K 进行乔列斯基分解是标准做法。对数边缘似然计算使用 log det(K) = 2 sum(log(diag(L)))。

**神经网络初始化。** 正交初始化使用 QR 分解来创建列向量标准正交的权重矩阵。这能防止深层网络中的信号坍缩。

**预处理。** 大规模优化器使用不完全乔列斯基分解或不完全 LU 分解作为共轭梯度求解器的预处理器。

**特征工程。** X^T X 的条件数能告诉你特征是否存在共线性。如果 kappa 很大，应剔除部分特征或添加正则化项。

## 构建实现

### 步骤 1：带部分主元的高斯消元法 (Gaussian elimination with partial pivoting)

import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]

    return x

### 步骤 2：LU 分解 (LU decomposition)

def lu_decompose(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    P = np.eye(n)

    for k in range(n):
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U

def lu_solve(P, L, U, b):
    n = len(b)
    Pb = P @ b.astype(float)

    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x

### 步骤 3：乔列斯基分解 (Cholesky decomposition)

def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L

### 步骤 4：基于正规方程的最小二乘法 (Least squares via normal equations)

def least_squares_normal(A, b):
    AtA = A.T @ A
    Atb = A.T @ b
    return gaussian_elimination(AtA, Atb)

def ridge_regression(A, b, lam):
    n = A.shape[1]
    AtA = A.T @ A + lam * np.eye(n)
    Atb = A.T @ b
    L = cholesky(AtA)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (Atb[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L.T[i, i+1:] @ x[i+1:]) / L.T[i, i]
    return x

### 步骤 5：条件数 (Condition number)

def condition_number(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0] / S[-1]

## 实际应用

将线性回归（Linear Regression）和岭回归（Ridge Regression）的各个部分整合到真实数据上：

np.random.seed(42)
X_raw = np.random.randn(100, 3)
w_true = np.array([2.0, -1.0, 0.5])
y = X_raw @ w_true + np.random.randn(100) * 0.1

X = np.column_stack([np.ones(100), X_raw])

w_ols = least_squares_normal(X, y)
print(f"OLS weights (ours):    {w_ols}")

w_np = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"OLS weights (numpy):   {w_np}")
print(f"Max difference: {np.max(np.abs(w_ols - w_np)):.2e}")

w_ridge = ridge_regression(X, y, lam=1.0)
print(f"Ridge weights (ours):  {w_ridge}")

from sklearn.linear_model import Ridge
ridge_sk = Ridge(alpha=1.0, fit_intercept=False)
ridge_sk.fit(X, y)
print(f"Ridge weights (sklearn): {ridge_sk.coef_}")

## 交付上线

本课程将产出：
- `code/linear_systems.py` 文件，包含从零实现的高斯消元法（Gaussian Elimination）、LU分解（LU Decomposition）、乔列斯基分解（Cholesky Decomposition）、最小二乘法（Least Squares）和岭回归的代码
- 一个可运行的演示，证明正规方程（Normal Equations）与 sklearn 的 `LinearRegression` 会产生相同的权重

## 练习

1. 使用你实现的高斯消元法、LU求解器以及 `np.linalg.solve` 求解方程组 `[[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27]`。验证这三种方法在浮点数容差范围内给出的结果是否一致。

2. 生成一个 50x5 的随机矩阵 X 和目标向量 y = X @ w_true + noise。分别使用正规方程、QR分解（通过 `np.linalg.qr`）、奇异值分解（SVD，通过 `np.linalg.svd`）以及 `np.linalg.lstsq` 求解 w。对比这四种解法。计算 X^T X 的条件数（Condition Number），并解释它如何影响你对不同方法的信任程度。

3. 通过使两列几乎相同（例如，第2列 = 第1列 + 1e-10 * noise）来构造一个近似奇异矩阵（Nearly Singular Matrix）。计算其条件数。分别在无正则化（Regularization）和添加正则化项（加上 0.01 * I）的情况下求解 Ax = b。对比解和残差（Residuals）。解释为什么正则化能够改善结果。

4. 针对一个 100x100 的随机对称正定矩阵（Symmetric Positive Definite Matrix）实现共轭梯度法（Conjugate Gradient Algorithm）。统计其收敛到容差 1e-8 所需的迭代次数。将其与理论最大迭代次数 n 进行对比。

5. 对尺寸分别为 10、50、200、500 的对称正定矩阵，分别测试你实现的乔列斯基求解器、LU求解器以及 `np.linalg.solve` 的运行时间。绘制结果图表。验证乔列斯基分解的速度是否约为 LU 分解的两倍。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 线性系统 (Linear System) | “求解 x” | 一组线性方程组 Ax = b。求解 x 意味着寻找在变换 A 下能产生输出 b 的输入。 |
| 高斯消元法 (Gaussian Elimination) | “行化简” | 通过行变换系统地将主对角线以下的元素消为零，生成一个可通过回代求解的上三角方程组。时间复杂度为 O(n^3)。 |
| 部分主元法 (Partial Pivoting) | “交换行以提高稳定性” | 在第 k 列消元前，将该列中绝对值最大的元素所在行交换至主元位置。可避免除以极小数值导致的数值不稳定。 |
| LU分解 (LU Decomposition) | “分解为三角矩阵” | 将矩阵表示为 A = LU，其中 L 是下三角矩阵（存储乘数），U 是上三角矩阵（消元后的矩阵）。在多次求解时可分摊 O(n^3) 的计算成本。 |
| QR分解 (QR Decomposition) | “正交分解” | 将矩阵表示为 A = QR，其中 Q 的列向量标准正交，R 为上三角矩阵。在求解最小二乘问题时比 LU 分解更稳定。 |
| 乔列斯基分解 (Cholesky Decomposition) | “矩阵的平方根” | 针对对称正定矩阵 A，将其表示为 A = LL^T。计算成本仅为 LU 分解的一半。常用于协方差矩阵、核矩阵和岭回归。 |
| 最小二乘法 (Least Squares) | “无法精确求解时的最佳拟合” | 当方程组为超定系统（方程数多于未知数）时，最小化残差平方和 ||Ax - b||^2。 |
| 正规方程 (Normal Equations) | “微积分捷径” | A^T A x = A^T b。通过将 ||Ax - b||^2 的梯度设为零推导得出。这正是线性回归的闭式解 (Closed-form Solution)。 |
| 伪逆 (Pseudoinverse) | “非方阵的求逆” | 通过奇异值分解 (SVD) 计算 A+ = V Sigma+ U^T。可为任意矩阵（无论方阵或矩形、奇异或非奇异）提供最小范数最小二乘解。 |
| 条件数 (Condition Number) | “答案有多可靠” | kappa = sigma_max / sigma_min。衡量解对输入扰动的敏感程度。大约会损失 log10(kappa) 位有效数字精度。 |
| 岭回归 (Ridge Regression) | “正则化最小二乘” | 求解 (X^T X + lambda I) w = X^T y。添加 lambda I 可改善矩阵条件数并将权重向零收缩，从而防止过拟合 (Overfitting)。 |
| 共轭梯度法 (Conjugate Gradient) | “大矩阵的迭代求解 Ax=b” | 针对对称正定系统的迭代求解器。最多在 n 步内收敛。适用于因分解成本过高而难以处理的稀疏大型系统。 |
| 超定系统 (Overdetermined System) | “数据多于参数” | 在 m×n 系统中满足 m > n。不存在精确解。最小二乘法可找到最佳近似解。这正是所有回归问题的本质。 |
| 回代 (Back Substitution) | “自底向上求解” | 给定上三角方程组，先求解最后一个方程，然后依次向前代入。时间复杂度为 O(n^2)。 |
| 前代 (Forward Substitution) | “自顶向下求解” | 给定下三角方程组，先求解第一个方程，然后依次向后代入。时间复杂度为 O(n^2)。用于 LU 分解求解中的 L 步骤。 |

## 进一步阅读

- [MIT 18.06：线性代数（Linear Algebra）](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)（Gilbert Strang）—— 讲解线性系统（Linear Systems）与矩阵分解（Matrix Factorizations）的权威课程
- [数值线性代数（Numerical Linear Algebra）](https://people.maths.ox.ac.uk/trefethen/text.html)（Trefethen & Bau）—— 理解数值稳定性（Numerical Stability）、条件数（Conditioning）以及算法失效原因的标准参考书
- [矩阵计算（Matrix Computations）](https://www.cs.cornell.edu/cv/GolubVanLoan4/golubandvanloan.htm)（Golub & Van Loan）—— 涵盖各类矩阵算法的百科全书式参考书
- [3Blue1Brown：逆矩阵（Inverse Matrices）](https://www.3blue1brown.com/lessons/inverse-matrices) —— 从几何视角直观阐释求解 Ax = b 的含义
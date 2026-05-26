# 凸优化 (Convex Optimization)

> 凸问题只有一个谷底。神经网络则有数百万个。了解这一差异至关重要。

**类型：** 构建
**语言：** Python
**先修要求：** 第一阶段，第 04 课（机器学习微积分）、第 08 课（优化）
**时长：** 约 90 分钟

## 学习目标

- 使用定义、二阶导数和海森矩阵 (Hessian Matrix) 准则检验函数是否为凸函数
- 实现牛顿法 (Newton's Method)，并将其二次收敛性与梯度下降法进行对比
- 使用拉格朗日乘子法 (Lagrange Multipliers) 求解约束优化问题，并解释 KKT 条件 (Karush-Kuhn-Tucker Conditions)
- 解释为何神经网络的损失地形 (Loss Landscape) 是非凸的，但随机梯度下降 (Stochastic Gradient Descent, SGD) 仍能找到优质解

## 问题背景

第 08 课讲解了梯度下降法 (Gradient Descent)、动量法 (Momentum) 和 Adam 优化器。这些优化器可以在任意曲面上“下坡”寻优，但它们无法提供理论保证。在非凸地形上运行梯度下降法可能会陷入糟糕的局部极小值 (Local Minimum)，卡在鞍点 (Saddle Point)，或者永远震荡。尽管如此，你依然会使用它们，因为神经网络本质上是非凸的，且目前别无选择。

然而，机器学习中的许多问题实际上是凸的。例如线性回归 (Linear Regression)、逻辑回归 (Logistic Regression)、支持向量机 (Support Vector Machines, SVM)、LASSO 和岭回归 (Ridge Regression)。针对这些问题，存在更强大的工具：带有数学保证的优化方法。凸问题恰好只有一个谷底。任何沿下坡方向搜索的算法都能抵达全局最小值 (Global Minimum)。无需重启，无需设计学习率调度策略，也无需祈祷。

理解凸性 (Convexity) 能带来三方面的好处。首先，它能帮你判断当前问题是容易求解的（凸问题）还是难以求解的（非凸问题）。其次，它为你提供了更快的工具，例如用于凸问题的牛顿法。最后，它能解释贯穿机器学习始终的核心概念：将正则化视为一种约束、支持向量机中的对偶性 (Duality)，以及深度学习为何在违背凸性所赋予的所有优良性质的情况下依然有效。

## 核心概念

### 凸集 (Convex Sets)

如果集合 S 中任意两点之间的线段完全包含在该集合内，则称 S 为凸集。

| 凸集 | 非凸集 |
|---|---|
| **矩形**：内部任意两点均可由一条完全位于内部的线段连接 | **星形/月牙形**：内部两点间的连线可能穿过集合外部 |
| **三角形**：所有内部点均满足相同性质 | **圆环/甜甜圈形**：中间的孔洞导致部分线段会穿出集合 |
| 任意两点间的线段始终位于集合内 | 部分点对之间的线段会穿出集合 |

形式化检验：对于 S 中的任意两点 x, y 以及任意 t ∈ [0, 1]，点 tx + (1-t)y 也必须属于 S。

凸集示例：
- 直线、平面、整个 R^n 空间
- 球体（圆、球面、超球面）
- 半空间：{x : a^T x <= b}
- 任意数量凸集的交集

非凸集示例：
- 圆环（甜甜圈形）
- 两个不相交圆的并集
- 任何带有“凹陷”或“孔洞”的集合

### 凸函数 (Convex Functions)

如果函数 f 的定义域是凸集，且对于定义域内的任意两点 x, y 以及任意 t ∈ [0, 1]，满足：

f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)

几何意义：函数图像上任意两点间的线段始终位于图像上方或与图像重合。

| 性质 | 凸函数 | 非凸函数 |
|---|---|---|
| **线段检验** | 图像上任意两点间的连线始终**位于曲线上方或与之重合** | 图像上部分点间的连线会**下凹至曲线下方** |
| **形状** | 单一向上弯曲的碗状/谷状 | 具有多个峰谷且曲率混合 |
| **局部极小值** | 每个局部极小值都是全局极小值 | 可能存在多个高度不同的局部极小值 |

常见凸函数：
- f(x) = x^2（抛物线）
- f(x) = |x|（绝对值函数）
- f(x) = e^x（指数函数）
- f(x) = max(0, x)（ReLU，尽管是分段线性函数）
- f(x) = -log(x)（x > 0 时的负对数函数）
- 任意线性函数 f(x) = a^T x + b（既是凸函数也是凹函数）

### 凸性检验

三种实用检验方法，按从易到严的顺序排列。

**检验 1：二阶导数检验（一维）**。若对所有 x 均有 f''(x) >= 0，则 f 为凸函数。

- f(x) = x^2：f''(x) = 2 >= 0。凸函数。
- f(x) = x^3：f''(x) = 6x。当 x < 0 时为负。非凸函数。
- f(x) = e^x：f''(x) = e^x > 0。凸函数。

**检验 2：海森矩阵检验（多元）**。若对所有 x，海森矩阵（Hessian Matrix）H(x) 均为半正定矩阵，则 f 为凸函数。海森矩阵是由二阶偏导数组成的矩阵。

**检验 3：定义检验**。直接验证不等式 f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)。适用于导数难以计算的函数。

### 凸性的重要性

凸优化（Convex Optimization）的核心定理：

**对于凸函数，每个局部极小值都是全局极小值。**

这意味着梯度下降（Gradient Descent）不会陷入局部陷阱。任何下坡路径都会导向同一个答案。该算法保证能收敛到最优解。

graph LR
    subgraph "Convex: ONE answer"
        direction TB
        C1["Loss surface has a single valley"] --> C2["Gradient descent ALWAYS finds the global minimum"]
    end
    subgraph "Non-convex: MANY traps"
        direction TB
        N1["Loss surface has multiple valleys and peaks"] --> N2["Gradient descent may get stuck in a local minimum"]
        N2 --> N3["Global minimum might be missed"]
    end

带来的影响：
- 无需随机重启
- 无需复杂的学习率调度策略
- 可证明收敛性（收敛速率取决于函数性质）
- 解是唯一的（平坦区域除外）

### 机器学习中的凸与非凸问题

| 问题 | 是否为凸？ | 原因 |
|---------|---------|-----|
| 线性回归（均方误差 MSE） | 是 | 损失函数关于权重是二次的 |
| 逻辑回归 | 是 | 对数损失关于权重是凸的 |
| 支持向量机（SVM，铰链损失） | 是 | 线性函数的最大值 |
| LASSO（L1 回归） | 是 | 凸函数之和仍为凸函数 |
| 岭回归（L2） | 是 | 二次函数 + 二次函数 = 凸函数 |
| 神经网络（任意损失） | 否 | 非线性激活函数导致非凸的损失地形 |
| k-means 聚类 | 否 | 离散的分配步骤 |
| 矩阵分解 | 否 | 未知量的乘积 |

具有凸损失函数的线性模型是凸的。一旦引入带有非线性激活函数的隐藏层，凸性即被破坏。

### 海森矩阵 (Hessian Matrix)

函数 f: R^n -> R 的海森矩阵 H 是一个由二阶偏导数组成的 n x n 矩阵。

H[i][j] = d^2 f / (dx_i dx_j)

对于 f(x, y) = x^2 + 3xy + y^2：

df/dx = 2x + 3y       d^2f/dx^2 = 2      d^2f/dxdy = 3
df/dy = 3x + 2y       d^2f/dydx = 3      d^2f/dy^2 = 2

H = [ 2  3 ]
    [ 3  2 ]

海森矩阵揭示了函数的曲率信息：
- 特征值全为正：函数在所有方向上均向上弯曲（在该点处为凸）
- 特征值全为负：函数在所有方向上均向下弯曲（凹函数，局部极大值）
- 符号混合：鞍点（某些方向向上弯曲，其他方向向下弯曲）
- 存在零特征值：在该方向上平坦（退化）

要保证凸性，海森矩阵必须在所有位置均为半正定（所有特征值 >= 0），而不仅仅是在某一点。

### 牛顿法 (Newton's Method)

梯度下降使用一阶信息（梯度）。牛顿法使用二阶信息（海森矩阵）。它在当前点拟合一个二次近似，并直接跳跃至该二次函数的极小值点。

Update rule:
  x_new = x - H^(-1) * gradient

Compare to gradient descent:
  x_new = x - lr * gradient

牛顿法用海森矩阵的逆矩阵替代了标量学习率。这会根据局部曲率自动调整步长和方向。

graph TD
    subgraph "Gradient Descent"
        GD1["Start"] --> GD2["Step 1"]
        GD2 --> GD3["Step 2"]
        GD3 --> GD4["..."]
        GD4 --> GD5["Step ~500: Converged"]
        GD_note["Follows gradient blindly — many small steps"]
    end
    subgraph "Newton's Method"
        NM1["Start"] --> NM2["Step 1"]
        NM2 --> NM3["..."]
        NM3 --> NM4["Step ~5: Converged"]
        NM_note["Uses curvature for optimal steps"]
    end

优势：
- 在极小值附近具有二次收敛性（每步误差平方级下降）
- 无需调节学习率
- 尺度不变性（无论问题如何参数化均有效）

劣势：
- 计算海森矩阵需要 O(n^2) 内存，求逆需要 O(n^3) 计算量
- 对于具有 100 万权重的神经网络，这将产生 10^12 个矩阵元素和 10^18 次运算
- 不适用于深度学习

### 约束优化 (Constrained Optimization)

无约束优化：在所有 x 上最小化 f(x)。
约束优化：在满足约束条件的情况下最小化 f(x)。

实际问题通常带有约束。你希望最小化成本，但预算有限；你希望最小化误差，但模型复杂度受限。

graph LR
    subgraph "Unconstrained"
        U1["Loss function"] --> U2["Free minimum: lowest point of the loss surface"]
    end
    subgraph "Constrained"
        C1["Loss function"] --> C2["Constrained minimum: lowest point within the feasible region"]
        C3["Constraint boundary limits the search space"]
    end

### 拉格朗日乘子法 (Lagrange Multipliers)

拉格朗日乘子法将约束问题转化为无约束问题。

问题：在 g(x) = 0 的约束下最小化 f(x)。

解法：引入一个新变量（拉格朗日乘子 lambda），并求解无约束问题：

L(x, lambda) = f(x) + lambda * g(x)

在最优解处，L 的梯度为零：

dL/dx = df/dx + lambda * dg/dx = 0
dL/dlambda = g(x) = 0

几何直观：在约束极小值点，f 的梯度必须与约束 g 的梯度平行。若不平行，你就可以沿约束曲面移动并进一步降低 f 的值。

graph LR
    A["Contours of f(x,y): concentric ellipses"] --- S["Solution point"]
    B["Constraint curve g(x,y) = 0"] --- S
    S --- C["At the solution, gradient of f is parallel to gradient of g"]

示例：在 x + y = 1 的约束下最小化 f(x,y) = x^2 + y^2。

L = x^2 + y^2 + lambda(x + y - 1)

dL/dx = 2x + lambda = 0  =>  x = -lambda/2
dL/dy = 2y + lambda = 0  =>  y = -lambda/2
dL/dlambda = x + y - 1 = 0

From first two: x = y
Substituting: 2x = 1, so x = y = 0.5, lambda = -1

直线 x + y = 1 上距离原点最近的点是 (0.5, 0.5)。

### KKT 条件 (Karush-Kuhn-Tucker Conditions)

KKT 条件将拉格朗日乘子法推广至不等式约束。

问题：在 g_i(x) <= 0 (i = 1, ..., m) 的约束下最小化 f(x)。

KKT 条件（最优性的必要条件）：

1. Stationarity:    df/dx + sum(lambda_i * dg_i/dx) = 0
2. Primal feasibility:  g_i(x) <= 0  for all i
3. Dual feasibility:    lambda_i >= 0  for all i
4. Complementary slackness:  lambda_i * g_i(x) = 0  for all i

互补松弛性（Complementary Slackness）是核心洞察：要么约束是活跃的（g_i = 0，解位于边界上），要么乘子为零（该约束不起作用）。不影响解的约束其 lambda = 0。

KKT 条件是支持向量机（SVM）的核心。支持向量正是那些约束活跃的数据点（lambda > 0）。所有其他数据点的 lambda = 0，不影响决策边界。

### 作为约束优化的正则化

L1 和 L2 正则化并非随意的技巧，它们本质上是伪装成无约束形式的约束优化问题。

**L2 正则化（岭回归 Ridge）：**

minimize  Loss(w)  subject to  ||w||^2 <= t

Equivalent unconstrained form:
minimize  Loss(w) + lambda * ||w||^2

约束 ||w||^2 <= t 定义了一个球体（2D 中为圆，3D 中为球面）。最优解是损失等高线首次接触该球体的位置。

**L1 正则化（LASSO）：**

minimize  Loss(w)  subject to  ||w||_1 <= t

Equivalent unconstrained form:
minimize  Loss(w) + lambda * ||w||_1

约束 ||w||_1 <= t 定义了一个菱形（2D 中为旋转的正方形）。

| 性质 | L2 约束（圆形） | L1 约束（菱形） |
|---|---|---|
| **约束形状** | 圆形（高维为球体） | 菱形（2D 中为旋转正方形） |
| **损失等高线接触位置** | 平滑边界——圆上的任意点 | 尖角——与坐标轴对齐 |
| **解的行为** | 权重较小但非零 | 部分权重恰好为零（稀疏） |
| **结果** | 权重收缩 | 特征选择 |

这解释了为何 L1 能产生稀疏模型（特征选择），而 L2 仅收缩权重。菱形的尖角与坐标轴对齐，损失等高线更有可能接触到尖角，从而将一个或多个权重精确置零。

### 对偶性 (Duality)

每个约束优化问题（原问题 Primal）都有一个对应的问题（对偶问题 Dual）。对于凸问题，原问题与对偶问题具有相同的最优值。这称为强对偶性（Strong Duality）。

拉格朗日对偶函数：

Primal: minimize f(x) subject to g(x) <= 0
Lagrangian: L(x, lambda) = f(x) + lambda * g(x)
Dual function: d(lambda) = min_x L(x, lambda)
Dual problem: maximize d(lambda) subject to lambda >= 0

对偶性的重要性：
- 对偶问题有时比原问题更容易求解
- SVM 通常在对偶形式下求解，此时问题仅依赖于数据点之间的点积（从而启用核技巧 Kernel Trick）
- 对偶问题为原问题最优值提供下界，可用于检验解的质量

针对 SVM 的具体情况：

Primal: find w, b that maximize the margin 2/||w|| subject to
        y_i(w^T x_i + b) >= 1 for all i

Dual:   maximize sum(alpha_i) - 0.5 * sum_ij(alpha_i * alpha_j * y_i * y_j * x_i^T x_j)
        subject to alpha_i >= 0 and sum(alpha_i * y_i) = 0

The dual only involves dot products x_i^T x_j.
Replace x_i^T x_j with K(x_i, x_j) to get the kernel trick.

### 尽管非凸，深度学习为何依然有效

神经网络的损失函数具有极强的非凸性。按照所有经典标准，优化它们理应失败。然而，随机梯度下降（Stochastic Gradient Descent, SGD）却能可靠地找到优质解。以下几个因素解释了这一现象。

**大多数局部极小值已经足够好。** 在高维空间中，随机临界点（梯度为零的点）绝大多数是鞍点，而非局部极小值。少数存在的局部极小值，其损失值通常也接近全局极小值。当参数空间高达数百万维时，陷入极差的局部极小值的概率微乎其微。

**鞍点而非局部极小值才是真正的障碍。** 在具有 n 个参数的函数中，鞍点同时包含正曲率和负曲率方向。对于高维空间中的随机临界点，所有 n 个特征值均为正（即局部极小值）的概率约为 2^(-n)。几乎所有临界点都是鞍点。SGD 的噪声有助于逃离这些鞍点。

**过参数化平滑了损失地形。** 参数量超过训练样本量的网络具有更平滑、连通性更好的损失曲面。更宽的网络具有更少的劣质局部极小值。这虽然反直觉，但在经验上是一致的。

**损失地形结构：**

| 性质 | 低维空间 | 高维空间 |
|---|---|---|
| **地形** | 大量孤立的峰谷 | 平滑连通的谷地 |
| **极小值** | 大量孤立的局部极小值 | 劣质局部极小值极少；大多数接近最优 |
| **寻优难度** | 难以找到全局极小值 | 多条路径均可导向优质解 |
| **临界点** | 局部极小值与鞍点混合 | 绝大多数为鞍点，而非局部极小值 |

**随机噪声充当隐式正则化。** 小批量 SGD 引入的噪声可防止模型陷入尖锐极小值。尖锐极小值会导致过拟合；平坦极小值则泛化能力更强。该噪声使优化过程偏向损失地形的平坦区域。

### 实践中的二阶方法

纯牛顿法对于大型模型并不实用。几种近似方法使得二阶信息得以应用。

**L-BFGS（有限内存 BFGS）**：利用最近 m 次梯度差来近似海森矩阵的逆。仅需 O(mn) 内存而非 O(n^2)。适用于参数量约 10,000 以内的问题。常用于经典机器学习（逻辑回归、条件随机场 CRF），但不用于深度学习。

**自然梯度（Natural Gradient）**：使用费雪信息矩阵（Fisher Information Matrix，对数似然的期望海森矩阵）替代标准海森矩阵。这考虑了概率分布的几何特性。K-FAC（克罗内克因子近似曲率）将费雪矩阵近似为克罗内克积，使其在神经网络中具备实用性。

**无海森优化（Hessian-free Optimization）**：使用共轭梯度法求解 Hx = g，而无需显式构造 H。仅需计算海森向量积，可通过自动微分在 O(n) 时间内完成。

**对角近似**：Adam 的二阶矩是对海森矩阵对角线的对角近似。AdaHessian 通过 Hutchinson 估计器使用真实的海森对角元素对此进行了扩展。

| 方法 | 内存占用 | 单步计算成本 | 适用场景 |
|--------|--------|--------------|-------------|
| 梯度下降 | O(n) | O(n) | 基线方法、大型模型 |
| 牛顿法 | O(n^2) | O(n^3) | 小型凸问题 |
| L-BFGS | O(mn) | O(mn) | 中型凸问题 |
| Adam | O(n) | O(n) | 深度学习默认选择 |
| K-FAC | O(n) | 每层 O(n) | 学术研究、大批量训练 |

## 动手实现

### 步骤 1：凸性 (Convexity) 检查器

编写一个函数，通过随机采样点并验证数学定义来经验性地测试凸性。

import random
import math

def check_convexity(f, dim, bounds=(-5, 5), samples=1000):
    violations = 0
    for _ in range(samples):
        x = [random.uniform(*bounds) for _ in range(dim)]
        y = [random.uniform(*bounds) for _ in range(dim)]
        t = random.uniform(0, 1)
        mid = [t * xi + (1 - t) * yi for xi, yi in zip(x, y)]
        lhs = f(mid)
        rhs = t * f(x) + (1 - t) * f(y)
        if lhs > rhs + 1e-10:
            violations += 1
    return violations == 0, violations

### 步骤 2：二维牛顿法 (Newton's Method)

使用显式海森矩阵 (Hessian Matrix) 实现牛顿法。将其收敛速度与梯度下降法 (Gradient Descent) 进行对比。

def newtons_method(f, grad_f, hessian_f, x0, steps=50, tol=1e-12):
    x = list(x0)
    history = [x[:]]
    for _ in range(steps):
        g = grad_f(x)
        H = hessian_f(x)
        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
        if abs(det) < 1e-15:
            break
        H_inv = [
            [H[1][1] / det, -H[0][1] / det],
            [-H[1][0] / det, H[0][0] / det],
        ]
        dx = [
            H_inv[0][0] * g[0] + H_inv[0][1] * g[1],
            H_inv[1][0] * g[0] + H_inv[1][1] * g[1],
        ]
        x = [x[0] - dx[0], x[1] - dx[1]]
        history.append(x[:])
        if sum(gi ** 2 for gi in g) < tol:
            break
    return history

### 步骤 3：拉格朗日乘子法 (Lagrange Multiplier) 求解器

通过对拉格朗日函数 (Lagrangian) 应用梯度下降法来求解约束优化 (Constrained Optimization) 问题。

def lagrange_solve(f_grad, g_val, g_grad, x0, lr=0.01,
                   lr_lambda=0.01, steps=5000):
    x = list(x0)
    lam = 0.0
    history = []
    for _ in range(steps):
        fg = f_grad(x)
        gv = g_val(x)
        gg = g_grad(x)
        x = [
            xi - lr * (fgi + lam * ggi)
            for xi, fgi, ggi in zip(x, fg, gg)
        ]
        lam = lam + lr_lambda * gv
        history.append((x[:], lam, gv))
    return history

### 步骤 4：一阶与二阶方法对比

在同一个二次函数 (Quadratic Function) 上运行梯度下降法和牛顿法。统计达到收敛所需的迭代步数。

def quadratic(x):
    return 5 * x[0] ** 2 + x[1] ** 2

def quadratic_grad(x):
    return [10 * x[0], 2 * x[1]]

def quadratic_hessian(x):
    return [[10, 0], [0, 2]]

牛顿法仅需 1 步即可收敛（它对二次函数是精确求解的）。梯度下降法则需要数百步，因为海森矩阵的特征值 (Eigenvalues) 相差 5 倍，导致目标函数呈现狭长的山谷形状。

## 实际应用

在选择机器学习（Machine Learning）模型和求解器（Solver）时，凸性分析（Convexity Analysis）具有直接的指导意义。

对于凸问题（Convex Problems）（如逻辑回归（Logistic Regression）、支持向量机（Support Vector Machines, SVM）、LASSO）：
- 使用专用求解器（如 `liblinear`、`CVXPY`、`scipy.optimize.minimize` 并设置 `method='L-BFGS-B'`）
- 预期存在唯一的全局解（Global Solution）
- 二阶方法（Second-order Methods）既实用又高效

对于非凸问题（Non-convex Problems）（如神经网络（Neural Networks））：
- 使用一阶方法（First-order Methods）（如随机梯度下降（Stochastic Gradient Descent, SGD）、Adam）
- 接受解依赖于初始化和随机性这一事实
- 利用过参数化（Overparameterization）、噪声（Noise）和学习率调度（Learning Rate Schedules）作为隐式正则化（Implicit Regularization）
- 不要浪费时间寻找全局最小值（Global Minimum）。一个良好的局部最小值（Local Minimum）已足够。

from scipy.optimize import minimize

result = minimize(
    fun=lambda w: sum((y - X @ w) ** 2) + 0.1 * sum(w ** 2),
    x0=np.zeros(d),
    method='L-BFGS-B',
    jac=lambda w: -2 * X.T @ (y - X @ w) + 0.2 * w,
)

对于支持向量机（SVM），其对偶形式（Dual Formulation）允许你使用核技巧（Kernel Trick）：

from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
print(f"Support vectors: {svm.n_support_}")

## 练习（Exercises）

1. **凸性示例集（Convexity Gallery）。** 使用检查器（Checker）测试以下函数的凸性：`f(x) = x^4`、`f(x) = sin(x)`、`f(x,y) = x^2 + y^2`、`f(x,y) = x*y`、`f(x) = max(x, 0)`。解释为什么每个结果都是合理的。

2. **牛顿法与梯度下降法对比（Newton vs Gradient Descent Race）。** 从起点 `(10, 10)` 开始，在 `f(x,y) = 50*x^2 + y^2` 上运行这两种方法。每种方法需要多少步才能使损失（Loss）小于 `1e-10`？当条件数（Condition Number，即最大与最小海森矩阵（Hessian Matrix）特征值之比）增大时，梯度下降法会发生什么变化？

3. **拉格朗日乘子几何（Lagrange Multiplier Geometry）。** 在约束条件 `x + 2y = 4` 下最小化 `f(x,y) = (x-3)^2 + (y-3)^2`。通过验证在解处 `f` 的梯度（Gradient）与 `g` 的梯度平行来确认该解。

4. **正则化约束（Regularization Constraint）。** 实现 L1 约束优化（L1-constrained Optimization）：在约束条件 `|x| + |y| <= 1` 下最小化 `(x-3)^2 + (y-2)^2`。证明解中有一个坐标为零（菱形约束带来的稀疏性（Sparsity））。

5. **海森矩阵特征值分析（Hessian Eigenvalue Analysis）。** 计算罗森布罗克函数（Rosenbrock Function）在 `(1,1)` 和 `(-1,1)` 处的海森矩阵（Hessian）。计算这两点的特征值（Eigenvalues）。特征值能告诉你最小值处与远离最小值处的曲率（Curvature）有何不同？

## 关键术语（Key Terms）

| 术语 | 含义 |
|------|------|
| 凸集 (Convex set) | 集合中任意两点之间的线段完全包含在该集合内 |
| 凸函数 (Convex function) | 函数图像上任意两点之间的连线始终位于图像上方或与之重合。等价于其海森矩阵（Hessian matrix）处处半正定 |
| 局部极小值 (Local minimum) | 低于其所有邻近点的点。对于凸函数而言，每个局部极小值即为全局极小值 |
| 全局极小值 (Global minimum) | 函数在整个定义域内的最低点 |
| 海森矩阵 (Hessian matrix) | 由所有二阶偏导数组成的矩阵，用于编码曲率信息 |
| 半正定 (Positive semidefinite) | 特征值全部非负的矩阵。可视为“二阶导数 >= 0”在多维空间中的推广 |
| 条件数 (Condition number) | 海森矩阵最大特征值与最小特征值的比值。条件数过高意味着损失函数呈狭长山谷状，会导致梯度下降（Gradient descent）收敛缓慢 |
| 牛顿法 (Newton's method) | 一种二阶优化算法，利用海森矩阵的逆矩阵来确定步长方向和大小。在极小值点附近具有二次收敛性 |
| 拉格朗日乘子 (Lagrange multiplier) | 引入的辅助变量，用于将带约束的优化问题转化为无约束优化问题 |
| KKT条件 (KKT conditions) | 处理不等式约束时最优解的必要条件。是拉格朗日乘子法的推广 |
| 互补松弛条件 (Complementary slackness) | 在最优解处，要么约束处于激活状态，要么其对应的乘子为零，两者不会同时非零 |
| 对偶性 (Duality) | 每个带约束的原问题都有一个对应的对偶问题。对于凸问题，两者的最优值相同 |
| 强对偶性 (Strong duality) | 原问题与对偶问题的最优值相等。在满足斯莱特条件（Slater's condition）的凸问题中成立 |
| L-BFGS算法 (L-BFGS) | 一种近似二阶优化方法，通过存储最近 m 次的梯度差值来替代完整的海森矩阵，以节省内存 |
| 鞍点 (Saddle point) | 梯度为零的点，但在某些方向上是极小值，在另一些方向上是极大值 |
| 过参数化 (Overparameterization) | 模型参数数量超过训练样本数量。能够平滑损失曲面（Loss landscape）并减少不良的局部极小值 |

## 扩展阅读

- [Boyd & Vandenberghe: Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) - 凸优化（Convex Optimization）领域的标准教材，可免费在线获取
- [Bottou, Curtis, Nocedal: Optimization Methods for Large-Scale Machine Learning (2018)](https://arxiv.org/abs/1606.04838) - 架起了凸优化理论与深度学习（Deep Learning）实践之间的桥梁
- [Choromanska et al.: The Loss Surfaces of Multilayer Networks (2015)](https://arxiv.org/abs/1412.0233) - 解释了为何非凸（Non-convex）神经网络的损失曲面并不像看起来那样糟糕
- [Nocedal & Wright: Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) - 关于牛顿法、L-BFGS 以及约束优化（Constrained Optimization）的全面参考书
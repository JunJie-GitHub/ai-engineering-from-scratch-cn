# 支持向量机 (Support Vector Machines)

> 在两类数据之间找到最宽的“街道”。这就是全部的核心思想。

**类型：** 实战构建
**语言：** Python
**先修要求：** 第一阶段（第08课 优化、第14课 范数与距离、第18课 凸优化）
**时长：** 约90分钟

## 学习目标

- 使用合页损失 (Hinge Loss) 和梯度下降 (Gradient Descent) 在原始形式 (Primal Formulation) 上从零实现线性支持向量机
- 解释最大间隔原则 (Maximum Margin Principle)，并能从训练好的模型中识别出支持向量 (Support Vectors)
- 比较线性核 (Linear Kernel)、多项式核 (Polynomial Kernel) 和径向基函数核 (RBF Kernel)，并解释核技巧 (Kernel Trick) 如何避免显式的高维映射 (High-Dimensional Mapping)
- 评估由 C 参数 (C Parameter) 控制的间隔宽度与分类错误之间的权衡 (Tradeoff)

## 问题描述

假设你有两类数据点，需要画一条直线（或超平面）将它们分开。理论上存在无数条可行的直线，你应该选择哪一条？

答案是间隔 (Margin) 最大的那一条。间隔指的是决策边界 (Decision Boundary) 与两侧最近数据点之间的距离。间隔越宽，意味着分类器的置信度越高，在未见数据上的泛化能力 (Generalization) 也越强。

这一直觉引出了支持向量机，它是机器学习 (Machine Learning) 领域数学上最优雅的算法之一。在深度学习兴起之前，SVM 曾是主流的分类方法；如今，对于小样本数据集、高维数据，以及需要具有理论保证、原理清晰且易于理解的模型的场景，它依然是最佳选择。

SVM 与第一阶段的内容紧密相连：其优化过程是凸的 (Convex)（第18课），间隔通过范数 (Norms) 进行度量（第14课），而核技巧则利用点积 (Dot Products) 来处理非线性边界 (Nonlinear Boundaries)，全程无需在高维空间中进行实际计算。

## 核心概念

### 最大间隔分类器 (Maximum Margin Classifier)

给定线性可分的数据，其标签 y_i 属于 {-1, +1}，特征向量为 x_i，我们希望找到一个超平面 (Hyperplane) w^T x + b = 0 来分隔这些类别。

点 x_i 到超平面的距离为：

distance = |w^T x_i + b| / ||w||

对于正确分类的点：y_i * (w^T x_i + b) > 0。间隔 (Margin) 是超平面到两侧最近点距离的两倍。

graph LR
    subgraph Margin
        direction TB
        A["w^T x + b = +1"] ~~~ B["w^T x + b = 0"] ~~~ C["w^T x + b = -1"]
    end
    D["+ class points"] --> A
    E["- class points"] --> C
    B --- F["Decision boundary"]

优化问题如下：

maximize    2 / ||w||     (the margin width)
subject to  y_i * (w^T x_i + b) >= 1  for all i

等价地（最小化 ||w||^2 更易于优化）：

minimize    (1/2) ||w||^2
subject to  y_i * (w^T x_i + b) >= 1  for all i

这是一个凸二次规划 (Convex Quadratic Program) 问题，具有唯一的全局最优解。恰好位于间隔边界上（即满足 y_i * (w^T x_i + b) = 1）的数据点被称为支持向量 (Support Vectors)。它们是唯一决定决策边界 (Decision Boundary) 的点。移动或移除任何非支持向量的点，决策边界都不会发生改变。

### 支持向量：关键的少数

graph TD
    subgraph Classification
        SV1["Support Vector (+ class)<br>y(w'x+b) = 1"] --- DB["Decision Boundary<br>w'x+b = 0"]
        DB --- SV2["Support Vector (- class)<br>y(w'x+b) = 1"]
    end
    O1["Other + points<br>(do not affect boundary)"] -.-> SV1
    O2["Other - points<br>(do not affect boundary)"] -.-> SV2

大多数训练样本是无关紧要的，只有支持向量起作用。这就是为什么支持向量机 (Support Vector Machine, SVM) 在预测时具有内存效率：你只需存储支持向量，而无需保留整个训练集。

支持向量的数量也为泛化误差 (Generalization Error) 提供了界限。相对于数据集规模，支持向量越少，通常意味着模型的泛化能力越好。

### 软间隔 (Soft Margin)：使用 C 参数处理噪声

真实数据很少是完美线性可分的。某些点可能位于边界的错误一侧，或者落在间隔内部。软间隔公式通过引入松弛变量 (Slack Variables) 来允许一定程度的违规。

minimize    (1/2) ||w||^2 + C * sum(xi_i)
subject to  y_i * (w^T x_i + b) >= 1 - xi_i
            xi_i >= 0  for all i

松弛变量 xi_i 衡量了第 i 个点违反间隔的程度。参数 C 控制着以下权衡：

| C 值 | 行为表现 |
|---------|----------|
| 较大的 C | 对违规行为施加严厉惩罚。间隔较窄，误分类较少。容易过拟合 (Overfit) |
| 较小的 C | 允许更多违规。间隔较宽，误分类较多。容易欠拟合 (Underfit) |

C 与正则化强度 (Regularization Strength) 成反比。较大的 C 意味着较弱的正则化，较小的 C 意味着较强的正则化。

### 合页损失 (Hinge Loss)：SVM 的损失函数

软间隔 SVM 可以重写为无约束优化问题：

minimize    (1/2) ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))

项 max(0, 1 - y_i * f(x_i)) 即为合页损失。当点被正确分类且位于间隔之外时，其值为零；当点位于间隔内部或被误分类时，其值呈线性增长。

Hinge loss for a single point:

loss
  |
  | \
  |  \
  |   \
  |    \
  |     \_______________
  |
  +-----|-----|-------->  y * f(x)
       0     1

Zero loss when y*f(x) >= 1 (correctly classified, outside margin).
Linear penalty when y*f(x) < 1.

与逻辑损失 (Logistic Loss，用于逻辑回归) 对比：

Hinge:     max(0, 1 - y*f(x))          Hard cutoff at margin
Logistic:  log(1 + exp(-y*f(x)))        Smooth, never exactly zero

合页损失会产生稀疏解（只有支持向量对模型有非零贡献）。而逻辑损失会用到所有数据点。这使得 SVM 在预测时更加节省内存。

### 使用梯度下降 (Gradient Descent) 训练线性 SVM

你可以直接在合页损失加上 L2 正则化 (L2 Regularization) 上使用梯度下降来训练线性 SVM，而无需求解带约束的二次规划问题：

L(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum(max(0, 1 - y_i * (w^T x_i + b)))

Gradient with respect to w:
  If y_i * (w^T x_i + b) >= 1:  dL/dw = lambda * w
  If y_i * (w^T x_i + b) < 1:   dL/dw = lambda * w - y_i * x_i

Gradient with respect to b:
  If y_i * (w^T x_i + b) >= 1:  dL/db = 0
  If y_i * (w^T x_i + b) < 1:   dL/db = -y_i

这被称为原始形式 (Primal Formulation)。每个训练周期 (Epoch) 的时间复杂度为 O(n * d)，其中 n 是样本数量，d 是特征数量。对于大规模、稀疏的高维数据（如文本分类），这种方法非常高效。

### 对偶形式 (Dual Formulation) 与核技巧 (Kernel Trick)

SVM 问题的拉格朗日对偶 (Lagrangian Dual，参考第一阶段第18课，KKT条件) 如下：

maximize    sum(alpha_i) - (1/2) * sum_ij(alpha_i * alpha_j * y_i * y_j * (x_i . x_j))
subject to  0 <= alpha_i <= C
            sum(alpha_i * y_i) = 0

对偶形式仅涉及数据点之间的点积 x_i . x_j。这是关键所在。将每个点积替换为核函数 (Kernel Function) K(x_i, x_j)，SVM 就能学习非线性边界，而无需显式计算特征变换。

Linear kernel:      K(x, z) = x . z
Polynomial kernel:  K(x, z) = (x . z + c)^d
RBF (Gaussian):     K(x, z) = exp(-gamma * ||x - z||^2)

径向基函数核 (Radial Basis Function Kernel, RBF) 将数据映射到无限维空间。在输入空间中距离较近的点，其核函数值接近 1；距离较远的点，核函数值接近 0。它能够学习任意平滑的决策边界。

graph LR
    subgraph "Input Space (not separable)"
        A["Data points in 2D<br>circular boundary"]
    end
    subgraph "Feature Space (separable)"
        B["Data points in higher dim<br>linear boundary"]
    end
    A -->|"Kernel trick<br>K(x,z) = phi(x).phi(z)"| B

核技巧无需实际进入高维空间，即可计算其中的点积。对于 D 维空间中 d 次的多项式核，显式特征空间的维度为 O(D^d)。但计算 K(x, z) 的时间复杂度仅为 O(D)。

### 用于回归的 SVM：支持向量回归 (Support Vector Regression, SVR)

支持向量回归会在数据周围拟合一个宽度为 epsilon 的管道 (Tube)。位于管道内部的点损失为零，位于管道外部的点则受到线性惩罚。

minimize    (1/2) ||w||^2 + C * sum(xi_i + xi_i*)
subject to  y_i - (w^T x_i + b) <= epsilon + xi_i
            (w^T x_i + b) - y_i <= epsilon + xi_i*
            xi_i, xi_i* >= 0

参数 epsilon 控制管道的宽度。管道越宽 = 支持向量越少 = 拟合越平滑。管道越窄 = 支持向量越多 = 拟合越紧密。

### 为什么 SVM 败给了深度学习 (Deep Learning)（以及它们何时依然胜出）

从 20 世纪 90 年代末到 21 世纪初，SVM 在机器学习 (Machine Learning, ML) 领域占据主导地位。深度学习在以下几个方面超越了它们：

| 因素 | SVM | 深度学习 |
|--------|------|---------------|
| 特征工程 (Feature Engineering) | 需要人工设计 | 自动学习特征 |
| 可扩展性 (Scalability) | 核方法复杂度为 O(n^2) 到 O(n^3) | 使用随机梯度下降 (SGD) 每周期复杂度为 O(n) |
| 图像/文本/音频处理 | 需要手工设计特征 | 直接从原始数据学习 |
| 大规模数据集（>10万） | 速度慢 | 扩展性良好 |
| GPU 加速 | 收益有限 | 大幅提速 |

在以下场景中，SVM 依然具有优势：
- 小型数据集（数百至数千个样本）
- 高维稀疏数据（如使用 TF-IDF 特征的文本）
- 需要数学理论保证（如间隔界限）时
- 训练时间必须极短时（线性 SVM 速度非常快）
- 具有清晰间隔结构的二分类问题
- 异常检测（单类支持向量机，One-Class SVM）

## 动手实现

### 步骤 1：合页损失（Hinge Loss）与梯度（Gradient）

奠定基础。计算单个批次（Batch）的合页损失及其梯度。

def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n

### 步骤 2：基于梯度下降（Gradient Descent）的线性支持向量机（Linear SVM）

通过最小化正则化合页损失（Regularized Hinge Loss）进行训练。无需使用二次规划（Quadratic Programming, QP）求解器。

class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]

### 步骤 3：核函数（Kernel Functions）

实现线性核（Linear Kernel）、多项式核（Polynomial Kernel）与径向基函数核（Radial Basis Function, RBF Kernel）。

def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))

### 步骤 4：间隔（Margin）与支持向量（Support Vector）识别

训练完成后，识别哪些数据点属于支持向量，并计算间隔宽度。

def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors

完整实现及所有演示代码请参见 `code/svm.py`。

## 使用它

使用 scikit-learn：

from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Support vectors: {clf['svm'].n_support_}")

重要提示：在训练支持向量机（Support Vector Machine, SVM）之前，务必对特征进行缩放。SVM 对特征的量级非常敏感，因为间隔（margin）的大小取决于权重向量的范数 ||w||，未缩放的特征会扭曲特征空间的几何结构。

对于大规模数据集，建议使用 `LinearSVC`（原始形式（primal formulation），每轮迭代（epoch）复杂度为 O(n)）代替 `SVC`（对偶形式（dual formulation），复杂度为 O(n^2) 至 O(n^3)）：

from sklearn.svm import LinearSVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=10000)),
])

## 练习

1. 生成一个二维线性可分数据集。训练你的线性支持向量机并找出支持向量（support vectors）。验证这些支持向量是否为距离决策边界（decision boundary）最近的样本点。

2. 在含噪声的数据集上，将正则化参数 C 从 0.001 调整至 1000。绘制每个 C 值对应的决策边界。观察模型从宽间隔（欠拟合（underfitting））向窄间隔（过拟合（overfitting））的转变过程。

3. 创建一个类别边界呈圆形（非线性）的数据集。证明线性 SVM 在此场景下失效。计算径向基函数核（Radial Basis Function, RBF kernel）矩阵，并展示类别在核诱导特征空间（kernel-induced feature space）中变得线性可分。

4. 在同一数据集上对比铰链损失（hinge loss）与逻辑损失（logistic loss）。分别训练线性 SVM 和逻辑回归（logistic regression）模型。统计对各自模型决策边界产生影响的训练样本数量（支持向量 vs 所有样本点）。

5. 实现支持向量回归（Support Vector Regression, SVR，采用 epsilon 不敏感损失（epsilon-insensitive loss））。将其拟合至 y = sin(x) + noise 数据。绘制预测值周围的 epsilon 间隔带（epsilon tube），并高亮显示支持向量（位于间隔带之外的样本点）。

## 关键术语

| 术语 | 实际含义 |
|------|----------------------|
| 支持向量 (Support Vectors) | 距离决策边界最近的训练样本。唯一决定超平面的数据点 |
| 间隔 (Margin) | 决策边界与最近支持向量之间的距离。支持向量机旨在最大化该距离 |
| 铰链损失 (Hinge Loss) | max(0, 1 - y*f(x))。当样本被正确分类且位于间隔之外时损失为零，否则施加线性惩罚 |
| C 参数 (C Parameter) | 间隔宽度与分类错误之间的权衡。较大的 C 值对应较窄的间隔，较小的 C 值对应较宽的间隔 |
| 软间隔 (Soft Margin) | 通过引入松弛变量允许违反间隔的支持向量机公式。用于处理线性不可分数据 |
| 核技巧 (Kernel Trick) | 无需显式映射到高维特征空间，即可在该空间中计算点积的方法 |
| 线性核 (Linear Kernel) | K(x, z) = x . z。等价于标准点积。适用于线性可分数据 |
| 径向基函数核 (RBF Kernel) | K(x, z) = exp(-gamma * \|\|x-z\|\|^2)。将数据映射至无限维空间。能够学习任意平滑的决策边界 |
| 多项式核 (Polynomial Kernel) | K(x, z) = (x . z + c)^d。将数据映射至多项式组合构成的特征空间 |
| 对偶形式 (Dual Formulation) | 仅依赖于数据点之间点积的支持向量机问题重构形式。使得核方法得以应用 |
| 支持向量回归 (Support Vector Regression, SVR) | 在数据周围拟合一个 epsilon 管。位于管内的样本损失为零 |
| 松弛变量 (Slack Variables) | xi_i：衡量样本违反间隔的程度。对于正确分类且位于间隔之外的样本，其值为零 |
| 最大间隔 (Maximum Margin) | 选择超平面的原则，即最大化该超平面到各类别最近样本点的距离 |

## 扩展阅读

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1) - 支持向量机与统计学习领域的奠基性著作
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018) - 支持向量机的原始论文
- [Platt: Sequential Minimal Optimization (1998)](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) - 使支持向量机训练具备实用性的序列最小优化 (Sequential Minimal Optimization, SMO) 算法
- [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html) - 包含实现细节的实用指南
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - 大多数支持向量机实现背后的 C++ 库
# 降维 (Dimensionality Reduction)

> 高维数据具有内在结构。你需要从正确的角度去发现它。

**类型：** 构建 (Build)
**语言：** Python
**前置要求：** 第一阶段，课程 01（线性代数直觉）、02（向量、矩阵与运算）、03（特征值与特征向量）、06（概率与分布）
**时长：** 约 90 分钟

## 学习目标

- 从零实现主成分分析 (Principal Component Analysis, PCA)：数据中心化、计算协方差矩阵、特征分解以及投影
- 使用解释方差比 (Explained Variance Ratio) 和肘部法则 (Elbow Method) 选择主成分数量
- 比较 PCA、t-SNE 和 UMAP 在 2D 可视化 MNIST 数字时的效果，并解释它们的权衡取舍
- 应用带 RBF 核的核主成分分析 (Kernel PCA) 来分离标准 PCA 无法处理的非线性数据结构

## 问题背景

假设你有一个数据集，每个样本包含 784 个特征。它可能是手写数字的像素值，可能是基因表达水平，也可能是用户行为信号。你无法可视化 784 维空间，无法绘制它们，甚至难以在脑海中构想它们。

但这 784 个特征中大部分是冗余的。真正的信息存在于一个低得多的流形（表面）上。一个手写的“7”不需要 784 个独立的数字来描述。它只需要几个关键参数：笔画的角度、横杠的长度、倾斜的程度。其余的都是噪声。

降维技术正是为了找到这个低维表面。它将你的 784 维数据压缩到 2 维、10 维或 50 维，同时保留关键的结构信息。

## 核心概念

### 维度灾难 (Curse of Dimensionality)

高维空间是反直觉的。随着维度的增加，三件事会“失效”。

**距离失去意义。** 在高维空间中，任意两个随机点之间的距离会收敛到同一个值。如果每个点与其他点的距离都大致相同，最近邻搜索就会失效。

Dimension    Avg distance ratio (max/min between random points)
2            ~5.0
10           ~1.8
100          ~1.2
1000         ~1.02

**体积集中在角落。** d 维单位超立方体有 2^d 个角。在 100 维空间中，几乎所有的体积都集中在角落，远离中心。数据点会扩散到边缘，导致模型在内部区域缺乏数据支撑。

**你需要指数级增长的数据。** 为了在空间中保持相同的样本密度，从 2D 扩展到 20D 意味着你需要多出 10^18 倍的数据。你永远不会有足够的数据。降维可以将数据密度恢复到可操作的水平。

### PCA：寻找关键方向

主成分分析 (Principal Component Analysis, PCA) 旨在找到数据变化最大的轴。它会旋转你的坐标系，使第一个轴捕获最大的方差，第二个轴捕获次大的方差，依此类推。

算法步骤：

1. Center the data        (subtract the mean from each feature)
2. Compute covariance     (how features move together)
3. Eigendecomposition     (find the principal directions)
4. Sort by eigenvalue     (biggest variance first)
5. Project               (keep top k eigenvectors, drop the rest)

为什么使用特征分解？协方差矩阵 (Covariance Matrix) 是对称且半正定的。它的特征向量构成了特征空间中的正交方向。特征值告诉你每个方向捕获了多少方差。最大特征值对应的特征向量指向方差最大的方向。

graph LR
    A["Original data (2D)\nData spread in both\nx and y directions"] -->|"PCA rotation"| B["After PCA\nPC1 captures the elongated spread\nPC2 captures the narrow spread\nDrop PC2 and you lose little info"]

- **PCA 之前：** 数据云沿 x 轴和 y 轴对角线方向分布
- **PCA 之后：** 坐标系发生旋转，使第一主成分 (PC1) 与最大方差方向（拉伸分布）对齐，第二主成分 (PC2) 与最小方差方向（狭窄分布）对齐
- **降维：** 丢弃 PC2 将数据投影到 PC1 上，几乎不会丢失信息

### 解释方差比 (Explained Variance Ratio)

每个主成分捕获总方差的一部分。解释方差比告诉你具体占比是多少。

Component    Eigenvalue    Explained ratio    Cumulative
PC1          4.73          0.473              0.473
PC2          2.51          0.251              0.724
PC3          1.12          0.112              0.836
PC4          0.89          0.089              0.925
...

当累积解释方差达到 0.95 时，说明这些主成分已经捕获了 95% 的信息。之后的部分大多是噪声。

### 选择主成分数量

三种策略：

1. **阈值法。** 保留足够的主成分，使其解释 90-95% 的方差。
2. **肘部法则 (Elbow Method)。** 绘制每个主成分的解释方差图，寻找急剧下降的拐点。
3. **下游任务性能。** 将 PCA 作为预处理步骤。遍历 k 值并测量模型的准确率。最佳 k 值通常出现在准确率趋于平稳的位置。

### t-SNE：保留局部邻域

t-分布随机邻域嵌入 (t-Distributed Stochastic Neighbor Embedding, t-SNE) 专为可视化设计。它将高维数据映射到 2D（或 3D），同时保留点与点之间的邻近关系。

核心直觉：在原始空间中，基于点对之间的距离计算概率分布。距离近的点获得高概率，距离远的点获得低概率。然后在 2D 空间中寻找一种布局，使得相同的概率分布得以保持。在 784 维空间中相邻的点，在 2D 空间中依然相邻。

t-SNE 的关键特性：
- 非线性。它可以展开 PCA 无法处理的复杂流形 (Manifold)。
- 随机性。每次运行会产生不同的布局。
- 困惑度 (Perplexity) 参数控制考虑多少个邻居（典型范围：5-50）。
- 输出中簇与簇之间的距离没有实际意义。只有簇内部的相对结构是可靠的。
- 在大数据集上较慢。默认时间复杂度为 O(n^2)。

### UMAP：更快，更好的全局结构

统一流形逼近与投影 (Uniform Manifold Approximation and Projection, UMAP) 的工作原理与 t-SNE 类似，但具有两大优势：
- 速度更快。它使用近似最近邻图，而不是计算所有点对之间的距离。
- 更好的全局结构。输出中簇的相对位置通常比 t-SNE 更具实际意义。

UMAP 在高维空间中构建一个加权图（“模糊拓扑表示”），然后寻找一个低维布局，以尽可能好地保留该图的结构。

关键参数：
- `n_neighbors`：定义局部结构的邻居数量（类似于困惑度）。值越大，保留的全局结构越多。
- `min_dist`：输出中点聚集的紧密程度。值越小，生成的簇越密集。

### 何时使用哪种方法

| 方法 | 适用场景 | 保留特性 | 速度 |
|--------|----------|-----------|-------|
| PCA | 训练前的预处理 | 全局方差 | 快（精确解），适用于数百万样本 |
| PCA | 快速探索性可视化 | 线性结构 | 快 |
| t-SNE | 出版级 2D 图表 | 局部邻域 | 慢（理想样本数 < 1万） |
| UMAP | 大规模 2D 可视化 | 局部 + 部分全局结构 | 中等（可处理数百万样本） |
| PCA | 模型特征降维 | 按方差排序的特征 | 快 |
| t-SNE / UMAP | 理解聚类结构 | 簇分离度 | 中等到慢 |

经验法则：使用 PCA 进行预处理和数据压缩。当你需要在 2D 中可视化结构时，使用 t-SNE 或 UMAP。

### 核主成分分析 (Kernel PCA)

标准 PCA 寻找的是线性子空间。它旋转坐标系并丢弃某些轴。但如果数据位于非线性流形上呢？2D 空间中的一个圆环无法用任何直线分开。标准 PCA 对此无能为力。

核主成分分析 (Kernel PCA) 在由核函数诱导的高维特征空间中应用 PCA，而无需显式计算该空间中的坐标。这就是核技巧 (Kernel Trick)——与支持向量机 (SVM) 背后的思想相同。

算法步骤：
1. 计算核矩阵 K，其中 K_ij = k(x_i, x_j)
2. 在特征空间中对核矩阵进行中心化
3. 对中心化后的核矩阵进行特征分解
4. 前 k 个特征向量（按 1/sqrt(特征值) 缩放）即为投影结果

常用核函数：

| 核函数 | 公式 | 适用场景 |
|--------|---------|----------|
| RBF（高斯核） | exp(-gamma * \|\|x - y\|\|^2) | 大多数非线性数据、平滑流形 |
| 多项式核 | (x . y + c)^d | 多项式关系 |
| Sigmoid 核 | tanh(alpha * x . y + c) | 类似神经网络的映射 |

何时使用核 PCA 与标准 PCA：

| 评判标准 | 标准 PCA | 核 PCA |
|-----------|-------------|------------|
| 数据结构 | 线性子空间 | 非线性流形 |
| 速度 | O(min(n^2 d, d^2 n)) | O(n^2 d + n^3) |
| 可解释性 | 主成分是特征的线性组合 | 主成分缺乏直接的特征解释 |
| 可扩展性 | 适用于数百万样本 | 核矩阵为 n x n，受内存限制 |
| 重构 | 直接逆变换 | 需要原像近似 (Pre-image approximation) |

经典示例：2D 空间中的同心圆。两圈点，一圈在另一圈内部。标准 PCA 会将它们投影到同一条直线上——对分类毫无用处。而使用 RBF 核的核 PCA 会将内圈和外圈映射到不同的区域，使它们线性可分。

### 重构误差 (Reconstruction Error)

你的降维效果如何？你将 784 维压缩到了 50 维。你丢失了什么？

测量重构误差：
1. 将数据投影到 k 维：X_reduced = X @ W_k
2. 重构数据：X_hat = X_reduced @ W_k^T
3. 计算均方误差 (MSE)：mean((X - X_hat)^2)

对于 PCA，重构误差与解释方差有明确的数学关系：

Reconstruction error = sum of eigenvalues NOT included
Total variance = sum of ALL eigenvalues
Fraction lost = (sum of dropped eigenvalues) / (sum of all eigenvalues)

每个主成分的解释方差比为：

explained_ratio_k = eigenvalue_k / sum(all eigenvalues)

绘制累积解释方差与主成分数量的关系图，你会得到“肘部”曲线。最佳主成分数量通常出现在：
- 曲线趋于平缓（边际收益递减）
- 累积方差超过你的阈值（通常为 0.90 或 0.95）
- 下游任务性能趋于平稳

重构误差的用途不仅限于选择 k 值。你还可以将其用于异常检测：重构误差高的样本是偏离学习到的子空间的异常值。这是生产系统中基于 PCA 的异常检测的基础。

## 动手实现

### 步骤 1：从零实现 PCA

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

### 步骤 2：在合成数据上测试

np.random.seed(42)
n_samples = 500

t = np.random.uniform(0, 2 * np.pi, n_samples)
x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)

X_synthetic = np.column_stack([x1, x2, x3])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_synthetic)

print(f"Original shape: {X_synthetic.shape}")
print(f"Reduced shape:  {X_reduced.shape}")
print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")

### 步骤 3：MNIST 数字的 2D 可视化

from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist = mnist.data[:5000].astype(float)
y_mnist = mnist.target[:5000].astype(int)

pca_mnist = PCA(n_components=50)
X_pca50 = pca_mnist.fit_transform(X_mnist)
print(f"50 components capture {sum(pca_mnist.explained_variance_ratio_):.2%} of variance")

pca_2d = PCA(n_components=2)
X_pca2d = pca_2d.fit_transform(X_mnist)
print(f"2 components capture {sum(pca_2d.explained_variance_ratio_):.2%} of variance")

### 步骤 4：与 sklearn 对比

from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE

sklearn_pca = SklearnPCA(n_components=2)
X_sklearn_pca = sklearn_pca.fit_transform(X_mnist)

print(f"\nOur PCA explained variance:     {pca_2d.explained_variance_ratio_}")
print(f"Sklearn PCA explained variance: {sklearn_pca.explained_variance_ratio_}")

diff = np.abs(np.abs(X_pca2d) - np.abs(X_sklearn_pca))
print(f"Max absolute difference: {diff.max():.10f}")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_mnist)
print(f"\nt-SNE output shape: {X_tsne.shape}")

### 步骤 5：UMAP 对比

try:
    from umap import UMAP

    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_mnist)
    print(f"UMAP output shape: {X_umap.shape}")
except ImportError:
    print("Install umap-learn: pip install umap-learn")

## 实际应用

在分类器前将 PCA 作为预处理步骤：

from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X_mnist, y_mnist, test_size=0.2, random_state=42
)

results = {}
for k in [10, 30, 50, 100, 200]:
    pca_k = SklearnPCA(n_components=k)
    X_tr = pca_k.fit_transform(X_train)
    X_te = pca_k.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    var_captured = sum(pca_k.explained_variance_ratio_)
    results[k] = (acc, var_captured)
    print(f"k={k:>3d}  accuracy={acc:.4f}  variance={var_captured:.4f}")

模型性能在远未达到 784 维之前就会趋于平稳。这个平稳点就是你的最佳工作点。

## 交付成果

本课程的产出：
- `outputs/skill-dimensionality-reduction.md` - 一份技能指南，用于为特定任务选择合适的降维技术

## 练习

1. 修改 PCA 类以支持 `inverse_transform`。使用 10、50 和 200 个主成分重构 MNIST 数字。打印每个情况下的重构误差（与原始数据的均方误差）。

2. 在相同的 MNIST 子集上运行 t-SNE，分别设置困惑度 (perplexity) 为 5、30 和 100。描述输出结果的变化。为什么困惑度会影响簇的紧密程度？

3. 获取一个包含 50 个特征但仅有 5 个具有信息量的数据集（可使用 `sklearn.datasets.make_classification` 生成）。应用 PCA 并检查解释方差曲线是否能正确识别出该数据实际上是 5 维的。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 维度灾难 (Curse of Dimensionality) | “特征太多了” | 随着维度增加，距离、体积和数据密度的表现都变得反直觉。模型需要指数级增长的数据来弥补。 |
| PCA | “降低维度” | 旋转坐标系，使坐标轴与最大方差方向对齐，然后丢弃低方差轴。 |
| 主成分 (Principal Component) | “一个重要方向” | 协方差矩阵的特征向量。特征空间中数据变化最大的方向。 |
| 解释方差比 (Explained Variance Ratio) | “这个成分包含多少信息” | 单个主成分捕获的总方差比例。将前 k 个比例相加，即可看出 k 个主成分保留了多少信息。 |
| 协方差矩阵 (Covariance Matrix) | “特征如何相关” | 一个对称矩阵，其中 (i,j) 位置的元素衡量特征 i 和特征 j 的协同变化程度。对角线元素是各特征的方差。 |
| t-SNE | “那个聚类图” | 一种非线性方法，通过保留成对邻域概率将高维数据映射到 2D。适合可视化，不适合预处理。 |
| UMAP | “更快的 t-SNE” | 一种基于拓扑数据分析的非线性方法。同时保留局部和部分全局结构。比 t-SNE 具有更好的可扩展性。 |
| 困惑度 (Perplexity) | “t-SNE 的调节旋钮” | 控制每个点考虑的有效邻居数量。低困惑度聚焦于极局部结构。高困惑度捕获更广泛的模式。 |
| 流形 (Manifold) | “数据所在的表面” | 嵌入在高维空间中的低维曲面。一张在 3D 空间中揉皱的纸就是一个 2D 流形。 |

## 延伸阅读

- [A Tutorial on Principal Component Analysis](https://arxiv.org/abs/1404.1100) (Shlens) - 从零开始清晰推导 PCA
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) (Wattenberg et al.) - 交互式指南，讲解 t-SNE 的常见陷阱与参数选择
- [UMAP documentation](https://umap-learn.readthedocs.io/) - UMAP 作者提供的理论与实践指南
# K近邻（K-Nearest Neighbors）与距离

> 存储所有数据。通过观察邻居来进行预测。这是最简单却真正有效的算法。

**类型：** 实战构建
**语言：** Python
**前置要求：** 第一阶段（第14课 范数与距离）
**时长：** 约90分钟

## 学习目标

- 从零实现 KNN 分类与回归，支持可配置的 K 值与距离加权投票（distance-weighted voting）
- 比较 L1、L2、余弦（cosine）和闵可夫斯基（Minkowski）距离度量，并为给定数据类型选择合适的度量方式
- 解释维度灾难（curse of dimensionality），并演示为何 KNN 在高维空间中性能会下降
- 构建 KD树（KD-tree）以实现高效的最近邻搜索，并分析其在何种情况下优于暴力搜索（brute-force）

## 问题描述

你拥有一个数据集。当一个新的数据点到来时，你需要对其进行分类或预测其数值。与线性回归或支持向量机（Support Vector Machines, SVMs）等从数据中学习参数的方法不同，你只需找到距离新数据点最近的 K 个训练样本，并让它们进行投票。

这就是 K近邻（K-Nearest Neighbors, KNN）算法。它没有训练阶段，无需学习参数，也无需最小化损失函数。你只需存储整个训练集，并在预测时计算距离。

这听起来简单得似乎无法奏效。但 KNN 在许多问题上却表现出惊人的竞争力，尤其是在中小型数据集上。深入理解它能揭示一些核心概念：距离度量的选择（关联至第一阶段第14课）、维度灾难（curse of dimensionality），以及惰性学习（lazy learning）与急切学习（eager learning）之间的区别。

KNN 也广泛存在于现代人工智能的各个角落，只是换了不同的名称。向量数据库在嵌入向量（embeddings）上执行 KNN 搜索；检索增强生成（Retrieval-Augmented Generation, RAG）查找最接近的 K 个文档片段；推荐系统寻找相似的用户或物品。底层算法是相同的，区别仅在于数据规模与所使用的数据结构。

## 核心概念

### KNN（K-Nearest Neighbors）的工作原理

给定一个带标签的数据集和一个新的查询点（query point）：

1. 计算查询点到数据集中每个点的距离
2. 按距离排序
3. 选取距离最近的 K 个点
4. 对于分类任务：对 K 个邻居进行多数投票（majority vote）
5. 对于回归任务：计算 K 个邻居值的平均值（或加权平均值）

graph TD
    Q["Query point ?"] --> D["Compute distances<br>to all training points"]
    D --> S["Sort by distance"]
    S --> K["Select K nearest"]
    K --> C{"Classification<br>or Regression?"}
    C -->|Classification| V["Majority vote"]
    C -->|Regression| A["Average values"]
    V --> P["Prediction"]
    A --> P

这就是完整的算法。无需拟合（fitting）。无需梯度下降（gradient descent）。无需训练轮数（epochs）。

### 选择 K 值

K 是唯一的超参数（hyperparameter）。它控制着偏差-方差权衡（bias-variance trade-off）：

| K | 行为表现 |
|---|----------|
| K = 1 | 决策边界（decision boundary）紧贴每一个点。训练误差为零。方差高。容易过拟合（overfit） |
| 较小的 K (3-5) | 对局部结构敏感。能够捕捉复杂的边界 |
| 较大的 K | 边界更平滑。对噪声更鲁棒（robust）。可能欠拟合（underfit） |
| K = N | 对所有点都预测为多数类。偏差最大 |

对于包含 N 个点的数据集，一个常见的起始点是 K = sqrt(N)。在二分类任务中，建议使用奇数 K 以避免平票。

graph LR
    subgraph "K=1 (overfitting)"
        A["Jagged boundary<br>follows every point"]
    end
    subgraph "K=15 (good)"
        B["Smooth boundary<br>captures true pattern"]
    end
    subgraph "K=N (underfitting)"
        C["Flat boundary<br>predicts majority class"]
    end
    A -->|"increase K"| B -->|"increase K"| C

### 距离度量（Distance Metrics）

距离函数定义了“近”的含义。不同的度量方式会产生不同的邻居，进而导致不同的预测结果。

**L2 距离（欧几里得距离，Euclidean）** 是默认选项。即直线距离。

d(a, b) = sqrt(sum((a_i - b_i)^2))

对特征尺度敏感。在使用 KNN 配合 L2 距离前，务必对特征进行标准化（standardize）。

**L1 距离（曼哈顿距离，Manhattan）** 计算绝对差值之和。由于不对差值进行平方，它比 L2 距离对异常值（outliers）更鲁棒。

d(a, b) = sum(|a_i - b_i|)

**余弦距离（Cosine distance）** 衡量向量之间的夹角，忽略向量模长。对于文本和嵌入（embedding）数据至关重要。

d(a, b) = 1 - (a . b) / (||a|| * ||b||)

**闵可夫斯基距离（Minkowski）** 通过参数 p 对 L1 和 L2 距离进行泛化。

d(a, b) = (sum(|a_i - b_i|^p))^(1/p)

p=1: Manhattan
p=2: Euclidean
p->inf: Chebyshev (max absolute difference)

选择哪种度量方式取决于数据类型：

| 数据类型 | 最佳度量方式 | 原因 |
|-----------|------------|-----|
| 数值型特征，尺度相近 | L2（欧几里得距离） | 默认选项，适用于空间数据 |
| 数值型特征，存在异常值 | L1（曼哈顿距离） | 鲁棒性强，不会放大较大差异 |
| 文本嵌入（Text embeddings） | 余弦距离 | 模长是噪声，方向才代表语义 |
| 高维稀疏数据 | 余弦距离或 L1 | L2 距离会受维度灾难（curse of dimensionality）影响 |
| 混合类型 | 自定义距离 | 根据特征类型组合不同度量方式 |

### 加权 KNN（Weighted KNN）

标准 KNN 对所有 K 个邻居赋予相同的权重。但距离为 0.1 的邻居理应比距离为 5.0 的邻居更重要。

**距离加权 KNN（Distance-weighted KNN）** 根据距离的倒数对每个邻居进行加权：

weight_i = 1 / (distance_i + epsilon)

For classification: weighted vote
For regression:     weighted average = sum(w_i * y_i) / sum(w_i)

当查询点与某个训练点完全重合时，epsilon 可防止除以零。

加权 KNN 对 K 值的选择较不敏感，因为无论 K 如何设定，距离较远的邻居贡献都微乎其微。

### 维度灾难（Curse of Dimensionality）

KNN 在高维空间中的性能会下降。这并非模糊的担忧，而是数学事实。

**问题 1：距离收敛。** 随着维度增加，最大距离与最小距离的比值趋近于 1。所有点相对于查询点都变得同样“远”。

In d dimensions, for random uniform points:

d=2:    max_dist / min_dist = varies widely
d=100:  max_dist / min_dist ~ 1.01
d=1000: max_dist / min_dist ~ 1.001

When all distances are nearly equal, "nearest" is meaningless.

**问题 2：体积爆炸。** 为了在固定比例的数据中捕获 K 个邻居，你需要扩大搜索半径以覆盖特征空间中更大比例的区域。在高维空间中，“邻域”实际上涵盖了大部分空间。

**问题 3：角落主导。** 在 d 维单位超立方体中，大部分体积集中在角落附近，而非中心。随着维度 d 的增长，内切于立方体的球体所占体积比例趋近于零。

实际影响：KNN 在特征数量约为 20-50 个时表现良好。超过此范围，在应用 KNN 之前需要进行降维（dimensionality reduction，如 PCA、UMAP、t-SNE），或者需要使用基于树的搜索结构来利用数据内在的低维特性。

### KD 树（KD-trees）：快速最近邻搜索

暴力 KNN 会计算查询点到每个训练点的距离。每次查询的时间复杂度为 O(n * d)。对于大型数据集，这太慢了。

KD 树沿特征轴递归地划分空间。在每一层，它沿某一维度的中位数值进行分割。

graph TD
    R["Split on x1 at 5.0"] -->|"x1 <= 5.0"| L["Split on x2 at 3.0"]
    R -->|"x1 > 5.0"| RR["Split on x2 at 7.0"]
    L -->|"x2 <= 3.0"| LL["Leaf: 3 points"]
    L -->|"x2 > 3.0"| LR["Leaf: 4 points"]
    RR -->|"x2 <= 7.0"| RL["Leaf: 2 points"]
    RR -->|"x2 > 7.0"| RRR["Leaf: 5 points"]

为了找到最近邻，首先遍历树到达包含查询点的叶子节点，然后回溯，仅当相邻分区可能包含更近的点时才进行检查。

平均查询时间：在低维空间中为 O(log n)。但在高维空间（d > 20）中，KD 树会退化至 O(n)，因为回溯过程能剪枝的分支越来越少。

### 球树（Ball trees）：更适合中等维度

球树将数据划分为嵌套的超球体，而非轴对齐的矩形框。每个节点定义一个球体（中心点 + 半径），包含该子树中的所有点。

相较于 KD 树的优势：
- 在中等维度（最高约 50 维）表现更好
- 能够处理非轴对齐的结构
- 更紧密的边界体积意味着在搜索过程中能剪枝更多分支

KD 树和球树都是精确算法。对于真正的大规模搜索（数百万个点、数百个维度），通常会改用近似最近邻（approximate nearest neighbor）方法（如 HNSW、IVF、乘积量化）。这些内容将在第一阶段第 14 课中讲解。

### 惰性学习（Lazy Learning）与急切学习（Eager Learning）

KNN 是一种惰性学习器（lazy learner）：它在训练阶段不执行任何工作，所有计算都推迟到预测阶段。大多数其他算法（如线性回归、支持向量机 SVM、神经网络）属于急切学习器（eager learner）：它们在训练阶段进行大量计算以构建紧凑的模型，随后预测速度很快。

| 方面 | 惰性学习（KNN） | 急切学习（SVM、神经网络） |
|--------|------------|------------------------|
| 训练时间 | O(1)，仅存储数据 | O(n * epochs) |
| 预测时间 | 每次查询 O(n * d) | O(d) 或 O(参数量) |
| 预测时内存占用 | 存储整个训练集 | 仅存储模型参数 |
| 适应新数据 | 即时添加新点 | 需要重新训练模型 |
| 决策边界 | 隐式，实时计算 | 显式，训练后固定 |

惰性学习适用于以下场景：
- 数据集频繁变动（无需重新训练即可增删数据点）
- 只需对极少量查询进行预测
- 希望训练时间为零
- 数据集足够小，暴力搜索速度很快

### 用于回归的 KNN

与多数投票不同，用于回归的 KNN 会对 K 个邻居的目标值求平均。

prediction = (1/K) * sum(y_i for i in K nearest neighbors)

Or with distance weighting:
prediction = sum(w_i * y_i) / sum(w_i)
where w_i = 1 / distance_i

KNN 回归会产生分段常数（或在使用加权时呈分段平滑）的预测结果。它无法对训练数据范围之外的值进行外推（extrapolate）。如果训练目标值全部在 0 到 100 之间，KNN 永远不会预测出 200。

## 构建它

### 步骤 1：距离函数

实现 L1 距离、L2 距离、余弦距离（Cosine Distance）和闵可夫斯基距离（Minkowski Distance）。这些内容直接对应第一阶段第 14 课。

import math

def l2_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def l1_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))

def cosine_distance(a, b):
    dot_val = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai ** 2 for ai in a))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot_val / (norm_a * norm_b)

def minkowski_distance(a, b, p=2):
    if p == float('inf'):
        return max(abs(ai - bi) for ai, bi in zip(a, b))
    return sum(abs(ai - bi) ** p for ai, bi in zip(a, b)) ** (1 / p)

### 步骤 2：K 近邻（KNN）分类器与回归器

构建完整的 K 近邻（KNN）模型，支持自定义 K 值、距离度量（Distance Metric）以及可选的距离加权（Distance Weighting）。

class KNN:
    def __init__(self, k=5, distance_fn=l2_distance, weighted=False,
                 task="classification"):
        self.k = k
        self.distance_fn = distance_fn
        self.weighted = weighted
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict_one(x) for x in X]

### 步骤 3：用于高效搜索的 KD 树（KD-Tree）

从零开始构建 KD 树（KD-Tree），该树通过递归地沿每个维度的中位数进行数据划分。

class KDTree:
    def __init__(self, X, indices=None, depth=0):
        # Recursively partition the data
        self.axis = depth % len(X[0])
        # Split on median of the current axis
        ...

    def query(self, point, k=1):
        # Traverse to leaf, then backtrack
        ...

完整实现（包含所有辅助方法和演示代码）请参见 `code/knn.py`。

### 步骤 4：特征缩放（Feature Scaling）

KNN 算法需要进行特征缩放（Feature Scaling），因为距离计算对特征的数值量级非常敏感。例如，取值范围在 0 到 1000 的特征会完全主导取值范围在 0 到 1 的特征。

def standardize(X):
    n = len(X)
    d = len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    stds = [
        max(1e-10, (sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n) ** 0.5)
        for j in range(d)
    ]
    return [[((X[i][j] - means[j]) / stds[j]) for j in range(d)] for i in range(n)], means, stds

## 使用它

使用 scikit-learn：

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")

当数据集足够大且维度足够低时，scikit-learn 会自动使用 KD 树（KD-tree）或球树（ball tree）。对于高维数据，它会回退到暴力搜索（brute force）。你可以通过 `algorithm` 参数来控制这一行为。

对于大规模最近邻搜索（nearest neighbor search，涉及数百万向量），请使用 FAISS、Annoy 或向量数据库（vector database）：

import faiss

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
distances, indices = index.search(query_vectors, k=5)

## 练习

1. 在包含 3 个类别的二维数据集上实现 KNN 分类。绘制 K=1、K=5、K=15 和 K=N 时的决策边界（decision boundary）。观察模型从过拟合（overfitting）到欠拟合（underfitting）的转变过程。

2. 在 2、5、10、50、100 和 500 维空间中分别生成 1000 个随机点。针对每个维度，计算最大成对距离与最小成对距离的比值。绘制该比值随维度变化的曲线，以直观展示维度灾难（curse of dimensionality）。

3. 在文本分类问题（使用 TF-IDF 向量）中，比较 KNN 算法使用 L1 距离、L2 距离和余弦距离（cosine distance）的效果。哪种度量方式能带来最高的准确率？为什么余弦距离在文本任务中通常表现更优？

4. 实现一个 KD 树，并在 2D、10D 和 50D 空间中，针对包含 1k、10k 和 100k 个点的数据集，测量其查询时间与暴力搜索的对比。在什么维度下，KD 树的速度不再优于暴力搜索？

5. 针对 `y = sin(x) + noise` 构建一个加权 KNN 回归器（weighted KNN regressor）。将其与 K=3、10、30 时的未加权 KNN 进行对比。证明加权机制能够产生更平滑的预测结果，尤其是在 K 值较大时。

## 关键术语

| 术语 | 实际含义 |
|------|----------------------|
| K近邻 (K-nearest neighbors) | 一种非参数算法 (non-parametric algorithm)，通过查找距离查询点最近的 K 个训练样本进行预测 |
| 惰性学习 (Lazy learning) | 训练阶段不进行计算，所有工作均在预测阶段完成。KNN 是其典型代表 |
| 急切学习 (Eager learning) | 在训练阶段进行大量计算以构建紧凑模型。大多数机器学习算法属于此类 |
| 维度灾难 (Curse of dimensionality) | 在高维空间中，距离趋于收敛，邻域扩大至覆盖大部分空间，导致 KNN 失效 |
| KD树 (KD-tree) | 沿特征轴递归划分空间的二叉树。在低维空间中查询复杂度为 O(log n) |
| 球树 (Ball tree) | 由嵌套超球体构成的树结构。在中等维度（最高约 50 维）下表现优于 KD树 |
| 加权 KNN (Weighted KNN) | 邻居权重与距离成反比。距离越近的邻居对预测结果的影响越大 |
| 特征缩放 (Feature scaling) | 将特征归一化至可比较的范围。KNN 等基于距离的方法必需此步骤 |
| 多数投票 (Majority vote) | 通过统计 K 个邻居中出现频率最高的类别来进行分类 |
| 暴力搜索 (Brute force search) | 计算与每个训练样本的距离。每次查询复杂度为 O(n*d)。结果精确但在 n 较大时速度较慢 |
| 近似最近邻 (Approximate nearest neighbor) | 比精确搜索快得多的算法（如 HNSW、LSH、IVF），用于查找近似最近点 |
| 维诺图 (Voronoi diagram) | 空间划分方式，每个区域包含距离某一训练样本比距离其他任何样本都近的所有点。K=1 的 KNN 会生成维诺边界 |

## 延伸阅读

- [Cover & Hart: Nearest Neighbor Pattern Classification (1967)](https://ieeexplore.ieee.org/document/1053964) - KNN 的奠基性论文，证明了其错误率至多为贝叶斯最优 (Bayes optimal) 的两倍
- [Friedman, Bentley, Finkel: An Algorithm for Finding Best Matches in Logarithmic Expected Time (1977)](https://dl.acm.org/doi/10.1145/355744.355745) - KD树 (KD-tree) 的原始论文
- [Beyer et al.: When Is "Nearest Neighbor" Meaningful? (1999)](https://link.springer.com/chapter/10.1007/3-540-49257-7_15) - 对最近邻搜索中维度灾难 (curse of dimensionality) 的形式化分析
- [scikit-learn Nearest Neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - 包含算法选择指南的实用文档
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 开发的用于十亿级规模近似最近邻搜索的库
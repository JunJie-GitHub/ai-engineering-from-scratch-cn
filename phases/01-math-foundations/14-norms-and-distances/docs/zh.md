# 范数与距离 (Norms and Distances)

> 距离函数 (Distance Function) 定义了“相似”的含义。一旦选错，下游的所有环节都会崩溃。

**类型：** 项目构建
**语言：** Python
**前置要求：** 第一阶段，第 01 课（线性代数直觉）、第 02 课（向量、矩阵与运算）
**时长：** 约 90 分钟

## 学习目标

- 从零开始实现 L1 范数 (L1 Norm)、L2 范数 (L2 Norm)、余弦相似度 (Cosine Similarity)、马哈拉诺比斯距离 (Mahalanobis Distance)、杰卡德距离 (Jaccard Distance) 以及编辑距离 (Edit Distance) 函数
- 为给定的机器学习 (Machine Learning, ML) 任务选择合适的距离度量 (Distance Metric)，并解释为何其他替代方案不适用
- 将 L1 和 L2 范数与 LASSO 和岭回归 (Ridge) 正则化 (Regularization) 及其几何约束区域联系起来
- 演示同一数据集在不同度量标准下如何产生不同的最近邻 (Nearest Neighbors)

## 核心问题

你有两个向量。它们可能是词嵌入 (Word Embeddings)，可能是用户画像，也可能是像素数组。你需要弄清楚：它们到底有多接近？

答案完全取决于你选择的距离函数。两个数据点在一种度量下可能是最近邻，在另一种度量下却可能相距甚远。你的 K 近邻 (K-Nearest Neighbors, KNN) 分类器、推荐引擎、向量数据库、聚类算法 (Clustering Algorithm) 以及损失函数 (Loss Function)——全都依赖于这一选择。一旦选错，你的模型就会朝着错误的方向进行优化。

不存在放之四海而皆准的最佳距离度量。L2 范数适用于空间数据；余弦相似度在自然语言处理 (Natural Language Processing, NLP) 领域占据主导；杰卡德距离用于处理集合；编辑距离用于处理字符串；马哈拉诺比斯距离考虑了特征间的相关性；Wasserstein 距离 (Wasserstein Distance) 则用于衡量概率质量的移动。每一种度量都蕴含了对“相似”这一概念的不同假设。

本课程将从零开始构建所有主要的距离函数，向你展示每种函数在何种场景下最为适用，并演示同一组数据如何因所选度量标准的不同而得出截然不同的最近邻结果。

## 核心概念

### 范数（Norms）：衡量向量大小

范数用于衡量向量的“大小”。任意两个向量之间的距离函数都可以表示为它们差值的范数：d(a, b) = ||a - b||。因此，理解范数就是理解距离。

### L1 范数（L1 Norm，曼哈顿距离 Manhattan distance）

L1 范数是所有分量绝对值的总和。

||x||_1 = |x_1| + |x_2| + ... + |x_n|

它被称为曼哈顿距离，因为它模拟了在城市网格中只能沿坐标轴方向行走的距离，不允许走对角线。

Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.

何时使用 L1 范数：
- 高维稀疏数据（文本特征、独热编码 one-hot encodings）
- 需要抗异常值干扰时（单个巨大的差异不会主导整体结果）
- 特征选择问题（L1 正则化 L1 regularization 会促进稀疏性）

与 L1 正则化（Lasso）的联系：在损失函数中加入 ||w||_1 会对权重绝对值之和进行惩罚。这会将较小的权重直接推向零，从而实现自动特征选择。L1 惩罚项在权重空间中形成菱形的约束区域，菱形的顶点恰好落在坐标轴上，对应某些权重为零的情况。

与损失函数的联系：平均绝对误差（Mean Absolute Error, MAE）是预测值与目标值之间 L1 距离的平均值。它对所有误差进行线性惩罚，因此相比均方误差（MSE），它对异常值更具鲁棒性。

### L2 范数（L2 Norm，欧几里得距离 Euclidean distance）

L2 范数即直线距离，是各分量平方和的平方根。

||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)

这就是你在几何课上学到的距离，相当于 n 维空间中的勾股定理。

Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.

何时使用 L2 范数：
- 低至中维度的连续型数据
- 特征量纲/尺度相近时
- 物理距离（空间数据、传感器读数）
- 像素级的图像相似度计算

与 L2 正则化（Ridge）的联系：在损失函数中加入 ||w||_2^2 会对较大的权重进行惩罚。与 L1 不同，它不会将权重推向零，而是按比例将所有权重向零收缩。L2 惩罚项形成圆形的约束区域，因此没有落在坐标轴上的顶点。权重会变小，但极少恰好为零。

与损失函数的联系：均方误差（Mean Squared Error, MSE）是 L2 距离平方的平均值。平方操作会对大误差施加比小误差更重的惩罚。

MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.

### Lp 范数（Lp Norms）：通用族

L1 和 L2 是 Lp 范数的特例：

||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)

不同的 p 值会生成不同形状的“单位球”（即距离原点为 1 的所有点的集合）：

p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)

### L-无穷范数（L-infinity Norm，切比雪夫距离 Chebyshev distance）

当 p 趋近于无穷大时，Lp 范数收敛于最大绝对分量。

||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)

两点之间的距离仅由差异最大的那个维度决定，其他所有维度均被忽略。

Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4

何时使用 L-无穷范数：
- 关注任意单一维度上的最坏情况偏差时
- 棋盘游戏（国际象棋中的国王按 L-无穷距离移动：向任意方向走一步代价均为 1）
- 制造公差控制（每个维度都必须符合规格）

### 余弦相似度（Cosine Similarity）与余弦距离（Cosine Distance）

余弦相似度衡量两个向量之间的夹角，忽略它们的模长（大小）。

cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)

其取值范围为 -1（方向完全相反）到 +1（方向完全相同）。垂直向量的余弦相似度为 0。

余弦距离将其转换为距离度量：cosine_distance = 1 - cosine_similarity。取值范围为 0（方向相同）到 2（方向相反）。

a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293

为什么余弦相似度在自然语言处理（NLP）和嵌入（Embeddings）中占据主导地位：在文本处理中，文档长度不应影响相似度判断。一篇关于猫的文档即使长度是另一篇的两倍，它们依然应该是“相似”的。余弦相似度忽略模长（长度），只关注方向。两篇词分布相同但长度不同的文档指向同一方向，余弦相似度为 1.0。

何时使用余弦相似度：
- 文本相似度（TF-IDF 向量、词嵌入 word embeddings、句嵌入 sentence embeddings）
- 任何模长代表噪声而方向代表信号的领域
- 推荐系统（用户偏好向量）
- 嵌入检索（向量数据库几乎总是使用余弦相似度或点积）

### 点积相似度（Dot Product Similarity）与余弦相似度

两个向量的点积为：

a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)

余弦相似度是经过双方模长归一化后的点积。当两个向量均已进行单位归一化（模长 = 1）时，点积与余弦相似度完全相同。

If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)

两者的区别在于：点积包含了模长信息。模长较大的向量会获得更高的点积得分。这在某些检索系统中很重要，因为你可能希望“热门”项目排名更高。此时模长充当了隐式的质量或重要性信号。

a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.

实际应用建议：
- 需要纯粹的方向相似度时使用余弦相似度
- 当模长携带有意义信息时使用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许你在两者之间选择
- 如果你的嵌入向量已进行 L2 归一化，则选择哪种方式无关紧要

### 马哈拉诺比斯距离（Mahalanobis Distance）

欧几里得距离平等对待所有维度。但如果特征之间存在相关性或量纲不同，L2 距离会给出误导性的结果。

马哈拉诺比斯距离考虑了数据的协方差结构。

d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))

其中 S 是数据的协方差矩阵。

直观理解：马哈拉诺比斯距离首先对数据进行去相关和归一化（白化 whitening），然后在该变换后的空间中计算 L2 距离。如果 S 是单位矩阵（特征不相关且方差为 1），马哈拉诺比斯距离就退化为欧几里得距离。

Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.

何时使用马哈拉诺比斯距离：
- 异常值检测（与均值马哈拉诺比斯距离较大的点即为异常值）
- 特征量纲不同且存在相关性时的分类任务
- 拥有足够数据以估计可靠协方差矩阵时
- 制造业质量控制（多变量过程监控）

### 杰卡德相似度（Jaccard Similarity，适用于集合）

杰卡德相似度衡量两个集合之间的重叠程度。

J(A, B) = |A intersect B| / |A union B|

取值范围为 0（无重叠）到 1（集合完全相同）。杰卡德距离 = 1 - 杰卡德相似度。

A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6

何时使用杰卡德相似度：
- 比较标签、类别或特征集合
- 基于词项是否存在（而非频率）的文档相似度计算
- 近似重复检测（使用 MinHash 近似计算杰卡德相似度）
- 比较二元特征向量（存在/缺失数据）
- 评估分割模型（交并比 Intersection over Union, IoU 即杰卡德相似度）

### 编辑距离（Edit Distance，莱文斯坦距离 Levenshtein Distance）

编辑距离计算将一个字符串转换为另一个字符串所需的最少单字符操作次数。操作包括：插入、删除或替换。

"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3

通常使用动态规划计算。填充一个矩阵，其中位置 (i, j) 的值表示字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的编辑距离。

        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3

何时使用编辑距离：
- 拼写检查与纠错
- DNA 序列比对（带权重的操作）
- 模糊字符串匹配
- 杂乱文本数据的去重

### KL 散度（KL Divergence，非距离度量但常被当作距离使用）

KL 散度衡量一个概率分布与另一个概率分布的差异。第 09 课已详细讲解，但在此仍需讨论，因为尽管它不是真正的距离，人们仍常将其作为“距离”使用。

D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))

关键性质：KL 散度不具备对称性。

D_KL(P || Q) != D_KL(Q || P)

这意味着它不满足距离度量的基本要求。它也不满足三角不等式。它是一种散度（divergence），而非距离。

正向 KL 散度（D_KL(P || Q)）具有“均值趋向”特性：Q 会尝试覆盖 P 的所有众数（modes）。
反向 KL 散度（D_KL(Q || P)）具有“众数趋向”特性：Q 会聚焦于 P 的单一众数。

常见应用场景：
- 变分自编码器（VAEs）（证据下界 ELBO 中的 KL 项推动潜在分布逼近先验分布）
- 知识蒸馏（Knowledge distillation）（学生模型尝试匹配教师模型的分布）
- 基于人类反馈的强化学习（RLHF）（KL 惩罚项确保微调后的模型与基础模型保持接近）
- 策略梯度方法（约束策略更新幅度）

### 瓦瑟斯坦距离（Wasserstein Distance，推土机距离 Earth Mover's Distance）

瓦瑟斯坦距离衡量将一个概率分布转换为另一个分布所需的最小“功”。可以这样理解：如果一个分布是一堆土，另一个是一个坑，你需要移动多少土以及移动多远？

W(P, Q) = inf over all transport plans gamma of E[d(x, y)]

对于一维分布，它简化为累积分布函数绝对差值的积分：

W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx

为什么瓦瑟斯坦距离很重要：
- 它是真正的度量（对称且满足三角不等式）
- 即使分布没有重叠，它也能提供梯度（而 KL 散度会趋于无穷大）
- 这一特性使其成为瓦瑟斯坦生成对抗网络（WGANs）的核心，解决了原始 GAN 训练不稳定的问题

Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.

何时使用瓦瑟斯坦距离：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能无重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同的距离度量

| 任务 | 最佳距离度量 | 原因 |
|------|--------------|-----|
| 文本相似度 | 余弦相似度 | 模长是噪声，方向代表语义 |
| 图像像素比较 | L2 距离 | 空间关系重要，特征尺度可比 |
| 稀疏高维特征 | L1 距离 | 鲁棒性强，不会放大罕见的大差异 |
| 集合重叠（标签、类别） | 杰卡德相似度 | 数据天然为集合形式，非向量形式 |
| 字符串匹配 | 编辑距离 | 操作映射符合人类编辑直觉 |
| 异常值检测 | 马哈拉诺比斯距离 | 考虑了特征相关性与量纲 |
| 分布比较 | KL 散度 | 衡量使用 Q 代替 P 所损失的信息量 |
| GAN 训练 | 瓦瑟斯坦距离 | 即使分布无重叠也能提供梯度 |
| 嵌入向量（向量数据库） | 余弦相似度或点积 | 嵌入向量训练时将语义编码在方向中 |
| 推荐系统 | 点积 | 模长可编码流行度或置信度 |
| DNA 序列 | 加权编辑距离 | 替换代价因核苷酸对而异 |
| 制造质量控制 | L-无穷距离 | 任意维度上的最坏偏差至关重要 |

### 与损失函数的联系

损失函数本质上是应用于预测值与目标值之间的距离函数。

Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin

### 与正则化的联系

正则化在损失函数中加入了针对权重的范数惩罚项。

L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.

为什么 L1 产生稀疏性而 L2 不会：想象二维权重空间中的约束区域。L1 是菱形，L2 是圆形。损失函数的等高线（椭圆）最有可能在菱形的顶点处与之相切，此时某个权重恰好为零。而与圆形相切时，切点通常是平滑的，两个权重均不为零。

### 最近邻搜索（Nearest Neighbor Search）

每种距离函数都对应一个最近邻搜索问题：给定一个查询点，在数据集中找到距离最近的点。

精确最近邻搜索在包含 n 个点、d 维的数据集中，每次查询的时间复杂度为 O(n * d)。对于大规模数据集，这太慢了。

近似最近邻（Approximate Nearest Neighbor, ANN）算法通过牺牲少量精度来换取巨大的速度提升：

Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space

HNSW（分层可导航小世界图 Hierarchical Navigable Small World）是现代向量数据库中的主流算法。它构建了一个多层图，其中每个节点连接到其近似最近邻。搜索从顶层开始（稀疏，长距离跳跃），逐层下降到底层（密集，短距离跳跃）。

## 动手构建 (Build It)

### 步骤 1：所有范数 (norm) 与距离函数 (distance functions)

完整实现请参见 `code/distances.py`。每个函数均仅使用基础 Python 数学运算从零开始构建。

### 步骤 2：相同数据，不同距离，不同邻居

`distances.py` 中的演示代码会创建一个数据集，选取一个查询点 (query point)，并展示最近邻 (nearest neighbor) 如何随距离度量 (distance metric) 的不同而变化。在 L1 距离 (L1 distance) 下“最近”的点，在 L2 距离 (L2 distance) 或余弦相似度 (cosine similarity) 下未必最近。

### 步骤 3：嵌入相似度搜索 (embedding similarity search)

代码中包含一个模拟的嵌入相似度搜索，它分别使用余弦相似度与 L2 距离来查找与查询最相似的“文档”，从而展示两者的排序结果可能存在差异。

## 实际应用 (Use It)

最常见的实际应用场景：在向量数据库 (vector database) 中查找相似项。

import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")

当你调用 `model.encode(text)` 并在向量数据库中进行搜索时，底层正是这样运作的。嵌入模型 (embedding model) 将文本映射为向量。向量数据库会计算你的查询向量与每个存储向量之间的余弦相似度（或点积 (dot product)），并利用近似最近邻算法 (Approximate Nearest Neighbor, ANN) 来避免逐一比对所有向量。

## 练习 (Exercises)

1. 计算点 (1, 2, 3) 与 (4, 0, 6) 之间的 L1 距离、L2 距离和 L无穷范数距离 (L-infinity distance)。验证对于任意点对，始终满足 L-inf <= L2 <= L1。证明为何该大小关系必然成立。

2. 构造两个向量，使其余弦相似度较高（> 0.9），但 L2 距离较大（> 10）。从几何角度解释这一现象。接着再构造两个向量，使其余弦相似度较低（< 0.3），但 L2 距离较小（< 0.5）。

3. 实现一个函数，接收一个数据集和一个查询点，并分别返回在 L1 距离、L2 距离、余弦相似度和马哈拉诺比斯距离 (Mahalanobis distance) 度量下的最近邻。寻找一个数据集，使得这四种度量对“最近点”的判断结果各不相同。

4. 使用累积分布函数法 (Cumulative Distribution Function, CDF) 手动计算 [0.5, 0.5, 0, 0] 与 [0, 0, 0.5, 0.5] 之间的瓦瑟斯坦距离 (Wasserstein distance)。接着计算 [0.25, 0.25, 0.25, 0.25] 与 [0, 0, 0.5, 0.5] 之间的该距离。哪一个更大？为什么？

5. 实现 MinHash 算法以近似计算杰卡德相似度 (Jaccard similarity)。生成 100 个随机集合，计算所有集合对的精确杰卡德相似度，并将其与分别使用 50、100 和 200 个哈希函数 (hash functions) 的 MinHash 近似结果进行对比。绘制近似误差图。

## 关键术语 (Key Terms)

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 范数 (Norm) | “向量的大小” | 将向量映射为非负标量的函数，满足三角不等式、绝对齐次性，且仅当向量为零向量时值为零 |
| L1 范数 (L1 Norm) | “曼哈顿距离” | 各分量绝对值之和。在优化中会产生稀疏性。对异常值具有鲁棒性 |
| L2 范数 (L2 Norm) | “欧几里得距离” | 各分量平方和的平方根。欧几里得空间中的直线距离 |
| Lp 范数 (Lp Norm) | “广义范数” | 各分量绝对值的 p 次方之和的 p 次方根。L1 和 L2 范数是其特例 |
| L-无穷范数 (L-infinity Norm) | “最大范数”或“切比雪夫距离” | 各分量绝对值的最大值。当 p 趋近于无穷大时 Lp 范数的极限 |
| 余弦相似度 (Cosine Similarity) | “向量间的夹角” | 经两个向量模长归一化后的点积。取值范围为 -1 到 +1。忽略向量长度 |
| 余弦距离 (Cosine Distance) | “1 减去余弦相似度” | 将余弦相似度转换为距离度量。取值范围为 0 到 2 |
| 点积 (Dot Product) | “未归一化的余弦” | 对应分量乘积之和。等于余弦相似度乘以两个向量的模长 |
| 马哈拉诺比斯距离 (Mahalanobis Distance) | “考虑相关性的距离” | 在使用数据协方差矩阵进行白化（去相关和归一化）处理后的空间中的 L2 距离 |
| 杰卡德相似度 (Jaccard Similarity) | “集合重叠度” | 交集大小除以并集大小。适用于集合而非向量 |
| 编辑距离 (Edit Distance) | “莱文斯坦距离” | 将一个字符串转换为另一个字符串所需的最少插入、删除和替换操作次数 |
| KL 散度 (KL Divergence) | “分布间的距离” | 并非真正的距离度量（不满足对称性）。衡量使用分布 Q 编码分布 P 时所需的额外信息量（比特数） |
| 瓦瑟斯坦距离 (Wasserstein Distance) | “推土机距离” | 将一个分布的质量转移到另一个分布所需的最小功。是一种真正的度量（满足距离公理） |
| 近似最近邻 (Approximate Nearest Neighbor) | “ANN 搜索” | 比精确搜索快得多的近似最近点查找算法（如 HNSW、LSH、IVF） |
| HNSW | “向量数据库算法” | 分层可导航小世界图（Hierarchical Navigable Small World）。用于快速近似最近邻搜索的多层图结构 |
| L1 正则化 (L1 Regularization) | “Lasso” | 将权重的 L1 范数添加到损失函数中。促使权重趋近于零（产生稀疏性） |
| L2 正则化 (L2 Regularization) | “Ridge 回归”或“权重衰减” | 将权重的 L2 范数平方添加到损失函数中。使权重向零收缩但不产生稀疏性 |
| 弹性网络 (Elastic Net) | “L1 + L2” | 结合 L1 和 L2 正则化。比单独使用其中一种更能有效处理高度相关的特征组 |

## 进一步阅读

- [FAISS：高效相似性搜索库](https://github.com/facebookresearch/faiss) - Meta 开发的用于十亿级近似最近邻（Approximate Nearest Neighbor, ANN）搜索的库
- [Wasserstein GAN（Arjovsky 等, 2017）](https://arxiv.org/abs/1701.07875) - 首次将推土机距离（Earth Mover's Distance, EMD）引入生成对抗网络（Generative Adversarial Networks, GANs）的论文
- [局部敏感哈希（Indyk & Motwani, 1998）](https://dl.acm.org/doi/10.1145/276698.276876) - 奠基性的近似最近邻（ANN）算法
- [词表示的高效估计（Mikolov 等, 2013）](https://arxiv.org/abs/1301.3781) - Word2Vec 模型，该模型确立了余弦相似度（Cosine Similarity）作为嵌入（Embeddings）的默认度量标准
- [sklearn.neighbors 文档](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距离度量（Distance Metrics）与近邻算法（Neighbor Algorithms）的实用指南
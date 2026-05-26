# 无监督学习 (Unsupervised Learning)

> 没有标签，没有导师。算法自行发现数据结构。

**类型：** 实战构建
**语言：** Python
**前置要求：** 第一阶段（范数与距离、概率与分布），第二阶段第 1-6 课
**时长：** 约 90 分钟

## 学习目标

- 从零实现 K均值 (K-Means)、DBSCAN 和高斯混合模型 (Gaussian Mixture Models)，并对比它们的聚类行为
- 使用轮廓系数 (Silhouette Score) 和肘部法则 (Elbow Method) 评估聚类质量，以选择最优的 K 值
- 解释 DBSCAN 在何种情况下优于 K均值，并识别哪种算法能够处理非球形簇和异常值
- 利用聚类方法构建异常检测 (Anomaly Detection) 流水线，以标记偏离正常模式的数据点

## 问题背景

迄今为止的机器学习课程都基于一个假设：数据是带标签的 (Labeled Data)，即“给定输入，就有对应的正确输出”。然而在现实世界中，获取标签的成本极高。医院拥有数百万份患者病历，却无人手动为每一份记录标注疾病类别；电商平台积累了数百万次用户会话，却无人逐一标注客户分群；安全团队掌握着海量网络日志，却无人标记出所有的异常行为。

无监督学习无需被告知具体目标，便能自行发现数据中的规律。它将相似的数据点归为一类，挖掘隐藏的结构，并让异常现象浮出水面。如果说监督学习 (Supervised Learning) 是拿着带答案的教科书进行学习，那么无监督学习就是直接凝视原始数据，直到其中的规律自行显现。

难点在于：由于缺乏标签，你无法直接衡量结果的“对”与“错”。你需要借助不同的评估工具，来判断算法所发现的结构是否具有实际意义。

## 核心概念

### 聚类（Clustering）：将相似的事物分组

聚类（Clustering）将每个数据点划分到一个组（簇，cluster）中，使得同一组内的数据点彼此之间的相似度高于与其他组数据点的相似度。核心问题始终是：“相似”究竟意味着什么？

flowchart LR
    A[Raw Data] --> B{Choose Method}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[Hierarchical]
    B --> F[GMM]
    C --> G[Flat, spherical clusters]
    D --> H[Arbitrary shapes, noise detection]
    E --> I[Tree of nested clusters]
    F --> J[Soft assignments, elliptical clusters]

### K-Means（K均值）：主力算法

K-Means（K均值）将数据精确划分为 K 个簇。每个簇都有一个质心（centroid，即其质量中心），且每个数据点都属于距离其最近的质心。

Lloyd 算法（Lloyd's algorithm）：

1. 随机选取 K 个点作为初始质心
2. 将每个数据点分配给最近的质心
3. 重新计算每个质心，将其更新为所属数据点的均值
4. 重复步骤 2-3，直到数据点的分配不再发生变化

目标函数（惯性，inertia）衡量的是每个数据点到其所属质心的平方距离总和。K-Means 致力于最小化该值，但通常只能收敛至局部最优解（local minimum）。不同的初始化策略可能会产生不同的聚类结果。

### 选择 K 值

两种常用方法：

**肘部法则（Elbow method）：** 依次运行 K = 1, 2, 3, ..., n 的 K-Means 算法。绘制惯性（inertia）随 K 变化的曲线。寻找曲线的“肘部”拐点，即继续增加簇数量不再显著降低惯性的位置。

**轮廓系数（Silhouette score）：** 针对每个数据点，计算其与自身所在簇的相似度（a）以及与最近其他簇的相似度（b）。轮廓系数（silhouette coefficient）的计算公式为 (b - a) / max(a, b)，取值范围为 -1（分配错误）到 +1（聚类效果极佳）。对所有数据点的轮廓系数求平均值，即可得到全局评分。

### DBSCAN：基于密度的聚类

K-Means 假设簇呈球形分布，且需要预先指定 K 值。DBSCAN 则无需这两项假设。它将簇识别为由稀疏区域分隔开的密集区域。

两个核心参数：
- **eps**：邻域半径
- **min_samples**：形成密集区域所需的最小数据点数量

三种数据点类型：
- **核心点（Core point）**：在 eps 距离范围内至少包含 min_samples 个数据点
- **边界点（Border point）**：位于某个核心点的 eps 距离范围内，但其自身不满足核心点条件
- **噪声点（Noise point）**：既非核心点也非边界点。这些点通常被视为异常值（outliers）。

DBSCAN 将彼此距离在 eps 范围内的核心点连接至同一个簇中。边界点会被归入附近核心点所在的簇。噪声点则不属于任何簇。

优势：能够发现任意形状的簇，自动确定簇的数量，并能识别异常值。劣势：难以处理密度差异较大的簇。

### 层次聚类（Hierarchical Clustering）

构建嵌套簇的树状结构（树状图，dendrogram）。

凝聚式（Agglomerative，自底向上）：
1. 初始时将每个数据点视为一个独立的簇
2. 合并距离最近的两个簇
3. 重复上述步骤，直到所有数据点合并为单一簇
4. 在树状图的指定高度进行切割，从而获得 K 个簇

簇之间的“距离”（closeness）可通过以下方式衡量：
- **单链接（Single linkage）**：两个簇中任意两点之间的最小距离
- **全链接（Complete linkage）**：两个簇中任意两点之间的最大距离
- **平均链接（Average linkage）**：两个簇中所有点对之间的平均距离
- **Ward 方法（Ward's method）**：选择使簇内总方差增加量最小的合并方式

### 高斯混合模型（Gaussian Mixture Models, GMM）

K-Means 采用硬分配（hard assignment）：每个数据点严格属于且仅属于一个簇。GMM 则采用软分配（soft assignment）：每个数据点属于各个簇的概率均不相同。

GMM 假设数据由 K 个高斯分布（Gaussian distribution）混合生成，每个分布拥有独立的均值（mean）和协方差（covariance）。期望最大化算法（Expectation-Maximization, EM）在以下两步之间交替迭代：

- **E步（期望步，E-step）**：计算每个数据点属于各个高斯分布的概率
- **M步（最大化步，M-step）**：更新每个高斯分布的均值、协方差和混合权重，以最大化数据的似然（likelihood）

GMM 能够对椭圆形簇进行建模（而非像 K-Means 那样仅限于球形），并能自然地处理相互重叠的簇。

### 算法选择指南

| 方法 | 适用场景 | 避免场景 |
|--------|----------|------------|
| K-Means | 大规模数据集、球形簇、K 值已知 | 簇形状不规则、存在异常值 |
| DBSCAN | K 值未知、任意形状簇、异常值检测 | 簇密度差异大、维度极高 |
| Hierarchical | 小规模数据集、需要树状图、K 值未知 | 大规模数据集（内存复杂度为 O(n^2)） |
| GMM | 簇相互重叠、需要软分配结果 | 数据集极大、维度过高 |

### 基于聚类的异常检测（Anomaly Detection）

聚类算法天然适用于异常检测：
- **K-Means**：距离所有质心均较远的数据点被视为异常值
- **DBSCAN**：根据定义，噪声点即为异常值
- **GMM**：在所有高斯分布下概率均较低的数据点被视为异常值

## 构建

### 步骤 1：从零实现 K-均值聚类 (K-Means)

import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids

### 步骤 2：肘部法则 (Elbow Method) 与轮廓系数 (Silhouette Score)

def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias

### 步骤 3：从零实现 DBSCAN 聚类算法 (DBSCAN)

def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels

### 步骤 4：高斯混合模型 (Gaussian Mixture Model) 与期望最大化算法 (EM Algorithm)

def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities

### 步骤 5：生成测试数据并运行完整流程

def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")


## 上手使用

借助 scikit-learn，实现相同的算法只需一行代码：

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)

从零实现的版本能让你清楚地了解这些底层库究竟在计算什么。K均值聚类 (K-Means) 在样本分配与中心点重计算之间交替迭代。DBSCAN (基于密度的空间聚类算法) 从高密度核心点向外生长簇。高斯混合模型 (Gaussian Mixture Model) 在期望步与最大化步之间交替。而现成的库版本在此基础上增加了数值稳定性、更智能的初始化策略（如 K-Means++）以及 GPU 加速，但核心逻辑保持一致。

## 交付上线

本节课程提供了从零开始实现的 K均值聚类 (K-Means)、DBSCAN 和高斯混合模型 (Gaussian Mixture Model) 的可用代码。这些聚类代码可作为基础，复用于更高级的无监督学习 (Unsupervised Learning) 方法中。

## 练习

1. 实现 K-Means++ 初始化策略：不再随机选取初始中心点，而是随机选择第一个中心点，后续每个中心点的选取概率与其到最近已有中心点的平方距离成正比。将其收敛速度与随机初始化进行对比。
2. 在代码中添加层次凝聚聚类 (Hierarchical Agglomerative Clustering)。实现 Ward 链接法 (Ward's Linkage) 并生成树状图 (Dendrogram)（以嵌套的合并列表形式表示）。在不同层级进行切割，并与 K均值聚类 (K-Means) 的结果进行对比。
3. 构建一个简单的异常检测 (Anomaly Detection) 流水线：在同一数据集上运行 DBSCAN 和 GMM，标记出两种方法均判定为异常值的点（DBSCAN 中的噪声点，GMM 中的低概率点）。计算两者的重合度，并探讨两种方法产生分歧的场景。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 聚类 (Clustering) | “将相似的事物分组” | 将数据划分为若干子集，使得子集内的相似度高于子集间的相似度，通常通过特定的距离度量 (distance metric) 来衡量 |
| 质心 (Centroid) | “簇的中心” | 分配给该簇的所有数据点的均值；K-Means (K均值) 算法将其作为簇的代表 |
| 惯性 (Inertia) | “簇的紧密程度” | 每个点到其所属质心的平方距离之和；数值越小表示簇越紧密 |
| 轮廓系数 (Silhouette Score) | “簇之间的分离程度” | 针对每个点计算 `(b - a) / max(a, b)`，其中 `a` 为簇内平均距离，`b` 为到最近邻簇的平均距离 |
| 核心点 (Core Point) | “密集区域中的点” | 在 DBSCAN 算法中，指在 `eps` 距离范围内至少包含 `min_samples` 个邻居的点 |
| EM算法 (EM Algorithm) | “软K-Means” | 期望最大化 (Expectation-Maximization) 算法：迭代计算样本归属概率（E步）并更新分布参数（M步） |
| 树状图 (Dendrogram) | “簇的树形结构” | 一种树形图，用于展示层次聚类 (Hierarchical Clustering) 中簇合并的顺序与距离 |
| 异常点 (Anomaly) | “离群值” | 不符合预期模式的数据点，在 DBSCAN 中被识别为噪声，或在 GMM (高斯混合模型) 中表现为低概率点 |

## 延伸阅读

- [Stanford CS229 - 无监督学习](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) - 吴恩达关于聚类与 EM 算法的讲义笔记
- [scikit-learn 聚类指南](https://scikit-learn.org/stable/modules/clustering.html) - 所有聚类算法的实用对比，附带可视化示例
- [DBSCAN 原始论文 (Ester et al., 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - 提出基于密度的聚类算法的开创性论文
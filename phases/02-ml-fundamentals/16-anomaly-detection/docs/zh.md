# 异常检测 (Anomaly Detection)

> “正常”很容易定义。“异常”就是任何不符合常规的事物。

**类型：** 构建实践
**语言：** Python
**前置要求：** 第二阶段，第 01-09 课
**时长：** 约 75 分钟

## 学习目标

- 从零开始实现 Z 分数 (Z-score)、四分位距 (IQR) 和孤立森林 (Isolation Forest) 异常检测方法
- 区分点异常 (Point Anomaly)、上下文异常 (Contextual Anomaly) 和集体异常 (Collective Anomaly)，并为每种类型选择合适的检测方法
- 解释为何异常检测被构建为对正常数据进行建模，而非对异常进行分类
- 比较无监督异常检测 (Unsupervised Anomaly Detection) 与监督分类 (Supervised Classification)，并评估在新型异常覆盖率与精确率 (Precision) 之间的权衡

## 问题描述

一张信用卡在下午 2 点于纽约使用，随后在 2 点 05 分又出现在东京。工厂传感器的读数为 150 度，而正常范围是 80-120 度。一台服务器每秒发送 50,000 次请求，而日均值仅为 200 次。

这些都是异常现象。发现它们至关重要。欺诈行为会造成数十亿美元的损失。设备故障会导致停机。网络入侵会致使数据泄露。

挑战在于：你极少拥有带标签的异常样本。欺诈交易仅占总交易量的 0.1%。设备故障每年也只发生几次。由于“异常”类别中几乎没有可供学习的数据，你无法训练标准的分类器 (Classifier)。即使你有一些标签，你见过的异常类型也并非未来会遇到的全部。明天的欺诈手法很可能与今天截然不同。

异常检测 (Anomaly Detection) 的思路则完全相反。与其学习什么是异常，不如学习什么是正常。任何偏离正常模式的事物都值得怀疑。这种方法无需标签即可运行，能够适应新型异常，并且可以轻松扩展到海量数据集。

## 核心概念

### 异常类型 (Anomaly Types)

并非所有异常都是相同的：

- **点异常 (Point Anomalies)**。无论上下文如何，单个数据点表现异常。例如 500 度的温度读数，或一个通常消费 50 美元的账户突然产生 5 万美元的交易。
- **上下文异常 (Contextual Anomalies)**。在特定上下文中表现异常的数据点。例如 90 度的温度在夏季是正常的，但在冬季则属于异常。数值相同，但上下文不同。
- **集体异常 (Collective Anomalies)**。作为一组数据序列表现异常，尽管其中每个单独的数据点可能都是正常的。例如 5 次登录失败是正常的，但连续 50 次失败则属于暴力破解攻击。

大多数方法检测的是点异常。上下文异常需要时间或位置特征。集体异常则需要具备序列感知能力的方法。

flowchart TD
    A[Anomaly Types] --> B[Point Anomaly]
    A --> C[Contextual Anomaly]
    A --> D[Collective Anomaly]

    B --> B1["Single unusual value<br/>Temperature: 500F"]
    C --> C1["Unusual in context<br/>90F in January"]
    D --> D1["Unusual sequence<br/>50 failed logins"]

    style B fill:#fdd,stroke:#333
    style C fill:#ffd,stroke:#333
    style D fill:#fdf,stroke:#333

### 无监督框架 (Unsupervised Framing)

在标准分类任务中，你拥有两个类别的标签。而在异常检测中，通常会面临以下三种情况之一：

1. **完全无监督 (Fully Unsupervised)**。没有任何标签。你将检测器拟合到所有数据上，并希望异常足够稀少，以至于不会破坏“正常”模型。
2. **半监督 (Semi-supervised)**。你拥有一个仅包含正常数据的干净数据集。你在此干净集上进行拟合，并对其他所有数据进行评分。如果条件允许，这是最理想的设置。
3. **弱监督 (Weakly Supervised)**。你只有少量带标签的异常数据。将它们仅用于评估而非训练。先进行无监督训练，然后在带标签的子集上测量精确率 (Precision) 和召回率 (Recall)。

核心洞察：异常检测与分类任务有着本质区别。你是在对正常数据的分布进行建模，而不是寻找两个类别之间的决策边界。

### 监督与无监督：权衡 (Supervised vs Unsupervised: The Tradeoff)

如果你确实拥有带标签的异常数据，应该将它们用于训练（监督分类）还是仅用于评估（无监督检测）？

**监督方法（视为分类任务）：**
- 能够精准捕捉你之前见过的异常类型
- 对已知异常类型具有更高的精确率
- 会完全漏掉新型异常
- 出现新异常类型时需要重新训练
- 需要足够的异常样本（通常样本量过少）

**无监督方法（对正常数据建模，标记偏离）：**
- 能够捕捉任何偏离正常模式的情况，包括新型异常
- 不需要带标签的异常数据
- 误报率较高（并非所有不寻常的情况都是有害的）
- 对数据分布偏移 (Distribution Shift) 更具鲁棒性

在实践中，最佳系统通常结合两者：使用无监督检测实现广泛覆盖，使用监督模型处理已知的高优先级异常类型，并对模糊案例进行人工审查。

### Z 分数法 (Z-Score Method)

最简单的方法。计算每个特征的均值 (Mean) 和标准差 (Standard Deviation)。标记任何偏离均值超过 k 个标准差的数据点。

z_score = (x - mean) / std
anomaly if |z_score| > threshold

默认阈值为 3.0（对于高斯分布 (Gaussian Distribution)，99.7% 的正常数据会落在 3 个标准差范围内）。

**优点：** 简单、快速、可解释性强（“该值偏离正常水平 4.5 个标准差”）。

**缺点：** 假设数据服从正态分布。对训练数据中的异常值敏感（异常值会拉偏均值并放大标准差，导致自身更难被检测）。在多峰分布 (Multimodal Distributions) 上表现不佳。

**适用场景：** 数据大致呈钟形曲线的单特征监控。例如服务器响应时间、制造公差、具有稳定基线的传感器读数。

**失效场景：** 多聚类数据（例如两个办公地点具有不同的温度基线）、偏态数据（例如交易金额中 1000 美元虽罕见但并非异常）、训练集中包含异常值的数据。

### 四分位距法 (IQR Method)

比 Z 分数法更稳健。使用四分位距 (Interquartile Range, IQR) 代替均值和标准差。

Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
lower_bound = Q1 - factor * IQR
upper_bound = Q3 + factor * IQR
anomaly if x < lower_bound or x > upper_bound

默认系数为 1.5。

**优点：** 对异常值具有鲁棒性（百分位数不受极端值影响）。适用于偏态分布。无需正态分布假设。

**缺点：** 仅适用于单变量（独立应用于每个特征）。无法检测仅在特征组合考虑时才显得异常的情况（一个点在每个单独特征上可能都正常，但在联合空间中却是异常的）。

**实践提示：** IQR 中的 1.5 系数对应箱线图 (Box Plot) 中的须线。须线之外的点为潜在异常值。将系数从 1.5 改为 3.0 会使检测器更保守（标记更少，误报更少）。合适的系数取决于你对误报的容忍度。

### 孤立森林 (Isolation Forest)

核心洞察：异常数据数量少且特征不同。在对数据进行随机划分时，异常点更容易被隔离——它们只需要更少的随机分割就能与其余数据分离。

flowchart TD
    A[All Data Points] --> B{Random Feature + Random Split}
    B --> C[Left Partition]
    B --> D[Right Partition]
    C --> E{Random Feature + Random Split}
    E --> F[Normal Point - deep in tree]
    E --> G[More splits needed...]
    D --> H["Anomaly - isolated quickly (short path)"]

    style H fill:#fdd,stroke:#333
    style F fill:#dfd,stroke:#333

**工作原理：**
1. 构建大量随机树（即孤立森林）
2. 在每个节点，随机选择一个特征，并在该特征的最小值和最大值之间随机选择一个分割值
3. 持续分割，直到每个数据点都被隔离（位于其自己的叶子节点中）
4. 异常点在所有树中的平均路径长度更短

**为何有效：** 正常点位于密集区域。需要多次随机分割才能将其与邻居分离。异常点位于稀疏区域。一两次随机分割就足以将其隔离。

异常分数基于所有树中的平均路径长度，并通过随机二叉搜索树的预期路径长度进行归一化：

score(x) = 2^(-average_path_length(x) / c(n))

其中 `c(n)` 是 n 个样本的预期路径长度。分数接近 1 表示异常。分数接近 0.5 表示正常。分数接近 0 表示非常正常（位于密集聚类深处）。

**优点：** 无分布假设。适用于高维数据。扩展性好（由于每棵树使用子样本，计算复杂度随样本量呈次线性增长）。可处理混合特征类型。

**缺点：** 难以检测密集区域中的异常（掩蔽效应）。当存在大量无关特征时，随机分割的效果会下降。

**关键超参数：**
- `n_estimators`：树的数量。通常 100 棵已足够。更多的树能提供更稳定的分数，但计算更慢。
- `max_samples`：每棵树的样本数量。原始论文中的默认值为 256。较小的值会降低单棵树的准确性，但能增加多样性。子采样正是孤立森林速度快的原因——每棵树仅看到数据的一小部分。
- `contamination`：预期的异常比例。仅用于设置阈值。不影响分数本身的计算。

### 局部异常因子 (Local Outlier Factor, LOF)

LOF 将某个点周围的局部密度与其邻居周围的密度进行比较。位于稀疏区域且被密集区域包围的点即为异常点。

**工作原理：**
1. 对于每个点，找到其 k 个最近邻 (k-Nearest Neighbors)
2. 计算局部可达密度（邻域的密集程度）
3. 将每个点的密度与其邻居的密度进行比较
4. 如果某点的密度远低于其邻居，则该点为异常值

**LOF 分数：**
- LOF 接近 1.0 表示与邻居密度相似（正常）
- LOF 大于 1.0 表示密度低于邻居（可能异常）
- LOF 远大于 1.0（例如 2.0+）表示密度显著更低（很可能是异常）

“局部”这一概念至关重要。考虑一个包含两个聚类的数据集：一个包含 1000 个点的密集聚类，和一个包含 50 个点的稀疏聚类。位于稀疏聚类边缘的点在全局范围内并不异常——它有 50 个邻居。但如果其直接邻居的密度高于它，那么它在局部就是异常的。LOF 能够捕捉到全局方法所忽略的这一细微差别。

**优点：** 能够检测局部异常（在其邻域内异常，即使在全局范围内不异常的点）。适用于不同密度的聚类。

**缺点：** 在大型数据集上运行缓慢（朴素实现的时间复杂度为 O(n^2)）。对 k 值的选择敏感。在极高维空间中表现不佳（维度灾难会影响距离计算）。

### 方法对比

| 方法 | 假设 | 速度 | 处理高维数据 | 检测局部异常 |
|--------|------------|-------|-------------------|------------------------|
| Z 分数法 | 正态分布 | 极快 | 是（按特征） | 否 |
| IQR 法 | 无（按特征） | 极快 | 是（按特征） | 否 |
| 孤立森林 | 无 | 快 | 是 | 部分支持 |
| LOF | 距离具有意义 | 慢 | 较差 | 是 |

### 评估挑战

评估异常检测器比评估分类器更困难：

- **极端的类别不平衡**。当异常仅占 0.1% 时，将所有样本预测为“正常”即可获得 99.9% 的准确率。此时准确率毫无意义。
- **AUROC 具有误导性**。在严重不平衡的情况下，即使模型在实际阈值下漏掉了大多数异常，AUROC 看起来也可能很好。
- **更优的指标：** Precision@k（在前 k 个被标记的项目中，有多少是真正的异常）、AUPRC（精确率-召回率曲线下面积），以及固定误报率下的召回率。

flowchart LR
    A[Raw Data] --> B[Train on Normal Data Only]
    B --> C[Score All Test Data]
    C --> D[Rank by Anomaly Score]
    D --> E[Evaluate Top-K Flagged Items]
    E --> F[Precision at K / AUPRC]

    style A fill:#f9f,stroke:#333
    style F fill:#9f9,stroke:#333

### 异常检测流水线 (Anomaly Detection Pipeline)

在实践中，异常检测通常遵循以下工作流：

1. **收集基线数据**。理想情况下，选择一段已知没有（或极少）异常的时间段。
2. **特征工程**。原始特征加上衍生特征（滚动统计量、时间特征、比率等）。
3. **训练检测器**。在基线数据上进行拟合。模型将学习“正常”数据的模样。
4. **对新数据评分**。每个新观测值都会获得一个异常分数。
5. **阈值选择**。确定分数截断点。这是一个业务决策：阈值越高，误报越少，但漏报的异常越多。
6. **告警与调查**。被标记的数据点将交由人工审查或触发自动响应。
7. **反馈收集**。记录被标记的项目是真实异常还是误报。利用这些数据来评估检测器，并随时间推移调整阈值。

该流水线永远不会“完成”。数据分布会发生偏移，新型异常会出现，阈值也需要不断调整。应将异常检测视为一个持续演进的系统，而非一次性的模型。

## 构建

`code/anomaly_detection.py` 中的代码从零开始实现了 Z分数（Z-score）、四分位距（IQR）和孤立森林（Isolation Forest）算法。

### Z分数检测器（Z-Score Detector）

def zscore_detect(X, threshold=3.0):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    z = np.abs((X - mean) / std)
    return z.max(axis=1) > threshold

实现简洁且已向量化。只要任意特征值超过阈值，就会将该数据点标记为异常。

### 四分位距检测器（IQR Detector）

def iqr_detect(X, factor=1.5):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outside = (X < lower) | (X > upper)
    return outside.any(axis=1)

### 从零实现孤立森林（Isolation Forest）

该从零开始的实现通过构建孤立树（Isolation Tree）来随机划分特征空间：

class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, depth=0):
        n, p = X.shape
        if depth >= self.max_depth or n <= 1:
            self.is_leaf = True
            self.size = n
            return self
        self.is_leaf = False
        self.feature = np.random.randint(p)
        x_min = X[:, self.feature].min()
        x_max = X[:, self.feature].max()
        if x_min == x_max:
            self.is_leaf = True
            self.size = n
            return self
        self.threshold = np.random.uniform(x_min, x_max)
        left_mask = X[:, self.feature] < self.threshold
        self.left = IsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
        self.right = IsolationTree(self.max_depth).fit(X[~left_mask], depth + 1)
        return self

隔离某个数据点所需的路径长度决定了其异常得分（Anomaly Score）。路径越短，表示该点越可能是异常值。

`IsolationForest` 类封装了多棵孤立树：

class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, seed=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples

    def fit(self, X):
        sample_size = min(self.max_samples, X.shape[0])
        max_depth = int(np.ceil(np.log2(sample_size)))
        for _ in range(self.n_estimators):
            idx = rng.choice(X.shape[0], size=sample_size, replace=False)
            tree = IsolationTree(max_depth=max_depth)
            tree.fit(X[idx])
            self.trees.append(tree)

    def anomaly_score(self, X):
        avg_path = average path length across all trees
        scores = 2.0 ** (-avg_path / c(max_samples))
        return scores

归一化因子 `c(n)` 表示在包含 n 个元素的二叉搜索树（Binary Search Tree）中进行不成功搜索的预期路径长度。其计算公式为 `2 * H(n-1) - 2*(n-1)/n`，其中 `H` 为调和数（Harmonic Number）。该归一化处理确保了不同规模数据集之间的异常得分具有可比性。

### 演示场景（Demo Scenarios）

代码生成了多种测试场景：

1. **含异常值的单簇数据。** 一个二维高斯分布簇，并在远离中心的位置注入异常点。所有方法在此场景下均应有效。
2. **多模态数据。** 包含三个大小和密度各异的簇。位于簇之间的数据点被视为异常。由于各特征的取值范围较广，Z分数方法在此场景下表现不佳。
3. **高维数据。** 包含 50 个特征，但异常点仅在其中的 5 个特征上表现出差异。用于测试各方法能否在特征子集中发现异常。

每个演示场景均使用精确率（Precision）、召回率（Recall）、F1分数（F1 Score）以及 Precision@k 指标对所有方法进行比较。

## Use It

使用 sklearn（使用库实现，而非从零编写）：

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_train)
predictions = iso.predict(X_test)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_train)
predictions = lof.predict(X_test)

注意，`contamination` 参数用于设定预期的异常值比例。正确配置该参数至关重要——设置过低会导致漏报异常，设置过高则会引发大量误报。

`anomaly_detection.py` 中的代码在同一数据集上对比了从零实现的算法与 sklearn 库的效果。

### sklearn 的 `contamination` 参数

sklearn 中的 `contamination` 参数决定了将连续异常分数 (anomaly scores) 转换为二分类预测的阈值 (threshold)。它不会改变底层的分数。

iso_5 = IsolationForest(contamination=0.05)
iso_10 = IsolationForest(contamination=0.10)

两者生成的异常分数完全相同。但 `iso_5` 会标记排名前 5% 的数据点，而 `iso_10` 则标记前 10%。如果你不知道真实的异常率（通常确实不知道），请将 `contamination` 设置为 `"auto"`，并直接基于原始分数进行工作。你可以根据误报 (false positives) 与漏报 (false negatives) 之间的成本权衡，自行设定决策阈值。

### 单类支持向量机 (One-Class SVM)

另一种值得了解的无监督异常检测器 (unsupervised anomaly detector)。单类支持向量机会在高维特征空间 (high-dimensional feature space) 中围绕正常数据拟合一个决策边界（利用核技巧 (kernel trick)）。

from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
oc_svm.fit(X_train)
predictions = oc_svm.predict(X_test)

`nu` 参数用于近似异常值的比例。单类支持向量机在中小型数据集上表现良好，但难以扩展到超大规模数据（因为核矩阵 (kernel matrix) 的大小会呈二次方增长）。

### 自编码器方法 (Autoencoder Approach)（预览）

自编码器 (Autoencoders) 是一种学习如何压缩并重建数据的神经网络 (neural networks)。模型仅在正常数据上进行训练。在测试阶段，异常数据会产生较高的重建误差 (reconstruction error)，因为网络只学会了重建正常模式。

这部分内容将在第三阶段（深度学习 (Deep Learning)）详细讲解，但其核心原理是一致的：对正常模式进行建模，并标记偏离该模式的数据。

### 集成异常检测 (Ensemble Anomaly Detection)

正如集成方法 (ensemble methods) 能提升分类效果（第 11 课）一样，组合多个异常检测器也能提升检测性能。最简单的方法如下：

1. 运行多个检测器（Z-score、IQR、孤立森林 (Isolation Forest)、局部异常因子 (LOF)）
2. 将每个检测器的分数归一化 (normalize) 到 [0, 1] 区间
3. 计算归一化后分数的平均值
4. 标记平均分数超过阈值的数据点

这种方法能有效降低误报率，因为不同算法的失效模式 (failure modes) 各不相同。被所有四种方法同时标记的数据点几乎可以确定是异常值。而仅被单一方法标记的数据点，可能只是该算法自身的特性所致。

更复杂的集成方案会根据每个检测器的预估可靠性进行加权（如果存在包含已知异常值的验证集 (validation set)，可在其上评估可靠性）。

### 生产环境考量

1. **阈值漂移 (Threshold drift)。** 随着数据分布发生偏移，固定的阈值会逐渐失效。需要持续监控异常分数的分布情况，并定期进行调整。
2. **警报疲劳 (Alert fatigue)。** 误报过多会导致运维人员逐渐忽视警报。建议初始阶段设置较高的阈值（警报数量少但更可靠），随着系统可信度的提升再逐步降低阈值。
3. **集成策略。** 在生产环境中，应组合使用多个检测器。仅当多种方法一致判定为异常时才进行标记。这能显著降低误报率。
4. **特征工程 (Feature engineering)。** 原始特征通常不足以支撑检测。建议加入滚动统计量、比率、距上次事件的时间间隔以及领域特定特征。优质的特征集比选择何种检测器更为重要。
5. **反馈循环 (Feedback loop)。** 当运维人员调查被标记的样本并确认或排除异常时，应将结果反馈回系统。随时间积累带标签的数据，以便持续评估并优化检测器。

## 部署上线

本章节将产出：
- `outputs/skill-anomaly-detector.md` -- 用于选择合适检测器的决策技能说明
- `code/anomaly_detection.py` -- 从零实现 Z分数（Z-score）、四分位距（IQR）和孤立森林（Isolation Forest），并与 sklearn 进行对比

### 选择阈值

异常分数（anomaly score）是连续值。你需要设定一个阈值（threshold）来做出二元决策。这本质上是一个业务决策，而非技术决策。

考虑以下两种场景：
- **欺诈检测（Fraud detection）。** 漏报欺诈的代价高昂（如拒付、客户信任流失）。误报（false alarms）的成本是人工分析师花费 5 分钟进行调查。应调低阈值以捕获更多欺诈行为，同时接受更多的误报。
- **设备维护。** 误报意味着不必要的停机，成本高达 5 万美元。漏报故障则意味着 50 万美元的维修费用。应设定阈值以平衡这两类成本。

在这两种情况下，最佳阈值取决于假阳性（false positives）与假阴性（false negatives）之间的成本比例。绘制不同阈值下的精确率（precision）和召回率（recall），叠加成本函数曲线，并选择成本最低的点。

### 扩展至生产环境

在生产环境中进行实时异常检测时：

1. **批量训练，在线评分（Batch training, online scoring）。** 定期（如每天或每周）使用近期的正常数据训练模型。每当新观测数据到达时，立即对其进行评分。
2. **特征计算必须保持一致。** 如果训练时使用了 30 天的滚动统计量（rolling statistics），那么为新观测数据计算特征时也需要 30 天的历史数据。请缓存所需的历史数据。
3. **分数分布监控。** 持续跟踪异常分数随时间的分布情况。如果中位数分数向上漂移，则说明数据分布发生了变化或模型已过时。
4. **可解释性（Explainability）。** 标记异常时，需说明原因。Z分数：“特征 X 高于正常水平 4.2 个标准差。” 孤立森林：“该数据点平均经过 3.1 次分割即被孤立（正常数据点平均需要 8.5 次）。”

## 练习

1. **阈值调优（Threshold tuning）。** 使用 Z分数检测器，将阈值从 1.0 到 5.0 以 0.5 为步长进行遍历。绘制每个阈值下的精确率和召回率。针对你的数据，最佳平衡点在哪里？

2. **多变量异常（Multivariate anomalies）。** 构建二维数据，使得每个特征单独看都正常，但组合起来呈现异常（例如，远离主簇对角线的点）。证明单特征 Z分数会漏检这些异常，而孤立森林能够捕获它们。

3. **从零实现 LOF。** 使用 k-近邻（k-nearest neighbors）算法实现局部异常因子（Local Outlier Factor, LOF）。在相同数据上与 sklearn 的 `LocalOutlierFactor` 进行对比。分别使用 k=10 和 k=50，观察 k 值的选择如何影响结果？

4. **流式异常检测（Streaming anomaly detection）。** 修改 Z分数检测器以适配流式场景：随着新数据点的到来，动态更新运行均值和方差（使用 Welford 在线算法（Welford's online algorithm））。在相同数据上与批量 Z分数进行对比。

5. **真实场景评估。** 选取一个包含已知异常的数据集（例如 Kaggle 上的信用卡欺诈数据集）。使用 precision@100、precision@500 和 AUPRC 评估上述四种方法。哪种方法效果最佳？原因是什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 异常 (Anomaly) | “离群点、异常点” | 显著偏离正常数据预期模式的数据点 |
| 点异常 (Point Anomaly) | “单个奇怪的值” | 无论上下文如何都显得异常的单个观测值 |
| 上下文异常 (Contextual Anomaly) | “值正常，但上下文不对” | 在特定上下文（如时间、地点等）中显得异常，但在其他上下文中可能正常的观测值 |
| 孤立森林 (Isolation Forest) | “通过随机划分来寻找离群点” | 一种由随机树组成的集成模型，它通过比正常点更少的划分次数来隔离异常点 |
| 局部异常因子 (Local Outlier Factor) | “与邻居比较密度” | 一种标记局部密度显著低于其邻居密度的数据点的方法 |
| Z分数 (Z-score) | “距离均值的标准差倍数” | (x - mean) / std，用于衡量数据点距离中心位置的标准差单位数 |
| 四分位距 (IQR) | “四分位范围” | Q3 - Q1，用于衡量中间 50% 数据的离散程度，常用于鲁棒的离群点检测 |
| 污染率 (Contamination) | “预期的异常比例” | 一个超参数，用于告知检测器应将多大比例的数据标记为异常 |
| 精确率@k (Precision@k) | “在前 k 个标记中，有多少是真实的” | 仅针对最可疑的 k 个点计算的精确率，适用于类别不平衡的异常检测任务 |
| 精确率-召回率曲线下面积 (AUPRC) | “精确率-召回率曲线下面积” | 一种汇总所有阈值下精确率-召回率性能的指标，在类别不平衡数据上优于 AUROC |

## 延伸阅读

- [Liu et al., Isolation Forest (2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) -- 孤立森林的原始论文
- [Breunig et al., LOF: Identifying Density-Based Local Outliers (2000)](https://dl.acm.org/doi/10.1145/342009.335388) -- 局部异常因子 (LOF) 的原始论文
- [scikit-learn Outlier Detection docs](https://scikit-learn.org/stable/modules/outlier_detection.html) -- scikit-learn 中所有异常检测器的概览
- [Chandola et al., Anomaly Detection: A Survey (2009)](https://dl.acm.org/doi/10.1145/1541880.1541882) -- 异常检测方法的全面综述
- [Goldstein and Uchida, A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms (2016)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152173) -- 在真实数据集上对 10 种方法进行的实证对比
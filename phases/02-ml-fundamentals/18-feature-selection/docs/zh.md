# 特征选择 (Feature Selection)

> 特征并非越多越好。合适的特征才是最好的。

**类型：** 构建
**语言：** Python
**前置条件：** 第二阶段，第 01-09 课，第 08 课（特征工程 (Feature Engineering)）
**时长：** 约 75 分钟

## 学习目标

- 从零开始实现过滤法 (Filter Methods)（方差阈值 (Variance Threshold)、互信息 (Mutual Information)、卡方检验 (Chi-squared)）与包装法 (Wrapper Methods)（递归特征消除 (RFE)、前向选择 (Forward Selection)）
- 解释为何互信息能够捕捉相关性 (Correlation) 所遗漏的特征与目标变量之间的非线性关系
- 比较 L1 正则化 (L1 Regularization)（嵌入法 (Embedded Selection)）与 RFE（包装法），并评估两者的计算权衡
- 构建结合多种方法的特征选择流水线 (Pipeline)，并在留出数据 (Held-out Data) 上验证其泛化能力 (Generalization) 的提升

## 问题描述

你手头有 500 个特征。模型训练缓慢，频繁过拟合 (Overfitting)，且无人能解释它究竟学到了什么。你试图添加更多特征以期提升性能，结果却适得其反。

这正是维度灾难 (Curse of Dimensionality) 的典型表现。随着特征数量的增加，特征空间的体积呈指数级膨胀。数据点变得极其稀疏，点与点之间的距离趋于收敛。模型需要呈指数级增长的数据量才能发现真实的模式。噪声特征淹没了信号特征，过拟合成了默认状态。

特征选择正是破解之道。剔除噪声，消除冗余，仅保留那些真正包含目标变量信息的特征。其结果是：训练速度更快、泛化能力更强，且模型具备可解释性。

我们的目标并非利用所有可用信息，而是精准使用正确的信息。

## 核心概念

### 特征选择（Feature Selection）的三大类别

每种特征选择方法都属于以下三大类别之一：

flowchart TD
    A[Feature Selection Methods] --> B[Filter Methods]
    A --> C[Wrapper Methods]
    A --> D[Embedded Methods]

    B --> B1["Variance Threshold"]
    B --> B2["Mutual Information"]
    B --> B3["Chi-squared Test"]
    B --> B4["Correlation Filtering"]

    C --> C1["Recursive Feature Elimination"]
    C --> C2["Forward Selection"]
    C --> C3["Backward Elimination"]

    D --> D1["L1 / Lasso Regularization"]
    D --> D2["Tree-based Importance"]
    D --> D3["Elastic Net"]

**过滤法（Filter Methods）**使用统计指标独立地对每个特征进行评分。它们不依赖模型。计算速度快，但会忽略特征之间的交互作用。

**包装法（Wrapper Methods）**通过训练模型来评估特征子集。它们以模型性能作为评分标准。效果通常更好，但计算成本高昂，因为需要多次重新训练模型。

**嵌入法（Embedded Methods）**将特征选择作为模型训练过程的一部分。L1 正则化（L1 Regularization）会将权重推向零。决策树（Decision Trees）会基于最有用的特征进行分裂。特征选择发生在模型拟合（Fitting）过程中，而非独立的步骤。

### 方差阈值（Variance Threshold）

最简单的过滤方法。如果一个特征在样本间几乎没有变化，那么它几乎不携带任何信息。

假设某个特征在 1000 个样本中有 999 个的值为 0.0，其方差接近于零。没有任何模型能利用它来区分类别。直接将其剔除。

variance(x) = mean((x - mean(x))^2)

设定一个阈值（例如 0.01）。剔除所有方差低于该阈值的特征。这种方法无需查看目标变量（Target Variable），即可移除常量或近似常量的特征。

适用场景：作为其他方法之前的预处理步骤。它能以近乎零的成本快速筛除明显无用的特征。

局限性：高方差的特征仍可能是纯噪声。方差阈值是必要条件，但非充分条件。

### 互信息（Mutual Information）

互信息用于衡量已知特征 X 的值能在多大程度上降低对目标变量 Y 的不确定性。

I(X; Y) = sum_x sum_y p(x, y) * log(p(x, y) / (p(x) * p(y)))

如果 X 和 Y 相互独立，则 p(x, y) = p(x) * p(y)，此时对数项为零，I(X; Y) = 0。X 提供的关于 Y 的信息越多，互信息值就越高。

相较于相关性（Correlation）的主要优势：互信息能够捕捉非线性关系。某个特征可能与目标变量的相关系数为零，但由于存在二次或周期性关系，其互信息值可能很高。

对于连续型特征，需先进行离散化分箱（基于直方图的估计）。分箱数量会影响估计结果——分箱过少会丢失信息，分箱过多则会引入噪声。常见选择：$\sqrt{n}$ 个分箱或 Sturges 公式（$1 + \log_2(n)$）。

flowchart LR
    A[Feature X] --> B[Discretize into Bins]
    B --> C["Compute Joint Distribution p(x,y)"]
    C --> D["Compute MI = sum p(x,y) * log(p(x,y) / p(x)p(y))"]
    D --> E["Rank Features by MI Score"]
    E --> F[Select Top K]

### 递归特征消除（Recursive Feature Elimination, RFE）

RFE 是一种包装法。它利用模型自身的特征重要性进行迭代剪枝：

1. 使用所有特征训练模型
2. 按重要性对特征排序（线性模型看系数，树模型看不纯度减少量）
3. 移除重要性最低的特征
4. 重复上述步骤，直到保留所需数量的特征

flowchart TD
    A["Start: All N Features"] --> B["Train Model"]
    B --> C["Rank Feature Importances"]
    C --> D["Remove Least Important"]
    D --> E{"Features == Target Count?"}
    E -->|No| B
    E -->|Yes| F["Return Selected Features"]

RFE 会考虑特征交互作用，因为模型会同时观察所有剩余特征。移除一个特征会改变其他特征的重要性。这使得它比过滤法更为全面。

代价：需要训练模型 N - 目标数量 次。若有 500 个特征且目标保留 10 个，则需训练 490 次。对于计算成本高的模型，这会很慢。可以通过每步移除多个特征来加速（例如每轮移除排名最后的 10%）。

### L1（Lasso）正则化（L1 Regularization）

L1 正则化将权重的绝对值加入损失函数（Loss Function）：

loss = prediction_error + alpha * sum(|w_i|)

参数 alpha 控制特征剪枝的强度。alpha 值越高，越多的权重会精确变为零。

为何会精确为零？L1 惩罚项在权重空间中形成了一个菱形约束区域。最优解往往落在该菱形的顶点上，此时一个或多个权重恰好为零。而 L2 正则化（L2 Regularization，岭回归 Ridge Regression）形成的是圆形约束，权重会缩小但极少精确为零。

这属于嵌入法特征选择：模型在训练过程中自动学习忽略哪些特征。权重为零的特征实际上已被剔除。

优势：只需单次训练；能处理共线性特征（保留其中一个，将其他权重置零）；已内置于大多数线性模型实现中。

局限性：仅适用于线性模型。无法捕捉非线性特征重要性。

### 基于树的特征重要性（Tree-Based Feature Importance）

决策树及其集成模型（随机森林 Random Forests、梯度提升 Gradient Boosting）天然具备特征排序能力。每次分裂都会降低不纯度（Impurity）（分类任务使用基尼系数 Gini 或熵 Entropy，回归任务使用方差 Variance）。能带来更大不纯度下降的特征更为重要。

对于包含 T 棵树的随机森林：

importance(feature_j) = (1/T) * sum over all trees of
    sum over all nodes splitting on feature_j of
        (n_samples * impurity_decrease)

这会为每个特征输出一个归一化的重要性得分。它能自动处理非线性关系和特征交互作用。

注意：基于树的重要性评估会偏向具有大量唯一值的特征（高基数 High Cardinality）。随机 ID 列会显得非常重要，因为它能完美区分每个样本。建议使用排列重要性（Permutation Importance）进行合理性验证。

### 排列重要性（Permutation Importance）

一种与模型无关（Model-Agnostic）的方法：

1. 训练模型并记录其在验证集上的基线性能
2. 针对每个特征：随机打乱其取值，测量性能下降幅度
3. 性能下降幅度越大，说明该特征越重要

如果打乱某个特征后性能未受损，说明模型不依赖该特征。如果性能急剧下降，则该特征至关重要。

排列重要性避免了基于树方法的基数偏差。但计算较慢：每个特征都需要进行一次完整评估，且为保证稳定性通常需重复多次。

### 方法对比表

| 方法 | 类型 | 速度 | 非线性 | 特征交互 |
|--------|------|-------|-----------|---------------------|
| 方差阈值 | 过滤法 | 极快 | 否 | 否 |
| 互信息 | 过滤法 | 快 | 是 | 否 |
| 相关性过滤 | 过滤法 | 快 | 否 | 否 |
| RFE | 包装法 | 慢 | 取决于模型 | 是 |
| L1 / Lasso | 嵌入法 | 快 | 否（仅线性） | 否 |
| 树重要性 | 嵌入法 | 中等 | 是 | 是 |
| 排列重要性 | 模型无关 | 慢 | 是 | 是 |

### 决策流程图

flowchart TD
    A[Start: Feature Selection] --> B{How many features?}
    B -->|"< 50"| C["Start with variance threshold + mutual information"]
    B -->|"50-500"| D["Variance threshold, then L1 or tree importance"]
    B -->|"> 500"| E["Variance threshold, then mutual info filter, then RFE on survivors"]

    C --> F{Using linear model?}
    D --> F
    E --> F

    F -->|Yes| G["L1 regularization for final selection"]
    F -->|No - trees| H["Tree importance + permutation importance"]
    F -->|No - other| I["RFE with your model"]

    G --> J[Validate: compare selected vs all features]
    H --> J
    I --> J

    J --> K{Performance improved?}
    K -->|Yes| L["Ship with selected features"]
    K -->|No| M["Try different method or keep all features"]


## 构建

### 步骤 1：生成具有已知特征结构的合成数据 (Synthetic Data)

import numpy as np


def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)

    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)

    informative = np.column_stack([x1, x2, x3, x4, x5])

    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])

    noise = rng.randn(n_samples, 10) * 0.5

    X = np.hstack([informative, correlated, noise])
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)

    feature_names = (
        [f"info_{i}" for i in range(5)]
        + [f"corr_{i}" for i in range(5)]
        + [f"noise_{i}" for i in range(10)]
    )

    return X, y, feature_names

我们已知基准真相 (Ground Truth)：特征 0-4 为信息特征 (Informative Features)（其中特征 3 和 4 是特征 0 和 1 的相关副本），特征 5-9 与信息特征相关，特征 10-19 为纯噪声 (Pure Noise)。优秀的特征选择方法应将 0-4 排在最高位，将 10-19 排在最低位。

### 步骤 2：方差阈值 (Variance Threshold)

def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances

### 步骤 3：互信息（离散）(Mutual Information)

def discretize(x, n_bins=10):
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned


def mutual_information(X, y, n_bins=10):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples

    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))

        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi

    return mi_scores

### 步骤 4：递归特征消除 (Recursive Feature Elimination, RFE)

def simple_logistic_importance(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        w -= lr * (X.T @ error) / n_samples
        b -= lr * np.mean(error)

    return w, b


def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total

    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)

        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)

    for idx in remaining:
        rankings[idx] = 1

    selected_mask = rankings == 1
    return selected_mask, rankings

### 步骤 5：L1 特征选择 (L1 Feature Selection)

def soft_threshold(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y

        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)

        w -= lr * gradient_w
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b

    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w

### 步骤 6：基于树的重要性评估（简单决策树 (Decision Tree)）

def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_idx):
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0

    best_threshold = None
    best_gain = -1.0
    parent_gini = gini_impurity(y)
    n = len(y)

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            continue

        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[right_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def tree_importance(X, y, n_trees=50, max_depth=5, seed=42):
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    for _ in range(n_trees):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        feature_subset = rng.choice(n_features, size=max(1, int(np.sqrt(n_features))), replace=False)

        X_boot = X[sample_idx]
        y_boot = y[sample_idx]

        tree_imp = _build_tree_importance(X_boot, y_boot, feature_subset, max_depth)
        importances += tree_imp

    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


def _build_tree_importance(X, y, feature_subset, max_depth, depth=0):
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    if depth >= max_depth or len(np.unique(y)) <= 1 or len(y) < 4:
        return importances

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    for f in feature_subset:
        threshold, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_threshold = threshold

    if best_feature is None or best_gain <= 0:
        return importances

    importances[best_feature] += best_gain * len(y)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    importances += _build_tree_importance(X[left_mask], y[left_mask], feature_subset, max_depth, depth + 1)
    importances += _build_tree_importance(X[right_mask], y[right_mask], feature_subset, max_depth, depth + 1)

    return importances

### 步骤 7：运行所有方法并进行比较

该代码文件将在同一合成数据集上运行上述五种方法，并打印对比表格，展示每种方法所筛选出的特征。

## 上手实践

在 scikit-learn 中，特征选择（Feature Selection）已内置于流水线（Pipeline）中：

from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)

mi_scores = mutual_info_classif(X, y)
top_k = np.argsort(mi_scores)[-10:]

rfe_selector = RFE(LogisticRegression(), n_features_to_select=10)
rfe_selector.fit(X, y)
X_rfe = rfe_selector.transform(X)

lasso_selector = SelectFromModel(Lasso(alpha=0.01))
lasso_selector.fit(X, y)
X_lasso = lasso_selector.transform(X)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_

从零实现的代码清晰地展示了每种方法内部的运行机制。方差阈值（Variance Threshold）仅仅是计算 `var(X, axis=0)` 并应用掩码（Mask）。互信息（Mutual Information）则是在列联表（Contingency Table）中统计联合频率与边缘频率。递归特征消除（Recursive Feature Elimination, RFE）是一个包含训练、排序和剪枝的循环过程。L1 正则化（L1 Regularization）是带有软阈值（Soft-thresholding）步骤的梯度下降（Gradient Descent）。树模型特征重要性（Tree Feature Importance）会累加各个分裂节点上的不纯度（Impurity）减少量。这里没有魔法，只有统计学原理和循环逻辑。

scikit-learn 的版本则增强了鲁棒性（例如 `mutual_info_classif` 使用 k 近邻（k-NN）密度估计而非分箱（Binning））、运行速度（基于 C 语言实现）以及流水线集成（Pipeline Integration）能力。

## 交付成果

本章节将产出：
- `outputs/skill-feature-selector.md` -- 用于选择合适特征选择方法的快速参考决策树

## 练习

1. **前向选择（Forward Selection）**：实现与递归特征消除（Recursive Feature Elimination, RFE）相反的过程。从零个特征开始，在每一步中，添加能最大程度提升模型性能的特征。当继续添加特征不再带来性能提升时停止。将所选特征与 RFE 的结果进行对比。哪种方法更快？哪种方法效果更好？

2. **稳定性选择（Stability Selection）**：运行 L1 特征选择（L1 Feature Selection）50 次，每次使用数据的随机 80% 子样本，并采用略有差异的 alpha 值。统计每个特征被选中的频次。在超过 80% 的运行中被选中的特征即为“稳定”特征。将稳定特征与单次运行的 L1 特征选择结果进行对比。哪种方法更可靠？

3. **多重共线性检测（Multicollinearity Detection）**：计算所有特征的相关系数矩阵（Correlation Matrix）。实现一个函数，在给定相关系数阈值（例如 0.9）的情况下，从每一对高度相关的特征中移除其中一个（保留与目标变量互信息（Mutual Information）更高的那个）。在合成数据集上进行测试，验证其是否成功移除了冗余的相关特征。

4. **特征选择流水线（Feature Selection Pipeline）**：将方差阈值（Variance Threshold）、互信息过滤器（Mutual Information Filter）和 RFE 串联成一个单一的流水线。首先移除近零方差特征，然后保留互信息排名前 50% 的特征，最后在剩余特征上运行 RFE。将此流水线与直接对所有特征单独运行 RFE 的结果进行对比。该流水线是否更快？其准确率是否相当？

5. **从零实现排列重要性（Permutation Importance）**：实现排列重要性算法。对每个特征，将其值随机打乱 10 次，测量 F1 分数（F1 Score）的平均下降幅度。将得到的特征排名与基于树模型的重要性（Tree-based Importance）进行对比。找出两者排名不一致的情况并解释原因（提示：考虑相关特征）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 过滤法 (Filter method) | “独立评估特征得分” | 一种特征选择方法，无需训练模型，仅通过统计指标对特征进行排序，独立评估每个特征 |
| 包装法 (Wrapper method) | “利用模型来挑选特征” | 一种特征选择方法，通过训练模型来评估特征子集，并将模型性能作为选择标准 |
| 嵌入法 (Embedded method) | “模型在训练过程中自动选择特征” | 在模型拟合过程中同步进行的特征选择，例如 L1 正则化 (L1 regularization) 会将不重要特征的权重压缩至零 |
| 互信息 (Mutual information) | “一个变量能透露多少关于另一个变量的信息” | 衡量在已知 X 的情况下 Y 的不确定性减少程度，能够同时捕捉线性与非线性依赖关系 |
| 递归特征消除 (Recursive Feature Elimination) | “训练、排序、剪枝、循环” | 一种迭代的包装法，通过训练模型、剔除最不重要的特征，并重复此过程直至达到目标特征数量 |
| L1 / Lasso 正则化 (L1 / Lasso regularization) | “能‘杀死’特征的惩罚项” | 在损失函数中加入权重绝对值之和，促使不重要特征的权重精确收敛至零 |
| 方差阈值 (Variance threshold) | “剔除恒定特征” | 丢弃样本间方差低于指定阈值的特征，从而过滤掉不携带任何信息的特征 |
| 特征重要性 (Feature importance) | “哪些特征最关键” | 衡量每个特征对模型预测贡献程度的分数，通常基于分裂增益 (split gains)（树模型）或系数大小 (coefficient magnitudes)（线性模型）计算得出 |
| 排列重要性 (Permutation importance) | “打乱数据并评估性能损失” | 通过随机打乱单个特征的值，并测量模型性能随之下降的程度来评估特征重要性 |
| 维度灾难 (Curse of dimensionality) | “特征太多，数据太少” | 随着特征增加，特征空间体积呈指数级膨胀，导致数据变得稀疏且距离度量失去意义的现象 |

## 扩展阅读

- [变量与特征选择导论 (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) -- 特征选择方法的基础性综述文献，至今仍被广泛引用
- [scikit-learn 特征选择指南](https://scikit-learn.org/stable/modules/feature_selection.html) -- 提供过滤法、包装法和嵌入法的实用参考及代码示例
- [稳定性选择 (Stability Selection) (Meinshausen & Buhlmann, 2010)](https://arxiv.org/abs/0809.2932) -- 将子采样与特征选择相结合，以获得稳健且可复现的结果
- [警惕随机森林默认的重要性评估 (Beware Default Random Forest Importances) (Strobl et al., 2007)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25) -- 揭示了基于树模型的重要性评估中存在的基数偏差 (cardinality bias)，并提出条件重要性作为替代方案
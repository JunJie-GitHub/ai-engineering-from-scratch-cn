# 处理不平衡数据 (Imbalanced Data)

> 当 99% 的数据都是“正常”样本时，准确率 (Accuracy) 就是一个谎言。

**类型：** 构建
**语言：** Python
**前置要求：** 第二阶段，第 01-09 课（尤其是评估指标 (Evaluation Metrics)）
**时长：** 约 90 分钟

## 学习目标

- 从零实现合成少数类过采样技术 (SMOTE)，并解释合成过采样 (Synthetic Oversampling) 与随机复制 (Random Duplication) 的区别
- 使用 F1 分数 (F1 Score)、AUPRC (Area Under the Precision-Recall Curve) 和马修斯相关系数 (Matthews Correlation Coefficient) 而非准确率来评估不平衡分类器 (Imbalanced Classifiers)
- 比较类别权重 (Class Weighting)、阈值调整 (Threshold Tuning) 和重采样策略 (Resampling Strategies)，并针对给定的不平衡比例 (Imbalance Ratio) 选择合适的方法
- 构建一个完整的不平衡数据处理流水线 (Pipeline)，结合 SMOTE、类别权重和阈值优化 (Threshold Optimization)

## 问题描述

你构建了一个欺诈检测模型，准确率高达 99.9%。你为此庆祝，随后却发现它对每一笔交易都预测为“非欺诈”。

这不是程序漏洞。当仅有 0.1% 的交易存在欺诈时，这是模型做出的理性选择。模型学到的是：始终预测多数类 (Majority Class) 能够最小化整体误差。这在技术上完全正确，但毫无实际用处。

在真正需要分类的实际场景中，这种现象无处不在。疾病诊断：阳性率仅 1%。网络入侵：攻击占比 0.01%。制造缺陷：次品率 0.5%。垃圾邮件过滤：垃圾邮件占 20%。用户流失预测：流失用户占 5%。少数类 (Minority Class) 的影响越重大，它往往就越罕见。

准确率之所以失效，是因为它对所有正确的预测一视同仁。正确标记一笔正常交易和成功揪出一笔欺诈交易，在计算中都只算作一分。但识别欺诈才是该模型存在的根本意义。我们需要引入特定的评估指标、技术手段和训练策略，迫使模型去关注那些罕见却至关重要的类别。

## 核心概念

### 为什么准确率 (Accuracy) 会失效

假设有一个包含 1000 个样本的数据集：990 个负样本，10 个正样本。如果一个模型始终预测为负类：

|  | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际为正类 | 0 (TP) | 10 (FN) |
| 实际为负类 | 0 (FP) | 990 (TN) |

准确率 (Accuracy) = (0 + 990) / 1000 = 99.0%

该模型未能识别出任何欺诈、疾病或缺陷。但准确率却显示为 99%。这就是为什么在处理类别不平衡 (Class Imbalance) 问题时，准确率具有误导性。

### 更优的评估指标

**精确率 (Precision)** = TP / (TP + FP)。在所有被标记为正类的样本中，有多少是真正的正类？高精确率意味着误报 (False Alarm) 较少。

**召回率 (Recall)** = TP / (TP + FN)。在所有实际为正类的样本中，我们成功找出了多少？高召回率意味着漏报的正类较少。

**F1 分数 (F1 Score)** = 2 * precision * recall / (precision + recall)。它是精确率和召回率的调和平均数。相比于算术平均数，它对两者之间的极端不平衡惩罚更重。

**F-beta 分数 (F-beta Score)** = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)。当 beta > 1 时，更看重召回率；当 beta < 1 时，更看重精确率。F2 分数常用于欺诈检测（漏报欺诈比误报的代价更大）。

**AUPRC**（精确率-召回率曲线下面积，Area Under Precision-Recall Curve）。类似于 AUC-ROC，但在类别不平衡数据上能提供更多信息。随机分类器的 AUPRC 等于正类比例（不像 ROC 那样固定为 0.5）。这使得模型性能的提升更容易被观察到。

**马修斯相关系数 (Matthews Correlation Coefficient, MCC)** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。取值范围为 -1 到 +1。仅当模型在两个类别上都表现良好时才会给出高分。即使类别规模差异巨大，该指标也能保持平衡。

对于上述“始终预测为负类”的模型：精确率 = 0/0（未定义，通常设为 0），召回率 = 0/10 = 0，F1 = 0，MCC = 0。这些指标正确地指出该模型毫无价值。

### 类别不平衡数据处理流程

flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B

### SMOTE：合成少数类过采样技术 (Synthetic Minority Oversampling Technique)

随机过采样 (Random Oversampling) 会直接复制现有的少数类样本。这种方法虽然有效，但存在过拟合 (Overfitting) 风险，因为模型会反复看到完全相同的样本点。

SMOTE 会生成合理但非复制的新合成少数类样本。其算法步骤如下：

1. 对于每个少数类样本 x，在其他少数类样本中找到其 k 个最近邻 (k-Nearest Neighbors)
2. 随机选择其中一个邻居
3. 在 x 与该邻居之间的线段上创建一个新样本

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

该方法在真实的少数类点之间进行插值，从而在特征空间的相同区域内生成新样本，而不仅仅是复制现有数据。

flowchart LR
    subgraph Original["Original Minority Points"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE Generation"]
        direction TB
        S1["Pick x1, neighbor x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["Augmented Set"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result

### 采样策略对比

**随机过采样 (Random Oversampling)**：复制少数类样本，使其数量与多数类匹配。
- 优点：简单，无信息损失
- 缺点：完全相同的副本会导致过拟合，增加训练时间

**随机欠采样 (Random Undersampling)**：移除多数类样本，使其数量与少数类匹配。
- 优点：训练速度快，实现简单
- 缺点：丢弃了可能有用的多数类数据，模型方差较高

**SMOTE**：通过插值生成合成的少数类样本。
- 优点：生成新数据点，相比随机过采样能降低过拟合风险
- 缺点：可能在决策边界附近生成噪声样本，未考虑多数类的分布情况

| 策略 | 数据变化 | 风险 | 适用场景 |
|----------|-------------|------|-------------|
| 过采样 (Oversample) | 复制少数类 | 过拟合 | 小型数据集，中度不平衡 |
| 欠采样 (Undersample) | 移除多数类 | 信息丢失 | 大型数据集，追求快速训练 |
| SMOTE | 添加合成少数类 | 边界噪声 | 中度不平衡，有足够少数类样本用于 k-NN |

### 类别权重 (Class Weights)

不改变数据本身，而是改变模型处理错误的方式。为误分类少数类样本分配更高的权重。

对于一个包含 950 个负样本和 50 个正样本的二分类问题：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类的权重是负类的 19 倍。误分类一个正样本的代价相当于误分类 19 个负样本。这迫使模型更加关注少数类。

在逻辑回归 (Logistic Regression) 中，这会修改损失函数：

weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])

其中 w_i 取决于样本 i 所属的类别。

从数学期望上看，类别权重与过采样是等价的，但无需创建新数据点。这使得计算更快，并避免了因样本重复带来的过拟合风险。

### 阈值调优 (Threshold Tuning)

大多数分类器会输出一个概率值。默认阈值为 0.5：如果 P(正类) >= 0.5，则预测为正类。但 0.5 是人为设定的。当类别不平衡时，最优阈值通常要低得多。

具体流程如下：
1. 训练模型
2. 获取验证集上的预测概率
3. 在 0.0 到 1.0 范围内遍历阈值
4. 计算每个阈值下的 F1 分数（或你选定的指标）
5. 选择使该指标最大化的阈值

flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]

模型可能对一笔欺诈交易输出 P(欺诈) = 0.15。在阈值 0.5 下，它会被判定为非欺诈。但在阈值 0.10 下，它就能被正确识别。相比于概率校准，排序能力更为关键——只要欺诈样本的概率普遍高于非欺诈样本，就必然存在一个阈值能将它们区分开来。

### 代价敏感学习 (Cost-Sensitive Learning)

类别权重的泛化形式。不再使用统一代价，而是为不同的误分类分配特定代价：

| | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际为正类 | 0（正确） | C_FN = 100 |
| 实际为负类 | C_FP = 1 | 0（正确） |

漏报一笔欺诈交易（FN）的代价是误报（FP）的 100 倍。模型优化的是总代价，而非总错误数量。

当你能估算现实世界中的代价时，这是最符合理论依据的方法。漏诊癌症的代价与导致额外活检的误报代价截然不同。明确这些代价能迫使模型做出正确的权衡。

### 决策流程图

flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["Mild: try class weights first"]
    B -->|"70/30 to 95/5"| D["Moderate: SMOTE + class weights"]
    B -->|"> 95/5"| E["Severe: combine multiple strategies"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["Oversample or SMOTE, avoid undersampling"]
    F -->|"1000-10000"| H["SMOTE + threshold tuning"]
    F -->|"> 10000"| I["Undersampling OK, or class weights"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]


## 构建

### 步骤 1：生成不平衡数据集 (Imbalanced Dataset)

import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]

### 步骤 2：从零实现 SMOTE (Synthetic Minority Over-sampling Technique)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)

### 步骤 3：随机过采样 (Random Oversampling) 与欠采样 (Undersampling)

def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]

### 步骤 4：带类别权重 (Class Weights) 的逻辑回归 (Logistic Regression)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])

### 步骤 5：阈值调优 (Threshold Tuning)

def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score

### 步骤 6：评估函数 (Evaluation Functions)

def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }

### 步骤 7：对比所有方法 (Compare All Approaches)

X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)

该代码文件将所有步骤整合在一个脚本中运行，并打印输出结果。

## 使用方法

借助 scikit-learn 和 imbalanced-learn 库，这些技术只需一行代码即可实现：

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))

手写实现的代码清晰地展示了每种技术的具体原理。合成少数类过采样技术（SMOTE）本质上只是对少数类（minority class）进行 k近邻（k-NN）插值。类别权重（class weights）通过缩放损失函数来发挥作用。阈值调优（threshold tuning）不过是在不同截断值上遍历的 for 循环。没有任何黑魔法。

## 交付成果

本课程的产出物包括：
- `outputs/skill-imbalanced-data.md` -- 用于处理类别不平衡分类问题（imbalanced classification problems）的决策检查清单

## 练习

1. **边界SMOTE（Borderline-SMOTE）**：修改 SMOTE 的实现，使其仅针对靠近决策边界（decision boundary）的少数类样本生成合成样本（synthetic samples）（即那些 k 近邻中包含多数类（majority class）样本的点）。在类别存在重叠的数据集上，将其结果与标准 SMOTE 进行对比。

2. **代价矩阵优化（Cost matrix optimization）**：实现代价敏感学习（cost-sensitive learning），其中代价矩阵（cost matrix）作为参数传入。创建一个函数，接收代价矩阵并返回使期望代价（expected cost）最小化的最优预测结果。使用不同的代价比例（1:10、1:100、1:1000）进行测试，并绘制精确率-召回率权衡（precision-recall tradeoff）的变化曲线。

3. **阈值校准（Threshold calibration）**：实现 Platt 缩放（Platt scaling）（即在模型的原始输出上拟合逻辑回归以生成校准概率（calibrated probabilities））。对比校准前后的精确率-召回率曲线（precision-recall curve）。证明校准不会改变样本的排序（曲线下面积（AUC）保持不变），但会使输出的概率更具实际意义。

4. **基于平衡装袋的集成学习（Ensemble with balanced bagging）**：训练多个模型，每个模型均使用平衡的自助采样样本（bootstrap sample）（包含全部少数类样本与随机抽取的多数类子集）。对它们的预测结果取平均。将此方法与使用 SMOTE 的单一模型进行对比。测量多次运行下的性能表现与方差（variance）。

5. **不平衡比例实验（Imbalance ratio experiment）**：选取一个平衡数据集，逐步提高类别不平衡比例（imbalance ratio）（50/50、70/30、90/10、95/5、99/1）。针对每个比例，分别在使用和不使用 SMOTE 的情况下进行训练。绘制两种方法的 F1 分数（F1 score）随不平衡比例变化的曲线。SMOTE 在何种比例下开始产生显著的性能差异？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 类别不平衡 (Class Imbalance) | “某一类的样本数量远多于其他类” | 数据集中各类别的分布严重倾斜，导致模型倾向于预测多数类 |
| SMOTE (Synthetic Minority Over-sampling Technique) | “合成过采样” | 通过对现有少数类样本及其 k 个最近邻少数类样本进行插值，生成新的少数类样本 |
| 类别权重 (Class Weights) | “让模型在稀有类别上犯错付出更高代价” | 将损失函数乘以特定类别的权重，使模型对少数类的误分类施加更重的惩罚 |
| 阈值调整 (Threshold Tuning) | “移动决策边界” | 将分类的概率阈值从默认的 0.5 调整为能优化目标指标的特定值 |
| 精确率-召回率权衡 (Precision-Recall Tradeoff) | “无法同时兼顾两者” | 降低阈值能捕获更多正例（提高召回率），但也会标记更多假阳性（降低精确率），反之亦然 |
| AUPRC (Area Under the Precision-Recall Curve) | “PR 曲线下面积” | 将精确率-召回率曲线汇总为单一数值；在类别严重不平衡时，比 AUC-ROC 更具参考价值 |
| 马修斯相关系数 (Matthews Correlation Coefficient) | “均衡型指标” | 预测标签与实际标签之间的相关系数，仅当模型在两个类别上均表现良好时才会得出高分 |
| 代价敏感学习 (Cost-Sensitive Learning) | “不同类型的错误代价不同” | 将现实世界中的误分类成本纳入训练目标，使模型优化总成本而非单纯的错误数量 |
| 随机过采样 (Random Oversampling) | “复制少数类样本” | 通过重复少数类样本来平衡各类别数量；方法简单，但存在对重复样本过拟合的风险 |

## 延伸阅读

- [SMOTE：合成少数类过采样技术 (Chawla 等, 2002)](https://arxiv.org/abs/1106.1813) -- SMOTE 的原始论文，至今仍是类别不平衡学习领域被引用最多的著作
- [从不平衡数据中学习 (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) -- 全面综述了采样、代价敏感及算法层面的各类方法
- [imbalanced-learn 官方文档](https://imbalanced-learn.org/stable/) -- 提供 SMOTE 变体、欠采样策略及流水线集成的 Python 库
- [精确率-召回率图比 ROC 图包含更多信息 (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- 阐述了在处理不平衡问题时，为何以及何时应优先使用 PR 曲线而非 ROC 曲线
# 模型评估 (Model Evaluation)

> 模型的好坏完全取决于你评估它的方式。

**类型：** 实战构建
**编程语言：** Python
**前置知识：** 第一阶段（概率与分布、机器学习统计学），第二阶段第 1-8 课
**预计时长：** 约 90 分钟

## 学习目标

- 从零实现 K 折交叉验证 (K-fold Cross-Validation) 与分层 K 折交叉验证 (Stratified K-fold Cross-Validation)，并解释为何分层处理对类别不平衡数据 (Imbalanced Data) 至关重要
- 从零计算精确率 (Precision)、召回率 (Recall)、F1 分数 (F1 Score)、AUC-ROC 以及回归指标 (Regression Metrics)（包括均方误差 MSE、均方根误差 RMSE、平均绝对误差 MAE、决定系数 R-squared）
- 解读学习曲线 (Learning Curves)，以诊断模型是否存在高偏差 (High Bias) 或高方差 (High Variance) 问题
- 识别常见的评估错误，包括数据泄露 (Data Leakage)、指标选择不当以及测试集污染 (Test Set Contamination)

## 问题剖析

你训练了一个模型，它在你的数据上达到了 95% 的准确率 (Accuracy)。这算好吗？

不一定。如果你的数据中有 95% 都属于同一个类别，那么一个永远只预测该类别的模型也能达到 95% 的准确率，但它实际上毫无用处。如果你使用训练集本身进行评估，这 95% 的数值毫无意义，因为模型仅仅记住了答案。如果你的数据集包含时间维度，且在划分前进行了随机打乱，你的模型可能会利用未来数据来预测过去。

模型评估是大多数机器学习 (Machine Learning) 项目最容易出错的环节。错误的评估指标会让劣质模型看起来表现优异。错误的数据划分方式会让模型“作弊”。错误的对比方法会让你选中更差的模型。做好评估并非可选项，而是决定模型能否在生产环境中稳定运行，还是在接触真实数据时瞬间崩溃的关键分水岭。

## 核心概念

### 训练集、验证集与测试集

flowchart LR
    A[Full Dataset] --> B[Train Set 60-70%]
    A --> C[Validation Set 15-20%]
    A --> D[Test Set 15-20%]
    B --> E[Fit Model]
    E --> C
    C --> F[Tune Hyperparameters]
    F --> E
    F --> G[Final Model]
    G --> D
    D --> H[Report Performance]

三个数据集划分，对应三种不同用途：

- **训练集 (Training Set)**：模型从中学习的数据。在训练过程中，模型会看到这些样本。
- **验证集 (Validation Set)**：用于调整超参数 (Hyperparameters) 并在不同模型之间进行选择。模型绝不会在此数据上进行训练，但你的决策会受到它的影响。
- **测试集 (Test Set)**：仅在最后阶段使用一次，用于报告最终性能。如果你查看了测试集性能并据此回头修改模型，它就不再是测试集，而是变成了第二个验证集。

测试集是你独立保留的保障，用于确保报告的性能能够真实反映模型在完全未见过的数据上的表现。

### K折交叉验证 (K-Fold Cross-Validation)

对于小型数据集，单一的划分方式会浪费数据并产生波动较大的评估结果。K折交叉验证会让所有数据都参与训练和验证：

flowchart TB
    subgraph Fold1["Fold 1"]
        direction LR
        V1["Val"] --- T1a["Train"] --- T1b["Train"] --- T1c["Train"] --- T1d["Train"]
    end
    subgraph Fold2["Fold 2"]
        direction LR
        T2a["Train"] --- V2["Val"] --- T2b["Train"] --- T2c["Train"] --- T2d["Train"]
    end
    subgraph Fold3["Fold 3"]
        direction LR
        T3a["Train"] --- T3b["Train"] --- V3["Val"] --- T3c["Train"] --- T3d["Train"]
    end
    subgraph Fold4["Fold 4"]
        direction LR
        T4a["Train"] --- T4b["Train"] --- T4c["Train"] --- V4["Val"] --- T4d["Train"]
    end
    subgraph Fold5["Fold 5"]
        direction LR
        T5a["Train"] --- T5b["Train"] --- T5c["Train"] --- T5d["Train"] --- V5["Val"]
    end
    Fold1 --> R["Average scores"]
    Fold2 --> R
    Fold3 --> R
    Fold4 --> R
    Fold5 --> R

1. 将数据划分为 K 个大小相等的子集（折）
2. 对于每一折，使用其余 K-1 折进行训练，并在剩余的那一折上进行验证
3. 计算 K 次验证得分的平均值

通常选择 K=5 或 K=10。每个数据点恰好会被用作验证集一次。平均得分比任何单次划分都能提供更稳定的评估结果。

**分层 K 折交叉验证 (Stratified K-Fold)**：保持每一折中的类别分布比例一致。如果你的数据集中类别 A 占 70%，类别 B 占 30%，那么每一折都会保持大致相同的比例。这对于类别不平衡的数据集尤为重要，因为随机划分可能会导致所有少数类样本都集中在某一折中。

### 分类评估指标 (Classification Metrics)

**混淆矩阵 (Confusion Matrix)**：所有指标的基础。对于二分类问题：

|  | 预测为正类 (Predicted Positive) | 预测为负类 (Predicted Negative) |
|--|---|---|
| 实际为正类 (Actually Positive) | 真阳性 (True Positive, TP) | 假阴性 (False Negative, FN) |
| 实际为负类 (Actually Negative) | 假阳性 (False Positive, FP) | 真阴性 (True Negative, TN) |

基于该矩阵，可推导出其他所有指标：

- **准确率 (Accuracy)** = (TP + TN) / (TP + TN + FP + FN)。正确预测的样本比例。在类别不平衡时具有误导性。
- **精确率 (Precision)** = TP / (TP + FP)。在所有被预测为正类的样本中，实际为正类的比例。当假阳性代价较高时使用（例如：垃圾邮件过滤器将正常邮件误判为垃圾邮件）。
- **召回率 (Recall)**（又称灵敏度）= TP / (TP + FN)。在所有实际为正类的样本中，模型成功找出的比例。当假阴性代价较高时使用（例如：癌症筛查漏诊肿瘤）。
- **F1 分数 (F1 Score)** = 2 * 精确率 * 召回率 / (精确率 + 召回率)。精确率与召回率的调和平均数。当两者重要性相当且无明显主次时，用于平衡二者。
- **AUC-ROC**：受试者工作特征曲线 (Receiver Operating Characteristic Curve) 下的面积。绘制不同分类阈值下的真阳性率与假阳性率关系。AUC = 0.5 表示随机猜测，AUC = 1.0 表示完美区分。该指标与阈值无关：它衡量的是模型将正类样本排在负类样本之前的能力，与你选择的具体截断值无关。

### 回归评估指标 (Regression Metrics)

- **MSE (均方误差, Mean Squared Error)** = mean((y_true - y_pred)^2)。对较大误差进行平方惩罚。对异常值敏感。
- **RMSE (均方根误差, Root Mean Squared Error)** = sqrt(MSE)。与目标变量单位相同。比 MSE 更易于解释。
- **MAE (平均绝对误差, Mean Absolute Error)** = mean(|y_true - y_pred|)。对所有误差进行线性处理。比 MSE 对异常值更具鲁棒性。
- **R² (决定系数, R-squared)** = 1 - SS_res / SS_tot，其中 SS_res = sum((y_true - y_pred)^2)，SS_tot = sum((y_true - y_mean)^2)。模型所解释的方差比例。R² = 1.0 表示完美拟合。R² = 0.0 表示模型效果等同于始终预测均值。若模型表现比预测均值还差，R² 可能为负值。

### 学习曲线 (Learning Curves)

绘制训练得分与验证得分随训练集大小变化的曲线：

- **高偏差/欠拟合 (High Bias / Underfitting)**：两条曲线均收敛于较低得分。增加更多数据无济于事，你需要使用更复杂的模型。
- **高方差/过拟合 (High Variance / Overfitting)**：训练得分很高，但验证得分明显较低，两者之间存在较大差距。增加更多数据通常会有所帮助。

### 验证曲线 (Validation Curves)

绘制训练得分与验证得分随某一超参数变化的曲线：

- 在低复杂度时：两项得分均较低（欠拟合）
- 在合适复杂度时：两项得分均较高且彼此接近
- 在高复杂度时：训练得分保持高位，但验证得分下降（过拟合）

验证得分达到峰值时所对应的超参数值即为最优值。

### 常见评估误区 (Common Evaluation Mistakes)

- **数据泄露 (Data Leakage)**：测试集的信息泄露到了训练过程中。例如：在划分数据集前使用完整数据拟合缩放器 (Scaler)、在时间序列预测中混入未来数据、使用了由目标变量衍生出的特征。务必先划分数据集，再进行预处理。
- **类别不平衡 (Class Imbalance)**：99% 的交易为正常交易，1% 为欺诈交易。一个始终预测“正常”的模型也能获得 99% 的准确率。此时应改用精确率、召回率、F1 分数或 AUC-ROC。
- **指标选择错误**：在应优化召回率（如医疗诊断）时却优化准确率，或在数据存在严重异常值时优化 RMSE（此时应改用 MAE）。
- **未使用分层划分**：对于不平衡数据，随机划分可能导致验证集中少数类样本极少，从而产生不稳定的评估结果。
- **频繁测试**：每次查看测试集性能并据此调整模型，都会导致模型对测试集过拟合。测试集仅限使用一次。

## 构建

### 步骤 1：训练集/验证集/测试集划分 (Train/Validation/Test Split)

import random
import math


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test

### 步骤 2：K折交叉验证 (K-fold Cross-Validation) 与分层K折交叉验证 (Stratified K-fold Cross-Validation)

def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))

    return folds


def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]

    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)

    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)

    return scores

### 步骤 3：混淆矩阵 (Confusion Matrix) 与分类评估指标 (Classification Metrics)

def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height

    return area

### 步骤 4：回归评估指标 (Regression Metrics)

def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot

### 步骤 5：学习曲线 (Learning Curves)

def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores

### 步骤 6：用于测试的简单分类器及完整演示 (A Simple Classifier for Testing, Plus the Full Demo)

class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    def __init__(self, lr=0.001, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values], mean, std


def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


if __name__ == "__main__":
    X_clf, y_clf = make_classification_data(300)

    print("=== Train/Validation/Test Split ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train class distribution: {sum(y_train)}/{len(y_train)} positive")
    print(f"  Val class distribution: {sum(y_val)}/{len(y_val)} positive")

    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)

    print("\n=== Classification Metrics ===")
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy:  {accuracy(y_test, y_pred):.4f}")
    print(f"  Precision: {precision(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")

    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC:   {auc:.4f}")

    print("\n=== K-Fold Cross-Validation (K=5) ===")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  Fold scores: {[round(s, 4) for s in cv_scores]}")
    print(f"  Mean: {mean_cv:.4f} (+/- {std_cv:.4f})")

    print("\n=== Stratified K-Fold Cross-Validation (K=5) ===")
    strat_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
        stratified=True,
    )
    strat_mean = sum(strat_scores) / len(strat_scores)
    strat_std = math.sqrt(sum((s - strat_mean) ** 2 for s in strat_scores) / len(strat_scores))
    print(f"  Fold scores: {[round(s, 4) for s in strat_scores]}")
    print(f"  Mean: {strat_mean:.4f} (+/- {strat_std:.4f})")

    print("\n=== Imbalanced Data: Why Accuracy Lies ===")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    positives = sum(y_imb)
    print(f"  Class distribution: {positives} positive, {len(y_imb) - positives} negative ({positives/len(y_imb)*100:.1f}% positive)")

    always_negative = [0] * len(y_imb)
    print(f"  Always-negative baseline:")
    print(f"    Accuracy:  {accuracy(y_imb, always_negative):.4f}")
    print(f"    Precision: {precision(y_imb, always_negative):.4f}")
    print(f"    Recall:    {recall(y_imb, always_negative):.4f}")
    print(f"    F1 Score:  {f1_score(y_imb, always_negative):.4f}")

    X_tr_i, y_tr_i, X_v_i, y_v_i, X_te_i, y_te_i = train_val_test_split(X_imb, y_imb)
    model_imb = SimpleLogistic(lr=0.5, epochs=500)
    model_imb.fit(X_tr_i, y_tr_i)
    y_pred_imb = [model_imb.predict(x) for x in X_te_i]
    print(f"\n  Trained model on imbalanced data:")
    print(f"    Accuracy:  {accuracy(y_te_i, y_pred_imb):.4f}")
    print(f"    Precision: {precision(y_te_i, y_pred_imb):.4f}")
    print(f"    Recall:    {recall(y_te_i, y_pred_imb):.4f}")
    print(f"    F1 Score:  {f1_score(y_te_i, y_pred_imb):.4f}")

    print("\n=== Regression Metrics ===")
    X_reg, y_reg = make_regression_data(200)

    col0 = [x[0] for x in X_reg]
    col1 = [x[1] for x in X_reg]
    col0_s, m0, s0 = standardize(col0)
    col1_s, m1, s1 = standardize(col1)
    X_reg_scaled = [[col0_s[i], col1_s[i]] for i in range(len(X_reg))]

    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg_scaled, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R-squared: {r_squared(y_te_r, y_pred_r):.4f}")

    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  Mean baseline:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R-squared: {r_squared(y_te_r, mean_baseline):.4f}")

    print("\n=== Learning Curve ===")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'Size':>6} {'Train':>8} {'Val':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")

    print("\n=== Statistical Model Comparison ===")
    model_a_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    model_b_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=500),
        k=5, metric_fn=accuracy,
    )
    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs))) if std_diff > 0 else 0.0
    print(f"  Model A (100 epochs) mean: {sum(model_a_scores)/len(model_a_scores):.4f}")
    print(f"  Model B (500 epochs) mean: {sum(model_b_scores)/len(model_b_scores):.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Paired t-statistic: {t_stat:.4f}")
    print(f"  (|t| > 2.78 for significance at p<0.05 with df=4)")


## 实践应用

借助 scikit-learn，模型评估（Model Evaluation）已无缝集成到标准工作流中：

from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")

从零实现的版本清晰地展示了交叉验证（Cross-Validation）的具体工作原理（没有黑魔法，仅是 for 循环与索引追踪）、各项评估指标（Evaluation Metrics）的计算方式（仅统计 TP/FP/TN/FN），以及分层抽样（Stratification）的重要性（在每次数据划分中保持类别比例）。而调用库函数的版本则额外提供了并行计算（Parallelism）、更丰富的评分选项以及与机器学习流水线（Pipeline）的集成。

## 交付产出

本章节将生成以下文件：
- `outputs/skill-evaluation.md` - 一份技能模块，涵盖分类（Classification）与回归（Regression）模型的评估策略

## 练习

1. 实现精确率-召回率曲线（Precision-Recall Curve）：绘制不同阈值下的精确率（Precision）与召回率（Recall）关系图。计算平均精确率（Average Precision，即 PR 曲线下面积）。在类别不平衡数据集（Imbalanced Dataset）上对比 PR 曲线与 ROC 曲线（Receiver Operating Characteristic Curve），并解释各自在何种场景下更具参考价值。
2. 构建嵌套交叉验证（Nested Cross-Validation）循环：外层循环用于评估模型性能，内层循环用于调优超参数（Hyperparameters）。利用该方法在不发生验证数据泄露（Data Leakage）的前提下，公平地对比两个模型。
3. 实现用于模型对比的置换检验（Permutation Test）：打乱标签（Labels）后重新训练模型并测量性能。重复此过程 100 次以构建零分布（Null Distribution）。计算观测到的模型性能相对于该分布的 p 值（p-value）。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 过拟合 (Overfitting) | “记住了训练数据” | 模型捕捉到了训练数据中的噪声，导致在训练集上表现优异，但在未见数据上表现不佳 |
| 交叉验证 (Cross-validation) | “在不同的子集上进行测试” | 系统地轮换用于验证的数据子集，并对所有轮换结果进行平均 |
| 精确率 (Precision) | “预测为正类的样本中有多少是正确的” | TP / (TP + FP)：在所有预测为正类的样本中，实际为正类的比例 |
| 召回率 (Recall) | “我们找到了多少实际的正类样本” | TP / (TP + FN)：在所有实际为正类的样本中，被正确识别的比例 |
| ROC曲线下面积 (AUC-ROC) | “模型区分不同类别的能力有多强” | 真正率 (TPR) 与假正率 (FPR) 曲线在所有阈值下的面积，取值范围为 0.5（随机猜测）至 1.0（完美分类） |
| 决定系数 (R-squared) | “解释了多少方差” | 1 - (残差平方和 / 总平方和)：模型所解释的目标变量方差占比 |
| 数据泄露 (Data Leakage) | “模型作弊了” | 训练时使用了预测阶段无法获取的信息，从而导致评估结果虚高 |
| 学习曲线 (Learning Curve) | “性能如何随数据量增加而变化” | 训练得分与验证得分随训练集规模变化的曲线图，用于诊断欠拟合或过拟合 |
| 分层划分 (Stratified Split) | “保持类别比例平衡” | 划分数据时，确保每个子集中各类别的比例与原始数据集完全一致 |

## 延伸阅读

- [scikit-learn 模型选择指南](https://scikit-learn.org/stable/model_selection.html) - 关于交叉验证 (Cross-validation)、评估指标 (Metrics) 和超参数调优 (Hyperparameter Tuning) 的全面参考资料
- [超越准确率：精确率与召回率（Google 机器学习速成课程）](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) - 配有交互式示例的清晰讲解
- [交叉验证方法综述（Arlot & Celisse, 2010）](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) - 深入探讨不同交叉验证 (CV) 策略的适用场景及其原理
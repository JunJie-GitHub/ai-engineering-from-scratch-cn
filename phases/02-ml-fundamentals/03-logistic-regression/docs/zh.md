# 逻辑回归 (Logistic Regression)

> 逻辑回归将直线“弯曲”成 S 形曲线，从而以概率形式回答“是”或“否”的问题。

**类型：** 实战构建
**编程语言：** Python
**前置知识：** 第二阶段 第1-2课（什么是机器学习、线性回归）
**预计时长：** 约 90 分钟

## 学习目标

- 使用 S 型函数（Sigmoid function）与二元交叉熵损失（Binary cross-entropy loss）从零实现逻辑回归
- 计算并解读二分类任务中的精确率（Precision）、召回率（Recall）、F1 分数（F1 score）以及混淆矩阵（Confusion matrix）
- 解释均方误差（Mean Squared Error, MSE）为何不适用于分类任务，以及二元交叉熵为何能产生凸代价曲面（Convex cost surface）
- 构建用于多分类任务的 Softmax 回归（Softmax regression）模型，并评估阈值调整（Threshold tuning）中的权衡取舍

## 问题描述

假设你想根据肿瘤的大小来预测它是恶性还是良性。你尝试使用线性回归（Linear regression），但它输出的数值可能是 0.3、1.7 或 -0.5。这些数字代表什么？1.7 表示“极度恶性”吗？-0.5 表示“极度良性”吗？线性回归的输出是无界的。而分类任务需要的是介于 0 和 1 之间的有界概率，以及一个明确的决策：是或否。

逻辑回归正是为了解决这一问题而生。它采用相同的线性组合（wx + b），并将其输入 S 型函数（Sigmoid function），该函数会将任意数值压缩至 (0, 1) 区间内。输出结果即为概率。你只需设定一个阈值（Threshold，通常为 0.5），即可做出分类决策。

这是实践中应用最广泛的算法之一。尽管名字中带有“回归”，但逻辑回归实际上是一种分类算法（Classification algorithm），而非回归算法。其名称来源于它所采用的逻辑函数（Logistic function，即 S 型函数）。

## 核心概念

### 为什么线性回归（Linear Regression）不适用于分类任务

假设我们根据学习时长来预测考试是否通过（1/0）。线性回归会尝试用一条直线来拟合数据：

hours:  1   2   3   4   5   6   7   8   9   10
actual: 0   0   0   0   1   1   1   1   1   1

线性拟合可能会在第1小时预测出-0.2，在第10小时预测出1.3。这些值并不是概率，它们会超出0到1的范围。更糟糕的是，单个异常值（例如某个学习了50小时的人）会拉扯整条拟合直线，从而改变对所有样本的预测结果。

分类任务需要一个具备以下特性的函数：
- 输出值介于0和1之间（表示概率）
- 能够形成清晰的过渡区域（即决策边界）
- 不会受到远离边界的异常值的干扰

### Sigmoid 函数（Sigmoid Function）

Sigmoid 函数恰好满足这些要求：

sigmoid(z) = 1 / (1 + e^(-z))

特性：
- 当 z 为较大的正数时，sigmoid(z) 趋近于 1
- 当 z 为较大的负数时，sigmoid(z) 趋近于 0
- 当 z = 0 时，sigmoid(z) = 0.5
- 输出值始终介于 0 和 1 之间
- 函数处处平滑且可导

其导数具有非常简洁的形式：sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))。这使得梯度计算非常高效。

### 逻辑回归（Logistic Regression）= 线性模型 + Sigmoid

该模型首先计算 z = wx + b（与线性回归相同），然后应用 Sigmoid 函数：

flowchart LR
    X[Input features x] --> L["Linear: z = wx + b"]
    L --> S["Sigmoid: p = 1/(1+e^-z)"]
    S --> D{"p >= 0.5?"}
    D -->|Yes| P[Predict 1]
    D -->|No| N[Predict 0]

输出值 p 被解释为 P(y=1 | x)，即输入样本属于类别 1 的概率。决策边界位于 wx + b = 0 处，此时 Sigmoid 的输出恰好为 0.5。

### 二元交叉熵损失（Binary Cross-Entropy Loss）

逻辑回归不能使用均方误差（MSE）。将 MSE 与 Sigmoid 结合会产生非凸的损失曲面，并存在多个局部极小值。因此，应改用二元交叉熵（对数损失，Log Loss）：

Loss = -(1/n) * sum(y * log(p) + (1-y) * log(1-p))

该损失函数有效的原因如下：
- 当 y=1 且 p 接近 1 时：log(1) = 0，损失接近 0（预测正确，代价低）
- 当 y=1 且 p 接近 0 时：log(0) 趋近于负无穷，损失极大（预测错误，代价高）
- 当 y=0 且 p 接近 0 时：log(1) = 0，损失接近 0（预测正确，代价低）
- 当 y=0 且 p 接近 1 时：log(0) 趋近于负无穷，损失极大（预测错误，代价高）

对于逻辑回归而言，该损失函数是凸函数（Convex Function），能够保证存在唯一的全局最小值。

### 逻辑回归的梯度下降（Gradient Descent）

结合 Sigmoid 的二元交叉熵梯度具有非常简洁的形式：

dL/dw = (1/n) * sum((p - y) * x)
dL/db = (1/n) * sum(p - y)

这些公式看起来与线性回归的梯度完全相同。区别在于，这里的 p = sigmoid(wx + b)，而非 p = wx + b。Sigmoid 引入了非线性，但梯度更新规则保持不变。

flowchart TD
    A[Initialize w=0, b=0] --> B[Forward pass: z = wx+b, p = sigmoid z]
    B --> C[Compute loss: binary cross-entropy]
    C --> D["Compute gradients: dw = (1/n) * sum((p-y)*x)"]
    D --> E[Update: w = w - lr*dw, b = b - lr*db]
    E --> F{Converged?}
    F -->|No| B
    F -->|Yes| G[Model trained]

### 决策边界（Decision Boundary）

对于二维输入（两个特征），决策边界是满足以下条件的直线：

w1*x1 + w2*x2 + b = 0

边界一侧的点被分类为 1，另一侧的点被分类为 0。逻辑回归始终生成线性的决策边界。如果需要曲线边界，可以添加多项式特征（Polynomial Features）或使用非线性模型。

### 使用 Softmax 进行多分类（Multi-Class Classification）

二元逻辑回归仅处理两个类别。对于 k 个类别，需使用 Softmax 函数：

softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)

每个类别都有其对应的权重向量。模型会为每个类别计算一个得分 z_i，随后 Softmax 将这些得分转换为总和为 1 的概率值。预测类别即为概率最高的那个类别。

损失函数变为分类交叉熵（Categorical Cross-Entropy）：

Loss = -(1/n) * sum(sum(y_k * log(p_k)))

其中，y_k 在真实类别处为 1，其余类别均为 0（独热编码，One-Hot Encoding）。

### 评估指标（Evaluation Metrics）

仅凭准确率（Accuracy）是不够的。在一个负样本占 95%、正样本占 5% 的数据集中，一个始终预测为负样本的模型也能达到 95% 的准确率，但毫无实际用处。

**混淆矩阵（Confusion Matrix）**：

| | 预测为正类 | 预测为负类 |
|---|---|---|
| 实际为正类 | 真阳性（TP） | 假阴性（FN） |
| 实际为负类 | 假阳性（FP） | 真阴性（TN） |

**精确率（Precision）**：在所有预测为正类的样本中，有多少是真正的正类？
Precision = TP / (TP + FP)

**召回率（Recall）**（灵敏度）：在所有实际为正类的样本中，我们成功找出了多少？
Recall = TP / (TP + FN)

**F1 分数（F1 Score）**：精确率与召回率的调和平均数。用于平衡这两项指标。
F1 = 2 * (Precision * Recall) / (Precision + Recall)

适用场景：
- **精确率**：当假阳性代价较高时（例如垃圾邮件过滤，你不希望误拦正常邮件）
- **召回率**：当假阴性代价较高时（例如癌症筛查，你不希望漏诊肿瘤）
- **F1 分数**：当你需要一个综合平衡的单一指标时

## 构建

### 步骤 1：Sigmoid 函数 (Sigmoid function) 与数据生成

import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")

### 步骤 2：从零实现逻辑回归 (Logistic Regression)

class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("\n=== Training Logistic Regression ===")
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
print(f"Bias: {model.bias:.4f}")

### 步骤 3：从零实现混淆矩阵 (Confusion Matrix) 与评估指标 (Metrics)

class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


y_pred_test = [model.predict(x) for x in X_test]
print("\n=== Classification Report (Test Set) ===")
metrics = ClassificationMetrics(y_test, y_pred_test)
metrics.print_report()

### 步骤 4：决策边界 (Decision Boundary) 分析

print("\n=== Decision Boundary ===")
w1, w2 = model.weights
b = model.bias
print(f"Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Solved for x2:     x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

print("\nSample predictions near the boundary:")
test_points = [
    [3.0, 3.0],
    [3.5, 3.5],
    [4.0, 4.0],
    [2.5, 2.5],
    [5.0, 5.0],
]
for point in test_points:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f"  [{point[0]}, {point[1]}] -> prob={prob:.4f}, class={pred}")

### 步骤 5：基于 Softmax 的多分类 (Multi-class Classification)

class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0
            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


random.seed(42)
X_3class = []
y_3class = []

centers = [(1, 1), (5, 1), (3, 5)]
for label, (cx, cy) in enumerate(centers):
    for _ in range(50):
        X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
        y_3class.append(label)

combined = list(zip(X_3class, y_3class))
random.shuffle(combined)
X_3class, y_3class = zip(*combined)
X_3class = list(X_3class)
y_3class = list(y_3class)

split_3 = int(0.8 * len(X_3class))
X_train_3 = X_3class[:split_3]
y_train_3 = y_3class[:split_3]
X_test_3 = X_3class[split_3:]
y_test_3 = y_3class[split_3:]

print("\n=== Multi-class Softmax Regression (3 classes) ===")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

print("\nSample predictions:")
for i in range(5):
    probs = softmax_model.predict_proba(X_test_3[i])
    pred = softmax_model.predict(X_test_3[i])
    print(f"  True: {y_test_3[i]}, Predicted: {pred}, Probs: [{', '.join(f'{p:.3f}' for p in probs)}]")

### 步骤 6：阈值 (Threshold) 调优

print("\n=== Threshold Tuning ===")
print("Default threshold: 0.5. Adjusting the threshold trades precision for recall.\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in thresholds:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")


## 实践应用

现在，我们使用 scikit-learn 来实现相同的功能。

from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_0 = np.random.randn(100, 2) + [2, 2]
X_1 = np.random.randn(100, 2) + [5, 5]
X_sk = np.vstack([X_0, X_1])
y_sk = np.array([0] * 100 + [1] * 100)

X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = SklearnLR()
lr.fit(X_tr_sc, y_tr)
y_pred = lr.predict(X_te_sc)

print("=== Scikit-learn Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_te, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")

你从零开始编写的实现能够生成相同的决策边界（decision boundary）与评估指标（metrics）。Scikit-learn 在此基础上增加了求解器选项（solver options，如 liblinear、lbfgs、saga）、自动正则化（regularization）、多分类策略（multi-class strategies，如一对多（one-vs-rest）和多项式（multinomial）），以及数值稳定性优化（numerical stability optimizations）。

## 交付内容

本章节将生成以下文件：
- `code/logistic_regression.py` - 从零实现的逻辑回归（logistic regression）代码及评估指标计算

## 练习

1. 生成一个线性不可分（linearly separable）的数据集（例如两个同心圆）。训练逻辑回归模型并观察其失效情况。随后添加多项式特征（polynomial features，如 x1^2、x2^2、x1*x2）并重新训练。证明准确率（accuracy）得到了提升。
2. 为三分类 Softmax 模型实现一个多分类混淆矩阵（confusion matrix）。计算每个类别的精确率（precision）和召回率（recall）。哪个类别最难分类？
3. 从零开始构建 ROC 曲线（ROC curve）。针对 0 到 1 之间的 100 个阈值（threshold），计算真阳性率（true positive rate）和假阳性率（false positive rate）。使用梯形法则（trapezoidal rule）计算曲线下面积（AUC, area under the curve）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 逻辑回归 (Logistic Regression) | “用于分类的回归” | 线性模型后接 Sigmoid 函数，用于输出类别概率 |
| Sigmoid 函数 (Sigmoid Function) | “S 形曲线” | 将任意实数映射到 (0, 1) 区间的函数，公式为 1/(1+e^(-z)) |
| 二元交叉熵 (Binary Cross-Entropy) | “对数损失” | 损失函数 -[y*log(p) + (1-y)*log(1-p)]，会对高置信度的错误预测施加严厉惩罚 |
| 决策边界 (Decision Boundary) | “分界线” | 模型输出概率等于 0.5 的超平面（或曲面），用于分隔预测类别 |
| Softmax 函数 (Softmax) | “多分类 Sigmoid” | 将得分向量转换为总和为 1 的概率分布的函数 |
| 精确率 (Precision) | “选中的有多少是相关的” | TP / (TP + FP)，即预测为正的样本中实际为正的比例 |
| 召回率 (Recall) | “相关的有多少被选中” | TP / (TP + FN)，即实际为正的样本中被模型正确识别的比例 |
| F1 分数 (F1 Score) | “平衡准确率” | 精确率与召回率的调和平均数：2*P*R / (P+R) |
| 混淆矩阵 (Confusion Matrix) | “错误细分” | 展示各类别组合下 TP、TN、FP、FN 数量的表格 |
| 阈值 (Threshold) | “截断值” | 模型预测为类别 1 的概率临界值（默认 0.5，可调节） |
| 独热编码 (One-Hot Encoding) | “类别的二值列” | 将类别 k 表示为仅在位置 k 处为 1、其余位置为 0 的向量 |
| 分类交叉熵 (Categorical Cross-Entropy) | “多分类对数损失” | 二元交叉熵在 k 个类别上的扩展，使用独热编码标签进行计算 |
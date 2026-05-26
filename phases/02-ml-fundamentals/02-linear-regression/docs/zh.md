# 线性回归 (Linear Regression)

> 线性回归旨在为你的数据拟合出一条最佳直线。它是机器学习领域的“Hello World”。

**类型：** 构建
**语言：** Python
**前置知识：** 第一阶段（线性代数、微积分、优化），第二阶段第1课
**时长：** 约90分钟

## 学习目标

- 推导均方误差 (Mean Squared Error) 的梯度下降 (Gradient Descent) 更新规则，并从头实现线性回归
- 从计算复杂度和适用场景的角度，对比梯度下降与正规方程 (Normal Equation)
- 构建包含特征标准化 (Feature Standardization) 的多元线性回归 (Multiple Linear Regression) 模型，并解释学习到的权重
- 解释岭回归 (Ridge Regression)（即 L2 正则化 (L2 Regularization)）如何通过惩罚较大权重来防止过拟合 (Overfitting)

## 问题背景

你手头有一组数据：房屋面积及其对应的售价。你的目标是根据新房的面积来预测其价格。虽然你可以在散点图上凭肉眼估算，但你需要一个确切的公式。你需要一条能最佳拟合数据的直线，以便代入任意面积值即可得到价格预测。

线性回归正是为你提供这条直线的算法。更重要的是，它完整引入了机器学习 (Machine Learning) 的训练循环：定义模型、定义代价函数 (Cost Function)、优化参数。每一种机器学习算法都遵循这一相同模式。在此通过最简单的案例掌握它，你将在后续的任何地方认出它。

这绝不仅仅适用于简单问题。线性回归被广泛应用于生产系统中，用于需求预测、A/B 测试分析、金融建模，并作为所有回归任务的基线模型 (Baseline)。

## 核心概念

### 模型

线性回归 (Linear Regression) 假设输入 (x) 与输出 (y) 之间存在线性关系：

y = wx + b

- `w`（权重/斜率）：x 每增加 1 时 y 的变化量
- `b`（偏置/截距）：当 x = 0 时 y 的值

对于多个输入（特征），该公式可扩展为：

y = w1*x1 + w2*x2 + ... + wn*xn + b

或者写成向量形式：`y = w^T * x + b`

目标：找到 `w` 和 `b` 的值，使得在所有训练样本上，预测的 y 尽可能接近真实的 y。

### 代价函数 (Cost Function)（均方误差）

如何衡量“尽可能接近”？你需要一个单一的数值来量化预测的误差。最常用的选择是均方误差 (Mean Squared Error, MSE)：

MSE = (1/n) * sum((y_predicted - y_actual)^2)

为什么要平方？有两个原因。首先，它对大误差的惩罚远大于小误差（误差为 10 的代价是误差为 1 的 100 倍，而非 10 倍）。其次，平方函数处处平滑且可导，这使得优化过程更加直接。

代价函数会形成一个曲面。对于单个权重 `w` 和偏置 `b`，MSE 曲面看起来像一个碗（凸抛物面）。碗底即为 MSE 最小化的位置。训练的过程就是寻找这个碗底。

### 梯度下降 (Gradient Descent)

梯度下降通过沿下坡方向逐步迭代来寻找碗底。

flowchart TD
    A[Initialize w and b randomly] --> B[Compute predictions: y_hat = wx + b]
    B --> C[Compute cost: MSE]
    C --> D[Compute gradients: dMSE/dw, dMSE/db]
    D --> E[Update parameters]
    E --> F{Cost low enough?}
    F -->|No| B
    F -->|Yes| G[Done: optimal w and b found]

梯度提供了两个关键信息：每个参数应该朝哪个方向调整，以及调整的幅度有多大。

对于 `y_hat = wx + b` 的 MSE：

dMSE/dw = (2/n) * sum((y_hat - y) * x)
dMSE/db = (2/n) * sum(y_hat - y)

参数更新规则：

w = w - learning_rate * dMSE/dw
b = b - learning_rate * dMSE/db

学习率 (Learning Rate) 控制步长。过大：会越过最小值导致发散。过小：训练将极其缓慢。典型的初始值通常设为 0.01、0.001 或 0.0001。

### 正规方程 (Normal Equation)（闭式解）

针对线性回归，存在一个直接公式，无需迭代即可求出最优权重：

w = (X^T * X)^(-1) * X^T * y

该公式通过矩阵求逆一步解出 `w`。它在小型数据集上表现完美。但对于大型数据集（数百万行或数千个特征），更推荐使用梯度下降，因为矩阵求逆的时间复杂度随特征数量呈 O(n^3) 增长。

### 多元线性回归 (Multiple Linear Regression)

当存在多个特征时，模型变为：

y = w1*x1 + w2*x2 + ... + wn*xn + b

其余机制完全相同：MSE 仍作为代价函数，梯度下降同时更新所有权重。唯一的区别在于，此时拟合的是一个超平面 (Hyperplane) 而非一条直线。

特征缩放 (Feature Scaling) 在此至关重要。如果一个特征的取值范围是 0 到 1，而另一个是 0 到 1,000,000，梯度下降将难以收敛，因为代价曲面会被严重拉长。在训练前，应对特征进行标准化处理（减去均值，除以标准差）。

### 多项式回归 (Polynomial Regression)

如果变量间的关系不是线性的怎么办？你仍然可以通过构造多项式特征来使用线性回归：

y = w1*x + w2*x^2 + w3*x^3 + b

这依然属于“线性”回归，因为模型关于权重 (`w1`, `w2`, `w3`) 是线性的。你仅仅是使用了 `x` 的非线性特征。

高阶多项式能够拟合更复杂的曲线，但存在过拟合 (Overfitting) 的风险。一个 10 阶多项式会穿过包含 10 个数据点的样本集中的每一个点，但在新数据上的预测表现会很差。

### R 平方分数 (R-Squared Score)

MSE 能告诉你预测误差有多大，但该数值依赖于 y 的量纲。R 平方 (R-Squared, R^2) 则提供了一个与量纲无关的评估指标：

R^2 = 1 - (sum of squared residuals) / (sum of squared deviations from mean)
    = 1 - SS_res / SS_tot

- R^2 = 1.0：完美预测
- R^2 = 0.0：模型的预测效果不比直接预测均值更好
- R^2 < 0.0：模型的预测效果比直接预测均值还要差

### 正则化预览 (Regularization)（岭回归）

当特征数量较多时，模型可能会通过赋予权重过大的值而导致过拟合。岭回归 (Ridge Regression)（L2 正则化 (L2 Regularization)）通过引入惩罚项来解决此问题：

Cost = MSE + lambda * sum(w_i^2)

该惩罚项会抑制过大的权重。超参数 (Hyperparameter) `lambda` 控制着权衡：`lambda` 越大，权重越小，正则化程度越高。这将在后续课程中深入讲解。目前只需了解它的存在及其作用即可。

## 构建

### 步骤 1：生成样本数据 (Sample Data)

import random
import math

random.seed(42)

TRUE_W = 3.0
TRUE_B = 7.0
N_SAMPLES = 100

X = [random.uniform(0, 10) for _ in range(N_SAMPLES)]
y = [TRUE_W * x + TRUE_B + random.gauss(0, 2.0) for x in X]

print(f"Generated {N_SAMPLES} samples")
print(f"True relationship: y = {TRUE_W}x + {TRUE_B} (+ noise)")
print(f"First 5 points: {[(round(X[i], 2), round(y[i], 2)) for i in range(5)]}")

### 步骤 2：从零实现线性回归 (Linear Regression) 与梯度下降 (Gradient Descent)

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n
        return cost

    def compute_gradients(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        dw = (2 / n) * sum((pred - actual) * x for pred, actual, x in zip(predictions, y, X))
        db = (2 / n) * sum(pred - actual for pred, actual in zip(predictions, y))
        return dw, db

    def fit(self, X, y, epochs=1000, print_every=200):
        for epoch in range(epochs):
            dw, db = self.compute_gradients(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


print("=== Training Linear Regression (Gradient Descent) ===")
model = LinearRegression(learning_rate=0.005)
model.fit(X, y, epochs=1000, print_every=200)
print(f"\nLearned: y = {model.w:.4f}x + {model.b:.4f}")
print(f"True:    y = {TRUE_W}x + {TRUE_B}")
print(f"R-squared: {model.r_squared(X, y):.4f}")

### 步骤 3：正规方程 (Normal Equation)（闭式解 (Closed-Form Solution)）

class LinearRegressionNormal:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        n = len(X)
        x_mean = sum(X) / n
        y_mean = sum(y) / n
        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((X[i] - x_mean) ** 2 for i in range(n))
        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean
        return self

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


print("\n=== Normal Equation (Closed-Form) ===")
model_normal = LinearRegressionNormal()
model_normal.fit(X, y)
print(f"Learned: y = {model_normal.w:.4f}x + {model_normal.b:.4f}")
print(f"R-squared: {model_normal.r_squared(X, y):.4f}")

### 步骤 4：多元线性回归 (Multiple Linear Regression)

class MultipleLinearRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        return sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


random.seed(42)
N = 100
X_multi = []
y_multi = []
for _ in range(N):
    size = random.uniform(500, 3000)
    bedrooms = random.randint(1, 5)
    age = random.uniform(0, 50)
    price = 50 * size + 10000 * bedrooms - 1000 * age + 50000 + random.gauss(0, 20000)
    X_multi.append([size, bedrooms, age])
    y_multi.append(price)


def standardize(X):
    n_features = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(n_features)]
    stds = []
    for j in range(n_features):
        variance = sum((X[i][j] - means[j]) ** 2 for i in range(len(X))) / len(X)
        stds.append(variance ** 0.5)
    X_scaled = []
    for i in range(len(X)):
        row = [(X[i][j] - means[j]) / stds[j] if stds[j] > 0 else 0 for j in range(n_features)]
        X_scaled.append(row)
    return X_scaled, means, stds


y_mean_val = sum(y_multi) / len(y_multi)
y_std_val = (sum((yi - y_mean_val) ** 2 for yi in y_multi) / len(y_multi)) ** 0.5
y_scaled = [(yi - y_mean_val) / y_std_val for yi in y_multi]

X_scaled, x_means, x_stds = standardize(X_multi)

print("\n=== Multiple Linear Regression (3 features) ===")
print("Features: house size, bedrooms, age")
multi_model = MultipleLinearRegression(n_features=3, learning_rate=0.01)
multi_model.fit(X_scaled, y_scaled, epochs=1000, print_every=200)

print(f"\nWeights (standardized): {[round(w, 4) for w in multi_model.weights]}")
print(f"Bias (standardized): {multi_model.bias:.4f}")
print(f"R-squared: {multi_model.r_squared(X_scaled, y_scaled):.4f}")

### 步骤 5：多项式回归 (Polynomial Regression)

class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.01):
        self.degree = degree
        self.weights = [0.0] * degree
        self.bias = 0.0
        self.lr = learning_rate

    def make_features(self, X):
        return [[x ** (d + 1) for d in range(self.degree)] for x in X]

    def predict(self, X):
        features = self.make_features(X)
        return [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]

    def fit(self, X, y, epochs=1000, print_every=200):
        features = self.make_features(X)
        n = len(y)
        for epoch in range(epochs):
            predictions = [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(self.degree):
                grad = (2 / n) * sum(errors[i] * features[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost = sum(e ** 2 for e in errors) / n
                print(f"  Epoch {epoch:4d} | Cost: {cost:.6f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


random.seed(42)
X_poly = [x / 10.0 for x in range(0, 50)]
y_poly = [0.5 * x ** 2 - 2 * x + 3 + random.gauss(0, 1.0) for x in X_poly]

x_max = max(abs(x) for x in X_poly)
X_poly_norm = [x / x_max for x in X_poly]
y_poly_mean = sum(y_poly) / len(y_poly)
y_poly_std = (sum((yi - y_poly_mean) ** 2 for yi in y_poly) / len(y_poly)) ** 0.5
y_poly_norm = [(yi - y_poly_mean) / y_poly_std for yi in y_poly]

print("\n=== Polynomial Regression (degree 2 vs degree 5) ===")
print("True relationship: y = 0.5x^2 - 2x + 3")

print("\nDegree 2:")
poly2 = PolynomialRegression(degree=2, learning_rate=0.1)
poly2.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly2.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 5:")
poly5 = PolynomialRegression(degree=5, learning_rate=0.1)
poly5.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly5.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 2 fits the true curve well. Degree 5 fits training data slightly better")
print("but risks overfitting on new data.")

### 步骤 6：岭回归 (Ridge Regression)（L2 正则化 (L2 Regularization)）

class RidgeRegression:
    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            mse = sum(e ** 2 for e in errors) / n
            reg_term = self.alpha * sum(w ** 2 for w in self.weights)
            cost = mse + reg_term
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                grad += 2 * self.alpha * self.weights[j]
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | L2 penalty: {reg_term:.4f}")
        return self


print("\n=== Ridge Regression (L2 Regularization) ===")
print("Same data as multiple regression, with alpha=0.1")
ridge = RidgeRegression(n_features=3, learning_rate=0.01, alpha=0.1)
ridge.fit(X_scaled, y_scaled, epochs=1000, print_every=200)
print(f"\nRidge weights: {[round(w, 4) for w in ridge.weights]}")
print(f"Plain weights: {[round(w, 4) for w in multi_model.weights]}")
print("Ridge weights are smaller (shrunk toward zero) due to the L2 penalty.")


## 实际应用

现在使用 scikit-learn 实现相同的功能，这也是你在生产环境中实际会使用的工具。

from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

np.random.seed(42)
X_sk = np.random.uniform(0, 10, (100, 1))
y_sk = 3.0 * X_sk.squeeze() + 7.0 + np.random.normal(0, 2.0, 100)

X_train, X_test, y_train, y_test = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

lr = SklearnLR()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("=== Scikit-learn Linear Regression ===")
print(f"Coefficient (w): {lr.coef_[0]:.4f}")
print(f"Intercept (b): {lr.intercept_:.4f}")
print(f"R-squared (test): {r2_score(y_test, y_pred):.4f}")
print(f"MSE (test): {mean_squared_error(y_test, y_pred):.4f}")

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_sk = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

lr_poly = SklearnLR()
lr_poly.fit(X_poly_sk, y_train)
print(f"\nPolynomial degree 2 R-squared: {r2_score(y_test, lr_poly.predict(X_poly_test)):.4f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print(f"Ridge R-squared: {r2_score(y_test, ridge.predict(X_test_scaled)):.4f}")
print(f"Ridge coefficient: {ridge.coef_[0]:.4f}")

你从零开始实现的版本与 scikit-learn 会得出相同的结果。两者的区别在于：scikit-learn 能够妥善处理边界情况 (edge cases)、数值稳定性 (numerical stability) 以及性能优化。在生产环境中请使用现成的库；而使用从零实现的版本则是为了深入理解其底层原理。

## 部署发布

本章节将生成：
- `outputs/skill-regression.md` - 一项用于根据具体问题选择合适回归方法 (regression approach) 的技能指南

## 练习

1. 实现批量梯度下降 (batch gradient descent)、随机梯度下降 (stochastic gradient descent, SGD) 和小批量梯度下降 (mini-batch gradient descent)。在同一数据集上比较它们的收敛速度 (convergence speed)。哪种方法收敛最快？哪种方法的代价曲线 (cost curve) 最平滑？
2. 根据三次函数 (y = ax^3 + bx^2 + cx + d + noise) 生成数据。分别拟合 1 次、3 次和 10 次多项式 (polynomial)。比较训练集和测试集的 R^2 分数。在几次多项式时，过拟合 (overfitting) 现象会变得明显？
3. 实现 Lasso 回归 (Lasso regression)（L1 正则化 (L1 regularization)：惩罚项 = alpha * sum(|w_i|)）。在包含多特征的房屋数据集上进行训练。与岭回归 (Ridge regression) 对比，观察哪些权重会变为零。为什么 L1 正则化会产生稀疏解 (sparse solutions)，而 L2 不会？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 线性回归 (Linear Regression) | “在数据中画一条线” | 寻找权重 w 和偏置 b，使 wx+b 与实际 y 值之间的平方差之和最小化 |
| 代价函数 (Cost Function) | “模型有多差” | 将模型参数映射为单个数值的函数，用于衡量预测误差，优化过程旨在最小化该值 |
| 均方误差 (Mean Squared Error) | “误差平方的平均值” | (1/n) * sum of (predicted - actual)^2，对较大误差施加更重的惩罚 |
| 梯度下降 (Gradient Descent) | “沿下坡方向走” | 利用偏导数，沿降低代价函数的方向迭代调整参数 |
| 学习率 (Learning Rate) | “步长” | 控制每次梯度下降步骤中参数变化幅度的标量 |
| 正规方程 (Normal Equation) | “直接求解” | 闭式解 w = (X^T X)^-1 X^T y，无需迭代即可直接得出最优权重 |
| 决定系数 (R-squared) | “拟合效果有多好” | 模型所能解释的 y 的方差比例，取值范围为负无穷到 1.0 |
| 特征缩放 (Feature Scaling) | “让特征具有可比性” | 将特征转换到相近的数值范围（例如零均值、单位方差），以加快梯度下降的收敛速度 |
| 正则化 (Regularization) | “惩罚复杂度” | 在代价函数中添加一项以收缩权重，从而防止过拟合 |
| 岭回归 (Ridge Regression) | “L2 正则化” | 在均方误差基础上增加 lambda * sum(w_i^2) 惩罚项的线性回归 |
| 多项式回归 (Polynomial Regression) | “用线性数学拟合曲线” | 对多项式特征（x, x^2, x^3, ...）进行线性回归，其对权重而言仍是线性的 |
| 过拟合 (Overfitting) | “死记硬背训练数据” | 使用过于复杂的模型，导致其拟合了训练数据中的噪声，从而在新数据上表现不佳 |

## 扩展阅读

- [《统计学习导论》(An Introduction to Statistical Learning, ISLR)](https://www.statlearning.com/) -- 免费 PDF，第 3 章和第 6 章结合实用的 R 语言示例讲解了线性回归与正则化
- [《统计学习基础》(The Elements of Statistical Learning, ESL)](https://hastie.su.domains/ElemStatLearn/) -- 免费 PDF，ISLR 的数学进阶版，对岭回归和套索回归进行了更深入的探讨
- [斯坦福 CS229 线性回归讲义](https://cs229.stanford.edu/main_notes.pdf) -- Andrew Ng 的笔记，从基本原理推导了正规方程与梯度下降
- [scikit-learn LinearRegression 文档](https://scikit-learn.org/stable/modules/linear_model.html) -- 包含 LinearRegression、Ridge、Lasso 和 ElasticNet 的实用参考及代码示例
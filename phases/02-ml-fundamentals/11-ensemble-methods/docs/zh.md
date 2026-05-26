# 集成方法 (Ensemble Methods)

> 一组弱学习器 (Weak Learners) 若组合得当，便能成为强学习器 (Strong Learner)。这并非比喻，而是一条定理。

**类型：** 构建
**语言：** Python
**前置知识：** 第二阶段，第10课（偏差-方差权衡 (Bias-Variance Tradeoff)）
**时长：** 约120分钟

## 学习目标

- 从零实现 AdaBoost 和梯度提升 (Gradient Boosting)，并解释提升法 (Boosting) 如何逐步降低偏差
- 构建装袋法 (Bagging) 集成模型，并演示如何通过对去相关模型进行平均来降低方差，同时不增加偏差
- 比较 Bagging、Boosting 和堆叠法 (Stacking)，说明每种方法分别针对哪种误差分量
- 评估集成多样性 (Ensemble Diversity)，并解释为何随着独立弱学习器数量的增加，多数投票法 (Majority Voting) 的准确率会随之提升

## 问题背景

单棵决策树训练速度快且易于解释，但容易过拟合 (Overfitting)。单一线性模型在处理复杂决策边界时则容易欠拟合 (Underfitting)。你可以花费数天时间精心打磨完美的模型架构，也可以将一堆不完美的模型组合起来，从而得到优于其中任何一个单独模型的结果。

集成方法正是为此而生。它们是赢得表格数据类 Kaggle 竞赛最可靠的技术，支撑着大多数生产环境中的机器学习 (Machine Learning) 系统，并直观地展现了偏差-方差权衡的实际运作。Bagging 用于降低方差，Boosting 用于降低偏差，而 Stacking 则学习在不同输入下该信任哪些模型。

## 核心概念

### 集成方法 (Ensemble Methods) 为何有效

假设你有 N 个独立的分类器 (Classifiers)，每个的准确率 p > 0.5。多数投票 (Majority Vote) 的准确率为：

P(majority correct) = sum over k > N/2 of C(N,k) * p^k * (1-p)^(N-k)

对于 21 个准确率均为 60% 的分类器，多数投票的准确率约为 74%。当分类器数量增至 101 个时，准确率提升至 84%。当模型犯下不同的错误时，这些误差会相互抵消。

核心要求是**多样性 (Diversity)**。如果所有模型都犯相同的错误，组合它们将毫无益处。集成方法之所以有效，是因为它们通过以下方式生成多样化的模型：

- 不同的训练子集（Bagging）
- 不同的特征子集（Random Forests）
- 顺序误差修正（Boosting）
- 不同的模型族（Stacking）

### Bagging（Bootstrap Aggregating / 自助聚合）

Bagging 通过在训练数据的不同自助样本 (Bootstrap Sample) 上训练每个模型来创造多样性。

flowchart TD
    D[Training Data] --> B1[Bootstrap Sample 1]
    D --> B2[Bootstrap Sample 2]
    D --> B3[Bootstrap Sample 3]
    D --> BN[Bootstrap Sample N]

    B1 --> M1[Model 1]
    B2 --> M2[Model 2]
    B3 --> M3[Model 3]
    BN --> MN[Model N]

    M1 --> V[Average or Majority Vote]
    M2 --> V
    M3 --> V
    MN --> V

    V --> P[Final Prediction]

自助样本是从原始数据中有放回地抽取的，大小与原始数据相同。每个自助样本中大约包含 63.2% 的唯一原始样本。剩余的 36.8%（袋外样本 / Out-of-Bag Samples）则提供了一个免费的验证集。

Bagging 能在不显著增加偏差 (Bias) 的情况下降低方差 (Variance)。每棵单独的树都会对其自助样本过拟合 (Overfit)，但由于每棵树的过拟合模式不同，取平均值可以抵消这些噪声。

**随机森林 (Random Forests)** 是 Bagging 的一种变体，增加了一个关键机制：在每次节点分裂时，仅考虑特征的随机子集。这进一步增强了树之间的多样性。分类任务中候选特征的典型数量为 `sqrt(n_features)`，回归任务中为 `n_features / 3`。

### Boosting（Sequential Error Correction / 顺序误差修正）

Boosting 按顺序训练模型。每个新模型都会重点关注前序模型预测错误的样本。

flowchart LR
    D[Data with weights] --> M1[Model 1]
    M1 --> E1[Find errors]
    E1 --> W1[Increase weights on errors]
    W1 --> M2[Model 2]
    M2 --> E2[Find errors]
    E2 --> W2[Increase weights on errors]
    W2 --> M3[Model 3]
    M3 --> F[Weighted sum of all models]

Boosting 主要用于降低偏差。每个新模型都会修正当前集成模型的系统性误差。最终预测结果是所有模型的加权和，表现更好的模型会获得更高的权重。

权衡之处在于：如果迭代轮数过多，Boosting 可能会过拟合，因为它会持续拟合更难分类的样本，而其中一些样本可能只是噪声。

### AdaBoost

AdaBoost（Adaptive Boosting / 自适应提升）是首个具有实用价值的 Boosting 算法。它可以与任何基学习器 (Base Learner) 配合使用，通常采用决策树桩 (Decision Stumps，即深度为 1 的树)。

算法流程如下：

1. Initialize sample weights: w_i = 1/N for all i

2. For t = 1 to T:
   a. Train weak learner h_t on weighted data
   b. Compute weighted error:
      err_t = sum(w_i * I(h_t(x_i) != y_i)) / sum(w_i)
   c. Compute model weight:
      alpha_t = 0.5 * ln((1 - err_t) / err_t)
   d. Update sample weights:
      w_i = w_i * exp(-alpha_t * y_i * h_t(x_i))
   e. Normalize weights to sum to 1

3. Final prediction: H(x) = sign(sum(alpha_t * h_t(x)))

误差较低的模型会获得更高的 alpha 值。被错误分类的样本会获得更高的权重，从而使下一个模型能够重点关注它们。

### Gradient Boosting（梯度提升）

梯度提升 (Gradient Boosting) 将 Boosting 推广至任意损失函数 (Loss Function)。它不再重新加权样本，而是让每个新模型去拟合当前集成模型的残差 (Residuals，即损失函数的负梯度)。

1. Initialize: F_0(x) = argmin_c sum(L(y_i, c))

2. For t = 1 to T:
   a. Compute pseudo-residuals:
      r_i = -dL(y_i, F_{t-1}(x_i)) / dF_{t-1}(x_i)
   b. Fit a tree h_t to the residuals r_i
   c. Find optimal step size:
      gamma_t = argmin_gamma sum(L(y_i, F_{t-1}(x_i) + gamma * h_t(x_i)))
   d. Update:
      F_t(x) = F_{t-1}(x) + learning_rate * gamma_t * h_t(x)

3. Final prediction: F_T(x)

对于平方误差损失 (Squared Error Loss)，伪残差 (Pseudo-residuals) 就是实际的残差：`r_i = y_i - F_{t-1}(x_i)`。每棵树实际上都在拟合前序集成模型的误差。

学习率 (Learning Rate，也称收缩率 / Shrinkage) 控制着每棵树的贡献程度。较小的学习率需要更多的树，但泛化能力 (Generalization) 更好。典型取值范围为 0.01 到 0.3。

### XGBoost：为何它在表格数据 (Tabular Data) 中占据主导地位

XGBoost（eXtreme Gradient Boosting / 极端梯度提升）是经过工程优化的梯度提升算法，使其具备速度快、精度高且抗过拟合的特点：

- **正则化目标函数 (Regularized Objective)：** 对叶子节点权重施加 L1 和 L2 惩罚，防止单棵树过于自信（即过拟合）
- **二阶近似 (Second-order Approximation)：** 同时利用损失函数的一阶和二阶导数，从而做出更优的分裂决策
- **感知稀疏性的分裂 (Sparsity-aware Splits)：** 原生支持缺失值处理，通过在每次分裂时学习缺失数据的最佳走向来实现
- **列子采样 (Column Subsampling)：** 与随机森林类似，在每次分裂时对特征进行采样以增加多样性
- **加权分位数草图 (Weighted Quantile Sketch)：** 在分布式数据上高效寻找连续特征的分裂点
- **缓存感知块结构 (Cache-aware Block Structure)：** 针对 CPU 缓存行优化的内存布局

对于表格数据，XGBoost（及其后继者 LightGBM）的表现始终优于神经网络。这一现状在短期内不会改变。如果你的数据可以整理成行列分明的表格，请优先从梯度提升算法开始尝试。

### Stacking（Meta-Learning / 元学习）

Stacking 将多个基模型的预测结果作为特征，输入给一个元学习器 (Meta-Learner)。

flowchart TD
    D[Training Data] --> M1[Model 1: Random Forest]
    D --> M2[Model 2: SVM]
    D --> M3[Model 3: Logistic Regression]

    M1 --> P1[Predictions 1]
    M2 --> P2[Predictions 2]
    M3 --> P3[Predictions 3]

    P1 --> META[Meta-Learner]
    P2 --> META
    P3 --> META

    META --> F[Final Prediction]

元学习器会学习在不同输入下应该信任哪个基模型。如果随机森林在某些数据区域表现更好，而支持向量机 (SVM) 在另一些区域表现更好，元学习器将学会据此进行路由分配。

为避免数据泄露 (Data Leakage)，基模型的预测结果必须通过在训练集上执行交叉验证 (Cross-Validation) 来生成。绝不能在相同的数据上同时训练基模型并生成元特征。

### Voting（投票法）

最简单的集成方法。直接组合预测结果即可。

- **硬投票 (Hard Voting)：** 对类别标签进行多数投票。
- **软投票 (Soft Voting)：** 对预测概率取平均值，选择平均概率最高的类别。通常效果更好，因为它利用了模型的置信度信息。

## 构建

### 步骤 1：决策树桩（Decision Stump，基学习器 Base Learner）

`code/ensembles.py` 中的代码完全从零开始实现。我们首先从决策树桩（Decision Stump）入手：这是一种仅包含单次节点划分的树模型。

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float("inf")

        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for thresh in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    pred[polarity * X[:, f] < polarity * thresh] = -1
                    error = np.sum(weights[pred != y])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = f
                        self.threshold = thresh
                        self.polarity = polarity

    def predict(self, X):
        n = X.shape[0]
        pred = np.ones(n)
        idx = self.polarity * X[:, self.feature_idx] < self.polarity * self.threshold
        pred[idx] = -1
        return pred

### 步骤 2：从零实现 AdaBoost

class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.full(n, 1 / n)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            pred = stump.predict(X)

            err = np.sum(weights[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)
            weights *= np.exp(-alpha * y * pred)
            weights /= weights.sum()

            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        total = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.sign(total)

### 步骤 3：从零实现梯度提升（Gradient Boosting）

class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        current_pred = np.full(len(y), self.initial_pred)

        for _ in range(self.n_estimators):
            residuals = y - current_pred
            tree = SimpleRegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            current_pred += self.lr * update
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred

### 步骤 4：与 sklearn 进行对比

该代码验证了我们手动实现的模型与 `sklearn` 中的 `AdaBoostClassifier` 和 `GradientBoostingClassifier` 能够达到相近的准确率，并对所有方法进行了横向对比。

## 使用方法

### 何时使用每种方法

| 方法 | 缓解 | 适用场景 | 注意事项 |
|--------|---------|----------|---------------|
| Bagging / 随机森林 (Random Forest) | 方差 (Variance) | 含噪声的数据、特征数量多 | 无法改善偏差 (Bias) |
| AdaBoost | 偏差 | 干净的数据、简单的基学习器 (Base Learner) | 对异常值和噪声敏感 |
| 梯度提升 (Gradient Boosting) | 偏差 | 表格型数据 (Tabular Data)、机器学习竞赛 | 训练速度慢，若不进行调参容易过拟合 (Overfitting) |
| XGBoost / LightGBM | 两者 | 生产环境的表格型机器学习任务 | 超参数 (Hyperparameter) 众多 |
| 堆叠 (Stacking) | 两者 | 榨取最后 1-2% 的准确率提升 | 结构复杂，元学习器 (Meta-learner) 存在过拟合风险 |
| 投票法 (Voting) | 方差 | 快速组合多样化模型 | 仅在模型具有多样性时才有效 |

### 表格数据的生产环境技术栈

对于大多数表格型预测问题，建议按以下顺序尝试：

1. 使用默认参数的 **LightGBM 或 XGBoost**
2. 调整 n_estimators, learning_rate, max_depth, min_child_weight
3. 如果需要最后 0.5% 的性能提升，使用 3-5 个多样化模型构建堆叠集成 (Stacking Ensemble)
4. 全程使用交叉验证 (Cross-Validation)

尽管研究不断，但在表格数据上，神经网络 (Neural Network) 的表现几乎总是逊于梯度提升。TabNet、NODE 及类似架构偶尔能与之持平，但极少能超越经过精细调优的 XGBoost。

## 交付成果

本课时将生成 `outputs/prompt-ensemble-selector.md` —— 一个提示词 (Prompt)，可帮助你为给定数据集选择合适的集成方法 (Ensemble Method)。你只需描述你的数据（规模、特征类型、噪声水平、类别平衡情况）以及你要解决的问题。该提示词会引导你完成决策清单，推荐合适的方法，建议初始超参数，并提醒该方法常见的陷阱。同时还会生成包含完整选择指南的 `outputs/skill-ensemble-builder.md`。

## 练习

1. 修改 AdaBoost 的实现代码，以记录每一轮迭代后的训练准确率。绘制准确率与基学习器数量的关系图。模型在何时收敛？

2. 从零实现随机森林：在回归树的基础上加入随机特征子采样。使用 `max_features=sqrt(n_features)` 训练 100 棵树并对预测结果取平均。将其方差缩减效果与单棵树进行对比。

3. 在梯度提升的实现中加入早停 (Early Stopping) 机制：记录每轮迭代后的验证集损失，若连续 10 轮未改善则停止训练。模型实际需要多少棵树？

4. 构建一个堆叠集成模型，包含三个基模型（逻辑回归 (Logistic Regression)、决策树 (Decision Tree)、K近邻 (K-Nearest Neighbors)）和一个逻辑回归元学习器。使用 5 折交叉验证生成元特征。将其性能与各基模型单独运行时的表现进行对比。

5. 使用默认参数在同一数据集上运行 XGBoost。将其准确率与你从零实现的梯度提升进行对比。记录两者的运行时间。速度差异有多大？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 装袋法 (Bagging) | “在随机子集上训练” | 自助聚合 (Bootstrap Aggregating)：在自助采样 (bootstrap samples) 样本上训练模型，通过对预测结果取平均来降低方差 |
| 提升法 (Boosting) | “专注于难分样本” | 顺序训练模型，每个新模型负责修正当前集成模型的误差，从而降低偏差 |
| 自适应提升 (AdaBoost) | “对数据重新加权” | 通过更新样本权重实现提升；被误分类的样本在下一轮学习器中会获得更高的权重 |
| 梯度提升 (Gradient Boosting) | “拟合残差” | 通过将每个新模型拟合到损失函数的负梯度来实现提升 |
| 极端梯度提升 (XGBoost) | “Kaggle 竞赛利器” | 融合了正则化、二阶优化以及系统级加速技巧的梯度提升算法 |
| 堆叠法 (Stacking) | “模型之上叠加模型” | 将基模型 (base models) 的预测结果作为元学习器 (meta-learner) 的输入特征 |
| 随机森林 (Random Forest) | “大量随机化的决策树” | 基于决策树的装袋法，在每次节点分裂时引入随机特征子采样以增加模型多样性 |
| 集成多样性 (Ensemble Diversity) | “犯不同的错误” | 各模型的误差必须互不相关，集成模型的性能才能超越单个模型 |
| 袋外误差 (Out-of-Bag Error) | “免费的验证集” | 未被抽入自助采样集的样本（约 36.8%）可直接作为验证集，无需额外划分保留集 (holdout set) |

## 进一步阅读

- [Schapire & Freund: Boosting: Foundations and Algorithms](https://mitpress.mit.edu/9780262526036/) —— AdaBoost 作者撰写的专著
- [Friedman: Greedy Function Approximation: A Gradient Boosting Machine (2001)](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) —— 梯度提升算法的原始论文
- [Chen & Guestrin: XGBoost (2016)](https://arxiv.org/abs/1603.02754) —— XGBoost 算法的原始论文
- [Wolpert: Stacked Generalization (1992)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) —— 堆叠泛化 (Stacking) 的原始论文
- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) —— 实用参考指南
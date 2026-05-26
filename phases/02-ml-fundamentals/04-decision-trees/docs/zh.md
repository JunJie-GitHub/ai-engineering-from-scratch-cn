# 决策树 (Decision Trees) 与随机森林 (Random Forests)

> 决策树本质上只是一张流程图。但由它们组成的“森林”却是机器学习 (Machine Learning) 中最强大的工具之一。

**类型：** 构建
**语言：** Python
**前置知识：** 第一阶段（第09课 信息论 (Information Theory)、第06课 概率论 (Probability)）
**预计时长：** 约90分钟

## 学习目标

- 实现基尼不纯度 (Gini Impurity)、熵 (Entropy) 和信息增益 (Information Gain) 的计算，以寻找决策树的最佳分裂点 (Decision Tree Splits)
- 从零开始构建决策树分类器 (Decision Tree Classifier)，并实现预剪枝 (Pre-pruning) 控制（最大深度、最小样本数）
- 使用自助采样 (Bootstrap Sampling) 和特征随机化 (Feature Randomization) 构建随机森林，并解释其为何能降低方差 (Variance)
- 对比平均不纯度减少 (Mean Decrease Impurity, MDI) 特征重要性与排列重要性 (Permutation Importance)，并识别 MDI 产生偏差的场景

## 问题背景

你手头有一份表格型数据 (Tabular Data)。其中每一行代表一个样本，每一列代表一个特征，还有一列是你需要预测的目标变量。你当然可以直接套用神经网络 (Neural Network) 来解决。但对于表格数据而言，基于树的模型 (Tree-based Models)（决策树、随机森林、梯度提升树 (Gradient Boosted Trees)）的表现始终优于深度学习 (Deep Learning)。在 Kaggle 的结构化数据 (Structured Data) 竞赛中，占据主导地位的始终是 XGBoost 和 LightGBM，而非 Transformer。

为什么？树模型无需预处理即可处理混合特征类型（数值型与类别型 (Categorical)）。它们无需特征工程 (Feature Engineering) 就能捕捉非线性关系。它们具有可解释性 (Interpretability)：你可以直接查看树结构，清楚地了解模型做出某项预测的具体原因。此外，随机森林通过对多棵树的结果进行平均，在中等规模的数据集上表现出极强的抗过拟合 (Overfitting) 能力。

本课程将使用递归分裂 (Recursive Splitting) 从零开始构建决策树，并在此基础上搭建随机森林。你将亲手实现分裂准则（基尼不纯度、熵、信息增益）背后的数学计算，并理解为何弱学习器 (Weak Learners) 的集成 (Ensemble) 能够转化为强学习器。

## 核心概念

### 决策树（Decision Tree）的工作原理

决策树通过提出一系列是/否问题，将特征空间（Feature Space）划分为多个矩形区域。

graph TD
    A["Age < 30?"] -->|Yes| B["Income > 50k?"]
    A -->|No| C["Credit Score > 700?"]
    B -->|Yes| D["Approve"]
    B -->|No| E["Deny"]
    C -->|Yes| F["Approve"]
    C -->|No| G["Deny"]

每个内部节点（Internal Node）都会将某个特征与阈值进行比较。每个叶节点（Leaf Node）负责输出预测结果。要对新数据点进行分类，需从根节点（Root Node）出发，沿着分支向下遍历，直至到达叶节点。

树结构采用自顶向下的方式构建：在每个节点处，选择能够最佳划分数据的特征与阈值。“最佳”的标准由划分准则（Split Criterion）定义。

### 划分准则：衡量不纯度（Impurity）

在每个节点处，我们拥有一组样本。划分的目标是使生成的子节点尽可能“纯净”，即每个子节点主要包含单一类别的样本。

**基尼不纯度（Gini Impurity）**衡量的是：如果按照该节点的类别分布随机为样本打标签，随机选中的样本被错误分类的概率。

Gini(S) = 1 - sum(p_k^2)

where p_k is the proportion of class k in set S.

对于纯净节点（全为同一类别），基尼不纯度为 0。对于类别比例为 50/50 的二分类划分，基尼不纯度为 0.5。该值越低越好。

Example: 6 cats, 4 dogs

Gini = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48

**信息熵（Entropy）**衡量节点中的信息含量（或混乱程度）。相关内容已在第一阶段第 09 课中讲解。

Entropy(S) = -sum(p_k * log2(p_k))

对于纯净节点，信息熵为 0。对于类别比例为 50/50 的二分类划分，信息熵为 1.0。该值越低越好。

Example: 6 cats, 4 dogs

Entropy = -(0.6 * log2(0.6) + 0.4 * log2(0.4))
        = -(0.6 * -0.737 + 0.4 * -1.322)
        = 0.442 + 0.529
        = 0.971 bits

**信息增益（Information Gain）**是指划分前后不纯度（信息熵或基尼不纯度）的减少量。

IG(S, feature, threshold) = Impurity(S) - weighted_avg(Impurity(S_left), Impurity(S_right))

where the weights are the proportions of samples in each child.

每个节点处的贪心算法（Greedy Algorithm）流程如下：遍历所有特征及所有可能的阈值，选择能使信息增益最大化的（特征，阈值）组合。

### 划分的具体流程

假设当前节点的数据集包含 n 个特征和 m 个样本：

1. 遍历每个特征 j（j = 1 到 n）：
   - 按特征 j 对样本进行排序
   - 将相邻不同值之间的中点依次作为候选阈值
   - 计算每个阈值对应的信息增益
2. 选择信息增益最高的特征与阈值
3. 将数据划分为左子集（特征值 <= 阈值）和右子集（特征值 > 阈值）
4. 对每个子节点递归执行上述过程

这种贪心策略无法保证得到全局最优的树结构。寻找最优树属于 NP 难（NP-hard）问题。但在实际应用中，贪心划分通常表现良好。

### 停止条件

若不设置停止条件，树会一直生长直至每个叶节点完全纯净（每个叶节点仅含一个样本）。这会导致模型完美记忆训练数据，但泛化能力（Generalization）极差。

**预剪枝（Pre-pruning）**在树完全生长前提前停止：
- 最大深度（Maximum Depth）：当树达到预设深度时停止划分
- 叶节点最小样本数：若节点样本数少于 k 个则停止
- 最小信息增益：若最佳划分带来的不纯度降低量低于阈值则停止
- 最大叶节点数：限制叶节点的总数

**后剪枝（Post-pruning）**先生成完整的树，再进行修剪：
- 代价复杂度剪枝（Cost-Complexity Pruning，scikit-learn 采用）：引入与叶节点数量成正比的惩罚项。增大惩罚项可得到更小的树
- 误差降低剪枝（Reduced Error Pruning）：若移除某子树后验证误差未上升，则将其剪除

预剪枝更简单快捷。后剪枝通常能生成更优的树结构，因为它不会过早终止那些可能导向后续有效划分的分支。

### 用于回归任务的决策树

在回归任务中，叶节点的预测值为该节点内目标值的均值。划分准则也随之改变：

**方差减少量（Variance Reduction）**取代了信息增益：

VR(S, feature, threshold) = Var(S) - weighted_avg(Var(S_left), Var(S_right))

选择使方差降低最多的划分方式。树将输入空间划分为多个区域，并在每个区域内预测一个常数值（即均值）。

### 随机森林（Random Forest）：集成学习的力量

单棵决策树具有高方差（High Variance）特性。数据的微小变动可能导致生成完全不同的树。随机森林通过平均多棵树的预测结果来解决这一问题。

graph TD
    D["Training Data"] --> B1["Bootstrap Sample 1"]
    D --> B2["Bootstrap Sample 2"]
    D --> B3["Bootstrap Sample 3"]
    D --> BN["Bootstrap Sample N"]
    B1 --> T1["Tree 1<br>(random feature subset)"]
    B2 --> T2["Tree 2<br>(random feature subset)"]
    B3 --> T3["Tree 3<br>(random feature subset)"]
    BN --> TN["Tree N<br>(random feature subset)"]
    T1 --> V["Aggregate Predictions<br>(majority vote or average)"]
    T2 --> V
    T3 --> V
    TN --> V

两种随机性机制确保了树之间的多样性：

**Bagging（Bootstrap Aggregating，自助聚合）**：每棵树均在自助采样（Bootstrap Sample）数据集上训练，即从原始训练集中有放回地随机抽样。每个自助样本集大约包含原始数据 63% 的样本（剩余部分为袋外样本 Out-of-Bag Samples，可用于验证）。

**特征随机化（Feature Randomization）**：在每次划分时，仅考虑特征的随机子集。分类任务的默认值为 `sqrt(n_features)`，回归任务为 `n_features/3`。这避免了所有树都依赖同一个主导特征进行划分。

核心思想：对大量去相关（Decorrelated）的树进行平均，可以在不增加偏差（Bias）的前提下有效降低方差。单棵树的表现可能平平，但集成模型（Ensemble）却非常强大。

### 特征重要性（Feature Importance）

随机森林天然能够提供特征重要性评分。最常用的方法如下：

**平均不纯度减少量（Mean Decrease in Impurity, MDI）**：针对每个特征，累加其在所有树及所有使用该特征的节点处带来的不纯度总减少量。在早期划分中带来更大不纯度降低的特征更为重要。

importance(feature_j) = sum over all nodes where feature_j is used:
    (n_samples_at_node / n_total_samples) * impurity_decrease

该方法计算速度快（在训练过程中即可得出），但会偏向高基数（High-Cardinality）特征以及具有大量可能划分点的特征。

**置换重要性（Permutation Importance）**是另一种替代方案：打乱某一特征的值，并测量模型准确率下降的幅度。该方法更可靠，但计算速度较慢。

### 树模型何时优于神经网络

在处理表格数据（Tabular Data）时，树模型与森林模型通常优于神经网络。主要原因如下：

| 因素 | 树模型 | 神经网络 |
|--------|-------|----------------|
| 混合数据类型（数值型 + 类别型） | 原生支持 | 需要编码 |
| 小型数据集（< 1 万行） | 表现良好 | 容易过拟合 |
| 特征交互 | 通过划分自动发现 | 需要专门的网络架构设计 |
| 可解释性 | 完全透明 | 黑盒模型 |
| 训练时间 | 分钟级 | 小时级 |
| 超参数敏感度 | 低 | 高 |

当数据具有空间或序列结构（如图像、文本、音频）时，神经网络更具优势。而对于扁平化的特征表格，树模型通常是默认首选。

## 构建

### 步骤 1：基尼不纯度 (Gini Impurity) 与熵 (Entropy)

从零开始实现这两种分裂标准 (Split Criteria)，并验证它们在评估分裂优劣时是否得出一致结论。

import math

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )

### 步骤 2：寻找最佳分裂 (Best Split)

遍历所有特征 (Feature) 与阈值 (Threshold)，返回信息增益 (Information Gain) 最高的分裂方案。

def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels) +
        (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity

### 步骤 3：构建 DecisionTree 类

实现递归分裂 (Recursive Splitting)、预测 (Prediction) 以及特征重要性 (Feature Importance) 追踪。

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini",
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

### 步骤 4：构建 RandomForest 类

实现自助采样 (Bootstrap Sampling)、特征随机化 (Feature Randomization) 以及多数投票 (Majority Voting)。

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None,
                 min_samples_split=2, max_features="sqrt",
                 criterion="gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.n_trees):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        predictions = []
        for i in range(len(X)):
            votes = {}
            for preds in all_preds:
                v = preds[i]
                votes[v] = votes.get(v, 0) + 1
            predictions.append(max(votes, key=votes.get))
        return predictions

完整实现及所有辅助方法请参见 `code/trees.py`。

## 实际应用

使用 scikit-learn 训练随机森林（Random Forest）仅需三行代码：

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"Feature importances: {rf.feature_importances_}")

在实际应用中，梯度提升树（Gradient Boosted Trees，如 XGBoost、LightGBM、CatBoost）通常比随机森林表现更强，因为它们按顺序构建树，每棵树都会修正前一棵树的误差。但随机森林更不容易配置出错，且几乎不需要超参数调优（Hyperparameter Tuning）。

## 交付使用

本节将生成 `outputs/prompt-tree-interpreter.md` 文件——这是一个用于向业务利益相关者解释决策树（Decision Tree）分裂逻辑的提示词（Prompt）。只需输入已训练树的结构信息（深度、特征、分裂阈值、准确率），它就能将模型转化为通俗易懂的规则，对特征重要性（Feature Importance）进行排序，标记过拟合（Overfitting）或数据泄露（Data Leakage）问题，并给出后续建议。当你需要向不懂代码的人解释基于树的模型（Tree-based Model）时，随时可以使用它。

## 练习

1. 在包含 3 个类别的二维数据集上训练单棵决策树。手动追踪分裂过程并绘制矩形决策边界（Decision Boundary）。对比 max_depth=2 与 max_depth=10 时的边界差异。

2. 为回归树（Regression Tree）实现基于方差减少（Variance Reduction）的分裂算法。生成 200 个数据点 y = sin(x) + noise 并拟合你的回归树。将模型的分段常数预测结果与真实曲线绘制在同一图表中进行对比。

3. 分别构建包含 1、5、10、50 和 200 棵树的随机森林。绘制训练准确率与测试准确率随树数量变化的曲线。观察测试准确率趋于平稳但不会下降的现象（森林模型具有抵抗过拟合的特性）。

4. 在 5 个不同的数据集上，对比基尼不纯度（Gini Impurity）与信息熵（Entropy）作为分裂准则的效果。测量准确率与树深度。在大多数情况下，两者会产生几乎相同的结果。请解释其原因。

5. 实现排列重要性（Permutation Importance）。在一个包含高基数（High Cardinality）随机噪声特征的数据集上，将其与平均不纯度减少（Mean Decrease Impurity, MDI）重要性进行对比。MDI 会给予该噪声特征较高的排名，而排列重要性则不会。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 决策树 (Decision Tree) | “用于预测的流程图” | 一种通过习得一系列 if/else 划分规则，将特征空间划分为矩形区域的模型 |
| 基尼不纯度 (Gini Impurity) | “节点的混合程度” | 在节点处随机抽样样本被错误分类的概率。0 表示完全纯净，二分类问题中 0.5 表示不纯度最大 |
| 熵 (Entropy) | “节点中的混乱程度” | 节点处的信息量。0 表示完全纯净，二分类问题中 1.0 表示不确定性最大。源自信息论 |
| 信息增益 (Information Gain) | “划分的质量有多好” | 划分后不纯度的降低量。用于选择划分点的贪心准则 |
| 预剪枝 (Pre-pruning) | “提前停止树的生长” | 通过设置最大深度、最小样本数或最小增益阈值来提前终止树的生长 |
| 后剪枝 (Post-pruning) | “生成树后再进行修剪” | 先生成完整的树，随后移除那些无法提升验证集性能的子树 |
| 装袋法 (Bagging) | “在随机子集上训练” | 自助聚合（Bootstrap Aggregating）。使用有放回抽样生成的不同随机样本分别训练每个模型 |
| 随机森林 (Random Forest) | “一堆树” | 决策树的集成模型，每棵树均在自助样本上训练，且在每次划分时随机选取特征子集 |
| 特征重要性 (MDI) | “哪些特征更重要” | 每个特征在所有树和节点中贡献的不纯度减少总量 |
| 排列重要性 (Permutation Importance) | “打乱并检查” | 随机打乱某特征值后模型准确率的下降幅度。对于含噪声特征，比 MDI 更可靠 |
| 方差减少 (Variance Reduction) | “信息增益的回归版本” | 信息增益在回归树中的对应概念。选择能最大程度降低目标变量方差的划分点 |
| 自助样本 (Bootstrap Sample) | “带重复的随机样本” | 从原始数据集中有放回抽取的随机样本。样本量与原数据集相同，但包含重复项 |

## 扩展阅读

- [Breiman: Random Forests (2001)](https://link.springer.com/article/10.1023/A:1010933404324) - 随机森林的原始论文
- [Grinsztajn et al.: Why do tree-based models still outperform deep learning on tabular data? (2022)](https://arxiv.org/abs/2207.08815) - 针对表格数据任务，对树模型与神经网络进行的严谨对比
- [scikit-learn Decision Trees documentation](https://scikit-learn.org/stable/modules/tree.html) - 包含可视化工具的实用指南
- [XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)](https://arxiv.org/abs/1603.02754) - 在 Kaggle 竞赛中占据主导地位的梯度提升算法论文
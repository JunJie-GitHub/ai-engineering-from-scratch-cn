# 朴素贝叶斯 (Naive Bayes)

> “朴素”假设在数学上是错误的，但它依然有效。这正是它的精妙之处。

**类型：** 构建
**语言：** Python
**前置知识：** 第二阶段，第 01-07 课（分类 (Classification)、贝叶斯定理 (Bayes' Theorem)）
**预计时长：** 约 75 分钟

## 学习目标

- 从零开始实现用于文本分类的多项式朴素贝叶斯 (Multinomial Naive Bayes)，并加入拉普拉斯平滑 (Laplace Smoothing)
- 解释为何朴素独立性假设在数学上是错误的，但在实践中却能产生正确的类别排序
- 比较多项式、伯努利 (Bernoulli) 和高斯 (Gaussian) 朴素贝叶斯变体，并针对给定的特征类型选择合适的模型
- 在高维稀疏数据上将朴素贝叶斯与逻辑回归 (Logistic Regression) 进行对比评估，并解释其中起作用的偏差-方差权衡 (Bias-Variance Tradeoff)

## 问题描述

你需要对文本进行分类。例如，将电子邮件分为垃圾邮件或非垃圾邮件；将客户评价分为正面或负面；将支持工单划分到不同类别。你拥有成千上万的特征（每个词对应一个特征），但训练数据却十分有限。

大多数分类器在此处都会失效。逻辑回归需要足够的样本来可靠地估计成千上万的权重。决策树每次只能基于单个词进行划分，极易产生严重的过拟合 (Overfitting)。而在 10,000 维空间中使用 K 近邻 (K-Nearest Neighbors, KNN) 算法毫无意义，因为在该维度下，任意两点之间的距离都趋于相等。

朴素贝叶斯能够很好地应对这一挑战。它基于一个在数学上不成立的假设（即在给定类别的条件下，所有特征彼此独立），但在文本分类任务中，它依然能够超越许多“更智能”的模型，尤其是在训练集较小的情况下。它只需对数据进行一次遍历即可完成训练。它能够轻松扩展至数百万个特征。它还能输出概率估计值（尽管由于独立性假设，这些概率往往校准不佳）。

理解为何一个错误的假设却能带来良好的预测结果，能让你领悟机器学习的一个核心原理：最优的模型并非数学上最严谨的模型，而是最契合你当前数据偏差-方差权衡的模型。

## 核心概念

### 贝叶斯定理（快速回顾）

贝叶斯定理（Bayes' Theorem）将条件概率（Conditional Probabilities）进行了转换：

P(class | features) = P(features | class) * P(class) / P(features)

我们的目标是计算 `P(class | features)`——即在已知文档包含特定词汇的情况下，该文档属于某一类别的概率。我们可以通过以下公式计算：
- `P(features | class)`——似然（Likelihood），表示在该类别的文档中出现这些词汇的概率
- `P(class)`——先验概率（Prior Probability），即该类别的基础概率（例如，垃圾邮件在总体中有多常见？）
- `P(features)`——证据因子（Evidence），对所有类别都相同，因此在比较时可以忽略

具有最高 `P(class | features)` 的类别即为最终预测结果。

### 朴素独立性假设

精确计算 `P(features | class)` 需要估计所有特征联合出现的联合概率（Joint Probability）。如果词汇表包含 10,000 个词，你需要估计 $2^{10,000}$ 种可能组合的分布。这显然是不可能的。

朴素假设（Naive Assumption）：在给定类别的条件下，所有特征相互独立（Conditionally Independent）。

P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)

通过这一假设，我们无需估计一个不可能实现的联合分布，而是只需估计 $n$ 个简单的单特征分布。每个分布仅需统计词频即可。

这一假设显然不符合现实。在任何文档中，“machine”和“learning”这两个词的出现都不是独立的。但分类器并不需要绝对准确的概率估计值，它只需要正确的排序——即哪个类别的概率最高。独立性假设确实会引入系统性误差，但这些误差对所有类别的影响是相似的，因此最终的排序结果依然正确。

### 为什么它依然有效

原因有三：

1. **重排序，轻校准（Ranking over Calibration）**。分类任务只需确保排名第一的类别正确即可。即使真实概率为 0.7，而模型输出 P(spam) = 0.99999，分类器依然能正确判定为垃圾邮件。我们不需要精确的概率值，只需要选出正确的胜者。

2. **高偏差，低方差（High Bias, Low Variance）**。独立性假设是一种强先验（Strong Prior）。它对模型施加了严格约束，从而有效防止过拟合（Overfitting）。在训练数据有限的情况下，一个略有偏差但表现稳定的模型，往往优于理论上正确但极不稳定的模型。这正是偏差-方差权衡（Bias-Variance Tradeoff）的实际体现。

3. **特征冗余相互抵消**。相关的特征会提供重复的证据。分类器会对这些证据进行重复计算，但它在正确的类别上也会同样重复计算。如果“machine”和“learning”总是同时出现，它们都会为“tech”类别提供证据。朴素贝叶斯（Naive Bayes, NB）虽然将它们计算了两次，但这两次都指向了正确的类别。

第四个实际原因：朴素贝叶斯的速度极快。训练过程只需遍历一次数据并统计词频。预测过程本质上是一次矩阵乘法。你可以在几秒钟内完成百万级文档的训练。这种速度优势意味着你可以更快地迭代、尝试更多的特征组合，并运行比慢速模型更多的实验。

### 数学推导逐步解析

让我们通过一个具体示例来逐步推导。假设我们有两个类别：垃圾邮件（spam）和非垃圾邮件（not-spam）。我们的词汇表包含三个词：“free”、“money”、“meeting”。

训练数据：
- 垃圾邮件中提及“free”80次，“money”60次，“meeting”10次（共150个词）
- 非垃圾邮件中提及“free”5次，“money”10次，“meeting”100次（共115个词）
- 40% 的邮件是垃圾邮件，60% 是非垃圾邮件

使用拉普拉斯平滑（Laplace Smoothing，alpha=1）：

P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856

新邮件包含：“free”（2次）、“money”（1次）、“meeting”（0次）。

log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838

垃圾邮件类别以较大优势胜出。“free”出现两次是判定为垃圾邮件的强证据。请注意，“meeting”未出现对两个对数求和的贡献均为零（0 * log(P)）——在多项式朴素贝叶斯（Multinomial Naive Bayes）中，未出现的词不会产生任何影响。显式建模词汇缺失的是伯努利朴素贝叶斯（Bernoulli Naive Bayes）。

### 三种变体

朴素贝叶斯有三种常见变体。它们对 `P(feature | class)` 的建模方式各不相同。

#### 多项式朴素贝叶斯（Multinomial Naive Bayes）

将每个特征建模为计数值。最适合特征为词频或 TF-IDF 值的文本数据。

P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)

其中的 `alpha` 代表拉普拉斯平滑（详见下文）。该变体是文本分类任务的主力模型。

#### 高斯朴素贝叶斯（Gaussian Naive Bayes）

将每个特征建模为正态分布（Normal Distribution）。最适合连续型特征。

P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))

每个类别针对每个特征都有独立的均值和方差。当特征在每个类别内确实呈现钟形曲线分布时，该模型表现优异。

#### 伯努利朴素贝叶斯（Bernoulli Naive Bayes）

将每个特征建模为二值型（出现或未出现）。最适合短文本或二值特征向量。

P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)

与多项式变体不同，伯努利变体会显式地对词汇缺失进行惩罚。如果“free”通常出现在垃圾邮件中，但当前邮件并未包含该词，伯努利模型会将其视为反对垃圾邮件的证据。

### 何时使用哪种变体

| 变体 | 特征类型 | 最佳适用场景 | 示例 |
|---------|-------------|----------|---------|
| 多项式（Multinomial） | 计数或频率 | 文本分类、词袋模型（Bag-of-Words） | 邮件垃圾过滤、主题分类 |
| 高斯（Gaussian） | 连续值 | 具有近似正态分布特征的表格数据 | 鸢尾花分类、传感器数据 |
| 伯努利（Bernoulli） | 二值（0/1） | 短文本、二值特征向量 | 短信垃圾过滤、存在/缺失特征 |

### 拉普拉斯平滑（Laplace Smoothing）

当某个词在测试数据中出现，但在特定类别的训练数据中从未出现过时，会发生什么？

如果不进行平滑处理：`P(word | class) = 0/N = 0`。整个连乘式中只要出现一个零，就会导致 `P(class | features) = 0`，无论其他证据多么充分。一个未见过的词就会彻底摧毁整个预测结果。

拉普拉斯平滑会为每个特征的计数加上一个较小的值 `alpha`（通常为 1）：

P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)

当 alpha=1 时，每个词至少会获得一个极小的概率。测试邮件中出现“discombobulate”这种生僻词也不会再导致垃圾邮件概率归零。这种平滑方法具有贝叶斯解释：它等价于在词分布上放置了一个均匀狄利克雷先验（Uniform Dirichlet Prior）。

较高的 alpha 意味着更强的平滑效果（分布更均匀）。较低的 alpha 意味着模型更信任观测数据。Alpha 是一个需要调节的超参数（Hyperparameter）。

alpha 的影响：

| Alpha 值 | 效果 | 适用场景 |
|-------|--------|-------------|
| 0.001 | 几乎不平滑，高度信任数据 | 训练集极大，且预期不会出现未见特征 |
| 0.1 | 轻度平滑 | 大型训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起始值 |
| 10.0 | 重度平滑，使分布趋于平坦 | 训练集极小，且预期会出现大量未见特征 |

### 对数空间计算（Log-Space Computation）

将数百个概率值（每个都小于 1）相乘会导致浮点数下溢（Floating-Point Underflow）。即使真实值是一个非常小的正数，在浮点运算中乘积也会变成零。

解决方案：在对数空间中进行计算。将概率相乘改为对数相加：

log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)

这使得预测过程转化为点积运算：

log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)

矩阵乘法。这就是朴素贝叶斯预测速度如此之快的原因——其运算与单层线性模型完全相同。

### 朴素贝叶斯 vs 逻辑回归（Logistic Regression）

两者都是用于文本的线性分类器。区别在于它们建模的对象不同。

| 方面 | 朴素贝叶斯（Naive Bayes） | 逻辑回归（Logistic Regression） |
|--------|------------|-------------------|
| 类型 | 生成式（Generative，建模 P(X\|Y)） | 判别式（Discriminative，建模 P(Y\|X)） |
| 训练方式 | 统计词频 | 优化损失函数 |
| 小数据表现 | 更优（强先验有帮助） | 较差（数据不足以估计权重） |
| 大数据表现 | 较差（错误假设带来负面影响） | 更优（决策边界更灵活） |
| 特征处理 | 假设特征独立 | 能处理特征相关性 |
| 速度 | 单次遍历，极快 | 迭代优化 |
| 概率校准 | 概率估计较差 | 概率估计较好 |

经验法则：从朴素贝叶斯开始。如果数据量充足且朴素贝叶斯性能遇到瓶颈，再切换到逻辑回归。

### 分类流水线（Classification Pipeline）

flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333

在实际应用中，我们在对数空间中进行计算以避免浮点数下溢。我们将多个小概率相乘转换为对数相加：

log P(class | features) = log P(class) + sum_i log P(feature_i | class)


## 构建

`code/naive_bayes.py` 中的代码从零开始实现了 `MultinomialNB` 和 `GaussianNB`。

### MultinomialNB

从零实现的逻辑如下：

1. **fit(X, y)**：针对每个类别，统计每个特征的出现频率。添加拉普拉斯平滑（Laplace smoothing）。计算对数概率。存储类别先验概率（类别频率的对数）。

2. **predict_log_proba(X)**：针对每个样本，计算所有类别的 log P(class) + log P(feature_i | class) 之和。这本质上是一个矩阵乘法：`X @ log_probs.T + log_priors`。

3. **predict(X)**：返回对数概率最高的类别。

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self

核心要点：模型拟合完成后，预测过程仅涉及矩阵乘法加上偏置项。这正是朴素贝叶斯（Naive Bayes）算法速度极快的原因。

### GaussianNB

对于连续型特征，我们按类别和特征分别估计均值与方差：

class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self

预测时，对每个特征使用高斯概率密度函数（Gaussian PDF）进行计算，并将各特征的概率相乘（在对数空间中转换为相加）。

### 演示：文本分类

该代码生成模拟两类数据（科技文章与体育文章）的合成词袋模型（bag-of-words）数据。每个类别具有不同的词频分布。`MultinomialNB` 利用词频统计对其进行分类。

合成数据的生成逻辑如下：我们创建 200 个“词”（特征列）。索引 0-39 的词在科技文章中频率较高，在体育文章中频率较低；索引 80-119 的词在体育文章中频率较高，在科技文章中频率较低；索引 40-79 的词在两类中均为中等频率。这构建了一个贴近现实的场景：部分词汇是强烈的类别指示器，而其余词汇则充当噪声。

### 演示：连续型特征

该代码生成类似鸢尾花（Iris）的数据集（3 个类别，4 个特征，高斯分布簇）。`GaussianNB` 利用各类别的均值和方差进行分类。每个类别拥有不同的中心（均值向量）和不同的离散程度（方差），以此模拟现实世界中不同类别在测量值上存在系统性差异的数据。

该代码还演示了以下内容：
- **平滑参数对比**：使用不同的 `alpha` 值训练 `MultinomialNB`，以展示平滑强度对准确率的影响。
- **训练集规模实验**：观察当训练数据从 20 个样本增加到 1600 个样本时，朴素贝叶斯（NB）准确率的提升情况。即使在样本量极少的情况下，NB 也能达到不错的准确率——这是它的主要优势。
- **混淆矩阵**：计算每个类别的精确率（precision）、召回率（recall）和 F1 分数（F1 score），以直观展示 NB 模型的误判情况。

### 预测速度

朴素贝叶斯的预测过程本质上是矩阵乘法。对于包含 n 个样本、d 个特征和 k 个类别的数据：
- `MultinomialNB`：一次矩阵乘法 `(n x d) @ (d x k)`，时间复杂度为 `O(n * d * k)`
- `GaussianNB`：进行 `n * k` 次高斯概率密度函数评估，每次评估涉及 `d` 个特征，时间复杂度为 `O(n * d * k)`

两者在所有维度上均呈线性复杂度。相比之下，K近邻算法（KNN）需要计算与所有训练样本的距离，而使用径向基核函数（RBF kernel）的支持向量机（SVM）则需要针对所有支持向量进行核函数评估。在预测阶段，朴素贝叶斯的速度要快出数个数量级。

## Use It

使用 `sklearn`，这两种变体只需一行代码即可实现：

from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")

在 `sklearn` 中进行文本分类时：

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)

`naive_bayes.py` 中的代码在相同数据上对比了从零实现的版本与 `sklearn` 的结果，以验证正确性。

### 结合朴素贝叶斯 (Naive Bayes) 的 TF-IDF

原始词频统计为每个词的出现赋予相同的权重。但像 "the" 和 "is" 这样的常见词在每个类别中都频繁出现，它们并不携带有效信息。词频-逆文档频率 (Term Frequency - Inverse Document Frequency) 会降低常见词的权重，同时提升稀有且具有区分度词汇的权重。

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])

TF-IDF 的值均为非负数，因此可以与 `MultinomialNB` 配合使用。TF-IDF 结合 `MultinomialNB` 是文本分类任务中最强的基线模型之一。在训练样本少于 10,000 个的数据集上，它经常能击败更复杂的模型。

### 针对短文本的伯努利朴素贝叶斯 (BernoulliNB)

对于短文本（如推文、短信、聊天消息），`BernoulliNB` 的表现可能优于 `MultinomialNB`。短文本的词频较低，导致 `MultinomialNB` 所依赖的频率信息充满噪声。`BernoulliNB` 仅关注词汇是否出现，这在处理短文本时更为可靠。

from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])

`CountVectorizer` 中的 `binary=True` 参数会将所有词频转换为 0 或 1。如果不设置该参数，`BernoulliNB` 虽然仍能运行，但会接收到并非为其设计的词频数据。

### 校准朴素贝叶斯 (NB) 的概率输出

朴素贝叶斯 (NB) 输出的概率通常校准效果较差。当 NB 预测 P(spam) = 0.95 时，真实概率可能仅为 0.7。如果你需要可靠的概率估计（例如用于设定阈值或与其他模型融合），可以使用 `sklearn` 的 `CalibratedClassifierCV`：

from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)

该方法通过交叉验证在 NB 的原始得分之上拟合一个逻辑回归 (Logistic Regression) 模型。校准后得到的概率会更接近真实的类别频率。

### 常见陷阱与注意事项

1. **负特征值。** `MultinomialNB` 要求特征值为非负数。如果你的数据包含负值（例如特定设置下的 TF-IDF 或标准化后的特征），请改用 `GaussianNB`，或将特征值平移至正数范围。

2. **零方差特征。** `GaussianNB` 的计算涉及除以方差。如果某个特征在某一类别下的方差为零（即所有值完全相同），概率计算将会出错。代码中为所有方差添加了一个极小的平滑项（1e-9）以防止此问题。

3. **类别不平衡。** 如果 99% 的邮件都不是垃圾邮件，那么先验概率 P(not-spam) = 0.99 会过于强势，从而掩盖似然证据。你可以手动设置类别先验，或在 `sklearn` 中使用 `class_prior` 参数进行调整。

4. **特征缩放。** `MultinomialNB` 不需要特征缩放（它直接处理词频计数）。`GaussianNB` 同样不需要缩放（它会独立估计每个特征的统计量）。这是它们相较于逻辑回归 (Logistic Regression) 和支持向量机 (Support Vector Machine) 的一大优势，后两者对特征尺度非常敏感。

## 交付上线

本课时将产出：
- `outputs/skill-naive-bayes-chooser.md` -- 用于选择合适朴素贝叶斯 (Naive Bayes, NB) 变体的决策技能
- `code/naive_bayes.py` -- 从零实现 MultinomialNB 和 GaussianNB，并与 scikit-learn 进行对比

### 朴素贝叶斯 (Naive Bayes) 何时失效

当独立性假设 (independence assumption) 导致排序错误（而不仅仅是概率计算错误）时，NB 就会失效。具体发生在以下情况：

1. **强烈的特征交互 (feature interactions)**。如果类别取决于两个特征的组合，而非单独某个特征（类似异或 (XOR) 的模式），NB 将完全无法捕捉。单个特征本身不提供任何判别证据，且 NB 无法进行非线性组合。

2. **高度相关但提供相反证据的特征**。如果特征 A 指示“垃圾邮件”，而特征 B 指示“非垃圾邮件”，但 A 和 B 完全相关（在现实中它们总是一致出现），NB 会误以为存在相互矛盾的证据，而实际上并没有。

3. **极大的训练集**。当数据量足够大时，逻辑回归 (logistic regression) 等判别式模型 (discriminative models) 能够学习到真实的决策边界，从而性能超越 NB。原本在小数据量下有益的独立性假设，此时反而限制了模型的表现。

在实践中，这些失效模式在文本分类中较为罕见。文本特征数量庞大且单个特征判别力较弱，独立性假设带来的误差往往会相互抵消。对于包含少量强相关特征的表格型数据，建议优先考虑逻辑回归或基于树的模型 (tree-based models)。

## 练习

1. **平滑实验 (smoothing experiment)**。使用 alpha 值为 0.01、0.1、1.0、10.0 和 100.0 在文本数据上训练 MultinomialNB。绘制准确率 (accuracy) 随 alpha 变化的曲线。性能在何处达到峰值？为什么过高的 alpha 值会损害性能？

2. **特征独立性测试**。选取一个真实的文本数据集。挑选两个明显相关的词（如 "machine" 和 "learning"）。计算 P(word1 | class) * P(word2 | class)，并与 P(word1 AND word2 | class) 进行比较。独立性假设的偏差有多大？它是否会影响分类准确率？

3. **伯努利实现 (Bernoulli implementation)**。在代码中扩展一个 BernoulliNB 类。将词袋模型 (bag-of-words) 转换为二值形式（出现/未出现），并在文本数据上对比其与 MultinomialNB 的准确率。伯努利朴素贝叶斯在什么情况下表现更优？

4. **NB 与逻辑回归对比**。在文本数据上同时训练两者。从 100 个训练样本开始，逐步增加至 10,000 个。绘制两者的准确率随训练集规模变化的曲线。逻辑回归在什么节点会超越朴素贝叶斯？

5. **垃圾邮件过滤器**。构建一个完整的垃圾邮件分类器：对原始邮件文本进行分词 (tokenize)，构建词汇表，创建词袋特征，训练 MultinomialNB，并使用精确率 (precision) 和召回率 (recall) 进行评估（而不仅仅是准确率——为什么？）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯 (Naive Bayes) | “简单的概率分类器” | 一种应用贝叶斯定理的分类器，其核心假设是：在给定类别的条件下，各特征之间是条件独立的 |
| 条件独立 (Conditional independence) | “特征之间互不影响” | P(A, B \| C) = P(A \| C) * P(B \| C) —— 在已知 C 的前提下，获知 B 不会为 A 提供任何额外信息 |
| 拉普拉斯平滑 (Laplace smoothing) | “加一平滑” | 为每个特征计数添加一个较小的常数，以防止零概率在预测中占据主导地位 |
| 先验概率 (Prior) | “观测数据前的初始信念” | P(class) —— 在观测到任何特征之前，各个类别的初始概率 |
| 似然 (Likelihood) | “数据与模型的拟合程度” | P(features \| class) —— 在类别已知的情况下，观测到当前特征组合的概率 |
| 后验概率 (Posterior) | “观测数据后的更新信念” | P(class \| features) —— 在观测到特征之后，经过更新得到的类别概率 |
| 生成模型 (Generative model) | “对数据生成过程进行建模” | 一种通过学习 P(X \| Y) 和 P(Y)，再利用贝叶斯定理推导出 P(Y \| X) 的模型 |
| 判别模型 (Discriminative model) | “对决策边界进行建模” | 一种直接学习 P(Y \| X)，而无需对特征 X 的生成过程进行建模的模型 |
| 对数概率 (Log probability) | “防止数值下溢” | 使用 log P 替代 P 进行运算，以避免多个极小概率值连乘时在浮点数计算中归零 |

## 扩展阅读

- [scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 涵盖三种变体及其详细的数学推导
- [McCallum 和 Nigam，《朴素贝叶斯文本分类事件模型比较》(1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) —— 针对文本分类中多项式 (Multinomial) 模型与伯努利 (Bernoulli) 模型的经典对比
- [Rennie 等人，《解决朴素贝叶斯文本分类器的缺陷假设》(2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) —— 针对文本场景的朴素贝叶斯改进方案
- [Ng 和 Jordan，《论判别式分类器与生成式分类器》(2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) —— 证明了在数据量较少时，朴素贝叶斯比逻辑回归 (Logistic Regression) 收敛更快
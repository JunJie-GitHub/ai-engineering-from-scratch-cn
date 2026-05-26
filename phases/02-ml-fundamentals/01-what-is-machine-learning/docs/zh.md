# 什么是机器学习 (Machine Learning)

> 机器学习是教计算机从数据中发现模式，而不是手动编写规则。

**类型：** 学习
**编程语言：** Python
**前置要求：** 第一阶段（数学基础）
**预计耗时：** 约45分钟

## 学习目标

- 解释监督学习 (Supervised Learning)、无监督学习 (Unsupervised Learning) 和强化学习 (Reinforcement Learning) 之间的区别，并判断哪种类型适用于特定问题
- 从零实现最近质心分类器 (Nearest Centroid Classifier)，并对照随机基线 (Random Baseline) 评估其性能
- 区分分类 (Classification) 与回归 (Regression) 任务，并为每种任务选择合适的损失函数 (Loss Function)
- 评估给定的业务问题是否适合使用机器学习 (ML) 解决，还是更适合通过确定性规则 (Deterministic Rules) 处理

## 问题背景

假设你想构建一个垃圾邮件过滤器。传统做法是：坐下来编写成百上千条规则。“如果邮件包含‘免费赚钱’，标记为垃圾邮件。如果感叹号超过3个，标记为垃圾邮件。”你花了几周时间编写规则。随后，垃圾邮件发送者改变了措辞。你的规则失效了。你只能继续编写更多规则。这个循环永无止境。

机器学习彻底颠覆了这一模式。你不再手动编写规则，而是向计算机提供成千上万封已标注的邮件（“垃圾邮件”或“非垃圾邮件”），让它自行推导出规则。计算机会发现你根本想不到的模式。当垃圾邮件发送者改变策略时，你只需使用新数据重新训练模型，而无需重写代码。

这种从“编程规则”到“从数据中学习”的转变，正是机器学习的核心。所有的推荐引擎、语音助手、自动驾驶汽车和语言模型都是基于这一原理运行的。

## 核心概念

### 从数据中学习，而非规则

传统编程与机器学习（Machine Learning）解决问题的方向截然相反。

flowchart LR
    subgraph Traditional["Traditional Programming"]
        direction LR
        R[Rules] --> P1[Program]
        D1[Data] --> P1
        P1 --> O1[Output]
    end

    subgraph ML["Machine Learning"]
        direction LR
        D2[Data] --> P2[Learning Algorithm]
        O2[Expected Output] --> P2
        P2 --> M[Model / Rules]
    end

传统编程：由你编写规则。程序将这些规则应用于数据以生成输出。

机器学习：你提供数据和期望的输出。算法自行发现规则。

训练产生的“模型”本质上就是规则，只不过它们被编码成了数字（权重、参数）。模型能够从已见过的样本中进行泛化（Generalization），从而对从未见过的数据做出预测。

### 机器学习的三大类型

flowchart TD
    ML[Machine Learning] --> SL[Supervised Learning]
    ML --> UL[Unsupervised Learning]
    ML --> RL[Reinforcement Learning]

    SL --> C[Classification]
    SL --> R[Regression]

    UL --> CL[Clustering]
    UL --> DR[Dimensionality Reduction]

    RL --> PO[Policy Optimization]
    RL --> VL[Value Learning]

**监督学习（Supervised Learning）**：你拥有输入-输出对。模型学习如何将输入映射到输出。
- “这里有 10,000 张标注为猫或狗的照片。学会区分它们。”
- “这里有房屋特征和价格数据。学会预测房价。”

**无监督学习（Unsupervised Learning）**：你只有输入数据，没有标签。模型自行发现数据中的结构。
- “这里有 10,000 条客户购买记录。找出自然的分组。”
- “这里有 1,000 维的数据点。在保持结构的同时将其降至 2 维。”

**强化学习（Reinforcement Learning）**：智能体（Agent）在环境中采取行动并获得奖励或惩罚。它学习一种策略（Policy）以最大化总奖励。
- “玩这个游戏。赢 +1 分，输 -1 分。自己摸索策略。”
- “控制这个机械臂。抓取到物体 +1 分，每浪费一秒 -0.01 分。”

在实际开发中，你构建的大多数项目都将使用监督学习。无监督学习常用于数据预处理和探索性分析。强化学习则是游戏 AI、机器人控制以及大语言模型的人类反馈强化学习（RLHF）的核心驱动力。

### 超越三大基础类型

上述三大类别界限分明，但现实世界中的机器学习应用往往会模糊这些界限。

**半监督学习（Semi-supervised Learning）**结合了少量标注数据和大量未标注数据。例如，你可能只有 100 张标注的医学图像，却有 100,000 张未标注的。常用技术包括：

- **标签传播（Label Propagation）**：构建一个连接相似数据点的图。标签通过图结构从已标注节点传播到未标注的邻居节点。
- **伪标签（Pseudo-labeling）**：先在标注数据上训练模型，用该模型为未标注数据预测标签，然后使用所有数据重新训练。模型通过这种方式“自举”扩充自己的训练集。
- **一致性正则化（Consistency Regularization）**：模型对原始输入及其轻微扰动版本应给出相同的预测。即使没有标签，该方法也有效。

**自监督学习（Self-supervised Learning）**直接从数据本身生成监督信号，完全无需人工标注。模型根据数据的内在结构自行构造预测任务。

- **掩码语言建模（Masked Language Modeling, BERT）**：隐藏句子中 15% 的词汇，训练模型预测缺失的词。这里的“标签”直接来自原始文本。
- **对比学习（Contrastive Learning, SimCLR）**：选取一张图像，生成两个增强版本。训练模型识别它们源自同一张图像，同时将其与其他图像的增强版本区分开来。
- **下一词元预测（Next-token Prediction, GPT）**：根据前面所有的词预测下一个词。每篇文本文档都自动转化为训练样本。

它们并非独立于上述三大类别之外的新分类，而是融合了监督与无监督思想的策略。从技术上讲，自监督学习属于监督学习（因为模型确实在进行预测），但其标签是自动生成的，而非人工标注。

### 分类与回归

这是监督学习的两大核心任务。

| 方面 | 分类（Classification） | 回归（Regression） |
|--------|---------------|------------|
| 输出 | 离散类别 | 连续数值 |
| 示例 | “这封邮件是垃圾邮件吗？” | “这套房子的价格会是多少？” |
| 输出空间 | {猫, 狗, 鸟} | 任意实数 |
| 损失函数 | 交叉熵（Cross-entropy）、准确率 | 均方误差（Mean Squared Error）、平均绝对误差（MAE） |
| 决策边界 | 类别之间的分界线 | 拟合数据的曲线 |

分类回答“属于哪一类？”，回归回答“具体是多少？”。

某些问题可以按任意一种方式建模。预测股票涨跌属于分类问题，而预测具体股价则属于回归问题。

### 机器学习工作流

无论使用何种算法，每个机器学习项目都遵循相同的流水线（Pipeline）。

flowchart LR
    A[Collect Data] --> B[Clean & Explore]
    B --> C[Feature Engineering]
    C --> D[Split Data]
    D --> E[Train Model]
    E --> F[Evaluate]
    F -->|Not good enough| C
    F -->|Good enough| G[Deploy]
    G --> H[Monitor]
    H -->|Performance drops| A

**收集数据**：获取原始数据。数据量越大通常越好，但数据质量比数量更重要。

**清洗与探索**：处理缺失值、去除重复项、可视化数据分布、识别异常值。这一步通常会占用项目总时间的 60% 到 80%。

**特征工程（Feature Engineering）**：将原始数据转换为模型可用的特征。例如将日期转换为星期几、对数值列进行归一化、对类别变量进行编码。优质的特征比复杂的算法更重要。

**划分数据**：将数据划分为训练集、验证集和测试集。模型在训练集上学习，你在验证集上调整超参数（Hyperparameters），并在测试集上报告最终性能。

**训练模型**：将训练数据输入算法。算法通过调整内部参数来最小化损失函数（Loss Function）。

**评估**：在验证集/测试集上衡量模型性能。如果性能不达标，则返回上一步尝试不同的特征、算法或超参数。

**部署**：将模型投入生产环境，使其对新数据进行预测。

**监控**：持续跟踪模型随时间变化的性能。数据分布会发生变化（数据漂移，Data Drift），导致模型性能衰退。当性能下降时，需重新训练。

### 训练集、验证集与测试集的划分

这是初学者最容易误解的核心概念。你必须使用模型在训练期间从未见过的数据来评估它。否则，你衡量的只是死记硬背的能力，而非真正的学习能力。

flowchart LR
    subgraph Dataset["Full Dataset (100%)"]
        direction LR
        TR["Training Set (70%)"]
        VA["Validation Set (15%)"]
        TE["Test Set (15%)"]
    end

    TR -->|Train model| M[Model]
    M -->|Tune hyperparameters| VA
    VA -->|Final evaluation| TE

| 划分 | 用途 | 使用时机 | 典型比例 |
|-------|---------|-----------|-------------|
| 训练集 | 模型从中学习 | 训练阶段 | 60-80% |
| 验证集 | 调整超参数、对比模型 | 每次训练结束后 | 10-20% |
| 测试集 | 最终无偏性能评估 | 仅在项目最后阶段使用一次 | 10-20% |

测试集是神圣不可侵犯的。你只能查看它一次。如果你不断根据测试集的表现来调整模型，实际上就等于在测试集上进行了训练，此时报告的指标将毫无意义。

对于小型数据集，建议使用 K 折交叉验证（K-fold Cross-Validation）：将数据分为 K 份，轮流使用其中 K-1 份进行训练，剩余 1 份进行验证，最后对结果取平均值。

### 过拟合与欠拟合

flowchart LR
    subgraph UF["Underfitting"]
        U1["Model too simple"]
        U2["High bias"]
        U3["Misses patterns"]
    end

    subgraph GF["Good Fit"]
        G1["Right complexity"]
        G2["Balanced"]
        G3["Generalizes well"]
    end

    subgraph OF["Overfitting"]
        O1["Model too complex"]
        O2["High variance"]
        O3["Memorizes noise"]
    end

    UF -->|Increase complexity| GF
    GF -->|Too much complexity| OF

**欠拟合（Underfitting）**：模型过于简单，无法捕捉数据中的规律。就像试图用一条直线去拟合曲线关系。此时训练误差和测试误差都很高。

**过拟合（Overfitting）**：模型过于复杂，死记硬背了训练数据（包括其中的噪声）。就像一条剧烈波动的曲线穿过了每一个训练点，但在新数据上表现糟糕。此时训练误差很低，但测试误差很高。

**良好拟合**：模型捕捉到了真实规律，而没有死记噪声。训练误差和测试误差都处于合理的较低水平。

过拟合的迹象：
- 训练准确率远高于验证准确率
- 模型在训练数据上表现优异，但在新数据上表现糟糕
- 增加更多训练数据能提升性能（说明模型之前只是在死记硬背，而非真正学习）

解决过拟合的方法：
- 获取更多训练数据
- 降低模型复杂度（减少参数量、简化网络架构）
- 正则化（Regularization，对较大的权重施加惩罚）
- Dropout（在训练过程中随机将部分神经元输出置零）
- 早停法（Early Stopping，当验证误差开始上升时停止训练）

解决欠拟合的方法：
- 使用更复杂的模型
- 增加更多特征
- 降低正则化强度
- 延长训练时间

### 偏差-方差权衡

这是解释过拟合与欠拟合现象的数学框架。

**偏差（Bias）**：源于模型错误假设的误差。当真实关系是非线性时，线性模型会表现出高偏差。高偏差会导致欠拟合。

**方差（Variance）**：源于模型对训练数据微小波动的敏感性。高方差模型在使用不同数据子集训练时，会给出差异极大的预测结果。高方差会导致过拟合。

| 模型复杂度 | 偏差 | 方差 | 结果 |
|-----------------|------|----------|--------|
| 过低（用线性模型拟合曲线数据） | 高 | 低 | 欠拟合 |
| 适中 | 中 | 中 | 泛化能力良好 |
| 过高（用 20 次多项式拟合 10 个数据点） | 低 | 高 | 过拟合 |

总误差 = 偏差² + 方差 + 不可约误差（Irreducible Noise）

你无法降低不可约误差（它源于数据本身的随机性）。你的目标是找到偏差²与方差之和最小的最佳平衡点。

### 没有免费午餐定理

不存在一种在所有问题上都能表现最佳的算法。在某类问题上表现优异的算法，在另一类问题上可能表现糟糕。这就是为什么数据科学家通常会尝试多种算法并对比结果。

在实际应用中，算法的选择取决于：
- 可用数据量的大小
- 特征的数量
- 数据关系是线性还是非线性
- 是否需要模型可解释性
- 可承受的计算资源成本

### 何时不应使用机器学习

机器学习虽然强大，但并非万能工具。在决定使用模型之前，请先问自己是否真的需要它。

**以下情况请勿使用机器学习：**

- **规则简单且明确。** 例如税费计算、排序算法、单位换算。如果你能用几条 `if` 语句写出逻辑，引入模型只会徒增复杂度而毫无益处。
- **没有数据或数据极少。** 机器学习需要样本进行学习。只有 10 个数据点时，你无法训练出任何有意义的模型。请先收集数据。
- **出错代价是灾难性的，且必须保证绝对正确。** 例如医疗剂量计算、核反应堆控制、密码学验证。机器学习模型是概率性的，它们偶尔会出错。如果“偶尔出错”是不可接受的，请使用确定性方法。
- **查表法或启发式规则已能解决问题。** 如果简单的阈值或规则表能覆盖 99% 的情况，引入机器学习只会增加维护成本，却不会带来实质性提升。
- **无法解释决策过程，但业务要求必须具备可解释性。** 受监管行业（如信贷、保险、刑事司法）有时要求每一项决策都能被完全解释。部分机器学习模型具有可解释性（如线性回归、小型决策树），但大多数模型并非如此。
- **问题变化的速度快于模型重新训练的速度。** 如果规则每天都在变，而重新训练需要一周时间，那么模型将永远处于过时状态。

请参考以下决策流程图：

flowchart TD
    A["Do you have data?"] -->|No| B["Collect data first or use rules"]
    A -->|Yes| C["Can you write the rules explicitly?"]
    C -->|"Yes, and they are simple"| D["Use rules. Skip ML."]
    C -->|"No, or they are too complex"| E["Is the cost of errors acceptable?"]
    E -->|"No, need guaranteed correctness"| F["Use deterministic methods"]
    E -->|Yes| G["Do you need explainability?"]
    G -->|"Yes, strictly"| H["Use interpretable models only"]
    G -->|"No, or partially"| I["Use ML"]
    I --> J["Do you have enough labeled data?"]
    J -->|Yes| K["Supervised learning"]
    J -->|"Some labels"| L["Semi-supervised learning"]
    J -->|"No labels"| M["Unsupervised or self-supervised"]


## 构建项目

`code/ml_intro.py` 中的代码从零开始实现了一个最近质心分类器（Nearest Centroid Classifier），这是最简单的机器学习（Machine Learning, ML）算法。它展示了核心思想：从数据中学习，然后对新数据进行预测。

### 步骤 1：从零实现最近质心分类器

最近质心分类器会计算训练数据中每个类别的中心（均值）。在进行预测时，它会将每个新数据点分配给中心距离最近的类别。

class NearestCentroid:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = np.array([
            X[y == c].mean(axis=0) for c in self.classes
        ])

    def predict(self, X):
        distances = np.array([
            np.sqrt(((X - c) ** 2).sum(axis=1))
            for c in self.centroids
        ])
        return self.classes[distances.argmin(axis=0)]

这就是该算法的全部内容。`fit` 负责计算均值，`predict` 负责计算距离。无需梯度下降（Gradient Descent），无需迭代，也没有超参数（Hyperparameters）。

### 步骤 2：在合成数据上进行训练

我们生成一个包含两个类别的二维分类数据集，这两个类别略有重叠。质心分类器会在类别中心之间绘制一条线性决策边界（Linear Decision Boundary）。

rng = np.random.RandomState(42)
X_class0 = rng.randn(100, 2) + np.array([1.0, 1.0])
X_class1 = rng.randn(100, 2) + np.array([-1.0, -1.0])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * 100 + [1] * 100)

### 步骤 3：与基线模型进行对比

每个机器学习模型都应与一个简单的基线（Baseline）进行对比。在这里，基线模型会随机预测一个类别。如果你的模型表现还不如随机猜测，那说明一定出了问题。

baseline_preds = rng.choice([0, 1], size=len(y_test))
baseline_acc = np.mean(baseline_preds == y_test)

在这个干净的数据集上，质心分类器的准确率应达到 90% 以上。而随机基线的准确率约为 50%。

### 为什么这很重要

最近质心分类器极其简单。它没有超参数，没有迭代，也没有梯度下降。但它却捕捉到了机器学习的基本模式：

1. 从训练数据中**学习**一种表示（Representation）（即质心）
2. 使用该表示对新数据进行**预测**（基于最近距离）
3. 与基线进行**评估**对比（随机猜测）

从逻辑回归（Logistic Regression）到 Transformer，每一种机器学习算法都遵循这相同的三步模式。表示形式会变得更加复杂，但工作流程保持不变。

### 步骤 4：质心分类器的局限性

最近质心分类器假设每个类别都形成一个单一的簇（Cluster）。它绘制的是线性决策边界。在以下情况下它会失效：

- 类别包含多个簇（例如，数字“1”可以有多种不同的写法）
- 决策边界是非线性的（例如，一个类别环绕着另一个类别）
- 特征尺度差异很大（距离计算会被尺度最大的特征主导）

这些局限性正是你将要学习的其他所有算法的出发点。K近邻（K-Nearest Neighbors, KNN）算法处理多簇问题。决策树（Decision Trees）处理非线性边界。特征缩放（Feature Scaling）解决尺度问题。每一课都建立在前一课的局限性之上。

## 上手使用

sklearn 提供了 `NearestCentroid` 和合成数据生成器：

from sklearn.neighbors import NearestCentroid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = NearestCentroid()
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")

## 交付上线

本课时将生成 `outputs/prompt-ml-problem-framer.md` 文件——这是一个提示词（Prompt），能够将模糊的业务问题转化为具体的机器学习任务。只需输入问题描述（例如“我们希望降低客户流失率”或“预测下一季度的需求”），它就能自动识别学习类型、定义预测目标、列出候选特征、选择评估指标、建立基线，并标记出数据泄露（Data Leakage）或类别不平衡（Class Imbalance）等潜在陷阱。在任何机器学习项目启动时使用它，可以避免方向性错误，确保构建正确的模型。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 模型（Model） | “人工智能” | 一种带有可学习参数的数学函数，用于将输入映射为输出 |
| 训练（Training） | “教 AI 学习” | 运行优化算法以调整模型参数，使预测结果与已知输出相匹配 |
| 特征（Feature） | “输入列” | 数据中可供模型用于预测的可度量属性 |
| 标签（Label） | “标准答案” | 训练样本的已知输出，用于计算误差信号 |
| 超参数（Hyperparameter） | “需要手动调整的设置” | 在训练前设定的参数，用于控制学习过程（如学习率、网络层数） |
| 损失函数（Loss Function） | “模型错得有多离谱” | 用于衡量预测输出与实际输出之间差距的函数，训练过程旨在最小化该值 |
| 过拟合（Overfitting） | “它把训练数据背下来了” | 模型记住了训练数据中的特定噪声而非普遍规律，导致在新数据上表现不佳 |
| 欠拟合（Underfitting） | “它啥也没学会” | 模型过于简单，无法捕捉数据中的真实规律 |
| 泛化（Generalization） | “它能处理新数据” | 模型对未参与训练的数据做出准确预测的能力 |
| 交叉验证（Cross-Validation） | “分块反复测试” | 将数据反复划分为训练集和测试集折（folds）并取平均结果，从而获得更稳健的性能评估 |
| 正则化（Regularization） | “限制权重过大” | 在损失函数中添加惩罚项，以防止模型过于复杂 |
| 数据漂移（Data Drift） | “现实情况变了” | 输入数据的统计分布随时间发生变化，导致模型性能下降 |

## 练习

1. 任选一个数据集（如 Iris 或 Titanic），按 70/15/15 的比例将其划分为训练集（training set）、验证集（validation set）和测试集（test set）。请解释为何不应在测试集上调整超参数（hyperparameters）。
2. 列举三个现实世界中的问题。针对每个问题，判断其属于分类（classification）、回归（regression）还是聚类（clustering），并说明其属于监督（supervised）还是无监督（unsupervised）学习。
3. 某模型在训练数据上的准确率（accuracy）达到 99%，但在测试数据上仅为 60%。请诊断该问题，并列出三项你打算尝试的修复措施。

## 扩展阅读

- [An Introduction to Statistical Learning](https://www.statlearning.com/) - 免费教材，涵盖所有经典机器学习（machine learning）方法并配有实际案例
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) - 简明直观的机器学习概念入门指南
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) - 在 Python 中实现机器学习的实用参考手册
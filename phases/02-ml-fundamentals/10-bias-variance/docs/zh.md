# 偏差-方差权衡 (Bias-Variance Tradeoff)

> 每个模型的误差都来源于三个方面之一：偏差（bias）、方差（variance）或噪声（noise）。你只能控制前两者。

**Type:** 学习
**Language:** Python
**Prerequisites:** 第二阶段，第 01-09 课（机器学习基础、回归、分类、评估）
**Time:** 约 75 分钟

## 学习目标

- 推导期望预测误差（expected prediction error）的偏差-方差分解（bias-variance decomposition），并解释不可约噪声（irreducible noise）的作用
- 利用训练误差和测试误差的模式，诊断模型是存在高偏差还是高方差
- 解释正则化（regularization）技术（L1、L2、Dropout、早停法）如何在偏差与方差之间进行权衡
- 实现实验，可视化不同复杂度模型的偏差-方差权衡过程

## 问题背景

你训练了一个模型，它在测试数据上存在一定的误差。这些误差究竟从何而来？

如果模型过于简单（例如在非线性数据集上使用线性回归），它将始终无法捕捉到真实的潜在规律。这就是偏差。如果模型过于复杂（例如仅用 15 个数据点拟合 20 次多项式），它会在训练数据上表现完美，但在新数据上给出差异巨大的预测结果。这就是方差。

在模型容量（model capacity）固定的情况下，你无法同时最小化这两者。降低偏差会导致方差上升，而降低方差则会使偏差增加。理解这种权衡关系是机器学习中最核心、最实用的诊断技能。它能指导你判断：应该增加还是降低模型复杂度，应该获取更多数据还是进行更好的特征工程（feature engineering），以及应该加强还是减弱正则化。

## 核心概念

### 偏差（Bias）：系统性误差

偏差衡量的是模型的平均预测值与真实值之间的偏离程度。如果你从同一分布中抽取多个不同的训练集来训练同一个模型，并对预测结果取平均，那么偏差就是这个平均值与真实值之间的差距。

高偏差意味着模型过于僵化，无法捕捉真实的模式。无论提供多少数据，用一条直线去拟合抛物线永远无法贴合曲线。这就是欠拟合（Underfitting）。

High bias (underfitting):
  Model always predicts roughly the same wrong thing.
  Training error: HIGH
  Test error: HIGH
  Gap between them: SMALL

### 方差（Variance）：对训练数据的敏感度

方差衡量的是当使用不同的数据子集进行训练时，模型预测结果的变化程度。如果训练集的微小变化导致模型发生剧烈变化，则说明方差很高。

高方差意味着模型拟合的是训练数据中的噪声，而非底层信号。一个20次多项式会穿过每一个训练点，但在点之间剧烈震荡。这就是过拟合（Overfitting）。

High variance (overfitting):
  Model fits training data perfectly but fails on new data.
  Training error: LOW
  Test error: HIGH
  Gap between them: LARGE

### 误差分解

对于任意数据点 x，在平方损失（Squared Loss）下的期望预测误差可以精确分解为：

Expected Error = Bias^2 + Variance + Irreducible Noise

where:
  Bias^2   = (E[f_hat(x)] - f(x))^2
  Variance = E[(f_hat(x) - E[f_hat(x)])^2]
  Noise    = E[(y - f(x))^2]             (sigma^2)

- `f(x)` 是真实函数
- `f_hat(x)` 是模型的预测值
- `E[...]` 表示在不同训练集上的期望
- `y` 是观测标签（真实函数加上噪声）

噪声项是不可约减的。在含有噪声的数据上，没有任何模型的表现能优于 `sigma^2`。你的任务是在偏差平方（`bias^2`）和方差之间找到最佳平衡点。

### 模型复杂度与误差的关系

graph LR
    A[Simple Model] -->|increase complexity| B[Sweet Spot]
    B -->|increase complexity| C[Complex Model]

    style A fill:#f9f,stroke:#333
    style B fill:#9f9,stroke:#333
    style C fill:#f99,stroke:#333

经典的 U 型曲线：

| 复杂度 | 偏差 | 方差 | 总误差 |
|-----------|------|----------|-------------|
| 过低 | 高 | 低 | 高（欠拟合） |
| 适中 | 中等 | 中等 | 最低 |
| 过高 | 低 | 高 | 高（过拟合） |

### 正则化：控制偏差与方差

正则化（Regularization）通过有意增加偏差来降低方差。它对模型施加约束，使其无法去拟合噪声。

- **L2 正则化（岭回归，Ridge）：** 将所有权重向零收缩。保留所有特征，但降低它们的影响。
- **L1 正则化（套索回归，Lasso）：** 将部分权重精确压缩至零。实现特征选择。
- **随机失活（Dropout）：** 在训练期间随机禁用神经元。强制模型学习冗余表示。
- **早停法（Early Stopping）：** 在模型完全拟合训练数据之前停止训练。

正则化强度（如 `lambda`、dropout 率、训练轮数 `epochs`）直接决定了你在偏差-方差曲线上的位置。正则化越强，偏差越大，方差越小。

### 双重下降（Double Descent）：现代视角

经典理论认为：越过最佳点后，增加复杂度总是有害的。但自 2019 年以来的研究揭示了一个意外现象。如果你继续大幅增加模型容量，使其远超插值阈值（Interpolation Threshold，即模型拥有足够参数以完美拟合训练数据的临界点），测试误差反而会再次下降。

graph LR
    A[Underfit Zone] --> B[Classical Sweet Spot]
    B --> C[Interpolation Threshold]
    C --> D[Double Descent - Error Drops Again]

    style A fill:#fdd,stroke:#333
    style B fill:#dfd,stroke:#333
    style C fill:#fdd,stroke:#333
    style D fill:#dfd,stroke:#333

这种“双重下降”现象解释了为何参数量远超训练样本量的大规模过参数化神经网络（Overparameterized Neural Networks）依然具有良好的泛化能力。经典的偏差-方差权衡（Bias-Variance Tradeoff）并没有错，但在现代机器学习范式下它是不完整的。

关于双重下降的关键观察：
- 它出现在线性模型、决策树和神经网络中
- 在插值区域，增加数据量反而可能有害（样本维度的双重下降）
- 增加训练轮数也可能引发该现象（轮数维度的双重下降）
- 正则化可以平滑峰值，但无法消除它

为什么会发生这种情况？在插值阈值处，模型恰好具备拟合所有训练点的容量。它被迫寻找一个穿过每个点的特定解，此时数据的微小扰动会导致拟合结果发生巨大变化，这正是方差达到峰值的地方。越过该阈值后，模型存在无数种能完美拟合数据的解。学习算法（例如带有隐式正则化的梯度下降）倾向于从中选择最简单的那个。这种对简单解的隐式偏好（Implicit Bias），正是过参数化模型能够良好泛化的原因。

| 机制/阶段 | 参数与样本关系 | 表现 |
|--------|----------------------|----------|
| 欠参数化（Underparameterized） | p << n | 适用经典权衡理论 |
| 插值阈值 | p ~ n | 方差达到峰值，测试误差骤升 |
| 过参数化（Overparameterized） | p >> n | 隐式正则化生效，测试误差下降 |

在实际应用中：如果你使用神经网络或大型树集成模型，不要停留在插值阈值处。要么保持在远低于该阈值的位置（配合显式正则化），要么大幅超越它。最糟糕的情况就是恰好卡在阈值上。

### 诊断你的模型

flowchart TD
    A[Compare train error vs test error] --> B{Large gap?}
    B -->|Yes| C[High variance - overfitting]
    B -->|No| D{Both errors high?}
    D -->|Yes| E[High bias - underfitting]
    D -->|No| F[Good fit]

    C --> G[More data / Regularize / Simpler model]
    E --> H[More features / Complex model / Less regularization]
    F --> I[Deploy]

| 症状 | 诊断 | 解决方案 |
|---------|-----------|-----|
| 训练误差高，测试误差高 | 偏差（欠拟合） | 增加特征、使用更复杂的模型、降低正则化强度 |
| 训练误差低，测试误差高 | 方差（过拟合） | 增加数据、添加正则化、简化模型、使用 Dropout |
| 训练误差低，测试误差低 | 拟合良好 | 直接部署 |
| 训练误差下降，测试误差上升 | 正在发生过拟合 | 使用早停法 |

### 实用策略

**当偏差是主要问题时：**
- 添加多项式或交互特征
- 使用更灵活的模型（例如用树集成替代线性模型）
- 降低正则化强度
- 延长训练时间（如果尚未收敛）

**当方差是主要问题时：**
- 获取更多训练数据
- 使用 Bagging（如随机森林）
- 增强正则化（提高 `lambda`、增加 dropout 比例）
- 特征选择（剔除噪声特征）
- 使用交叉验证（Cross-Validation）尽早发现问题

### 集成方法与方差降低

集成方法（Ensemble Methods）是对抗方差最实用的工具。

**Bagging（自助聚合，Bootstrap Aggregating）** 在训练数据的不同自助采样集上训练多个模型，然后对它们的预测结果取平均。每个独立模型都具有高方差，但平均后的结果方差会大幅降低。随机森林（Random Forests）就是将 Bagging 应用于决策树的典型代表。

其数学原理在于：如果你对 N 个独立的预测取平均，且每个预测的方差为 `sigma^2`，那么平均值的方差将变为 `sigma^2 / N`。虽然这些模型并非完全独立（它们都基于相似的数据训练），因此方差降低幅度小于 `1/N`，但效果依然非常显著。

**Boosting（提升法）** 通过串行构建模型来降低偏差，每个新模型都专注于修正当前集成模型的误差。梯度提升（Gradient Boosting）和 AdaBoost 是主要代表。如果添加过多模型，Boosting 可能会过拟合，因此需要配合早停法或正则化。

| 方法 | 主要作用 | 偏差变化 | 方差变化 |
|--------|---------------|-------------|-----------------|
| Bagging | 降低方差 | 基本不变 | 降低 |
| Boosting | 降低偏差 | 降低 | 可能增加 |
| Stacking（堆叠法） | 同时降低两者 | 取决于元学习器 | 取决于基模型 |
| Dropout | 隐式 Bagging | 轻微增加 | 降低 |

**实用法则：** 如果你的基模型方差较高（如深层决策树、高次多项式），请使用 Bagging。如果基模型偏差较高（如浅层树桩、简单线性模型），请使用 Boosting。

### 学习曲线（Learning Curves）

学习曲线绘制了训练误差和验证误差随训练集大小变化的趋势。这是你最实用的诊断工具。与单次训练/测试对比不同，学习曲线能展示模型的变化轨迹，并告诉你增加数据是否会有帮助。

flowchart TD
    subgraph HB["High Bias Learning Curve"]
        direction LR
        HB1["Small N: both errors high"]
        HB2["Large N: both errors converge to HIGH error"]
        HB1 --> HB2
    end

    subgraph HV["High Variance Learning Curve"]
        direction LR
        HV1["Small N: train low, test high (big gap)"]
        HV2["Large N: gap shrinks but slowly"]
        HV1 --> HV2
    end

    subgraph GF["Good Fit Learning Curve"]
        direction LR
        GF1["Small N: some gap"]
        GF2["Large N: both converge to LOW error"]
        GF1 --> GF2
    end

如何解读学习曲线：

| 场景 | 训练误差 | 验证误差 | 差距 | 含义 | 应对措施 |
|----------|---------------|-----------------|-----|---------------|------------|
| 高偏差 | 高 | 高 | 小 | 模型无法捕捉数据模式 | 增加特征、使用更复杂的模型、降低正则化 |
| 高方差 | 低 | 高 | 大 | 模型死记硬背训练数据 | 增加数据、添加正则化、简化模型 |
| 拟合良好 | 中等 | 中等 | 小 | 模型泛化能力好 | 直接部署 |
| 高方差，持续改善 | 低 | 随数据增加而下降 | 缩小 | 数据可解决的方差问题 | 收集更多数据 |
| 高偏差，趋于平稳 | 高 | 高且平稳 | 小且平稳 | 增加数据无效 | 更改模型架构 |

核心洞察：如果两条曲线均已趋于平稳，且差距很小但误差都很高，那么增加数据毫无用处。你需要一个更好的模型。如果差距仍然很大且持续缩小，增加数据将会带来帮助。

### 如何生成学习曲线

有两种常用方法：

**方法一：改变训练集大小，固定模型。** 保持模型和超参数不变。在训练数据中不断增大的子集上进行训练。记录每个数据量下的训练误差和验证误差。这是标准的学习曲线。

**方法二：改变模型复杂度，固定数据。** 保持数据不变。扫描复杂度参数（如多项式次数、树深度、网络层数）。记录每个复杂度下的训练误差和验证误差。这称为验证曲线（Validation Curve），能直接展示偏差-方差权衡。

这两种方法相辅相成。第一种告诉你增加数据是否有效，第二种告诉你更换模型是否有效。在决定下一步行动之前，建议两者都运行一遍。

flowchart TD
    A[Model underperforming] --> B[Generate learning curve]
    B --> C{Gap between train and val?}
    C -->|Large gap, val still decreasing| D[More data will help]
    C -->|Small gap, both high| E[More data will NOT help]
    C -->|Large gap, val flat| F[Regularize or simplify]
    E --> G[Generate validation curve]
    G --> H[Try more complex model]


## 构建项目

`code/bias_variance.py` 中的代码运行了完整的偏差-方差分解（bias-variance decomposition）实验。以下是具体步骤。

### 步骤 1：从已知函数生成合成数据

我们使用带有高斯噪声（Gaussian noise）的 `f(x) = sin(1.5x) + 0.5x`。由于已知真实函数，我们可以精确计算偏差和方差。

def true_function(x):
    return np.sin(1.5 * x) + 0.5 * x

def generate_data(n_samples=30, noise_std=0.5, x_range=(-3, 3), seed=None):
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_function(x) + rng.normal(0, noise_std, n_samples)
    return x, y

### 步骤 2：自助法采样与多项式拟合

对于每个多项式阶数（polynomial degree），我们抽取多个自助法训练集，拟合多项式，并在固定的测试点集上记录预测结果。这为我们提供了每个测试点上预测值的分布。

def fit_polynomial(x_train, y_train, degree, lam=0.0):
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    if lam > 0:
        penalty = lam * np.eye(X.shape[1])
        penalty[0, 0] = 0
        w = np.linalg.solve(X.T @ X + penalty, X.T @ y_train)
    else:
        w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w

我们在 200 个不同的自助法样本上进行拟合。每个自助法样本均从相同的底层分布中抽取，但包含的数据点各不相同。

### 步骤 3：计算偏差平方与方差分解

利用每个测试点上的 200 组预测结果，我们可以直接根据定义计算分解结果：

mean_pred = predictions.mean(axis=0)
bias_sq = np.mean((mean_pred - y_true) ** 2)
variance = np.mean(predictions.var(axis=0))
total_error = np.mean(np.mean((predictions - y_true) ** 2, axis=1))

- `mean_pred` 是从自助法样本中估计出的 E[f_hat(x)]
- `bias_sq` 是平均预测值与真实值之间差距的平方
- `variance` 是各自助法样本预测值的平均离散程度
- `total_error` 应近似等于偏差平方 + 方差 + 噪声

### 步骤 4：学习曲线

学习曲线（learning curves）在固定模型复杂度（model complexity）的情况下遍历训练集大小。它们能够揭示模型是受限于数据量还是受限于模型容量。

def demo_learning_curves():
    sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
    degree = 5

    for n in sizes:
        train_errors = []
        test_errors = []
        for seed in range(50):
            x_train, y_train = generate_data(n_samples=n, seed=seed * 100)
            w = fit_polynomial(x_train, y_train, degree)
            train_pred = predict_polynomial(x_train, w)
            train_mse = np.mean((train_pred - y_train) ** 2)
            test_pred = predict_polynomial(x_test, w)
            test_mse = np.mean((test_pred - y_test) ** 2)
            train_errors.append(train_mse)
            test_errors.append(test_mse)
        # Average over runs gives the learning curve point

对于高方差模型（high-variance model，例如小数据量下的 5 阶多项式），你会观察到：
- 训练误差起初较低，但随着数据量增加、模型难以死记硬背数据，误差会逐渐上升
- 测试误差起初较高，但随着模型获取更多有效信号，误差会逐渐下降
- 随着数据量增加，两者之间的差距会缩小

对于高偏差模型（high-bias model，例如 1 阶多项式），两种误差会迅速收敛至相同的高值，且增加数据量也无济于事。

### 步骤 5：正则化强度遍历

代码中还包含 `demo_regularization_sweep()` 函数，它固定使用高阶多项式（15 阶），并将岭正则化（Ridge regularization）强度从 0.001 遍历至 100。这从另一个角度展示了偏差-方差权衡（bias-variance tradeoff）：我们不再改变模型复杂度，而是调整约束强度。

def demo_regularization_sweep():
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    for alpha in alphas:
        results = bias_variance_decomposition([15], lam=alpha)
        r = results[15]
        print(f"alpha={alpha:.3f}  bias={r['bias_sq']:.4f}  var={r['variance']:.4f}")

当 alpha 较低时，15 阶多项式几乎不受约束。由于模型会拟合每个自助法样本中的噪声，方差占据主导地位。当 alpha 较高时，惩罚项过强，模型实际上退化为近似常值函数，此时偏差占据主导地位。最优的 alpha 值介于这两个极端之间。

这与改变多项式阶数时得到的 U 型曲线本质相同，但此处是通过连续参数而非离散参数进行控制。在实际应用中，正则化是控制该权衡的首选方法，因为它无需更改特征集即可实现细粒度控制。

## 实践应用

`sklearn` 提供了 `learning_curve` 和 `validation_curve` 函数，无需手动编写自举循环（bootstrap loops）即可自动执行这些模型诊断。

### 验证曲线：遍历模型复杂度

from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

degrees = list(range(1, 16))
train_scores_all = []
val_scores_all = []

for d in degrees:
    pipe = make_pipeline(PolynomialFeatures(d), Ridge(alpha=0.01))
    train_scores, val_scores = validation_curve(
        pipe, X, y, param_name="polynomialfeatures__degree",
        param_range=[d], cv=5, scoring="neg_mean_squared_error"
    )
    train_scores_all.append(-train_scores.mean())
    val_scores_all.append(-val_scores.mean())

这将直接为你生成偏差-方差权衡（bias-variance tradeoff）曲线。当验证集得分相对于训练集得分表现最差时，说明模型受方差（variance）主导；当两者得分均较差时，说明模型受偏差（bias）主导。

### 学习曲线：遍历训练集规模

from sklearn.model_selection import learning_curve

pipe = make_pipeline(PolynomialFeatures(5), Ridge(alpha=0.01))
train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="neg_mean_squared_error"
)
train_mse = -train_scores.mean(axis=1)
val_mse = -val_scores.mean(axis=1)

将 `train_mse` 和 `val_mse` 针对 `train_sizes` 绘制成图。曲线的形状能够揭示关于你模型的所有关键信息。

### 结合正则化遍历的交叉验证

from sklearn.model_selection import cross_val_score

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
for alpha in alphas:
    pipe = make_pipeline(PolynomialFeatures(10), Ridge(alpha=alpha))
    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    print(f"alpha={alpha:>7.3f}  MSE={-scores.mean():.4f} +/- {scores.std():.4f}")

此方法在固定模型复杂度的情况下遍历正则化（regularization）强度。你将观察到同样的偏差-方差权衡现象：较低的 `alpha` 值意味着高方差，较高的 `alpha` 值意味着高偏差。

### 综合应用：完整的诊断工作流

在实际操作中，建议按以下顺序执行这些诊断步骤：

1. 训练模型。计算训练误差和测试误差。
2. 如果两者均较高：说明存在偏差问题。跳至第 4 步。
3. 如果训练误差低但测试误差高：说明存在方差问题。绘制学习曲线以判断增加数据量是否有效。若无效，则引入正则化。
4. 生成验证曲线，遍历你的主要复杂度参数。寻找最佳平衡点（sweet spot）。
5. 在最佳平衡点处生成学习曲线。如果训练与验证误差的差距依然很大，则需要更多数据或加强正则化。
6. 使用 `cross_val_score` 尝试不同 `alpha` 值的 Ridge/Lasso 模型。选择交叉验证（cross-validation）误差最低的 `alpha` 值。

对于大多数表格型数据集（tabular datasets），该流程仅需 10-15 分钟的计算时间，却能节省数小时盲目调参的时间。

## 交付产物

本教程将生成以下文件：`outputs/prompt-model-diagnostics.md`

## 练习题

1. 在 `noise_std=0`（无噪声）的情况下运行分解。不可约误差（irreducible error）项会发生什么变化？最优复杂度（optimal complexity）是否会改变？

2. 将训练集大小从 30 增加到 300。这对方差分量（variance component）有何影响？最优多项式次数（optimal polynomial degree）是否会发生变化？

3. 在实验中加入 L2 正则化（L2 regularization，即岭回归 Ridge regression）。对于固定的高阶多项式（次数为 15），将 lambda 从 0 遍历到 100。绘制偏差平方（bias^2）和方差（variance）随 lambda 变化的函数曲线。

4. 将真实函数（true function）从多项式修改为 `sin(x)`。偏差-方差分解（bias-variance decomposition）会发生怎样的变化？是否仍然存在明确的最优次数？

5. 实现一个简单的自助聚合（bootstrap aggregating，简称 bagging）包装器：在自助样本（bootstrap samples）上训练 10 个模型并对预测结果取平均。证明该方法能在不显著增加偏差（bias）的情况下降低方差（variance）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 偏差 (Bias) | “模型太简单了” | 由错误假设引起的系统性误差。即平均模型预测值与真实值之间的差距。 |
| 方差 (Variance) | “模型过拟合了” | 由对训练数据敏感性引起的误差。即预测结果在不同训练集上的波动程度。 |
| 不可约误差 (Irreducible error) | “数据中的噪声” | 由真实数据生成过程中的随机性引起的误差。任何模型都无法消除它。 |
| 欠拟合 (Underfitting) | “学得不够” | 模型具有高偏差。即使在训练数据上也无法捕捉到真实模式。 |
| 过拟合 (Overfitting) | “死记硬背数据” | 模型具有高方差。它拟合了训练数据中无法泛化的噪声。 |
| 正则化 (Regularization) | “约束模型” | 添加惩罚项以降低模型复杂度，通过略微增加偏差来换取更低的方差。 |
| 双重下降 (Double descent) | “更多参数会有帮助” | 当模型容量远超插值阈值时，测试误差会再次下降。 |
| 模型复杂度 (Model complexity) | “模型的灵活程度” | 模型拟合任意模式的能力。由架构、特征或正则化控制。 |

## 延伸阅读

- [Hastie, Tibshirani, Friedman: Elements of Statistical Learning, Ch. 7](https://hastie.su.domains/ElemStatLearn/) —— 偏差-方差分解（bias-variance decomposition）的权威论述
- [Belkin et al., Reconciling modern machine learning practice and the bias-variance trade-off (2019)](https://arxiv.org/abs/1812.11118) —— 双重下降（double descent）相关论文
- [Nakkiran et al., Deep Double Descent (2019)](https://arxiv.org/abs/1912.02292) —— 基于训练轮次（epoch-wise）和样本量（sample-wise）的双重下降现象
- [Scott Fortmann-Roe: Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html) —— 清晰的可视化解释
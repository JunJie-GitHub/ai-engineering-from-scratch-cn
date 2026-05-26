# 超参数调优 (Hyperparameter Tuning)

> 超参数是训练开始前需要调节的“旋钮”。调节得当与否，决定了模型是平庸还是卓越。

**类型：** 构建
**语言：** Python
**前置条件：** 第二阶段，第11课（集成学习方法 (Ensemble Methods)）
**时长：** 约90分钟

## 学习目标

- 从零实现网格搜索 (Grid Search)、随机搜索 (Random Search) 和贝叶斯优化 (Bayesian Optimization)，并比较它们的样本效率 (Sample Efficiency)
- 解释当大多数超参数的有效维度 (Effective Dimensionality) 较低时，为什么随机搜索的表现优于网格搜索
- 使用代理模型 (Surrogate Model) 和采集函数 (Acquisition Function) 构建贝叶斯优化循环，以指导搜索过程
- 设计一种超参数调优策略，通过合理的交叉验证 (Cross-Validation) 避免在验证集 (Validation Set) 上过拟合

## 问题描述

你的梯度提升 (Gradient Boosting) 模型包含学习率、树的数量、最大深度、叶节点最小样本数、子采样比例和列采样比例。这共计六个超参数。如果每个超参数有5个合理的取值，那么网格将包含 5^6 = 15,625 种组合。训练每个组合需要10秒。尝试所有组合将耗费43小时的计算时间。

网格搜索是最直观的方法，但在大规模场景下却是最差的选择。随机搜索能以更少的计算量取得更好的效果。贝叶斯优化则通过从历史评估中学习，表现更为出色。了解该采用哪种策略，以及哪些超参数真正关键，能为你节省数天被浪费的 GPU 计算时间。

## 核心概念

### 参数（Parameters）与超参数（Hyperparameters）

参数是在训练过程中学习得到的（如权重、偏置、分裂阈值）。超参数则在训练开始前设定，用于控制学习过程。

| 超参数（Hyperparameter） | 控制内容 | 典型范围 |
|---------------|-----------------|---------------|
| 学习率（Learning rate） | 每次更新的步长 | 0.001 至 1.0 |
| 树的数量/轮数（Number of trees/epochs） | 训练时长 | 10 至 10,000 |
| 最大深度（Max depth） | 模型复杂度 | 1 至 30 |
| 正则化系数（Regularization, lambda） | 防止过拟合 | 0.0001 至 100 |
| 批次大小（Batch size） | 梯度估计的噪声 | 16 至 512 |
| Dropout 率（Dropout rate） | 丢弃的神经元比例 | 0.0 至 0.5 |

### 网格搜索（Grid Search）

网格搜索会评估指定值的所有组合。它虽然全面且易于理解，但其计算量会随着超参数数量的增加呈指数级增长。

Grid for 2 hyperparameters:

  learning_rate: [0.01, 0.1, 1.0]
  max_depth:     [3, 5, 7]

  Evaluations: 3 x 3 = 9 combinations

  (0.01, 3)  (0.01, 5)  (0.01, 7)
  (0.1,  3)  (0.1,  5)  (0.1,  7)
  (1.0,  3)  (1.0,  5)  (1.0,  7)

网格搜索存在一个根本缺陷：如果某个超参数起关键作用而另一个无关紧要，那么大部分评估都会被浪费。在 9 次评估中，你只能得到重要参数的 3 个不同取值。

### 随机搜索（Random Search）

随机搜索从概率分布中采样超参数，而非使用固定网格。在同样 9 次评估的预算下，你可以获得每个超参数的 9 个不同取值。

flowchart LR
    subgraph Grid Search
        G1[3 unique learning rates]
        G2[3 unique max depths]
        G3[9 total evaluations]
    end

    subgraph Random Search
        R1[9 unique learning rates]
        R2[9 unique max depths]
        R3[9 total evaluations]
    end

为什么随机搜索优于网格搜索（Bergstra & Bengio, 2012）：

- 大多数超参数的有效维度较低。对于特定问题，通常只有 6 个超参数中的 1-2 个起关键作用。
- 网格搜索会在不重要的维度上浪费评估次数。
- 在相同预算下，随机搜索能更密集地覆盖重要维度。
- 进行 60 次随机试验后，你有 95% 的概率找到一个距离最优解 5% 以内的点（前提是搜索空间中存在该点）。

### 贝叶斯优化（Bayesian Optimization）

随机搜索会忽略历史结果。它无法“学习”到较高的学习率会导致模型发散，也无法发现深度为 3 的模型始终优于深度为 10 的模型。贝叶斯优化则利用过去的评估结果来决定下一步的搜索方向。

flowchart TD
    A[Define search space] --> B[Evaluate initial random points]
    B --> C[Fit surrogate model to results]
    C --> D[Use acquisition function to pick next point]
    D --> E[Evaluate the model at that point]
    E --> F{Budget exhausted?}
    F -->|No| C
    F -->|Yes| G[Return best hyperparameters found]

两个核心组件：

**代理模型（Surrogate model）：** 一种评估成本较低的模型（通常为高斯过程），用于近似昂贵的目标函数。它能在搜索空间的任意点同时提供预测值和不确定性估计。

**采集函数（Acquisition function）：** 通过平衡利用（在已知优良点附近搜索）与探索（在不确定性高的区域搜索）来决定下一步的评估位置。常见选择包括：

- **期望改进（Expected Improvement, EI）：** 在该点预期能比当前最优值提升多少？
- **置信上界（Upper Confidence Bound, UCB）：** 预测值加上不确定性的倍数。较高的 UCB 意味着该点要么潜力巨大，要么尚未被探索。
- **改进概率（Probability of Improvement, PI）：** 该点优于当前最优值的概率是多少？

与随机搜索相比，贝叶斯优化通常能以少 2-5 倍的评估次数找到更优的超参数。拟合代理模型的开销与实际模型训练相比微乎其微。

### 早停法（Early Stopping）

并非每次训练都需要完整运行。如果某个配置在 10 个轮次（epochs）后表现明显不佳，就应停止训练并转向下一个配置。这就是超参数搜索语境下的早停法。

策略：
- **基于耐心值（Patience-based）：** 如果验证损失（validation loss）连续 N 个轮次未改善，则停止。
- **中位数剪枝（Median pruning）：** 如果当前试验的中间结果差于同阶段已完成试验的中位数，则停止。
- **Hyperband：** 为大量配置分配较小的初始预算，随后逐步增加表现最佳配置的预算。

Hyperband 尤为高效。它首先为 81 个配置各分配 1 个轮次，保留排名前三分之一的配置，为其分配 3 个轮次，再次保留前三分之一，依此类推。这种方法找到优良配置的速度比用完整预算评估所有配置快 10-50 倍。

### 学习率调度器（Learning Rate Schedulers）

学习率几乎总是最重要的超参数。调度器会在训练过程中动态调整学习率，而非保持固定。

| 调度器（Scheduler） | 公式 | 适用场景 |
|-----------|---------|-------------|
| 阶梯衰减（Step decay） | 每 N 个轮次乘以 0.1 | 经典卷积神经网络（CNN）训练 |
| 余弦退火（Cosine annealing） | lr * 0.5 * (1 + cos(pi * t / T)) | 现代默认选择 |
| 预热+衰减（Warmup + decay） | 线性增长后接余弦衰减 | Transformer 模型 |
| 单周期（One-cycle） | 在一个周期内先增后减 | 快速收敛 |
| 平台期衰减（Reduce on plateau） | 指标停滞时按因子降低 | 安全默认选项 |

### 超参数重要性（Hyperparameter Importance）

并非所有超参数都同等重要。针对随机森林（Probst et al., 2019）和梯度提升（gradient boosting）的研究揭示了以下一致规律：

**高重要性：**
- 学习率（始终优先调整）
- 估计器数量/训练轮数（建议使用早停法代替手动调参）
- 正则化强度

**中等重要性：**
- 最大深度/层数
- 叶节点最小样本数/权重衰减（weight decay）
- 子采样比例

**低重要性：**
- 最大特征数（针对随机森林）
- 具体激活函数的选择
- 批次大小（在合理范围内）

优先调整重要超参数，其余保持默认值即可。

### 实用策略

flowchart TD
    A[Start with defaults] --> B[Coarse random search: 20-50 trials]
    B --> C[Identify important hyperparameters]
    C --> D[Fine random or Bayesian search: 50-100 trials in narrowed space]
    D --> E[Final model with best hyperparameters]
    E --> F[Retrain on full training data]

具体工作流：

1. **从库的默认值开始。** 这些默认值由经验丰富的从业者设定，通常已能达到 80% 的性能。
2. **粗略随机搜索。** 设置较宽的搜索范围，进行 20-50 次试验。利用早停法快速终止表现不佳的运行。
3. **分析结果。** 哪些超参数与性能相关？据此缩小搜索空间。
4. **精细搜索。** 在缩小后的空间内使用贝叶斯优化或针对性随机搜索。进行 50-100 次试验。
5. **使用找到的最佳超参数在全部训练数据上重新训练模型。**

### 交叉验证集成（Cross-Validation Integration）

仅在单一验证集划分上调参存在风险。找到的最佳超参数可能会过拟合到特定的验证折（fold）。嵌套交叉验证（Nested cross-validation）通过双层循环解决此问题：

- **外层循环（评估）：** 将数据划分为训练+验证集和测试集。报告无偏的性能指标。
- **内层循环（调参）：** 将训练+验证集进一步划分为训练集和验证集。用于寻找最佳超参数。

flowchart TD
    D[Full Dataset] --> O1[Outer Fold 1: Test]
    D --> O2[Outer Fold 2: Test]
    D --> O3[Outer Fold 3: Test]
    D --> O4[Outer Fold 4: Test]
    D --> O5[Outer Fold 5: Test]

    O1 --> I1[Inner 5-fold CV on remaining data]
    I1 --> T1[Best hyperparams for fold 1]
    T1 --> E1[Evaluate on outer test fold 1]

    O2 --> I2[Inner 5-fold CV on remaining data]
    I2 --> T2[Best hyperparams for fold 2]
    T2 --> E2[Evaluate on outer test fold 2]

每个外层折都会独立寻找其最佳超参数。外层得分是对模型泛化性能的无偏估计。

使用 `sklearn`：

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

inner_cv = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid={
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "n_estimators": [50, 100, 200],
    },
    cv=5,
    scoring="neg_mean_squared_error",
)

outer_scores = cross_val_score(
    inner_cv, X, y, cv=5, scoring="neg_mean_squared_error"
)

print(f"Nested CV MSE: {-outer_scores.mean():.4f} +/- {outer_scores.std():.4f}")

这种方法计算成本较高（5 个外层折 × 5 个内层折 × 27 个网格点 = 675 次模型拟合），但能提供可信的性能评估。建议在论文中报告最终结果或决策风险较高时使用。

### 实用建议

**优先调整学习率。** 对于基于梯度的方法，它始终是最重要的超参数。学习率设置不当会使其他调参工作失去意义。先将其他超参数固定为默认值，首先对学习率进行扫描。

**对学习率和正则化使用对数均匀分布（log-uniform distributions）。** 0.001 与 0.01 之间的差异，其重要程度等同于 0.1 与 1.0 之间的差异。线性搜索会在较大数值端浪费预算。

**使用早停法代替调整 `n_estimators`。** 对于提升树和神经网络，将 `n_estimators` 或轮数设得较高，交由早停法决定何时停止。这相当于从搜索空间中移除了一个超参数。

**预算分配。** 将 60% 的调参预算用于最重要的 2 个超参数，剩余 40% 用于其他参数。前 2 个参数通常决定了大部分的性能差异。

**尺度至关重要。** 切勿在对数尺度上搜索批次大小（16、32、64 即可）。务必在对数尺度上搜索学习率。搜索分布应与超参数影响模型的方式相匹配。

| 模型类型 | 核心超参数 | 推荐搜索方法 | 预算 |
|-----------|--------------------|--------------------|--------|
| 随机森林（Random Forest） | `n_estimators`, `max_depth`, `min_samples_leaf` | 随机搜索，50 次试验 | 低（训练快） |
| 梯度提升（Gradient Boosting） | `learning_rate`, `n_estimators`, `max_depth` | 贝叶斯优化，100 次试验 + 早停法 | 中 |
| 神经网络（Neural Network） | `learning_rate`, `weight_decay`, `batch_size` | 贝叶斯或随机搜索，100+ 次试验 | 高（训练慢） |
| 支持向量机（SVM） | `C`, `gamma`（RBF 核） | 对数尺度网格搜索，25-50 次试验 | 低（2 个参数） |
| Lasso/Ridge | `alpha` | 对数尺度一维搜索，20 次试验 | 极低 |
| XGBoost | `learning_rate`, `max_depth`, `subsample`, `colsample` | 贝叶斯优化，100-200 次试验 + 早停法 | 中 |

**拿不准时：** 采用随机搜索，试验次数设为超参数数量的 2 倍（例如 6 个超参数至少进行 12 次试验）。你会惊讶地发现，50 次试验的随机搜索往往能击败精心设计的网格搜索。

## 构建

### 步骤 1：从零实现网格搜索 (Grid Search)

文件 `code/tuning.py` 中的代码从零实现了网格搜索 (Grid Search)、随机搜索 (Random Search) 以及一个简单的贝叶斯优化器 (Bayesian Optimizer)。

def grid_search(model_fn, param_grid, X_train, y_train, X_val, y_val):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = -float("inf")
    best_params = None
    n_evals = 0

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)
        n_evals += 1

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_evals

### 步骤 2：从零实现随机搜索 (Random Search)

def random_search(model_fn, param_distributions, X_train, y_train,
                  X_val, y_val, n_iter=50, seed=42):
    rng = np.random.RandomState(seed)
    best_score = -float("inf")
    best_params = None

    for _ in range(n_iter):
        params = {k: sample(v, rng) for k, v in param_distributions.items()}
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_iter

### 步骤 3：贝叶斯优化 (Bayesian Optimization)（简化版）

核心思想：将高斯过程 (Gaussian Process) 拟合到已观测的（超参数，得分）数据对上，然后使用采集函数 (Acquisition Function) 来决定下一步的搜索位置。

class SimpleBayesianOptimizer:
    def __init__(self, search_space, n_initial=5):
        self.search_space = search_space
        self.n_initial = n_initial
        self.X_observed = []
        self.y_observed = []

    def _kernel(self, x1, x2, length_scale=1.0):
        dists = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / length_scale ** 2)

    def _fit_gp(self, X_new):
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        y_mean = y_obs.mean()
        y_centered = y_obs - y_mean

        K = self._kernel(X_obs, X_obs) + 1e-4 * np.eye(len(X_obs))
        K_star = self._kernel(X_new, X_obs)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
        mu = K_star @ alpha + y_mean

        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-6)

        return mu, var

    def _expected_improvement(self, mu, var, best_y):
        sigma = np.sqrt(var)
        z = (mu - best_y) / (sigma + 1e-10)
        ei = sigma * (z * norm_cdf(z) + norm_pdf(z))
        return ei

    def suggest(self):
        if len(self.X_observed) < self.n_initial:
            return sample_random(self.search_space)

        candidates = [sample_random(self.search_space) for _ in range(500)]
        X_cand = np.array([to_vector(c) for c in candidates])
        mu, var = self._fit_gp(X_cand)
        ei = self._expected_improvement(mu, var, max(self.y_observed))
        return candidates[np.argmax(ei)]

    def observe(self, params, score):
        self.X_observed.append(to_vector(params))
        self.y_observed.append(score)

高斯过程代理模型 (GP Surrogate) 会在每个候选点给出两个值：预测得分 (`mu`) 和不确定性 (`var`)。期望改进 (Expected Improvement) 策略会对这两者进行权衡：它更倾向于模型预测得分较高**或**不确定性较高的点。在优化初期，大多数点的不确定性都很高，因此优化器会进行广泛探索；随着迭代进行，它会逐渐聚焦于最有潜力的区域。

### 步骤 4：对比所有方法

在同一个合成目标函数 (Synthetic Objective) 上运行这三种方法并进行对比。此对比使用了一个简化的包装器，它直接调用目标函数（无需训练模型），因此其 API 与上述基于模型的实现有所不同：

def synthetic_objective(params):
    lr = params["learning_rate"]
    depth = params["max_depth"]
    return -(np.log10(lr) + 2) ** 2 - (depth - 4) ** 2 + 10

param_grid = {
    "learning_rate": [0.001, 0.01, 0.1, 1.0],
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
}

grid_best = None
grid_score = -float("inf")
grid_history = []
for combo in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), combo))
    score = synthetic_objective(params)
    grid_history.append((params, score))
    if score > grid_score:
        grid_score = score
        grid_best = params

param_dist = {
    "learning_rate": ("log_float", 0.001, 1.0),
    "max_depth": ("int", 2, 8),
}

rand_best = None
rand_score = -float("inf")
rand_history = []
rng = np.random.RandomState(42)
for _ in range(28):
    params = {k: sample(v, rng) for k, v in param_dist.items()}
    score = synthetic_objective(params)
    rand_history.append((params, score))
    if score > rand_score:
        rand_score = score
        rand_best = params

optimizer = SimpleBayesianOptimizer(param_dist, n_initial=5)
bayes_history = []
for _ in range(28):
    params = optimizer.suggest()
    score = synthetic_objective(params)
    optimizer.observe(params, score)
    bayes_history.append((params, score))
bayes_score = max(s for _, s in bayes_history)

print(f"{'Method':<20} {'Best Score':>12} {'Evaluations':>12}")
print("-" * 50)
print(f"{'Grid Search':<20} {grid_score:>12.4f} {len(grid_history):>12}")
print(f"{'Random Search':<20} {rand_score:>12.4f} {len(rand_history):>12}")
print(f"{'Bayesian Opt':<20} {bayes_score:>12.4f} {len(bayes_history):>12}")

在相同的计算预算下，贝叶斯优化通常能最快找到最佳得分，因为它不会在明显较差的区域浪费评估次数。随机搜索的探索范围比网格搜索更广。只有当超参数数量极少且能够承担穷举成本时，网格搜索才会占据优势。

## Use It

### Optuna 实战

Optuna 是进行专业级超参数调优（Hyperparameter Tuning）的推荐库。它开箱即用地支持剪枝（Pruning）、分布式搜索和可视化。

import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    n_est = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 10)

    model = GradientBoostingRegressor(
        learning_rate=lr,
        n_estimators=n_est,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return mean_squared_error(y_val, model.predict(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best MSE: {study.best_value:.4f}")

Optuna 的核心特性：
- 使用 `suggest_float(..., log=True)` 搜索最适合对数尺度的参数（如学习率、正则化系数）
- 使用 `suggest_int` 处理整数参数
- 使用 `suggest_categorical` 处理离散选项
- 内置 `MedianPruner` 用于提前终止表现不佳的试验（Trial）
- 使用 `study.trials_dataframe()` 进行结果分析

### 结合剪枝的 Optuna

剪枝（Pruning）能够提前终止没有希望的试验，从而节省大量计算资源。以下是标准用法：

import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-4, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }

    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3,
                             scoring="neg_mean_squared_error")
    mean_score = -scores.mean()

    trial.report(mean_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return mean_score

pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=200)

当某个试验的中间值劣于同一步骤下所有已完成试验的中位数时，`MedianPruner` 会终止该试验。实现剪枝需要调用 `trial.report()` 上报中间指标，并通过 `trial.should_prune()` 检查是否应停止试验。设置 `n_startup_trials=10` 可确保在剪枝机制生效前，至少有 10 个试验完整运行完毕。这通常能节省 40% 到 60% 的总计算量。

### sklearn 内置的调优器

对于快速实验，sklearn 提供了 `GridSearchCV`、`RandomizedSearchCV` 和 `HalvingRandomSearchCV`：

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    "learning_rate": loguniform(1e-4, 0.5),
    "max_depth": randint(2, 10),
    "n_estimators": randint(50, 500),
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(),
    param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV MSE: {-search.best_score_:.4f}")

使用 scipy 的 `loguniform` 处理学习率和正则化参数，使用 `randint` 处理整数超参数。`n_jobs=-1` 标志会利用所有 CPU 核心进行并行计算。

### 超参数调优中的常见错误

**预处理导致的数据泄露（Data Leakage）。** 如果在交叉验证（Cross-Validation）之前对整个数据集拟合缩放器，验证折（Validation Fold）的信息就会泄露到训练集中。务必将预处理步骤放入 `Pipeline` 中，确保它仅在训练折上进行拟合。

**对验证集过拟合。** 运行数千次试验实际上等同于在验证集上进行训练。对于最终的性能评估，请使用嵌套交叉验证（Nested Cross-Validation），或者预留一个在调优过程中绝不触碰的独立测试集。

**搜索范围过窄。** 如果最优值出现在搜索空间的边界上，说明你的搜索范围不够广。真正的最优值可能位于范围之外。务必检查最佳参数是否落在边界处。

**忽略参数间的交互效应。** 在提升算法（Boosting）中，学习率与估计器数量之间存在强烈的交互作用。较低的学习率通常需要更多的估计器。独立调优这两个参数的效果往往不如联合调优。

**未对迭代模型使用早停（Early Stopping）。** 对于梯度提升和神经网络，应将 `n_estimators` 或 `epochs` 设置为较大的值，并启用早停机制。这严格优于将迭代次数作为超参数进行调优。

## 练习

1. 在相同的总预算（例如 50 次评估）下运行网格搜索（Grid Search）和随机搜索（Random Search）。比较两者找到的最佳得分。使用不同的随机种子重复实验 10 次。随机搜索胜出的频率是多少？

2. 从零开始实现 Hyperband 算法。初始设置 81 个配置（Configurations），每个配置训练 1 个轮次（Epoch）。在每一轮中保留排名前 1/3 的配置，并将其预算增至三倍。将总计算量（所有配置的所有轮次之和）与让 81 个配置都跑满完整预算的情况进行对比。

3. 在第 11 课的梯度提升（Gradient Boosting）实现中添加一个学习率调度器（Learning Rate Scheduler，采用余弦退火 Cosine Annealing 策略）。与使用固定学习率相比，它是否有帮助？

4. 使用 Optuna 在真实数据集（例如 scikit-learn 的乳腺癌数据集）上对 `RandomForestClassifier` 进行调参。使用 `optuna.visualization.plot_param_importances(study)` 查看哪些超参数（Hyperparameters）最为关键。其结果是否与本节中的重要性排名一致？

5. 实现一个简单的采集函数（Acquisition Function，如期望改进 Expected Improvement），并演示探索与利用（Exploration vs Exploitation）的权衡。绘制代理模型（Surrogate Model）的均值与不确定性曲线，并展示 EI 下一步选择评估的位置。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------------|------|
| 超参数 (Hyperparameter) | “你手动选择的设置” | 在训练前设定的值，用于控制学习过程，而非从数据中学习得到 |
| 网格搜索 (Grid Search) | “尝试所有组合” | 在指定的参数网格上进行穷举搜索。计算成本呈指数级增长。 |
| 随机搜索 (Random Search) | “只是随机采样” | 从概率分布中采样超参数。相比网格搜索，能更有效地覆盖重要维度。 |
| 贝叶斯优化 (Bayesian Optimization) | “智能搜索” | 利用目标函数的代理模型来决定下一步的评估位置，从而平衡探索与利用。 |
| 代理模型 (Surrogate Model) | “廉价的近似” | 一种模型（通常为高斯过程），用于根据已观测的评估结果来近似计算成本高昂的目标函数。 |
| 采集函数 (Acquisition Function) | “下一步该看哪里” | 通过平衡期望改进与不确定性来为候选点打分。EI（期望改进）和 UCB（置信上界）是常见选择。 |
| 早停法 (Early Stopping) | “别浪费时间了” | 当验证集性能不再提升时，提前终止训练。 |
| Hyperband | “配置的淘汰赛” | 自适应资源分配：以较小的预算启动大量配置，保留表现最好的并逐步增加其预算。 |
| 学习率调度器 (Learning Rate Scheduler) | “训练期间动态调整学习率” | 一种在训练过程中调整学习率的函数，旨在实现更好的收敛效果。 |

## 延伸阅读

- [Bergstra & Bengio: Random Search for Hyper-Parameter Optimization (2012)](https://jmlr.org/papers/v13/bergstra12a.html) -- 证明了随机搜索（Random Search）优于网格搜索（Grid Search）的论文
- [Snoek et al., Practical Bayesian Optimization of Machine Learning Algorithms (2012)](https://arxiv.org/abs/1206.2944) -- 面向机器学习（Machine Learning）的贝叶斯优化（Bayesian Optimization）方法
- [Li et al., Hyperband: A Novel Bandit-Based Approach (2018)](https://jmlr.org/papers/v18/16-558.html) -- 介绍 Hyperband 算法的论文
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902) -- 介绍 Optuna 框架的论文
- [Probst et al., Tunability: Importance of Hyperparameters (2019)](https://jmlr.org/papers/v20/18-444.html) -- 探讨哪些超参数（Hyperparameters）更为关键的论文
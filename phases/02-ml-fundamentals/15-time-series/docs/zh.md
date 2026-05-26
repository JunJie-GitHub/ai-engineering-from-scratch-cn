# 时间序列基础 (Time Series Fundamentals)

> 过去的表现确实能够预测未来的结果——前提是你先检验平稳性 (Stationarity)。

**类型：** 构建 (Build)
**语言：** Python
**前置要求：** 第二阶段，第 01-09 课
**预计耗时：** 约 90 分钟

## 学习目标

- 将时间序列 (Time Series) 分解为趋势 (Trend)、季节性 (Seasonality) 和残差 (Residual) 分量，并检验其平稳性 (Stationarity)
- 实现滞后特征 (Lag Features) 和滚动统计量 (Rolling Statistics)，将时间序列转化为监督学习 (Supervised Learning) 问题
- 构建前向验证 (Walk-Forward Validation) 框架，防止未来数据泄露 (Data Leakage) 到训练集中
- 解释为何随机划分训练集/测试集 (Random Train/Test Split) 不适用于时间序列，并展示其与正确的时间划分 (Temporal Split) 之间的性能差距

## 问题所在

你手头有一批按时间顺序排列的数据：每日销售额、每小时气温、每分钟 CPU 使用率、每周股价。你的目标是预测下一个时间点的值、下一周或下一个季度的表现。

你习惯性地拿出标准的机器学习 (Machine Learning) 工具箱：随机划分训练集/测试集、交叉验证 (Cross-Validation)、输入特征矩阵、输出预测结果。然而，这里的每一步都是错误的。

时间序列打破了标准机器学习所依赖的基本假设。样本之间并非相互独立——今天的气温取决于昨天。随机划分会导致未来信息泄露到过去。那些在回测 (Backtest) 中表现优异的特征，在实际生产环境中往往会失效，因为它们依赖的模式会随时间发生漂移。

一个在随机交叉验证中能达到 95% 准确率的模型，在采用正确的时间序列评估方法时，准确率可能骤降至 55%。这种差异绝非细枝末节的技术问题，而是“纸上谈兵”的模型与“真正能在生产环境中落地”的模型之间的本质区别。

本课程将涵盖时间序列分析的基础知识：时间数据有何独特之处、如何客观公正地评估模型，以及如何将时间序列转化为标准机器学习模型可处理的特征。

## 核心概念

### 时间序列的独特之处

传统机器学习（Machine Learning, ML）假设数据满足独立同分布（independent and identically distributed, i.i.d.）——即每个样本都从相同的分布中独立抽取，且与其他样本无关。时间序列数据同时违背了这两点：

- **不独立。** 今天的股价取决于昨天的表现。本周的销售额与上周相关。
- **非同分布。** 数据分布会随时间推移发生偏移。12月的销售数据与3月的截然不同。

这些违背并非无关紧要。它们会彻底改变你构建特征、评估模型以及选择算法的方式。

flowchart LR
    subgraph IID["Standard ML (i.i.d.)"]
        direction TB
        S1[Sample 1] ~~~ S2[Sample 2]
        S2 ~~~ S3[Sample 3]
    end
    subgraph TS["Time Series (not i.i.d.)"]
        direction LR
        T1[t=1] --> T2[t=2]
        T2 --> T3[t=3]
        T3 --> T4[t=4]
    end

    style S1 fill:#dfd
    style S2 fill:#dfd
    style S3 fill:#dfd
    style T1 fill:#ffd
    style T2 fill:#ffd
    style T3 fill:#ffd
    style T4 fill:#ffd

在传统机器学习中，样本是可互换的，打乱顺序不会改变任何结果。而在时间序列中，顺序就是一切，打乱顺序会彻底破坏信号。

### 时间序列的组成成分

每个时间序列都是以下成分的组合：

flowchart TD
    A[Observed Time Series] --> B[Trend]
    A --> C[Seasonality]
    A --> D[Residual/Noise]

    B --> E[Long-term direction: up, down, flat]
    C --> F[Repeating patterns: daily, weekly, yearly]
    D --> G[Random variation after removing trend and seasonality]

- **趋势（Trend）**：长期走向。例如年收入增长10%，或全球气温持续上升。
- **季节性（Seasonality）**：固定间隔内重复出现的模式。例如零售业在12月销量激增，或空调使用量在7月达到峰值。
- **残差（Residual）**：剔除趋势和季节性后剩余的部分。如果残差看起来像白噪声（white noise），说明分解过程已成功提取了有效信号。

### 平稳性（Stationarity）

如果时间序列的统计特性（均值、方差、自相关性）不随时间变化，则该序列是平稳的。大多数预测方法都基于平稳性假设。

**为什么重要：** 非平稳序列的均值会发生漂移。在1月数据上训练的模型学到的均值，与2月实际呈现的均值不同，这会导致模型产生系统性偏差。

**如何检验：** 计算滑动窗口内的滚动均值（rolling mean）和滚动标准差（rolling standard deviation）。如果它们发生漂移，则序列是非平稳的。

**如何解决：** 差分（Differencing）。不再对原始值建模，而是对连续值之间的变化量进行建模：

diff[t] = value[t] - value[t-1]

如果一次差分未能使序列平稳，可再次应用（二阶差分）。大多数现实世界的时间序列最多只需要两次差分。

**示例：**

原始序列：[100, 102, 106, 112, 120]
一阶差分：  [2, 4, 6, 8]（仍呈上升趋势）
二阶差分：  [2, 2, 2]（保持恒定——已平稳）

原始序列具有二次趋势。一阶差分将其转化为线性趋势，二阶差分则使其变为水平。在实际应用中，通常不需要超过两次差分。

**正式检验：** 增强迪基-富勒检验（Augmented Dickey-Fuller, ADF）是检验平稳性的标准统计方法。其原假设（null hypothesis）为“序列是非平稳的”。若 p 值低于 0.05，则可拒绝原假设，认为序列平稳。我们不会从零实现 ADF 检验（因为它需要渐近分布表），但代码中的滚动统计方法提供了一种实用的可视化检验手段。

### 自相关性（Autocorrelation）

自相关性衡量的是时间 t 的值与时间 t-k（过去 k 步）的值之间的相关程度。自相关函数（Autocorrelation Function, ACF）会绘制出每个滞后阶数（lag）k 对应的自相关系数。

**ACF 能告诉你：**
- 序列的“记忆”有多长。如果 ACF 在滞后 5 阶后降至零，说明 5 步之前的数据已无关紧要。
- 是否存在季节性。如果 ACF 在滞后 12 阶（月度数据）处出现峰值，则表明存在年度季节性。
- 需要创建多少个滞后特征。应使用 ACF 降至可忽略水平之前的滞后阶数。

**偏自相关函数（Partial Autocorrelation Function, PACF）** 剔除了间接相关性。如果今天与3天前的数据相关仅仅是因为两者都与昨天相关，那么滞后3阶的 PACF 将为零，而 ACF 则不为零。

### 滞后特征：将时间序列转化为监督学习

传统机器学习模型需要特征矩阵 X 和目标变量 y。时间序列只提供单列数值，而滞后特征（lag features）正是连接两者的桥梁。

以序列 [10, 12, 14, 13, 15] 为例，创建滞后1阶和滞后2阶特征：

| lag_2 | lag_1 | target |
|-------|-------|--------|
| 10    | 12    | 14     |
| 12    | 14    | 13     |
| 14    | 13    | 15     |

现在你得到了一个标准的回归问题。任何机器学习模型（线性回归、随机森林、梯度提升）都可以通过滞后特征来预测目标值。

你可以构建的其他特征包括：
- **滚动统计量（Rolling statistics）**：过去 k 个值的均值、标准差、最小值、最大值
- **日历特征（Calendar features）**：星期几、月份、是否节假日、是否周末
- **差分值（Differenced values）**：相对于上一步的变化量
- **扩展统计量（Expanding statistics）**：累积均值、累积求和
- **比率特征（Ratio features）**：当前值 / 滚动均值（反映偏离近期平均值的程度）
- **交互特征（Interaction features）**：`lag_1 * day_of_week`（工作日对趋势动量的影响）

**需要多少个滞后阶数？** 参考自相关函数。如果 ACF 在滞后 10 阶内显著，则至少使用 10 个滞后特征。如果存在周季节性，应包含滞后 7 阶（可能还有 14 阶）。更多的滞后阶数能为模型提供更长的历史上下文，但也会增加待拟合的特征数量，从而提升过拟合风险。

**目标对齐陷阱。** 在构建滞后特征时，目标变量必须是时间 t 的值，而所有特征只能使用时间 t-1 或更早的值。如果不小心将时间 t 的值也作为特征输入，你将得到一个“完美预测器”——以及一个完全无用的模型。这是时间序列特征工程中最常见的错误。

### 滚动验证（Walk-Forward Validation）

这是本课程最重要的概念。传统的 k 折交叉验证（k-fold cross-validation）会随机将样本划分到训练集和测试集。对于时间序列而言，这种做法会导致未来信息泄露。

flowchart TD
    subgraph WRONG["Random Split (WRONG)"]
        direction LR
        W1[Jan] --> W2[Mar]
        W2 --> W3[Feb]
        W3 --> W4[May]
        W4 --> W5[Apr]
        style W1 fill:#fdd
        style W3 fill:#fdd
        style W5 fill:#fdd
        style W2 fill:#dfd
        style W4 fill:#dfd
    end

    subgraph RIGHT["Walk-Forward (CORRECT)"]
        direction LR
        R1["Train: Jan-Mar"] --> R2["Test: Apr"]
        R3["Train: Jan-Apr"] --> R4["Test: May"]
        R5["Train: Jan-May"] --> R6["Test: Jun"]
        style R1 fill:#dfd
        style R2 fill:#fdd
        style R3 fill:#dfd
        style R4 fill:#fdd
        style R5 fill:#dfd
        style R6 fill:#fdd
    end

滚动验证步骤：
1. 使用截至时间 t 的数据进行训练
2. 预测时间 t+1 的值（或多步预测 t+1 至 t+k）
3. 将窗口向前滑动
4. 重复上述过程

每个测试折仅包含所有训练数据之后的数据，杜绝了未来信息泄露。这能为你提供模型部署后真实性能的可靠预估。

**扩展窗口（Expanding window）** 使用全部历史数据进行训练（窗口逐渐扩大）。**滑动窗口（Sliding window）** 使用固定大小的训练窗口（窗口整体滑动）。当你认为早期数据依然具有参考价值时，使用扩展窗口；当环境发生变化且旧数据会产生干扰时，使用滑动窗口。

### ARIMA 模型直觉

ARIMA（自回归积分滑动平均模型）是经典的时间序列模型，包含三个组成部分：

- **AR（自回归，Autoregressive）**：基于历史值进行预测。AR(p) 使用最近的 p 个值。
- **I（差分/积分，Integrated）**：通过差分实现平稳性。I(d) 表示进行 d 阶差分。
- **MA（移动平均，Moving Average）**：基于历史预测误差进行预测。MA(q) 使用最近的 q 个误差项。

ARIMA(p, d, q) 将三者结合。你可以通过 ACF/PACF 分析或自动化搜索（如 auto-ARIMA）来确定 p、d、q 的值。

我们不会从零实现 ARIMA——它涉及的数值优化超出了本课程的范围。关键在于理解每个组件的作用，以便你能解读 ARIMA 的结果，并知道何时该使用它。

### 方法选择指南

| 方法 | 适用场景 | 处理季节性 | 处理外部特征 |
|----------|---------|-------------------|------------------------|
| 滞后特征 + 机器学习 | 包含大量外部特征的表格数据 | 需结合日历特征 | 支持 |
| ARIMA | 单变量序列、短期预测 | SARIMA 变体支持 | 不支持（ARIMAX 仅支持有限外部变量） |
| 指数平滑（Exponential smoothing） | 简单的趋势 + 季节性 | 支持（Holt-Winters） | 不支持 |
| Prophet | 商业预测、节假日效应 | 支持（傅里叶项） | 有限支持 |
| 神经网络（LSTM、Transformer） | 长序列、多序列预测 | 自动学习 | 支持 |

对于大多数实际问题，**滞后特征 + 梯度提升（Gradient Boosting）** 是最强的起点。它能自然处理外部特征，不要求数据平稳，且易于调试。

### 预测跨度与策略

单步预测（Single-step forecasting）仅预测下一个时间步。多步预测（Multi-step forecasting）则预测多个时间步。主要有三种策略：

**递归法（Recursive）**：预测下一步，并将该预测值作为下一步的输入。方法简单但误差会累积——因为每次预测都依赖上一次的预测结果，导致错误不断叠加。

**直接法（Direct）**：为每个预测跨度训练独立的模型。例如模型1预测 t+1，模型5预测 t+5。不会产生误差累积，但每个模型的训练样本较少，且模型间无法共享信息。

**多输出法（Multi-output）**：训练一个能同时输出所有预测跨度的模型。能在不同跨度间共享信息，但需要模型支持多输出（或自定义损失函数）。

对于大多数实际问题，建议短跨度（1-5步）从递归法开始，长跨度则使用直接法。

### 时间序列常见错误

| 错误 | 原因 | 解决方法 |
|---------|---------------|-----------|
| 随机划分训练/测试集 | 传统机器学习的习惯 | 使用滚动验证或按时间顺序划分 |
| 使用未来特征 | 误将时间 t 的特征纳入 | 审查每个特征的时间对齐情况 |
| 对季节性过拟合 | 模型死记硬背了日历模式 | 在测试集中预留完整的季节性周期 |
| 忽略量级变化 | 收入翻倍但模式未变 | 对百分比变化建模，而非绝对值 |
| 滞后特征过多 | “历史数据越多越好”的误区 | 使用 ACF 确定有效的滞后阶数 |
| 未进行差分 | “模型自己能搞定”的假设 | 树模型可处理趋势；线性模型需要平稳性 |

## 构建实现

文件 `code/time_series.py` 中的代码从零开始实现了核心构建模块。

### 滞后特征生成器 (Lag Feature Creator)

def make_lag_features(series, n_lags):
    n = len(series)
    X = np.full((n, n_lags), np.nan)
    for lag in range(1, n_lags + 1):
        X[lag:, lag - 1] = series[:-lag]
    valid = ~np.isnan(X).any(axis=1)
    return X[valid], series[valid]

该函数将一维时间序列转换为特征矩阵，其中每一行包含最近 `n_lags` 个值作为特征，当前值作为目标变量。

### 向前滚动交叉验证 (Walk-Forward Cross-Validation)

def walk_forward_split(n_samples, n_splits=5, min_train=50):
    assert min_train < n_samples, "min_train must be less than n_samples"
    step = max(1, (n_samples - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n_samples)
        if train_end >= n_samples:
            break
        yield slice(0, train_end), slice(train_end, test_end)

每次划分都确保训练数据严格位于测试数据之前。随着每一折（fold）的推进，训练窗口会不断扩展。

### 简单自回归模型 (Simple Autoregressive Model)

纯粹的自回归（Autoregressive, AR）模型本质上就是对滞后特征进行线性回归：

class SimpleAR:
    def __init__(self, n_lags=5):
        self.n_lags = n_lags
        self.weights = None
        self.bias = None

    def fit(self, series):
        X, y = make_lag_features(series, self.n_lags)
        # Solve via normal equations
        X_b = np.column_stack([np.ones(len(X)), X])
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.bias = theta[0]
        self.weights = theta[1:]
        return self

这在概念上与第 02 课中的线性回归完全一致，只是将其应用于同一变量的时间滞后版本。

### 平稳性检验 (Stationarity Check)

该代码通过计算滚动统计量（Rolling Statistics），从视觉和数值两个维度评估序列的平稳性：

def check_stationarity(series, window=50):
    rolling_mean = np.array([
        series[max(0, i - window):i].mean()
        for i in range(1, len(series) + 1)
    ])
    rolling_std = np.array([
        series[max(0, i - window):i].std()
        for i in range(1, len(series) + 1)
    ])
    return rolling_mean, rolling_std

如果滚动均值发生漂移或滚动标准差发生变化，则表明该序列是非平稳的（Non-Stationary）。此时应进行差分处理（Differencing）并重新检验。

代码还会通过比较序列的前半部分和后半部分来检验平稳性。如果两部分的均值差异超过半个标准差，或方差比率超过 2 倍，则该序列将被标记为非平稳。

### 自相关性 (Autocorrelation)

def autocorrelation(series, max_lag=20):
    n = len(series)
    mean = series.mean()
    var = series.var()
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        cov = np.mean((series[:n-k] - mean) * (series[k:] - mean))
        acf[k] = cov / var if var > 0 else 0
    return acf

## 实际应用

使用 `sklearn` 时，你可以直接将滞后特征（lag features）与任何回归器（regressor）配合使用：

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_lag_features(series, n_lags=10)

for train_idx, test_idx in walk_forward_split(len(X)):
    model = Ridge(alpha=1.0)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])

对于 ARIMA 模型，请使用 `statsmodels`：

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_series, order=(5, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=30)

`time_series.py` 中的代码演示了这两种方法，并使用前向验证（walk-forward validation）对它们进行了比较。

### sklearn 的 TimeSeriesSplit

`sklearn` 提供了 `TimeSeriesSplit` 类，用于实现前向验证：

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

这与我们从零实现的 `walk_forward_split` 功能等效，但已集成到 `sklearn` 的交叉验证（cross-validation）框架中。你可以将其与 `cross_val_score` 配合使用：

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
print(f"Mean score: {scores.mean():.4f} +/- {scores.std():.4f}")

### 评估指标

时间序列预测（time series forecasting）使用回归指标，但需要结合时间上下文进行考量：

- **平均绝对误差 (Mean Absolute Error)：** `|y_true - y_pred|` 的平均值。易于用原始单位进行解释。“平均而言，预测值偏差为 3.2 度。”
- **均方根误差 (Root Mean Squared Error)：** 均方误差的平方根。相比 MAE，它对较大误差的惩罚更重。当大误差比多个小误差更不可接受时，应使用该指标。
- **平均绝对百分比误差 (Mean Absolute Percentage Error)：** `|error / true_value| * 100` 的平均值。具有尺度无关性，适用于在不同序列之间进行比较。但当真实值为零时该指标无定义。
- **朴素基线对比 (Naive baseline comparison)：** 始终应与简单的基线模型进行对比。季节性朴素基线（seasonal naive baseline）会预测上一个周期（如昨天或上周）的值。如果你的模型无法优于朴素基线，则说明存在问题。

### 滚动特征

代码演示了如何将滚动统计量（rolling statistics）（如 7 天和 14 天窗口内的均值、标准差、最小值和最大值）添加到滞后特征中。这些特征为模型提供了近期趋势和波动性的信息，而仅靠滞后特征无法捕捉到这些信息。

例如，如果滚动均值呈上升趋势，则表明存在向上趋势。如果滚动标准差增大，则表明波动性正在加剧。这类模式是基于树的模型（tree-based models）能够学习到的，而线性模型（linear models）则无法捕捉。

## 部署上线

本课时将生成：
- `outputs/prompt-time-series-advisor.md` -- 用于构建时间序列问题（time series problem）的提示词（prompt）
- `code/time_series.py` -- 滞后特征（lag feature）、前向滚动验证（walk-forward validation）、自回归模型（AR model）、平稳性检验（stationarity check）

### 必须超越的基线（Baseline）

在构建任何模型之前，请先建立基线：

1. **最近值/持久性基线（Last value / persistence）。** 预测明天的值与今天相同。对于许多序列而言，这个基线出乎意料地难以超越。
2. **季节性朴素基线（Seasonal naive）。** 预测今天的值与上周（或去年）同一天相同。如果你的模型无法超越此基线，说明它除了季节性之外没有学到任何有用的模式。
3. **移动平均基线（Moving average）。** 预测过去 k 个值的平均值。它能平滑噪声，但无法捕捉突变。

如果你精心设计的机器学习（ML）模型输给了季节性朴素基线，那说明代码存在缺陷。最常见的原因包括：特征中存在未来数据泄露（future leakage）、评估方法错误，或者该序列本身就是真正随机且不可预测的。

### 实用建议

1. **从绘图开始。** 在进行任何建模之前，先绘制原始序列图。观察趋势（trend）、季节性（seasonality）、异常值（outlier）以及结构性断点（structural break，即行为模式的突然变化）。30秒的视觉检查通常比一小时的自动化分析能提供更多信息。

2. **先差分，后建模。** 如果序列存在明显趋势，在构建滞后特征之前先对其进行差分处理。基于树的模型（tree-based model）可以处理趋势，但线性模型（linear model）不行，而差分操作通常只有好处没有坏处。

3. **至少预留一个完整的季节性周期。** 如果存在周季节性，测试集至少需要包含完整的一周。如果是月季节性，则至少需要完整的一个月。否则，你将无法评估模型是否成功捕捉到了季节性模式。

4. **在生产环境中持续监控。** 随着现实世界的变化，时间序列模型的性能会随时间推移而下降。采用滚动方式跟踪预测误差。当误差开始上升时，使用近期数据重新训练模型。

5. **警惕机制转换（regime change）。** 使用疫情前数据训练的模型无法预测疫情后的行为。可以将已知机制转换的指标作为特征加入，或者使用会遗忘旧数据的滑动窗口（sliding window）。

6. **对偏态序列进行对数变换（log-transform）。** 收入、价格和计数数据通常呈右偏分布。取对数可以稳定方差，并将乘法模式转化为加法模式，从而便于线性模型处理。在对数空间中进行预测，然后通过指数运算还原为原始单位。

## 练习

1. **平稳性实验（Stationarity experiment）**。生成一个具有线性趋势的序列。使用滚动统计量（rolling statistics）检验其平稳性。应用一阶差分（first differencing）。再次检验。对于二次趋势，需要进行几轮差分才能使其平稳？

2. **滞后阶数选择（Lag selection）**。在季节性序列（周期=7）上计算自相关函数（ACF）。哪些滞后阶数的自相关性最高？仅使用这些特定的滞后阶数（而非连续阶数）构建滞后特征（lag features）。与使用滞后阶数 1 至 7 相比，模型准确率是否有所提升？

3. **前向验证与随机划分（Walk-forward vs random split）**。基于滞后特征训练岭回归（Ridge regression）模型。分别使用随机 80/20 划分和前向验证（walk-forward validation）进行评估。随机划分会高估多少模型性能？

4. **特征工程（Feature engineering）**。在滞后特征的基础上，添加滚动均值（rolling mean，窗口大小=7）、滚动标准差（rolling std，窗口大小=7）以及星期几特征。使用前向验证，对比添加与不添加这些额外特征时的模型准确率。

5. **多步预测（Multi-step forecasting）**。修改自回归模型（AR model），使其预测未来 5 步而非 1 步。对比两种策略：(a) 递归策略（recursive）：预测一步后，将该预测值作为下一步的输入；(b) 直接策略（direct）：为每个预测步长分别训练独立的模型。哪种策略的准确率更高？

## 关键术语

| 术语 | 通俗理解 | 严格定义 |
|------|----------------|----------------------|
| 平稳性（Stationarity） | “统计特性不随时间变化” | 均值、方差和自相关结构在时间上保持恒定的序列 |
| 差分（Differencing） | “减去相邻值” | 计算 y[t] - y[t-1] 以消除趋势并实现平稳性 |
| 自相关函数（Autocorrelation, ACF） | “序列与自身的相关性” | 时间序列与其滞后版本之间的相关性，作为滞后阶数的函数 |
| 偏自相关函数（Partial Autocorrelation, PACF） | “仅保留直接相关性” | 剔除所有较短滞后阶数的影响后，滞后 k 阶的自相关性 |
| 滞后特征（Lag features） | “将历史值作为输入” | 使用 y[t-1], y[t-2], ..., y[t-k] 作为特征来预测 y[t] |
| 前向验证（Walk-forward validation） | “尊重时间顺序的交叉验证” | 评估方法，其中训练数据在时间上始终早于测试数据 |
| ARIMA模型 | “经典的时间序列模型” | 自回归积分滑动平均模型（AutoRegressive Integrated Moving Average）：结合历史值（AR）、差分（I）和历史误差（MA） |
| 季节性（Seasonality） | “重复的日历模式” | 时间序列中与日历周期（日、周、年）相关的规律性、可预测的循环 |
| 趋势（Trend） | “长期走向” | 序列水平随时间持续上升或下降的现象 |
| 扩展窗口（Expanding window） | “使用全部历史数据” | 一种前向验证方法，训练集在每次划分时不断累积扩大 |
| 滑动窗口（Sliding window） | “固定大小的历史数据” | 一种前向验证方法，训练集为固定长度的窗口并随时间向前滑动 |

## 延伸阅读

- [Hyndman and Athanasopoulos, Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) -- 时间序列预测（Time Series Forecasting）领域最佳的免费教材
- [scikit-learn Time Series Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) -- scikit-learn 的前向滚动分割器（Walk-Forward Splitter）
- [statsmodels ARIMA docs](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) -- 附带模型诊断（Model Diagnostics）功能的自回归积分滑动平均模型（ARIMA）实现
- [Makridakis et al., The M5 Competition (2022)](https://www.sciencedirect.com/science/article/pii/S0169207021001874) -- 展示机器学习（Machine Learning）方法与统计方法（Statistical Methods）对比的大规模预测竞赛
# 机器学习统计学 (Statistics for Machine Learning)

> 统计学能让你判断模型是真正有效，还是仅仅碰巧运气好。

**类型：** 构建 (Build)
**语言：** Python
**前置要求：** 第一阶段，第 06 课（概率与分布 (Probability and Distributions)）、第 07 课（贝叶斯定理 (Bayes' Theorem)）
**时长：** 约 120 分钟

## 学习目标

- 从零开始计算描述性统计量 (Descriptive Statistics)、皮尔逊/斯皮尔曼相关系数 (Pearson/Spearman Correlation) 以及协方差矩阵 (Covariance Matrices)
- 执行假设检验 (Hypothesis Tests)（如 t 检验 (t-test)、卡方检验 (Chi-squared Test)），并正确解读 p 值 (p-values) 与置信区间 (Confidence Intervals)
- 使用自助法重采样 (Bootstrap Resampling) 为任意指标构建置信区间，且无需对数据分布进行假设
- 借助效应量 (Effect Size) 指标，区分统计显著性 (Statistical Significance) 与实际显著性 (Practical Significance)

## 问题场景

你训练了两个模型。模型 A 在测试集上的得分为 0.87，模型 B 为 0.89。于是你部署了模型 B。三周后，生产环境的指标反而不如从前。发生了什么？

模型 B 实际上并未优于模型 A。这 0.02 的差异只是噪声。你的测试集太小，或者方差过高，亦或是两者兼有。你部署的不过是被包装成性能提升的随机波动。

这种情况屡见不鲜。Kaggle 排行榜的剧烈洗牌、无法复现的学术论文、仅凭几百个样本就宣布胜者的 A/B 测试。其根本原因始终如一：有人跳过了统计学验证。

统计学为你提供了区分信号与噪声的工具。它能告诉你差异是否真实存在、你该有多大的把握，以及在信任某个结果之前需要多少数据。每一个机器学习流水线 (ML Pipeline)、每一次模型对比、每一项实验都离不开统计学。没有它，你只是在盲目猜测。

## 核心概念

### 描述性统计（Descriptive Statistics）：总结你的数据

在构建任何模型之前，你需要先了解数据的分布情况。描述性统计将数据集压缩为几个关键数值，以捕捉其整体形态。

**集中趋势度量（Measures of Central Tendency）**回答“中心在哪里？”

Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.

均值（Mean）是平衡点，中位数（Median）是中间位置。当两者偏离时，说明你的数据分布是偏态的（Skewed）。收入分布通常均值远大于中位数（受亿万富翁影响呈右偏）。训练过程中的损失分布（Loss Distribution）通常均值远小于中位数（受简单样本影响呈左偏）。

**离散程度度量（Measures of Spread）**回答“数据有多分散？”

Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.

**百分位数（Percentiles）**将排序后的数据划分为100个等份。第25百分位数（Q1）表示有25%的数值低于该点。第50百分位数即为中位数。第75百分位数为Q3。

For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)

在机器学习（Machine Learning, ML）中，你通常关注推理延迟（Inference Latency）、预测置信度分布以及误差分布的百分位数。一个平均误差很低但P99误差极差的模型，在安全关键型应用中可能毫无用处。

**样本统计与总体统计（Sample vs Population Statistics）。** 在计算样本方差时，分母应使用 (n-1) 而非 n。这称为贝塞尔校正（Bessel's Correction）。它用于补偿样本均值并非真实总体均值这一事实。若分母为 n，你会系统性地低估真实方差；使用 (n-1) 则能得到无偏估计。

Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)

实际应用中：如果 n 很大（数千个样本），差异可忽略不计；如果 n 很小（几十个样本），则必须考虑。

### 相关性（Correlation）：变量如何协同变化

相关性衡量两个变量之间线性关系的强度和方向。

**皮尔逊相关系数（Pearson Correlation Coefficient）**衡量线性关联：

r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]

皮尔逊相关系数假设变量间呈线性关系，且两个变量大致服从正态分布（Normal Distribution）。它对异常值非常敏感。单个极端值就可能使 r 从 0.1 飙升至 0.9。

**斯皮尔曼等级相关系数（Spearman Rank Correlation）**衡量单调关联：

1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.

**何时使用哪种方法：**

Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.

**黄金法则：** 相关性不等于因果性（Correlation does not imply causation）。冰淇淋销量与溺水死亡人数相关，是因为两者都在夏季上升。模型的准确率与参数量相关，但增加参数并不会自动提升准确率（参见：过拟合 Overfitting）。

### 协方差矩阵（Covariance Matrix）

两个变量之间的协方差（Covariance）衡量它们如何共同变化：

Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement

对于 d 个特征，协方差矩阵 C 是一个 d x d 的矩阵，其中 C[i][j] = Cov(feature_i, feature_j)。对角线元素 C[i][i] 即为每个特征的方差。

C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances

**与主成分分析（Principal Component Analysis, PCA）的联系。** PCA 对协方差矩阵进行特征分解（Eigendecomposition）。特征向量即为主成分（最大方差方向），特征值则告诉你每个成分捕获了多少方差。这正是第10课的内容，但现在你明白了为什么协方差矩阵是正确的分解对象：它编码了数据中所有成对的线性关系。

**与相关性的联系。** 相关矩阵（Correlation Matrix）是标准化变量（每个变量除以其标准差）的协方差矩阵。相关性对协方差进行了归一化，使所有值落在 [-1, 1] 区间内。

### 假设检验（Hypothesis Testing）

假设检验是在不确定性下进行决策的框架。你从一个假设出发，收集数据，并判断数据是否与该假设一致。

**基本设定：**

Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A

**P值（P-value）**是在原假设（H0）为真的前提下，观察到当前数据或更极端数据的概率。它**不是** H0 为真的概率。这是统计学中最常见的误解。

p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.

**置信区间（Confidence Intervals）**给出参数的合理取值范围：

95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.

置信区间的宽度反映了估计的精确度。区间越宽，不确定性越高；区间越窄，估计越精确（但如果数据存在偏差，精确并不意味着准确）。

### T检验（T-test）

T检验用于比较均值，有多种变体。

**单样本T检验（One-sample T-test）：** 总体均值是否与假设值不同？

t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1

**双样本T检验（独立）（Two-sample T-test (Independent)）：** 两组均值是否不同？

t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.

**配对T检验（Paired T-test）：** 当测量值成对出现时（同一模型在相同数据划分上评估）：

Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0

在机器学习中，配对T检验很常见：你在相同的10折交叉验证（Cross-validation）划分上运行两个模型，并逐对比较它们的得分。

### 卡方检验（Chi-squared Test）

卡方检验用于检查观测频数是否与期望频数匹配。适用于分类数据（Categorical Data）。

chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.

### 面向机器学习模型的A/B测试（A/B Testing）

机器学习中的A/B测试与网页A/B测试不同。模型比较面临特定挑战：

1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.

**操作流程：**

1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance

### 统计显著性与实际显著性（Statistical vs Practical Significance）

一个结果可能具有统计显著性，但在实际中毫无意义。只要有足够的数据，即使微不足道的差异也会变得统计显著。

Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.

**效应量（Effect Size）**量化差异的大小，且独立于样本量：

Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect

务必同时报告P值和效应量。P值告诉你差异是否真实存在，效应量告诉你差异是否重要。

### 多重比较问题（Multiple Comparison Problem）

当你检验多个假设时，部分结果会因偶然性而呈现“显著”。如果在 alpha = 0.05 下检验20个指标，即使没有任何真实效应，你也预期会出现1次假阳性（False Positive）。

P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.

**邦费罗尼校正（Bonferroni Correction）：** 将 alpha 除以检验次数。

Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.

在机器学习中，当你在多个指标上比较模型、测试大量超参数配置或在多个数据集上评估时，这一点尤为重要。

### 自助法（Bootstrap Methods）

自助法通过有放回地重采样数据来估计统计量的抽样分布（Sampling Distribution）。无需对底层分布做任何假设。

**算法流程：**

1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution

**自助法置信区间（百分位数法）：**

Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]

**为什么自助法对机器学习很重要：**

- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.

**用于模型比较的自助法：**

1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant

这比配对T检验更稳健，因为它不依赖任何分布假设。

### 参数检验与非参数检验（Parametric vs Non-parametric Tests）

**参数检验（Parametric Tests）**假设数据服从特定分布（通常为正态分布）：

t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality

**非参数检验（Non-parametric Tests）**不做分布假设：

Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)

**何时使用非参数检验：**

- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions

**何时使用参数检验：**

- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)

在机器学习实验中，样本量通常较小（5折或10折交叉验证），因此像威尔科克森符号秩检验（Wilcoxon Signed-rank Test）这样的非参数检验通常比T检验更合适。

### 中心极限定理（Central Limit Theorem, CLT）：实际意义

中心极限定理指出，随着样本量 n 增大，样本均值的分布将趋近于正态分布，无论总体分布如何。

If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.

**为什么这对机器学习很重要：**

1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model

**中心极限定理不能做什么：**

- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).

### 机器学习论文中常见的统计学错误

1. **在训练集上进行测试。** 这必然导致过拟合。务必保留模型在训练期间从未见过的数据作为测试集。

2. **缺乏置信区间。** 仅报告单一的准确率数值而不提供不确定性范围，会导致结果无法复现和验证。

3. **忽略多重比较。** 测试50种配置并直接报告最佳结果而不进行校正，会大幅推高假阳性率。

4. **混淆统计显著性与实际显著性。** 在0.01%的准确率提升上得到0.001的P值毫无实际意义。

5. **在类别不平衡数据上使用准确率。** 在负类占比99%的数据集上达到99%的准确率，意味着模型什么都没学到。应使用精确率（Precision）、召回率（Recall）、F1分数或AUC。

6. **挑拣指标（Cherry-picking）。** 仅报告模型表现最好的指标。诚实的评估应报告所有相关指标。

7. **在训练/测试划分间泄露信息。** 在划分前进行归一化，或使用未来数据预测过去。

8. **测试集过小且无方差估计。** 仅在100个样本上评估并声称提升了2%，这通常是噪声而非有效信号。

9. **在数据不独立时假设独立。** 同一患者的多张医学影像、同一文档中的多个句子。组内观测值是相关的。

10. **P值操纵（P-hacking）。** 不断尝试不同的检验方法、数据子集或排除标准，直到得到 p < 0.05。这种结果只是搜索过程的产物，而非真实发现。

## 构建实现

你将实现以下内容：

1. **从零实现描述性统计 (Descriptive Statistics)**（均值 (Mean)、中位数 (Median)、众数 (Mode)、标准差 (Standard Deviation)、百分位数 (Percentile)、四分位距 (IQR)）
2. **相关性函数 (Correlation Functions)**（皮尔逊相关系数 (Pearson Correlation) 与斯皮尔曼相关系数 (Spearman Correlation)，含协方差矩阵 (Covariance Matrix)）
3. **假设检验 (Hypothesis Tests)**（单样本 t 检验 (One-sample t-test)、双样本 t 检验 (Two-sample t-test)、卡方检验 (Chi-squared Test)）
4. **自助法置信区间 (Bootstrap Confidence Intervals)**（适用于任意统计量，无需分布假设）
5. **A/B 测试模拟器 (A/B Test Simulator)**（生成数据、执行测试、检查第一类错误 (Type I Error) 与第二类错误 (Type II Error)）
6. **统计显著性与实际显著性对比演示 (Statistical vs Practical Significance Demo)**（展示大样本量如何使任何结果都呈现“显著性”）

全部从零开始实现，仅使用 `math` 和 `random` 模块。不使用 numpy 或 scipy。

## 关键术语

| 术语 | 定义 |
|---|---|
| 均值 | 所有数值之和除以数量。对异常值敏感。 |
| 中位数 | 排序后数据的中间值。对异常值具有稳健性。 |
| 标准差 | 方差的平方根。以原始单位衡量数据的离散程度。 |
| 百分位数 | 数据中低于给定百分比的数值。 |
| 四分位距 | 第三四分位数 (Q3) 减去第一四分位数 (Q1)。表示中间 50% 数据的分布范围。 |
| 皮尔逊相关系数 | 衡量两个变量之间的线性相关性。取值范围为 [-1, 1]。 |
| 斯皮尔曼相关系数 | 基于秩次衡量变量间的单调相关性。 |
| 协方差矩阵 | 包含所有特征两两之间协方差的矩阵。 |
| 零假设 | 默认假设，即不存在效应或差异。 |
| p 值 | 在零假设成立的前提下，观测到当前或更极端数据的概率。 |
| 置信区间 | 在给定置信水平下，参数的合理取值范围。 |
| t 检验 | 检验均值是否存在显著差异。基于 t 分布。 |
| 卡方检验 | 检验观测频数与期望频数是否存在显著差异。 |
| 效应量 | 差异的大小，独立于样本量。常用科恩 d 值 (Cohen's d)。 |
| 邦费罗尼校正 | 将显著性阈值除以检验次数，以控制假阳性。 |
| 自助法 | 通过有放回重采样来估计抽样分布。 |
| 第一类错误 | 假阳性。零假设为真时错误地拒绝它。 |
| 第二类错误 | 假阴性。零假设为假时未能拒绝它。 |
| 统计功效 | 正确拒绝错误零假设的概率。功效 = 1 - 第二类错误率。 |
| 中心极限定理 | 随着样本量增大，样本均值将收敛于正态分布。 |
| 参数检验 | 假设数据服从特定分布（通常为正态分布）。 |
| 非参数检验 | 不依赖分布假设。基于秩次或符号进行检验。 |
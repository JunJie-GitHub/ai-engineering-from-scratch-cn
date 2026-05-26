---
name: 技能-统计检验
description: 为比较机器学习模型与评估实验选择合适的统计检验方法
version: 1.0.0
phase: 1
lesson: 15
tags: [统计学, 假设检验, 模型比较]
---

# 机器学习中的统计检验 (Statistical Testing)

在比较模型、进行 A/B 实验或验证结果时，如何选择合适的检验方法。

## 决策清单

1. 你要比较什么？均值 (Means)、比例 (Proportions)、分布 (Distributions) 还是相关性 (Correlations)？
2. 有多少个组？单样本与参考值对比、两组对比，还是多组对比？
3. 观测值是配对的（相同的测试集、相同的折）还是独立的？
4. 数据是否服从正态分布 (Normal Distribution)？如果样本量 n < 30 且明显非正态，请使用非参数检验 (Non-parametric Test)。
5. 数据是连续型 (Continuous)、有序型 (Ordinal) 还是分类型 (Categorical)？
6. 你要运行多少次检验？如果超过一次，请应用多重检验校正 (Multiple Testing Correction)。

## 决策树

Comparing means?
  Two groups?
    Paired (same data splits)? --> Paired t-test (or Wilcoxon signed-rank if non-normal)
    Independent? --> Welch's t-test (or Mann-Whitney U if non-normal)
  Multiple groups?
    Paired? --> Repeated measures ANOVA (or Friedman test)
    Independent? --> One-way ANOVA (or Kruskal-Wallis)

Comparing proportions?
  Two groups? --> Chi-squared test or Fisher's exact test (small n)
  Multiple groups? --> Chi-squared test

Comparing distributions?
  Is one distribution a reference? --> Kolmogorov-Smirnov test
  Are both empirical? --> Two-sample KS test

Measuring association?
  Both continuous, roughly normal? --> Pearson correlation
  Ordinal or non-normal? --> Spearman rank correlation
  Categorical x Categorical? --> Chi-squared test of independence

Running many tests?
  Apply Bonferroni correction: alpha_adjusted = alpha / number_of_tests
  Or use Holm-Bonferroni (less conservative, still controls family-wise error)

## 何时使用每种检验方法

| 检验方法 | 数据类型 | 假设条件 | 机器学习应用场景 |
|---|---|---|---|
| 配对 t 检验 (Paired t-test) | 连续型，配对 | 差值服从正态分布 | 在相同的 k 折交叉验证 (k-fold Cross-Validation) 划分上比较 2 个模型 |
| Wilcoxon 符号秩检验 (Wilcoxon signed-rank test) | 连续型/有序型，配对 | 无（非参数） | 比较 2 个模型，折数 k 较小（5-10 折） |
| Welch t 检验 (Welch's t-test) | 连续型，独立 | 大致服从正态分布 | 在两个独立数据集上比较模型 |
| Mann-Whitney U 检验 (Mann-Whitney U test) | 连续型/有序型，独立 | 无 | 比较延迟 (Latency) 分布 |
| 方差分析 (ANOVA) | 连续型，3 组及以上 | 正态分布，方差齐性 | 比较多种模型架构 (Model Architectures) |
| Kruskal-Wallis 检验 (Kruskal-Wallis test) | 连续型/有序型，3 组及以上 | 无 | 比较多个模型，指标非正态分布 |
| 卡方检验 (Chi-squared test) | 分类计数 | 期望频数 >= 5 | 比较类别分布、混淆矩阵 (Confusion Matrices) |
| Fisher 精确检验 (Fisher's exact test) | 分类计数 | 小样本 | 罕见事件比较 |
| KS 检验 (Kolmogorov-Smirnov test) | 连续型 | 无 | 检查预测值是否符合预期分布 |
| Bootstrap 置信区间 (Bootstrap CI) | 任意统计量 | 无 | 计算 AUC、F1 或任意指标的置信区间 (Confidence Interval) |
| McNemar 检验 (McNemar's test) | 配对二分类 | 无 | 在同一测试集上比较两个分类器 (Classifiers) |

## 模型比较操作指南

1. 在运行实验前，先定义评估指标 (Metric) 和显著性水平 (Significance Level, alpha = 0.05)。
2. 在相同的 k 折交叉验证划分上运行两个模型（k = 5 或 10）。
3. 收集配对得分：(a_1, b_1), (a_2, b_2), ..., (a_k, b_k)。
4. 计算差值：d_i = b_i - a_i。
5. 运行配对检验（k <= 10 时使用 Wilcoxon 检验，k > 10 或差值正态时使用配对 t 检验）。
6. 报告结果：p 值 (p-value)、均值差、95% 置信区间、效应量 (Effect Size, Cohen's d)。
7. 如果 p < alpha 且效应量具有实际意义，则说明差异是真实存在的，值得据此采取行动。

## 常见错误

- 当数据存在配对关系时使用独立检验。若两个模型均在相同的测试折（test folds）上进行了评估，则必须使用配对检验（paired test）。独立检验会忽略配对关系，从而损失统计功效（statistical power）。
- 仅报告 p < 0.05 而不提供效应量（effect size）。即使 0.1% 的准确率（accuracy）提升具有统计显著性，也往往不值得投入生产部署。务必计算 Cohen's d 或原始均值差（raw mean difference）。
- 在不同的测试集上比较模型。两个模型的测试集必须完全相同。使用不同的测试集会使比较失去意义。
- 进行 20 次比较并仅报告最佳结果，且未进行 Bonferroni 校正（Bonferroni correction）。在 alpha = 0.05 下进行 20 次检验，仅凭随机性就预期会出现 1 次假阳性（false positive）。
- 在类别不平衡数据（imbalanced data）上使用准确率。当多数类占比达 99% 时，一个简单分类器即可达到 99% 的准确率。建议改用 F1 分数（F1 score）、精确率-召回率 AUC（precision-recall AUC）或马修斯相关系数（Matthews correlation coefficient）。
- 将交叉验证折（cross-validation folds）视为独立样本。由于各折共享训练数据，这违反了独立性假设（independence assumption）。校正重采样 t 检验（corrected resampled t-test）专门用于处理此问题。

## 快速参考：效应量解读

| Cohen's d | 解读 |
|---|---|
| 0.2 | 小效应 |
| 0.5 | 中等效应 |
| 0.8 | 大效应 |
| > 1.0 | 极大效应 |

| 需报告的指标 | 原因 |
|---|---|
| p 值（p-value） | 差异是否真实存在？ |
| 置信区间（confidence interval） | 差异可能有多大？ |
| 效应量（Cohen's d） | 差异是否具有实际意义？ |
| 样本量（n 或 k 折） | 结果是否可信？ |
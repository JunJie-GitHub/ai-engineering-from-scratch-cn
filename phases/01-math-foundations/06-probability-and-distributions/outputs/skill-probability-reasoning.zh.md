---
name: skill-probability-reasoning
description: 为给定的机器学习问题选择合适的概率分布
version: 1.0.0
phase: 1
lesson: 6
tags: [概率, 分布, 建模]
---

# 概率分布选择 (Probability Distribution Selection)

在数据建模、设计损失函数 (loss functions) 或设置先验分布 (priors) 时，如何选择合适的分布。

## 决策清单

1. 结果是离散型 (discrete)（类别、计数）还是连续型 (continuous)（测量值、分数）？
2. 结果是有界的 (bounded)（例如 [0, 1]）还是无界的 (unbounded)？
3. 有多少种可能的结果？两种？k 种？还是无限种？
4. 数据是对称的 (symmetric) 还是偏斜的 (skewed)？
5. 事件是相互独立的 (independent) 还是相关的 (correlated)？
6. 你是在对速率、计数、比例还是测量值进行建模？

## 分布决策树

Is the variable discrete?
  Yes --> Only 2 outcomes? --> Bernoulli (p)
     |    k outcomes, one trial? --> Categorical (p1...pk)
     |    k outcomes, n trials? --> Multinomial (n, p1...pk)
     |    Count of successes in n trials? --> Binomial (n, p)
     |    Count of events per interval? --> Poisson (lambda)
     |    Count of trials until first success? --> Geometric (p)
     |    Count of trials until r successes? --> Negative Binomial (r, p)
  No --> Symmetric, bell-shaped? --> Normal (mu, sigma)
     |   Positive values, right-skewed? --> Log-normal or Exponential
     |   Bounded in [0, 1]? --> Beta (alpha, beta)
     |   Positive values, flexible shape? --> Gamma (alpha, beta)
     |   Time between events? --> Exponential (lambda)
     |   Heavy tails needed? --> Student's t (nu) or Cauchy
     |   Multivariate, bell-shaped? --> Multivariate Normal
     |   On a simplex (sums to 1)? --> Dirichlet (alpha)

## 将现实世界的机器学习场景映射到分布

| 场景 | 分布 | 参数 |
|---|---|---|
| 二分类输出 | 伯努利分布 (Bernoulli) | p = sigmoid(logit) |
| 多分类输出 | 类别分布 (Categorical) | p = softmax(logits) |
| 语言模型中的词元预测 | 基于词表的类别分布 | p 来自 softmax |
| 像素强度（归一化后） | Beta 分布或 [0, 1] 均匀分布 | 取决于图像统计特征 |
| 文档中的词频计数 | 泊松分布 (Poisson) | lambda = 平均词数 |
| 用户请求间隔时间 | 指数分布 (Exponential) | lambda = 请求速率 |
| 测量误差 | 正态分布 (Normal) | mu = 0, sigma 来自数据 |
| 权重初始化 | 正态分布或均匀分布 | Kaiming/Xavier 规则 |
| VAE 隐空间先验 | 标准正态分布 | mu = 0, sigma = 1 |
| 比例的贝叶斯先验 | Beta 分布 | alpha, beta 来自先验信念 |
| 类别权重的贝叶斯先验 | 狄利克雷分布 (Dirichlet) | alpha 向量 |
| 回归目标中的噪声 | 正态分布 | mu = 0, sigma 为估计值 |
| 对异常值鲁棒的回归 | 学生 t 分布 (Student's t) | 低自由度 |
| 持续时间/寿命建模 | 威布尔分布 (Weibull) 或 Gamma 分布 | 形状参数和尺度参数 |
| 文档主题分布 (LDA) | 狄利克雷分布 | alpha < 1 以实现稀疏性 |

## 分布误用的情况

- 当数据存在硬性下界（例如价格、距离）时使用正态分布 (Normal)。正态分布会为负值分配非零概率。应改用对数正态分布 (log-normal) 或 Gamma 分布。
- 当方差与均值不同时使用泊松分布 (Poisson)。泊松分布假设均值等于方差。若方差大于均值，应改用负二项分布 (Negative Binomial)。
- 将伯努利分布 (Bernoulli) 用于多分类问题。伯努利分布严格限于二分类。当 k > 2 时应使用类别分布 (Categorical)。
- 在观测值相关时假设独立性。时间序列、空间数据和分组数据均违反独立性假设。应改用自回归模型 (autoregressive models) 或分层模型 (hierarchical models)。

## 常见误区

- 混淆概率密度函数 (PDF) 值与概率。PDF 的值可以大于 1。概率是通过对 PDF 在某个区间上积分得到的。
- 忘记 softmax 输出的是类别概率，而非独立的伯努利概率。它们在构造上总和为 1。
- 在拥有领域知识时仍使用均匀先验 (uniform prior)。若选择得当，信息先验 (informative priors) 可以在不引入偏差的情况下降低方差。
- 将对数概率 (log-probabilities) 当作概率处理。对数概率始终为负值（或零）。它们的总和不等于 1。

## 快速参考：分布属性 (Distribution Properties)

| 分布 (Distribution) | 支撑集 (Support) | 均值 (Mean) | 方差 (Variance) | 关键特性 (Key property) |
|---|---|---|---|---|
| 伯努利分布 (Bernoulli(p)) | {0, 1} | p | p(1-p) | 最简单的离散分布 |
| 二项分布 (Binomial(n, p)) | {0..n} | np | np(1-p) | n 个伯努利试验之和 |
| 泊松分布 (Poisson(lam)) | {0, 1, 2, ...} | lam | lam | 均值等于方差 |
| 正态分布 (Normal(mu, s^2)) | (-inf, inf) | mu | s^2 | 给定均值/方差下的最大熵分布 |
| 指数分布 (Exponential(lam)) | [0, inf) | 1/lam | 1/lam^2 | 无记忆性 (Memoryless) |
| Beta分布 (Beta(a, b)) | [0, 1] | a/(a+b) | ab/((a+b)^2(a+b+1)) | 二项分布的共轭先验 |
| Gamma分布 (Gamma(a, b)) | (0, inf) | a/b | a/b^2 | 泊松分布的共轭先验 |
| 狄利克雷分布 (Dirichlet(alpha)) | 单纯形 (Simplex) | alpha_i/sum | (见公式) | 类别分布 (Categorical) 的共轭先验 |
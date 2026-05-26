# 概率与分布 (Probability and Distributions)

> 概率（Probability）是人工智能（AI）用于表达不确定性的语言。

**Type:** 学习
**Language:** Python
**Prerequisites:** 第一阶段，课程 01-04
**Time:** 约 75 分钟

## 学习目标

- 从零实现伯努利分布 (Bernoulli distribution)、分类分布 (Categorical distribution)、泊松分布 (Poisson distribution)、均匀分布 (Uniform distribution) 和正态分布 (Normal distribution) 的概率质量函数 (Probability Mass Functions, PMFs) 与概率密度函数 (Probability Density Functions, PDFs)
- 计算期望值 (Expected value) 与方差 (Variance)，并运用中心极限定理 (Central Limit Theorem) 解释为何高斯分布 (Gaussian distribution) 占据主导地位
- 构建 softmax 与 log-softmax 函数，并应用数值稳定性技巧（减去最大 logit (subtract max logit)）
- 根据 logits 计算交叉熵损失 (Cross-entropy loss)，并将其与负对数似然 (Negative log-likelihood) 建立联系

## 问题

分类器 (Classifier) 输出 `[0.03, 0.91, 0.06]`。语言模型 (Language Model) 从 50,000 个候选词中挑选下一个词。扩散模型 (Diffusion Model) 通过对学习到的分布进行采样来生成图像。所有这些都是概率 (Probability) 在实际中的体现。

模型做出的每一个预测都是一个概率分布 (Probability Distribution)。每一个损失函数 (Loss Function) 都在衡量预测分布与真实分布之间的差距。每一个训练步骤 (Training Step) 都在调整参数 (Parameters)，以使一个分布更接近另一个分布。如果不掌握概率知识，你将无法阅读任何一篇机器学习 (Machine Learning) 论文，无法调试 (Debug) 任何一个模型，也无法理解为什么你的训练损失 (Training Loss) 会变成 NaN。

## 概念

### Events, Sample Spaces, and Probability

The sample space S is the set of all possible outcomes. An event is a subset of the sample space. Probability maps events to numbers between 0 and 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Three axioms define all of probability:
1. P(A) >= 0 for any event A
2. P(S) = 1 (something always happens)
3. P(A or B) = P(A) + P(B) when A and B cannot both occur

Everything else (Bayes' theorem, expectations, distributions) follows from these three rules.

### Conditional Probability and Independence

P(A|B) is the probability of A given that B happened.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Two events are independent when knowing one tells you nothing about the other:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Coin flips are independent. Drawing cards without replacement is not.

### Probability Mass Functions vs Probability Density Functions

Discrete random variables have a probability mass function (PMF). Each outcome has a specific probability that you can read off directly.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Continuous random variables have a probability density function (PDF). The density at a single point is not a probability. Probability comes from integrating the density over an interval.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

This distinction matters in ML. Classification outputs are PMFs (discrete choices). VAE latent spaces use PDFs (continuous).

### Common Distributions

**Bernoulli:** one trial, two outcomes. Models binary classification.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:** one trial, k outcomes. Models multi-class classification (softmax output).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** all outcomes equally likely. Used for random initialization.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** the bell curve. Parameterized by mean (mu) and variance (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** counts of rare events in a fixed interval. Models event rates.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Expected Value and Variance

Expected value is the weighted average outcome.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Variance measures spread around the mean.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

In ML, expected value appears as the loss function (average loss over the data distribution). Variance tells you about model stability. High variance in gradients means noisy training.

### Joint and Marginal Distributions

A joint distribution P(X, Y) describes two random variables together.

Joint PMF example (X = weather, Y = umbrella):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

The marginal distribution sums out the other variable:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

The row and column totals in the table above are the marginals.

### Why the Normal Distribution Shows Up Everywhere

The Central Limit Theorem: the sum (or average) of many independent random variables converges to a normal distribution, regardless of the original distribution.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

This is why:
- Measurement errors are approximately normal (many small independent sources)
- Weight initializations in neural networks use normal distributions
- Gradient noise in SGD is approximately normal (sum of many sample gradients)
- The normal distribution is the maximum entropy distribution for a given mean and variance

### Log Probabilities

Raw probabilities cause numerical problems. Multiplying many small probabilities together quickly underflows to zero.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log probabilities fix this. Multiplications become additions.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Rules:
- log(a * b) = log(a) + log(b)
- log probabilities are always <= 0 (since 0 < P <= 1)
- More negative = less likely
- Cross-entropy loss is the negative log probability of the correct class

### Softmax as a Probability Distribution

Neural networks output raw scores (logits). Softmax converts them into a valid probability distribution.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

The softmax trick: subtract the max logit before exponentiating to prevent overflow.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax combines softmax and log for numerical stability. PyTorch uses this internally for cross-entropy loss.

### Sampling

Sampling means drawing random values from a distribution. In ML:
- Dropout randomly samples which neurons to zero out
- Data augmentation samples random transformations
- Language models sample the next token from the predicted distribution
- Diffusion models sample noise and progressively denoise

Sampling from arbitrary distributions requires techniques like inverse transform sampling, rejection sampling, or the reparameterization trick (used in VAEs).

## 构建它

### 步骤 1：概率基础 (Probability Basics)

import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")

### 步骤 2：从零实现概率质量函数 (Probability Mass Function, PMF) 与概率密度函数 (Probability Density Function, PDF)

def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)

### 步骤 3：期望值 (Expected Value) 与方差 (Variance)

def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")

### 步骤 4：分布采样 (Sampling from Distributions)

def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples

### 步骤 5：Softmax 函数与对数概率 (Log Probabilities)

def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]

### 步骤 6：中心极限定理 (Central Limit Theorem) 演示

def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages

### 步骤 7：可视化 (Visualization)

import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)

包含所有可视化图表的完整实现代码位于 `code/probability.py` 文件中。

## 使用方法

借助 NumPy 和 SciPy，上述所有操作均可通过单行代码 (one-liners) 实现：

import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")

你之前是从零开始构建这些功能的。现在，你已经清楚这些库函数调用 (library calls) 背后究竟在做什么了。

## 练习

1. 实现指数分布（Exponential Distribution）的逆变换采样（Inverse Transform Sampling）。通过生成 10,000 个样本值，并将直方图与真实的概率密度函数（Probability Density Function, PDF）进行对比来验证结果。

2. 为两个有偏骰子（Loaded Dice）构建联合分布表（Joint Distribution Table）。计算边缘分布（Marginal Distribution），并检验这两个骰子是否相互独立。

3. 计算一个 5 分类分类器的交叉熵损失（Cross-Entropy Loss），该分类器输出的对数几率（Logits）为 `[2.0, 0.5, -1.0, 3.0, 0.1]`，且正确类别的索引为 3。随后使用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，接收对数概率（Log Probabilities）列表，并返回最可能序列、总对数概率以及等效的原始概率。使用一个包含 50 个单词的句子进行测试，假设其中每个单词的概率均为 0.01。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 样本空间 (Sample Space) | “所有可能的情况” | 实验所有可能结果构成的集合 S |
| 概率质量函数 (Probability Mass Function, PMF) | “概率函数” | 给出每个离散结果确切概率的函数，且所有概率之和为 1 |
| 概率密度函数 (Probability Density Function, PDF) | “概率曲线” | 连续变量的密度函数。在某一区间上对其进行积分即可得到概率 |
| 条件概率 (Conditional Probability) | “在已知某条件下的概率” | P(A\|B) = P(A and B) / P(B)。贝叶斯思维与贝叶斯定理的基础 |
| 独立性 (Independence) | “它们互不影响” | P(A and B) = P(A) * P(B)。已知一个事件的发生无法提供关于另一个事件的任何信息 |
| 期望值 (Expected Value) | “平均值” | 所有结果按其概率加权求和的值。损失函数本质上就是一个期望值 |
| 方差 (Variance) | “数据的分散程度” | 偏离均值的平方的期望。高方差意味着估计结果噪声大、不稳定 |
| 正态分布 (Normal Distribution) | “钟形曲线” | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。得益于中心极限定理，它在各领域无处不在 |
| 中心极限定理 (Central Limit Theorem, CLT) | “平均值会趋于正态分布” | 无论原始分布如何，大量独立样本的均值都会收敛于正态分布 |
| 联合分布 (Joint Distribution) | “两个变量一起考虑” | P(X, Y) 描述了 X 和 Y 所有可能结果组合的概率 |
| 边缘分布 (Marginal Distribution) | “把另一个变量求和消去” | P(X) = sum_y P(X, Y)。从联合分布中还原出单个变量的分布 |
| 对数概率 (Log Probability) | “概率的对数” | log P(x)。将连乘转化为求和，防止长序列计算中的数值下溢 |
| Softmax 函数 (Softmax) | “将分数转化为概率” | softmax(z_i) = exp(z_i) / sum(exp(z_j))。将实数值的 logits 映射为有效的概率分布 |
| 交叉熵 (Cross-Entropy) | “损失函数” | -sum(p_true * log(p_predicted))。衡量两个分布之间的差异。值越小越好 |
| 未归一化分数 (Logits) | “模型的原始输出” | Softmax 之前的未归一化分数。名称源于 logistic 函数 |
| 采样 (Sampling) | “抽取随机值” | 根据概率分布生成数值。模型生成输出的具体方式 |

## 延伸阅读

- [3Blue1Brown：中心极限定理（Central Limit Theorem）究竟是什么？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 可视化证明为何均值会趋于正态分布
- [斯坦福 CS229 概率论复习](https://cs229.stanford.edu/section/cs229-prob.pdf) - 简明参考资料，涵盖此处所有内容及更多延伸知识
- [Log-Sum-Exp 技巧（Log-Sum-Exp Trick）](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为何数值稳定性（numerical stability）至关重要以及如何实现它
# 采样方法 (Sampling Methods)

> 采样 (Sampling) 是人工智能探索可能性空间的方式。

**类型：** 实战构建
**语言：** Python
**前置知识：** 第一阶段，第 06-07 课（概率论、贝叶斯定理 (Bayes' Theorem)）
**时长：** 约 120 分钟

## 学习目标

- 仅使用均匀分布随机数，从零实现逆累积分布函数 (Inverse CDF) 采样、拒绝采样 (Rejection Sampling) 和重要性采样 (Importance Sampling)
- 为语言模型词元 (Token) 生成构建温度采样 (Temperature Sampling)、Top-k 采样和 Top-p（核）采样 (Top-p / Nucleus Sampling)
- 解释重参数化技巧 (Reparameterization Trick)，并说明它为何能在变分自编码器 (VAEs) 中实现通过采样进行反向传播
- 运行 Metropolis-Hastings 马尔可夫链蒙特卡洛 (MCMC) 算法，从非归一化目标分布中进行采样

## 问题背景

语言模型处理完你的提示词 (Prompt) 后，会输出一个包含 50,000 个对数几率 (Logits) 的向量，词汇表中的每个词元 (Token) 对应一个值。现在它必须从中选出一个。该怎么选？

如果它总是选择概率最高的词元，每次生成的回复都会完全相同。确定性强，但枯燥乏味。如果完全均匀随机地选择，输出结果将是一堆乱码。真正的答案介于这两个极端之间，而这个“中间地带”正是由采样策略来控制的。

采样并不局限于文本生成。强化学习 (Reinforcement Learning) 通过采样轨迹来估计策略梯度 (Policy Gradients)。变分自编码器 (VAEs) 通过对学习到的分布进行采样，并在随机性中反向传播来学习潜在表示 (Latent Representations)。扩散模型 (Diffusion Models) 通过对噪声进行采样并迭代去噪来生成图像。蒙特卡洛方法 (Monte Carlo Methods) 用于估计没有解析解的积分。马尔可夫链蒙特卡洛 (MCMC) 算法则用于探索无法穷举的高维后验分布 (Posterior Distributions)。

每一个生成式人工智能系统本质上都是一个采样系统。采样策略直接决定了输出结果的质量、多样性和可控性。本课程将从零开始构建所有主流的采样方法，从最基础的均匀分布随机数出发，一直延伸到驱动现代大语言模型 (LLMs) 和生成式模型的核心技术。

## 核心概念

### 为什么采样至关重要

采样在人工智能（Artificial Intelligence）与机器学习（Machine Learning）中主要体现为四种基础作用：

**生成（Generation）。** 语言模型（Language Models）、扩散模型（Diffusion Models）和生成对抗网络（Generative Adversarial Networks, GANs）均通过采样生成输出。采样算法直接控制着生成内容的创造性、连贯性与多样性。温度（Temperature）、Top-k 采样和核采样（Nucleus Sampling）是工程师日常调节的“旋钮”。

**训练（Training）。** 随机梯度下降（Stochastic Gradient Descent）对迷你批次（Mini-batches）进行采样。Dropout 对神经元进行采样以将其停用。数据增强（Data Augmentation）对随机变换进行采样。重要性采样（Importance Sampling）通过对样本重新加权，来降低强化学习（Reinforcement Learning，如 PPO、TRPO）中的梯度方差。

**估计（Estimation）。** 机器学习中的许多量没有闭式解（Closed-form Solution）。例如数据分布上的期望损失、基于能量的模型（Energy-based Model）的配分函数（Partition Function）、贝叶斯推断（Bayesian Inference）中的证据（Evidence）。蒙特卡洛估计（Monte Carlo Estimation）通过对样本求平均来近似所有这些量。

**探索（Exploration）。** 马尔可夫链蒙特卡洛（Markov Chain Monte Carlo, MCMC）算法用于探索贝叶斯推断中的后验分布（Posterior Distribution）。进化策略（Evolutionary Strategies）对参数扰动进行采样。汤普森采样（Thompson Sampling）在多臂老虎机（Bandits）问题中平衡探索与利用（Exploration and Exploitation）。

核心挑战在于：你只能直接从简单分布（如均匀分布、正态分布）中进行采样。对于其他所有情况，你需要一种方法将简单样本转换为目标分布（Target Distribution）的样本。

### 均匀随机采样（Uniform Random Sampling）

所有采样方法都从这里开始。均匀随机数生成器（Uniform Random Number Generator）生成 [0, 1) 区间内的值，其中任意等长子区间的概率均相等。

U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12

若要从包含 n 个元素的离散集合中进行均匀采样，生成 U 并返回 `floor(n * U)`。若要从连续区间 [a, b] 中采样，计算 `a + (b - a) * U`。

核心洞见在于：单个均匀随机数恰好包含了生成任意分布中一个样本所需的随机性。关键在于找到正确的变换方法。

### 逆累积分布函数法（Inverse CDF Method / Inverse Transform Sampling）

累积分布函数（Cumulative Distribution Function, CDF）将数值映射为概率：

F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]

逆 CDF 将概率映射回数值。若 U ~ Uniform(0, 1)，则 X = F_inverse(U) 服从目标分布。

Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)

**指数分布（Exponential Distribution）示例：**

PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda

当你能写出 F_inverse 的闭式表达式时，该方法效果极佳。对于正态分布（Normal Distribution），不存在闭式逆 CDF，因此我们使用其他方法（如 Box-Muller 变换或数值近似）。

**离散版本：** 对于离散分布，将 CDF 构建为累积和，生成 U，并找到累积和首次超过 U 的索引。这正是第 06 课中 `sample_categorical` 的工作原理。

### 拒绝采样（Rejection Sampling）

当你无法求逆 CDF，但能计算目标概率密度函数（Probability Density Function, PDF）（可能未归一化）时，拒绝采样即可派上用场。

Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M

边界 M 越紧，接受率（Acceptance Rate）越高。在低维空间（1-3 维）中，拒绝采样效果良好。但在高维空间中，接受率会呈指数级下降，因为提议分布（Proposal Distribution）的大部分体积都会被拒绝。这就是拒绝采样面临的维度灾难（Curse of Dimensionality）。

**示例：从截断正态分布（Truncated Normal Distribution）中采样。** 在截断范围内使用均匀分布作为提议分布。包络常数 M 即为该范围内正态 PDF 的最大值。

**示例：从半圆中采样。** 在外接矩形内均匀提议。若点落在半圆内则接受。这正是蒙特卡洛方法计算圆周率 pi 的原理：接受率等于面积比 pi/4。

### 重要性采样（Importance Sampling）

有时你并不需要目标分布 p(x) 的样本。你需要估计 p(x) 下的期望（Expectation），而你手头只有来自另一个分布 q(x) 的样本。

Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)

这在强化学习中至关重要。在近端策略优化（Proximal Policy Optimization, PPO）中，你在旧策略 pi_old 下收集轨迹（Trajectories），但希望优化新策略 pi_new。重要性权重为 pi_new(a|s) / pi_old(a|s)。PPO 会对这些权重进行截断（Clipping），以防止新策略偏离旧策略过远。

重要性采样估计器的方差取决于 q 与 p 的相似程度。若 q 与 p 差异很大，少数样本会获得极大的权重并主导估计结果。自归一化重要性采样（Self-normalized Importance Sampling）通过除以权重之和来缓解此问题：

E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)

### 蒙特卡洛估计（Monte Carlo Estimation）

蒙特卡洛估计通过对随机样本求平均来近似积分。大数定律（Law of Large Numbers）保证了其收敛性。

Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension

其误差率与维度无关。这正是蒙特卡洛方法在高维空间（基于网格的积分无法实现）中占据主导地位的原因。

**估计 pi：**

Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)

**估计期望：**

E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N

### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings 算法

MCMC 构建一个马尔可夫链（Markov Chain），其平稳分布（Stationary Distribution）即为目标分布 p(x)。经过足够多的步数后，从该链中采样的样本即为（近似）来自 p(x) 的样本。

Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples

对于对称提议分布（q(x'|x) = q(x|x')），该比率简化为 p(x')/p(x)。这就是原始的 Metropolis 算法。

**为何有效。** 接受规则确保了细致平衡（Detailed Balance）：处于状态 x 并转移到 x' 的概率，等于处于状态 x' 并转移到 x 的概率。细致平衡意味着 p(x) 是该链的平稳分布。

**实际考量：**
- 预热期（Burn-in）：在链达到平衡前丢弃早期样本
- 稀释（Thinning）：每隔 k 个样本保留一个，以降低自相关性（Autocorrelation）
- 提议尺度（Proposal Scale）：过小会导致链移动缓慢（接受率高，探索慢）；过大会导致大多数提议被拒绝（接受率低，原地停滞）
- 高维空间中高斯提议分布的最优接受率约为 0.234

### 吉布斯采样（Gibbs Sampling）

吉布斯采样是 MCMC 针对多元分布（Multivariate Distributions）的一种特例。它不是一次性在所有维度上提议移动，而是每次从其条件分布（Conditional Distribution）中更新一个变量。

Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})

吉布斯采样要求你能从每个条件分布 p(x_i | x_{-i}) 中进行采样。这对许多模型来说很直接：
- 贝叶斯网络（Bayesian Networks）：条件分布由图结构决定
- 高斯混合模型（Gaussian Mixtures）：条件分布为高斯分布
- 伊辛模型（Ising Models）：每个自旋的条件分布仅取决于其邻居

接受率始终为 1（每个提议都被接受），因为从精确条件分布中采样自动满足细致平衡。

**局限性。** 当变量高度相关时，吉布斯采样的混合（Mixing）速度很慢，因为每次更新一个变量无法在分布中进行大幅度的对角线移动。

### 温度采样（Temperature Sampling，常用于大语言模型）

语言模型为词汇表中的每个词元（Token）输出 logits z_1, ..., z_V。Softmax 将其转换为概率。温度参数在 Softmax 之前对 logits 进行重新缩放：

p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)

**为何有效。** 将 logits 除以 T < 1 会放大 logits 之间的差异。若 z_1 = 2 且 z_2 = 1，除以 T = 0.5 后得到 z_1/T = 4 和 z_2/T = 2，差距变大。经过 Softmax 后，最高 logit 对应的词元将获得大得多的概率份额。

**实际应用：**
- T = 0.0：贪婪解码（Greedy Decoding），最适合事实性问答
- T = 0.3-0.7：略带创造性，适合代码生成
- T = 0.7-1.0：平衡型，适合日常对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：随机性过强，极少实用

温度参数不会改变哪些词元是可能的，它只改变分配给每个词元的概率质量（Probability Mass）。

### Top-k 采样（Top-k Sampling）

Top-k 采样将候选集限制为概率最高的 k 个词元，然后重新归一化并从该受限集中采样。

Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens

Top-k 采样可防止模型选择词汇分布长尾（Long Tail）中极不可能的词元（如拼写错误、无意义内容）。其问题在于：k 是固定的，不随上下文变化。当模型很确定时（某个词元概率达 95%），k = 40 仍会允许 39 个备选词元。当模型不确定时（概率分散在 1000 个词元上），k = 40 会截断合理的选项。

### Top-p（核）采样（Top-p / Nucleus Sampling）

Top-p 采样动态调整候选集大小。它不保留固定数量的词元，而是保留累积概率超过 p 的最小词元集合。

Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy

当模型确定时，核采样保留的词元很少（可能 2-3 个）。当模型不确定时，它会保留很多（可能 200 个）。这种自适应行为正是核采样通常比 Top-k 生成文本质量更高的原因。

**常见组合：**
- 温度 0.7 + top-p 0.9：良好的通用设置
- 温度 0.0（贪婪）：最适合确定性任务
- 温度 1.0 + top-k 50：Fan 等人（2018）原始论文的设置

Top-k 和 Top-p 可以结合使用。先应用 Top-k，再对剩余集合应用 Top-p。

### 重参数化技巧（Reparameterization Trick，常用于 VAE）

变分自编码器（Variational Autoencoders, VAEs）通过将输入编码为潜在空间（Latent Space）中的分布、从中采样，再将样本解码回来进行学习。问题在于：你无法通过采样操作进行反向传播（Backpropagation）。

Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???

重参数化技巧将随机性与参数分离开来：

Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.

该方法有效是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 具有相同的分布。核心洞见：将随机性移至无参数源（epsilon），然后将样本表示为参数的可微变换（Differentiable Transformation）。

**在 VAE 训练循环中：**
1. 编码器（Encoder）为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重构输入
5. 沿步骤 4、3、2、1 进行反向传播（可行是因为步骤 3 是可微的）

没有重参数化技巧，VAE 无法使用标准反向传播进行训练。正是这一洞见让 VAE 变得切实可行。

### Gumbel-Softmax（可微分类采样）

重参数化技巧适用于连续分布（如高斯分布）。对于离散分类分布（Discrete Categorical Distributions），我们需要不同的方法。Gumbel-Softmax 为分类采样提供了一种可微近似。

**Gumbel-Max 技巧（不可微）：**

To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.

**Gumbel-Softmax（可微近似）：**

Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation

Gumbel-Softmax 生成离散样本的连续松弛（Continuous Relaxation）。输出是一个概率向量（软独热编码，Soft One-hot）而非硬独热编码。梯度可通过 Softmax 流动。在训练的前向传播（Forward Pass）中，你可以使用“直通估计器”（Straight-through Estimator）：前向传播使用硬 argmax，反向传播使用软 Gumbel-Softmax 梯度。

**应用场景：**
- VAE 中的离散潜在变量（Discrete Latent Variables）
- 神经架构搜索（Neural Architecture Search，选择离散操作）
- 硬注意力机制（Hard Attention Mechanisms）
- 具有离散动作的强化学习

### 分层采样（Stratified Sampling）

标准蒙特卡洛采样可能会因偶然性在样本空间中留下空白。分层采样通过将空间划分为若干层（Strata）并从每层中采样，强制实现均匀覆盖。

Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1

与标准蒙特卡洛相比，分层采样的方差始终更低或相等：

Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.

**应用场景：**
- 数值积分（Numerical Integration，拟蒙特卡洛方法）
- 训练数据划分（确保每个折中的类别平衡）
- 结合分层的重要性采样（结合两种技术）
- NeRF（神经辐射场，Neural Radiance Fields）沿相机光线使用分层采样

### 与扩散模型的联系（Connection to Diffusion Models）

扩散模型（Diffusion Models）通过采样过程生成图像。前向过程（Forward Process）在 T 个步骤中向图像添加高斯噪声，直至其变为纯噪声。反向过程（Reverse Process）学习去噪，逐步恢复原始图像。

Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.

与本课程方法的联系：
- 每个去噪步骤都使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声调度表 {alpha_t} 控制着一种形式的温度退火（Temperature Annealing）
- 训练使用蒙特卡洛估计来近似证据下界（Evidence Lower Bound, ELBO）
- 扩散模型中的祖先采样（Ancestral Sampling）是一个马尔可夫链（每一步仅依赖于当前状态）

整个图像生成过程就是迭代采样（Iterative Sampling）：从噪声开始，在每一步中，基于学习到的去噪模型条件，采样一个噪声略少的版本。

## 构建

### 步骤 1：均匀分布采样（Uniform Sampling）与逆累积分布函数（Inverse CDF）采样

import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam

生成 10,000 个指数分布（Exponential Distribution）样本，并验证其均值是否为 1/lambda。

### 步骤 2：拒绝采样（Rejection Sampling）

def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x

使用拒绝采样从截断正态分布（Truncated Normal Distribution）中抽取样本。通过绘制样本直方图来验证其分布形状。

### 步骤 3：重要性采样（Importance Sampling）

def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n

使用均匀提议分布（Uniform Proposal Distribution），估计正态分布下的 E[X^2]。将其与已知解析解（mu^2 + sigma^2）进行对比。

### 步骤 4：蒙特卡洛方法（Monte Carlo Method）估算圆周率 pi

def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n

### 步骤 5：Metropolis-Hastings 马尔可夫链蒙特卡洛（Markov Chain Monte Carlo, MCMC）

def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples

从双峰分布（Bimodal Distribution，即两个高斯分布的混合）中进行采样。可视化马尔可夫链的轨迹。

### 步骤 6：吉布斯采样（Gibbs Sampling）

def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples

### 步骤 7：温度采样（Temperature Sampling）

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)

展示温度参数如何改变一组词元对数几率（Token Logits）的输出分布。

### 步骤 8：Top-k 与 Top-p 采样（Top-k and Top-p Sampling）

def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]

### 步骤 9：重参数化技巧（Reparameterization Trick）

def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma

证明梯度可以通过重参数化后的样本进行反向传播，而无法通过直接采样进行传播。

### 步骤 10：Gumbel-Softmax

def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])

展示降低温度参数如何使输出分布逐渐逼近独热向量（One-hot Vector）。

包含所有可视化功能的完整实现代码位于 `code/sampling.py` 中。

## 使用方法

结合 NumPy 和 SciPy 的生产环境版本：

import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")

对于大规模马尔可夫链蒙特卡洛（MCMC）任务，请使用专用库：
- PyMC：支持无转折采样器（NUTS，自适应哈密顿蒙特卡洛/HMC）的完整贝叶斯建模
- emcee：集成 MCMC 采样器
- NumPyro/JAX：GPU 加速的 MCMC

你已从零开始实现了这些算法。现在，你应该清楚这些库函数底层究竟在做什么。

## 练习

1. 为柯西分布实现逆累积分布函数（Inverse CDF）采样。其累积分布函数（CDF）为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，并绘制直方图与真实概率密度函数（PDF）进行对比。注意观察其重尾特性（即远离中心的极端值）。

2. 使用拒绝采样（Rejection sampling），以 Uniform(0, 1) 作为提议分布，从 Beta(2, 5) 分布中生成样本。将接受的样本与真实的 Beta 分布 PDF 进行对比绘图。理论接受率是多少？

3. 使用蒙特卡洛（Monte Carlo）方法，分别基于 1,000、10,000 和 100,000 个样本，估算 sin(x) 在 0 到 pi 区间上的积分。比较各样本量下的误差。验证误差是否按 O(1/sqrt(N)) 的比例缩放。

4. 实现 Metropolis-Hastings 算法，从二维分布 p(x, y) 中进行采样，该分布正比于 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)。绘制采样点及马尔可夫链轨迹。尝试使用不同的提议分布标准差进行实验。

5. 构建一个完整的文本生成演示：给定包含 10 个词及其对数几率（logits）的词表，分别使用 (a) 贪婪策略（greedy）、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成长度为 20 个词元（token）的序列。对比 5 次运行中输出结果的多样性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 采样 (Sampling) | “抽取随机值” | 依据概率分布生成数值。这是所有生成式人工智能 (Generative AI) 的底层机制。 |
| 均匀分布 (Uniform Distribution) | “所有结果等概率” | 区间 [a, b] 内的每个值具有相等的概率密度 1/(b-a)。所有采样方法的起点。 |
| 逆累积分布函数 (Inverse CDF) | “概率变换” | F_inverse(U) 将均匀分布样本转换为任意已知累积分布函数 (CDF) 的分布样本。精确且高效。 |
| 拒绝采样 (Rejection Sampling) | “提议并接受/拒绝” | 从简单的提议分布中生成样本，按目标分布与提议分布的比例概率进行接受。结果精确但会浪费部分样本。 |
| 重要性采样 (Importance Sampling) | “样本重加权” | 使用来自 q(x) 的样本，通过 p(x)/q(x) 的权重来估计 p(x) 下的期望。是强化学习 (RL) 中近端策略优化 (PPO) 算法的核心。 |
| 蒙特卡洛方法 (Monte Carlo) | “随机样本平均” | 将积分近似为样本均值。无论维度如何，误差均为 O(1/sqrt(N))。 |
| 马尔可夫链蒙特卡洛 (MCMC) | “收敛的随机游走” | 构建一个平稳分布为目标分布的马尔可夫链。Metropolis-Hastings 是其基础算法。 |
| Metropolis-Hastings 算法 | “接受上坡，偶尔下坡” | 提议状态转移，根据密度比率决定是否接受。细致平衡条件 (Detailed Balance) 确保收敛至目标分布。 |
| 吉布斯采样 (Gibbs Sampling) | “逐变量更新” | 在固定其他变量的情况下，从其条件分布中更新每个变量。接受率为 100%。 |
| 温度参数 (Temperature) | “置信度旋钮” | 在 softmax 前将 logits 除以 T。T<1 使分布更尖锐（更自信），T>1 使分布更平坦（更多样化）。 |
| Top-k 采样 | “保留前 k 个最佳” | 将除概率最高的 k 个词元 (token) 外的所有概率置零，重新归一化后采样。候选集大小固定。 |
| 核心采样 (Nucleus Sampling / Top-p) | “保留高概率项” | 保留累积概率超过 p 的最小词元集合。候选集大小自适应。 |
| 重参数化技巧 (Reparameterization Trick) | “将随机性外移” | 将 z 表示为 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。使采样过程可微。对变分自编码器 (VAE) 训练至关重要。 |
| Gumbel-Softmax | “软类别采样” | 使用 Gumbel 噪声结合带温度的 softmax，对类别采样进行可微近似。 |
| 分层采样 (Stratified Sampling) | “强制覆盖” | 将样本空间划分为若干层，从每层中分别采样。方差始终低于朴素蒙特卡洛方法。 |
| 预烧期 (Burn-in) | “预热阶段” | 在马尔可夫链达到平稳分布之前丢弃的初始 MCMC 样本。 |
| 细致平衡 (Detailed Balance) | “可逆性条件” | p(x) * T(x->y) = p(y) * T(y->x)。p 成为马尔可夫链平稳分布的充分条件。 |
| 扩散采样 (Diffusion Sampling) | “迭代去噪” | 从噪声出发，应用学习到的去噪步骤生成数据。每一步均为条件采样操作。 |

## 进一步阅读

- [Holbrook (2023): Metropolis-Hastings 算法](https://arxiv.org/abs/2304.07010) - 马尔可夫链蒙特卡洛 (Markov Chain Monte Carlo, MCMC) 基础的详细教程
- [Jang, Gu, Poole (2017): 基于 Gumbel-Softmax 的类别重参数化](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 的原始论文
- [Holtzman 等人 (2020): 神经文本退化的奇特案例](https://arxiv.org/abs/1904.09751) - 核采样 (nucleus sampling / top-p sampling) 的原始论文
- [Kingma & Welling (2014): 自动编码变分贝叶斯](https://arxiv.org/abs/1312.6114) - 引入重参数化技巧 (reparameterization trick) 的变分自编码器 (Variational Autoencoder, VAE) 论文
- [Ho, Jain, Abbeel (2020): 去噪扩散概率模型](https://arxiv.org/abs/2006.11239) - 将去噪扩散概率模型 (Denoising Diffusion Probabilistic Models, DDPM) 的采样过程与图像生成相连接
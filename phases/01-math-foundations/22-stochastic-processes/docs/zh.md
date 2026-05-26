# 随机过程 (Stochastic Processes)

> 具有结构的随机性。随机游走 (Random Walk)、马尔可夫链 (Markov Chain) 和扩散模型 (Diffusion Model) 背后的数学原理。

**Type:** 学习
**Language:** Python
**Prerequisites:** 第一阶段，第 06-07 课（概率论、贝叶斯）
**Time:** 约 75 分钟

## 学习目标

- 模拟一维和二维随机游走，并验证位移的 sqrt(n) 缩放规律
- 构建马尔可夫链模拟器，并通过特征分解 (Eigendecomposition) 计算其平稳分布 (Stationary Distribution)
- 实现 Metropolis-Hastings MCMC 和朗之万动力学 (Langevin Dynamics)，用于从目标分布中采样
- 将前向扩散过程 (Forward Diffusion Process) 与布朗运动 (Brownian Motion) 联系起来，并解释反向过程 (Reverse Process) 如何生成数据

## 问题背景

许多人工智能系统涉及随时间演化的随机性。这种随机性并非静态的，而是具有结构性和序列性的，其中每一步都依赖于前一步的结果。

语言模型逐个生成词元 (Token)。每个词元都依赖于先前的上下文。模型输出一个概率分布，从中进行采样，然后继续下一步。这就是一个随机过程。

扩散模型逐步向图像添加噪声，直到其变为纯噪声。随后，它们逆转该过程，逐步去噪，直至生成一幅新图像。前向过程是一个马尔可夫链。反向过程则是一个逆向运行的、经过学习的马尔可夫链。

强化学习智能体在环境中执行动作。每个动作都以一定的概率导致进入一个新状态。智能体在一个随机世界中遵循随机策略。整个过程构成了一个马尔可夫决策过程 (Markov Decision Process)。

马尔可夫链蒙特卡洛 (MCMC) 采样是贝叶斯推断 (Bayesian Inference) 的核心，它通过构建一个马尔可夫链来实现，该链的平稳分布正是你想要采样的目标后验分布 (Posterior)。

所有这些方法都建立在四个基础概念之上：
1. 随机游走 —— 最简单的随机过程
2. 马尔可夫链 —— 具有转移矩阵的结构化随机性
3. 朗之万动力学 —— 加入噪声的梯度下降
4. Metropolis-Hastings 算法 —— 从任意分布中进行采样

## 核心概念

### 随机游走 (Random Walks)

从位置 0 开始。每一步抛一枚均匀硬币。正面：向右移动 (+1)。反面：向左移动 (-1)。

经过 n 步后，你的位置是 n 个随机 +/-1 值的总和。期望位置为 0（该游走是无偏的）。但距离原点的期望距离随 sqrt(n) 增长。

这有些反直觉。游走是公平的——没有任何方向的漂移。但随着时间的推移，它会离起点越来越远。n 步后的标准差为 sqrt(n)。

Step 0:  Position = 0
Step 1:  Position = +1 or -1
Step 2:  Position = +2, 0, or -2
...
Step 100: Expected distance from origin ~ 10 (sqrt(100))
Step 10000: Expected distance from origin ~ 100 (sqrt(10000))

**在二维空间中**，游走以相等的概率向上、下、左或右移动。距离原点的缩放规律同样遵循 sqrt(n)。其路径会呈现出类似分形 (fractal) 的图案。

**为什么是 sqrt(n)？** 每一步以相等的概率取 +1 或 -1。经过 n 步后，位置为 S_n = X_1 + X_2 + ... + X_n，其中每个 X_i 为 +/-1。每一步的方差为 1，且各步相互独立，因此 Var(S_n) = n。标准差 = sqrt(n)。根据中心极限定理 (Central Limit Theorem)，S_n / sqrt(n) 会收敛于标准正态分布。

这种 sqrt(n) 的缩放规律在机器学习 (Machine Learning, ML) 中随处可见。随机梯度下降 (Stochastic Gradient Descent, SGD) 的噪声按 1/sqrt(batch_size) 缩放。嵌入 (Embedding) 维度按 sqrt(d) 缩放。平方根正是独立随机累加的典型特征。

**与布朗运动 (Brownian Motion) 的联系。** 考虑一个步长为 1/sqrt(n)、单位时间内进行 n 步的随机游走。当 n 趋于无穷大时，该游走收敛于布朗运动 B(t)——这是一个连续时间过程，其中 B(t) 服从均值为 0、方差为 t 的正态分布。

布朗运动是扩散 (Diffusion) 的数学基础。它用于模拟流体中粒子的随机抖动、股票价格的波动，以及——至关重要的一点——扩散模型中的噪声过程。

**赌徒破产问题 (Gambler's Ruin)。** 一个从位置 k 出发的随机游走者，在 0 和 N 处设有吸收壁 (absorbing barriers)。在到达 0 之前先到达 N 的概率是多少？对于公平游走：P(reach N) = k/N。这个结论出奇地简洁优雅。它与鞅 (Martingale) 理论密切相关——公平随机游走本身就是一个鞅（未来期望值等于当前值）。

### 马尔可夫链 (Markov Chains)

马尔可夫链是一个根据固定概率在状态之间转换的系统。其核心特性是：下一个状态仅取决于当前状态，而与历史状态无关。

P(X_{t+1} = j | X_t = i, X_{t-1} = ...) = P(X_{t+1} = j | X_t = i)

这就是马尔可夫性质 (Markov Property)。这意味着你可以用一个转移矩阵 (Transition Matrix) P 来描述整个动态过程：

P[i][j] = probability of going from state i to state j

P 的每一行之和为 1（你必然会转移到某个状态）。

**示例——天气：**

States: Sunny (0), Rainy (1), Cloudy (2)

P = [[0.7, 0.1, 0.2],    (if sunny: 70% sunny, 10% rainy, 20% cloudy)
     [0.3, 0.4, 0.3],    (if rainy: 30% sunny, 40% rainy, 30% cloudy)
     [0.4, 0.2, 0.4]]    (if cloudy: 40% sunny, 20% rainy, 40% cloudy)

从任意状态开始。经过多次转移后，状态分布会收敛于平稳分布 (Stationary Distribution) π，满足 π * P = π。这对应于 P 的特征值为 1 的左特征向量。

对于该天气链，平稳分布可能为 [0.53, 0.18, 0.29]——从长期来看，无论起始状态如何，晴天出现的概率均为 53%。

graph LR
    S["Sunny"] -->|0.7| S
    S -->|0.1| R["Rainy"]
    S -->|0.2| C["Cloudy"]
    R -->|0.3| S
    R -->|0.4| R
    R -->|0.3| C
    C -->|0.4| S
    C -->|0.2| R
    C -->|0.4| C

**计算平稳分布。** 有两种常用方法：

1. **幂迭代法 (Power Method)**：将任意初始分布反复乘以 P。经过足够多次迭代后，结果将收敛。
2. **特征值法 (Eigenvalue Method)**：寻找 P 特征值为 1 的左特征向量。这等价于寻找 P^T 特征值为 1 的特征向量。

这两种方法都要求马尔可夫链满足收敛条件。

**收敛条件。** 若马尔可夫链满足以下条件，它将收敛于唯一的平稳分布：
- **不可约性 (Irreducible)**：从任意状态均可到达其他任意状态
- **非周期性 (Aperiodic)**：链不会以固定周期循环

你在机器学习中遇到的大多数链都同时满足这两个条件。

**吸收态 (Absorbing States)。** 如果一旦进入某个状态就永远不会离开（即 P[i][i] = 1），则该状态为吸收态。吸收马尔可夫链用于对具有终止状态的过程进行建模——例如结束的游戏、流失的客户，或触发了文本结束 (End-of-Text) 标记的 Token 序列。

**混合时间 (Mixing Time)。** 链需要多少步才能“接近”平稳分布？形式上，这是指链的平稳分布总变差距离 (Total Variation Distance) 降至某个阈值以下所需的步数。快速混合 = 所需步数少。P 的谱隙 (Spectral Gap)（1 减去第二大特征值）控制着混合时间。谱隙越大 = 混合越快。

### 与语言模型的联系

语言模型中的 Token 生成近似为一个马尔可夫过程。给定当前上下文，模型会输出下一个 Token 的概率分布。温度参数 (Temperature) 控制分布的尖锐程度：

P(token_i) = exp(logit_i / temperature) / sum(exp(logit_j / temperature))

- Temperature = 1.0：标准分布
- Temperature < 1.0：更尖锐（更确定性）
- Temperature > 1.0：更平坦（更随机）
- Temperature -> 0：取最大值 (argmax，贪婪策略)

Top-k 采样 (Top-k Sampling) 仅保留概率最高的 k 个 Token。Top-p（核）采样 (Top-p / Nucleus Sampling) 仅保留累积概率超过 p 的最小 Token 集合。两者都会修改马尔可夫转移概率。

### 布朗运动 (Brownian Motion)

随机游走的连续时间极限。位置 B(t) 具有三个性质：
1. B(0) = 0
2. B(t) - B(s) 服从均值为 0、方差为 t - s 的正态分布（当 t > s 时）
3. 非重叠区间上的增量相互独立

布朗运动是连续的，但处处不可导——它在每一个尺度上都在抖动。其路径在平面上的分形维数为 2。

在离散模拟中，你可以通过以下方式近似布朗运动：

B(t + dt) = B(t) + sqrt(dt) * z,    where z ~ N(0, 1)

sqrt(dt) 的缩放比例至关重要。它源于应用于随机游走的中心极限定理。

### 朗之万动力学 (Langevin Dynamics)

梯度下降用于寻找函数的最小值。朗之万动力学用于寻找与 exp(-U(x)/T) 成正比的概率分布，其中 U 是能量函数，T 是温度。

x_{t+1} = x_t - dt * gradient(U(x_t)) + sqrt(2 * T * dt) * z_t

粒子受到两种力的作用：
1. **梯度力 (Gradient Force)**（-dt * gradient(U)）：推向低能量区域（类似于梯度下降）
2. **随机力 (Random Force)**（sqrt(2*T*dt) * z）：推向随机方向（用于探索）

当温度 T = 0 时，这等同于纯梯度下降。在高温下，它近似于随机游走。在合适的温度下，粒子会探索能量景观 (Energy Landscape)，并在低能量区域停留更长时间。

**与扩散模型的联系。** 扩散模型的前向过程 (Forward Process) 为：

x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise

这是一个将数据与噪声逐渐混合的马尔可夫链。经过足够多的步骤后，x_T 变为纯高斯噪声。

反向过程（从噪声恢复为数据）同样是一个马尔可夫链，但其转移概率由神经网络学习得到。该网络学习预测每一步添加的噪声，然后将其减去。

graph LR
    subgraph "Forward Process (add noise)"
        X0["x_0 (data)"] -->|"+ noise"| X1["x_1"]
        X1 -->|"+ noise"| X2["x_2"]
        X2 -->|"..."| XT["x_T (pure noise)"]
    end
    subgraph "Reverse Process (denoise)"
        XT2["x_T (noise)"] -->|"neural net"| XR2["x_{T-1}"]
        XR2 -->|"neural net"| XR1["x_{T-2}"]
        XR1 -->|"..."| XR0["x_0 (generated data)"]
    end

### MCMC：马尔可夫链蒙特卡洛 (Markov Chain Monte Carlo)

有时你需要从一个分布 p(x) 中采样，该分布你可以计算（最多差一个常数因子），但无法直接采样。贝叶斯后验分布 (Bayesian Posterior) 就是经典例子——你知道似然 (Likelihood) 乘以先验 (Prior) 的结果，但归一化常数难以计算。

**Metropolis-Hastings 算法** 构建一个平稳分布为 p(x) 的马尔可夫链：

1. 从某个位置 x 开始
2. 从提议分布 (Proposal Distribution) Q(x'|x) 中提议一个新位置 x'
3. 计算接受率 (Acceptance Ratio)：a = p(x') * Q(x|x') / (p(x) * Q(x'|x))
4. 以概率 min(1, a) 接受 x'。否则停留在 x。
5. 重复上述步骤。

如果 Q 是对称的（例如 Q(x'|x) = Q(x|x') = N(x, sigma^2)），则比率简化为 a = p(x') / p(x)。你只需要概率的比值——归一化常数会被抵消。

在温和条件下，该链保证收敛于 p(x)。但如果提议步长太小（类似随机游走）或太大（导致高拒绝率），收敛可能会很慢。调整提议分布是 MCMC 的艺术所在。

**为什么有效。** 接受率确保了细致平衡 (Detailed Balance)：处于状态 x 并转移到 x' 的概率，等于处于状态 x' 并转移到 x 的概率。细致平衡意味着 p(x) 是该链的平稳分布。因此，经过足够多的步骤后，采样结果将来自 p(x)。

**实际注意事项：**
- **预热期 (Burn-in)**：丢弃前 N 个样本。链需要时间从起点到达平稳分布。
- **稀释 (Thinning)**：每隔 k 个样本保留一个，以降低自相关性。
- **多链并行 (Multiple Chains)**：从不同起点运行多条链。如果它们收敛到相同的分布，则可作为收敛的证据。
- **接受率 (Acceptance Rate)**：对于 d 维高斯提议，最优接受率约为 23%（Roberts & Rosenthal, 2001）。过高意味着链几乎不移动。过低意味着它拒绝几乎所有提议。

### AI 中的随机过程

| 过程 | AI 应用 |
|---------|---------------|
| 随机游走 (Random Walk) | 强化学习 (Reinforcement Learning, RL) 中的探索、Node2Vec 嵌入 |
| 马尔可夫链 (Markov Chain) | 文本生成、MCMC 采样 |
| 布朗运动 (Brownian Motion) | 扩散模型（前向过程） |
| 朗之万动力学 (Langevin Dynamics) | 基于分数的生成模型 (Score-based Generative Models)、SGLD |
| 马尔可夫决策过程 (Markov Decision Process) | 强化学习 |
| Metropolis-Hastings 算法 | 贝叶斯推断、后验采样 |

## 构建

### 步骤 1：随机游走模拟器 (Random Walk Simulator)

import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # right
    dx[directions == 1] = -1  # left
    dy[directions == 2] = 1   # up
    dy[directions == 3] = -1  # down
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y

一维随机游走（1D Random Walk）记录累积和（Cumulative Sum）。每一步为 +1 或 -1。经过 n 步后，当前位置即为步长的累加值。方差（Variance）随 n 线性增长，因此标准差（Standard Deviation）按 sqrt(n) 增长。

### 步骤 2：马尔可夫链 (Markov Chain)

class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)

平稳分布（Stationary Distribution）是转移矩阵 P 对应特征值（Eigenvalue）为 1 的左特征向量（Left Eigenvector）。我们通过计算 P^T 的特征向量来求解它（矩阵转置可将左特征向量转换为右特征向量）。

### 步骤 3：朗之万动力学 (Langevin Dynamics)

def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)

梯度（Gradient）推动 x 向低能量状态移动。噪声（Noise）则防止其陷入局部最优。在平衡态（Equilibrium）下，样本的分布与 exp(-U(x)/temperature) 成正比。

### 步骤 4：Metropolis-Hastings 算法

def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate

该算法会提议（Propose）一个新状态点，检查其是否具有更高的概率（或以与该比率成正比的概率接受该提议），并重复此过程。为了获得良好的混合效果（Mixing），接受率（Acceptance Rate）应保持在 23% 到 50% 左右。

## 实践应用

在实际开发中，你通常会使用成熟的库来实现这些算法。但理解其底层机制对于调试和调优至关重要。

import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"Final position: {walk[-1]}")
print(f"Expected distance: {np.sqrt(10000):.1f}")
print(f"Actual distance: {abs(walk[-1])}")

### 使用 numpy 处理转移矩阵（Transition Matrices）

import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

distribution = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    distribution = distribution @ P

print(f"Stationary distribution: {np.round(distribution, 4)}")

将初始分布反复与矩阵 P 相乘。经过足够多次迭代后，无论初始状态如何，它都会收敛到平稳分布（Stationary Distribution）。这正是用于求解主左特征向量（Dominant Left Eigenvector）的幂迭代法（Power Method）。

### 与主流框架的关联

- **PyTorch 扩散模型：** Hugging Face `diffusers` 库中的 `DDPMScheduler` 实现了前向与反向马尔可夫链（Markov Chain）
- **NumPyro / PyMC：** 使用马尔可夫链蒙特卡洛方法（Markov Chain Monte Carlo, MCMC）（具体为 NUTS 采样器，它在 Metropolis-Hastings 算法基础上进行了改进）进行贝叶斯推断（Bayesian Inference）
- **Gymnasium（强化学习）：** 环境中的 `step` 函数定义了一个马尔可夫决策过程（Markov Decision Process, MDP）

### 验证马尔可夫链收敛性

import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"Eigenvalues: {eigenvalues}")
print(f"Spectral gap: {spectral_gap:.4f}")
print(f"Approximate mixing time: {1/spectral_gap:.1f} steps")

谱隙（Spectral Gap）反映了马尔可夫链遗忘初始状态的速度。谱隙为 0.2 意味着大约需要 5 步即可达到混合（Mixing）状态；谱隙为 0.01 则大约需要 100 步。在运行长时间模拟前务必检查该指标——混合缓慢的链会白白浪费计算资源。

## 产出内容

本节课程将产出：
- `outputs/prompt-stochastic-process-advisor.md` -- 一个提示词（Prompt），用于帮助识别针对特定问题应选用哪种随机过程（Stochastic Process）框架

## 关联内容

| 概念 | 应用场景 |
|---------|------------------|
| 随机游走 (Random walk) | Node2Vec 图嵌入、强化学习 (Reinforcement Learning, RL) 中的探索 |
| 马尔可夫链 (Markov chain) | 大语言模型 (Large Language Models, LLM) 中的词元生成、马尔可夫链蒙特卡洛 (Markov Chain Monte Carlo, MCMC) 采样 |
| 布朗运动 (Brownian motion) | 去噪扩散概率模型 (Denoising Diffusion Probabilistic Models, DDPM) 中的前向扩散过程、基于随机微分方程 (Stochastic Differential Equation, SDE) 的模型 |
| 朗之万动力学 (Langevin dynamics) | 基于分数的生成模型 (Score-based generative models)、随机梯度朗之万动力学 (Stochastic Gradient Langevin Dynamics, SGLD) |
| 平稳分布 (Stationary distribution) | MCMC 收敛目标、PageRank 算法 |
| 梅特罗波利斯-黑斯廷斯算法 (Metropolis-Hastings) | 贝叶斯后验采样 (Bayesian posterior sampling)、模拟退火 (Simulated annealing) |
| 温度参数 (Temperature) | LLM 采样、强化学习中的玻尔兹曼探索 (Boltzmann exploration)、模拟退火 |
| 混合时间 (Mixing time) | MCMC 收敛速度、谱隙分析 (Spectral gap analysis) |
| 吸收态 (Absorbing state) | 序列结束词元 (End-of-sequence token)、强化学习中的终止状态 |
| 细致平衡条件 (Detailed balance) | MCMC 采样器的正确性保证 |

扩散模型 (Diffusion models) 值得特别关注。DDPM（Ho 等人，2020）定义了一个前向马尔可夫链：

q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)

其中 `beta_t` 是噪声调度 (noise schedule)。经过 T 步后，`x_T` 近似服从 `N(0, I)` 分布。反向过程由一个预测噪声的神经网络进行参数化：

p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)

生成过程中的每一步，都是在一个已学习的马尔可夫链中迈出的一步。理解马尔可夫链，就意味着理解扩散模型如何以及为何能够生成数据。

SGLD 将小批量梯度下降 (mini-batch gradient descent) 与朗之万噪声 (Langevin noise) 相结合。它不计算完整梯度，而是使用随机估计值并添加校准后的噪声。随着学习率 (learning rate) 衰减，SGLD 会从优化过程过渡到采样过程——你可以免费获得近似的贝叶斯后验样本 (Bayesian posterior samples)。这是从神经网络中获取不确定性估计 (uncertainty estimates) 的最简单方法之一。

贯穿所有这些联系的核心洞见在于：随机过程 (stochastic processes) 不仅仅是理论工具，它们更是现代人工智能系统内部的计算机制。当你调整大语言模型的温度参数时，你实际上是在调整一个马尔可夫链。当你训练扩散模型时，你是在学习如何逆转一个类似布朗运动的过程。当你执行贝叶斯推断 (Bayesian inference) 时，你是在构建一个收敛至后验分布的链。

## 练习

1. **模拟 1000 次、每次 10000 步的随机游走（Random Walk）。** 绘制最终位置的分布图。验证其近似服从均值为 0、标准差为 sqrt(10000) = 100 的高斯分布（Gaussian Distribution）。

2. **使用马尔可夫链（Markov Chain）构建文本生成器。** 在小型语料库上进行训练：针对每个词，统计其转移到下一个词的频次。构建转移矩阵（Transition Matrix）。通过对该链进行采样来生成新句子。

3. **使用 Metropolis-Hastings 算法实现模拟退火（Simulated Annealing）。** 从高温开始（几乎接受所有状态），然后逐渐降温（仅接受更优状态）。利用它来寻找具有多个局部极小值（Local Minima）的函数的全局最小值。

4. **比较不同温度下的朗之万动力学（Langevin Dynamics）。** 从双势阱（Double-Well Potential）U(x) = (x^2 - 1)^2 中进行采样。在低温下，样本会聚集在其中一个势阱中；在高温下，样本会扩散至两个势阱。寻找链在两个势阱之间充分混合（Mixing）的临界温度。

5. **实现前向扩散过程（Forward Diffusion Process）。** 从一个一维信号（例如正弦波）开始。在 100 个步骤中，按照线性噪声调度（Linear Noise Schedule）逐步添加噪声。展示信号如何逐渐退化为纯噪声。然后实现一个简单的去噪器（Denoiser）来逆转该过程（即使是一个仅减去估计噪声的朴素版本也可以）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 随机游走（Random Walk） | “抛硬币式的移动” | 每一步的位置变化都由随机增量决定的过程 |
| 马尔可夫性质（Markov Property） | “无记忆性” | 未来状态仅取决于当前状态，而与历史状态无关 |
| 转移矩阵（Transition Matrix） | “概率表” | P[i][j] = 从状态 i 转移到状态 j 的概率 |
| 平稳分布（Stationary Distribution） | “长期平均值” | 满足 pi*P = pi 的分布 pi，即马尔可夫链的平衡态 |
| 布朗运动（Brownian Motion） | “随机抖动” | 随机游走在连续时间下的极限形式，B(t) ~ N(0, t) |
| 朗之万动力学（Langevin Dynamics） | “带噪声的梯度下降” | 结合确定性梯度与随机扰动的更新规则 |
| 马尔可夫链蒙特卡洛（MCMC） | “向目标漫步” | 构建一个马尔可夫链，使其平稳分布恰好是你想要的目标分布 |
| Metropolis-Hastings 算法 | “提议与接受/拒绝” | 一种利用接受率来确保收敛的 MCMC 算法 |
| 温度（Temperature） | “随机性旋钮” | 控制探索（Exploration）与利用（Exploitation）之间权衡的参数 |
| 扩散过程（Diffusion Process） | “进噪声，出噪声” | 前向过程：逐步添加噪声。反向过程：逐步去除噪声。用于生成数据。 |

## 扩展阅读

- **Ho, Jain, Abbeel (2020)** -- 《Denoising Diffusion Probabilistic Models》。这篇 DDPM 论文开启了扩散模型（Diffusion Model）的革命。文中清晰地推导了前向与反向马尔可夫链（Markov Chain）。
- **Song & Ermon (2019)** -- 《Generative Modeling by Estimating Gradients of the Data Distribution》。提出了一种基于分数的方法（Score-based approach），利用朗之万动力学（Langevin Dynamics）进行采样。
- **Roberts & Rosenthal (2004)** -- 《General state space Markov chains and MCMC algorithms》。深入探讨了马尔可夫链蒙特卡洛（Markov Chain Monte Carlo, MCMC）算法的适用条件及其生效的理论基础。
- **Norris (1997)** -- 《Markov Chains》。该领域的标准教材。内容涵盖收敛性（Convergence）、平稳分布（Stationary Distribution）以及首达时间（Hitting Time）。
- **Welling & Teh (2011)** -- 《Bayesian Learning via Stochastic Gradient Langevin Dynamics》。将随机梯度下降（Stochastic Gradient Descent, SGD）与朗之万动力学相结合，用于实现可扩展的贝叶斯推断（Bayesian Inference）。
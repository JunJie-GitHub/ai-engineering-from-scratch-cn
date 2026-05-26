# 策略梯度（Policy Gradient）—— 从零实现 REINFORCE

> 别再估计价值了。直接对策略进行参数化，计算期望回报（expected return）的梯度，沿梯度方向上升。Williams（1992）仅用一条定理就将其阐明。这正是近端策略优化（PPO）、组相对策略优化（GRPO）以及所有大语言模型强化学习（LLM-RL）循环得以存在的基石。

**Type:** 构建实践
**Languages:** Python
**Prerequisites:** 第 3 阶段 · 03（反向传播），第 9 阶段 · 03（蒙特卡洛方法），第 9 阶段 · 04（时序差分学习）
**Time:** 约 75 分钟

## 问题背景

Q-learning 和 DQN 对*价值函数（value function）*进行参数化。你通过 `argmax Q` 来选择动作。这在离散动作（discrete actions）和离散状态（discrete states）下表现良好。但当动作是连续的（continuous）（如何在 10 维扭矩空间上做 `argmax`？），或者你需要一个随机策略（stochastic policy）时，这种方法就会失效（因为 `argmax` 本质上是确定性的 deterministic）。

策略梯度（Policy Gradient）则直接对*策略（policy）*进行参数化。`π_θ(a | s)` 是一个神经网络（neural network），它输出动作的概率分布（distribution）。从中采样以执行动作。计算期望回报关于 `θ` 的梯度。沿梯度方向上升。无需 `argmax`。无需贝尔曼递归（Bellman recursion）。只需对 `J(θ) = E_{π_θ}[G]` 执行梯度上升（gradient ascent）。

REINFORCE 定理（Williams, 1992）指出该梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个回合（episode）。计算回报（return）。在每一步乘以 `∇ log π_θ(a | s)`。取平均。执行梯度上升。完成。

2026 年的每一个 LLM-RL 算法——PPO、直接偏好优化（DPO）、GRPO——都是对 REINFORCE 的改进。将其彻底掌握并形成直觉是学习本阶段后续内容，以及第 10 阶段 · 07（基于人类反馈的强化学习 RLHF 实现）和第 10 阶段 · 08（DPO）的先决条件。

## 核心概念

![策略梯度：Softmax策略、log-π梯度、回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理（Policy Gradient Theorem）。** 对于任意由参数 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 表示从第 `t` 步开始的折扣回报（Discounted Return）。该期望是针对从 `π_θ` 中采样得到的完整轨迹（Trajectory）`τ` 计算的。

**证明过程很简短。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。利用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧，Log-Derivative Trick）。将 `log P(τ; θ)` 分解为 `Σ log π_θ(a_t | s_t) + 不依赖于 θ 的环境项`。环境项会相互抵消。只需两行代数推导即可得出该定理。

**方差缩减技巧（Variance Reduction Tricks）。** 基础版 REINFORCE 算法的方差极大——回报（Returns）充满噪声，`∇ log π` 充满噪声，两者的乘积更是噪声巨大。两种标准的改进方法如下：

1. **基线减法（Baseline Subtraction）。** 将 `G_t` 替换为 `G_t - b(s_t)`，其中基线 `b(s_t)` 不依赖于动作 `a_t`。该操作是无偏的，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：由评论家网络（Critic）学习得到的 `b(s_t) = V̂(s_t)` → 演员-评论家算法（Actor-Critic）（第 07 课）。
2. **剩余回报（Reward-to-Go）。** 将 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。对于给定动作而言，只有未来的回报才有意义——过去的奖励只会贡献均值为零的噪声。

结合两者，可得：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE 算法——它是 A2C（第 07 课）和 PPO（第 08 课）的直接前身。

**Softmax 策略参数化（Softmax Policy Parameterization）。** 对于离散动作空间，标准的选择是：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任意能够为每个动作输出分数的神经网络。其梯度具有简洁的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即：实际执行动作的分数减去该策略下所有动作的期望分数。

**连续动作的高斯策略（Gaussian Policy）。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 具有闭式解（Closed-Form Solution）。这正是第 9 阶段 · 第 07 课中 SAC 算法所需的全部内容。

## 动手实现

### 步骤 1：Softmax 策略网络 (Softmax Policy Network)

def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]

对于表格型环境 (Tabular Environment)，请使用线性策略（每个动作对应一个权重向量）。若应用于 Atari 环境，可替换为卷积神经网络 (Convolutional Neural Network, CNN)，并保留 Softmax 输出层 (Softmax Head)。

### 步骤 2：采样与对数概率 (Sampling and Log-Probability)

def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)

### 步骤 3：记录对数概率的轨迹展开 (Rollout)

def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory

### 步骤 4：REINFORCE 更新

def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]

梯度 `∇ log π(a|s) = e_a - π(·|s)`（即动作 `a` 的独热编码 (One-Hot Encoding) 减去概率分布）是 Softmax 策略梯度 (Policy Gradient) 的核心。请务必将其熟记于心（形成肌肉记忆）。

### 步骤 5：基线 (Baseline)

对近期回合 (Episode) 的回报 `G` 计算滑动平均 (Running Mean)，足以降低方差并使 4×4 网格世界 (GridWorld) 环境顺利运行；大约需要 500 个回合即可收敛。若将基线升级为可学习的价值函数估计 `V̂(s)`，即可得到演员-评论家算法 (Actor-Critic)。

## 常见陷阱 (Pitfalls)

- **梯度爆炸 (Exploding Gradients)。** 回报值可能非常大。在乘以 `∇ log π` 之前，务必在批次 (Batch) 范围内将 `G` 归一化至近似标准正态分布 `~N(0, 1)`。
- **熵崩溃 (Entropy Collapse)。** 策略过早收敛至近乎确定性的动作，停止探索并陷入停滞。解决方法：在目标函数 (Objective Function) 中添加熵奖励 `β · H(π(·|s))`。
- **高方差 (High Variance)。** 原始 REINFORCE 算法 (Vanilla REINFORCE) 通常需要数千个回合。引入评论家基线 (Critic Baseline)（第 07 课）或使用 TRPO/PPO 的信任域 (Trust Region)（第 08 课）是标准的解决方案。
- **样本效率低下 (Sample Inefficiency)。** 在线策略 (On-Policy) 意味着每次更新后都会丢弃所有状态转移 (Transition) 数据。通过重要性采样 (Importance Sampling) 进行离线策略 (Off-Policy) 修正可以重新利用数据，但会以增加方差为代价（PPO 中的比率即为经过截断的重要性采样权重 (Clipped Importance Sampling Weight)）。
- **梯度非平稳性 (Non-Stationary Gradients)。** 100 个回合前的相同梯度是基于旧策略 `π` 计算的。正因如此，在线策略方法通常每隔几次轨迹展开 (Rollout) 就进行一次更新。
- **信用分配 (Credit Assignment)。** 若不采用未来回报 (Reward-to-Go)，过去的奖励会引入噪声。务必始终使用未来回报。

## 实践应用 (Use It)

到2026年，REINFORCE 算法已很少被直接运行，但其梯度公式却无处不在：

| 应用场景 | 衍生方法 |
|----------|---------------|
| 连续控制 (Continuous Control) | 采用高斯策略 (Gaussian Policy) 的 PPO / SAC |
| 大语言模型人类反馈强化学习 (LLM RLHF) | 带 KL 惩罚的 PPO，在词元级策略 (Token-level Policy) 上运行 |
| 大语言模型推理（DeepSeek） | GRPO —— 采用组相对基线 (Group-relative Baseline) 的 REINFORCE，无需价值网络 (Critic) |
| 多智能体 (Multi-agent) | 集中式价值网络 REINFORCE（MADDPG、COMA） |
| 离散动作机器人控制 | A2C、A3C、PPO |
| 仅偏好设置 (Preference-only Settings) | DPO —— 将 REINFORCE 重写为偏好似然损失 (Preference-likelihood Loss)，无需采样 |

当你在 2026 年的训练脚本中看到 `loss = -advantage * log_prob` 时，这实际上就是带基线的 REINFORCE。整篇论文（如 DPO、GRPO、RLOO）的核心，不过是基于这一行代码的方差缩减 (Variance Reduction) 技巧。

## 交付部署

保存为 `outputs/skill-policy-gradient-trainer.md`：

---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.

## 练习

1. **简单。** 在 4×4 网格世界 (GridWorld) 中使用线性 Softmax 策略实现 REINFORCE。在不使用基线的情况下训练 1,000 个回合 (Episodes)。绘制学习曲线 (Learning Curve)；测量方差（回报的标准差）。
2. **中等。** 添加滑动平均基线 (Running-mean Baseline)。重新训练。与基础版本 (Vanilla Run) 对比样本效率 (Sample Efficiency) 与方差。基线将收敛所需的步数减少了多少？
3. **困难。** 添加熵奖励项 (Entropy Bonus) `β · H(π)`。对 `β ∈ {0, 0.01, 0.1, 1.0}` 进行参数扫描。绘制最终回报与策略熵 (Policy Entropy) 的曲线。在该任务中，最佳平衡点位于何处？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 策略梯度 (Policy Gradient) | “直接训练策略” | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`；源自对数导数技巧 (log-derivative trick)。 |
| REINFORCE | “原始的策略梯度算法” | Williams (1992) 提出；将蒙特卡洛回报 (Monte Carlo returns) 与策略对数梯度相乘。 |
| 对数导数技巧 (Log-derivative trick) | “得分函数估计器 (Score function estimator)” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使期望的梯度计算变得可行。 |
| 基线 (Baseline) | “降低方差” | 从 `G` 中减去的任意 `b(s)`；由于 `E[b · ∇ log π] = 0`，该操作保持无偏。 |
| 后续回报 (Reward-to-go) | “仅计算未来回报” | 使用 `G_t^{from t}` 而非完整的 `G_0`；结果正确且方差更低。 |
| 熵奖励 (Entropy bonus) | “鼓励探索” | 添加 `+β · H(π(·|s))` 项，防止策略过早坍缩。 |
| 同策略 (On-policy) | “用刚看到的数据训练” | 梯度期望是针对当前策略的——无法直接复用旧数据。 |
| 优势函数 (Advantage) | “比平均水平好多少” | `A(s, a) = G(s, a) - V(s)`；带基线的 REINFORCE 算法所乘的带符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — REINFORCE 算法的原始论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 结合函数逼近 (function approximation) 的现代策略梯度定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书式的系统讲解。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 配有 PyTorch 代码的清晰教学讲解。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 探讨方差降低技术，并从自然梯度 (natural gradient) 视角揭示了 REINFORCE 与信任域方法族 (trust-region family, 如 TRPO、PPO) 之间的联系。
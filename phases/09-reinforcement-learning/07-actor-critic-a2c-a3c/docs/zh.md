# 演员-评论家（Actor-Critic）— A2C 与 A3C

> REINFORCE 算法的噪声较大。引入一个用于学习 `V̂(s)` 的评论家（critic），并将其从回报（return）中减去，即可得到优势（advantage）。该优势值具有相同的期望，但方差（variance）显著降低。这就是 Actor-Critic 的核心思想。A2C 采用同步方式运行该算法，而 A3C 则通过多线程并行运行。两者构成了所有现代深度强化学习（deep-RL）方法的基础心智模型。

**类型：** 项目构建
**编程语言：** Python
**前置知识：** 第 9 阶段 · 04（时序差分学习（TD Learning）），第 9 阶段 · 06（REINFORCE）
**预计耗时：** 约 75 分钟

## 核心问题

原始的 REINFORCE 算法虽然可行，但方差极高。蒙特卡洛（Monte Carlo）回报 `G_t` 在不同回合（episode）间的波动幅度可达十倍以上。将这种噪声与 `∇ log π` 相乘并取平均，所得到的梯度估计器（gradient estimator）需要经历数千个回合，才能使策略（policy）产生与少量 DQN 更新相同的移动距离。

高方差源于直接使用原始回报。若从中减去一个基线（baseline） `b(s_t)`（即状态（state）的任意函数，包含已学习的价值函数（value function）），期望值将保持不变，而方差会显著降低。最实用且易于计算的基线是 `V̂(s_t)`。此时，与 `∇ log π` 相乘的项即为*优势（advantage）*：

`A(s, a) = G - V̂(s)`

若某动作产生的回报高于平均水平，则该动作较优；反之则较差。引入已学习评论家的 REINFORCE 算法即为*Actor-Critic（演员-评论家）*。评论家为演员（actor）提供了低方差的指导信号。2015 年之后的所有深度策略方法（deep-policy method）（如 A2C、A3C、PPO、SAC、IMPALA）均以此为核心。

## 核心概念

![Actor-Critic：策略网络（Actor）与价值网络（Critic）结合，TD残差作为优势函数](../assets/actor-critic.svg)

**双网络，共享单一损失函数：**

- **策略网络（Actor）** `π_θ(a | s)`：表示策略。通过采样生成动作。采用策略梯度（Policy Gradient）进行训练。
- **价值网络（Critic）** `V_φ(s)`：估计从当前状态出发的期望回报。通过最小化 `(V_φ(s) - target)²` 进行训练。

**优势函数（Advantage）。** 两种标准形式：

- *蒙特卡洛优势（MC advantage）：* `A_t = G_t - V_φ(s_t)`。无偏估计，但方差较高。
- *时序差分优势（TD advantage）：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏估计（依赖 `V_φ`），但方差显著更低。也称为 *TD残差（TD residual）* `δ_t`。

**n步优势（n-step advantage）。** 在上述两者之间进行插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 对应纯时序差分（TD）方法。`n = ∞` 对应蒙特卡洛（MC）方法。大多数实现中，Atari 任务通常使用 `n = 5`，而在 MuJoCo 上运行 PPO 时则使用 `n = 2048`。

**广义优势估计（Generalized Advantage Estimation, GAE）。** Schulman 等人（2016）提出了一种对所有 n 步优势进行指数加权平均的方法：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 对应 TD（低方差，高偏差）。`λ = 1` 对应 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值——请根据实际需求调节偏差/方差权衡点。

**A2C：同步优势 Actor-Critic。** 在 `N` 个并行环境中收集 `T` 步数据。计算每一步的优势值。在合并后的批次上更新 Actor 和 Critic。重复此过程。它是 A3C 更简单、更易扩展的变体。

**A3C：异步优势 Actor-Critic。** Mnih 等人（2016）提出。生成 `N` 个工作线程，每个线程运行一个环境。每个工作线程在本地基于自身的轨迹（rollout）计算梯度，然后异步地将梯度应用到共享的参数服务器上。无需经验回放缓冲区（replay buffer）——工作线程通过运行不同的轨迹来实现去相关。A3C 证明了可以在 CPU 上进行大规模训练。到了 2026 年，基于 GPU 的 A2C（批处理并行环境）占据主导地位，因为 GPU 更适合处理大批量数据。

**组合损失函数。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

包含三项：策略梯度损失、价值回归、熵奖励（entropy bonus）。`c_v ~ 0.5` 和 `c_e ~ 0.01` 是标准的初始参考值。

## 动手实现

### 步骤 1：价值评估器（Critic）

使用均方误差（Mean Squared Error, MSE）更新的线性价值评估器 `V_φ(s) = w · features(s)`：

def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat

在表格型环境（tabular environment）中，该价值评估器通常在几百个回合（episode）内即可收敛。在 Atari 环境中，需将线性价值评估器替换为共享的卷积神经网络（Convolutional Neural Network, CNN）主干网络加价值头（value head）。

### 步骤 2：n 步优势函数（n-step Advantage）

给定长度为 `T` 的轨迹（rollout）以及经过自举（bootstrapped）的最终状态价值 `V(s_T)`：

def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns

`returns` 是价值评估器的目标值。`advantages` 则是用于乘以策略梯度 `∇ log π` 的系数。

### 步骤 3：联合更新（Combined Update）

for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]

采用同策略（on-policy）学习，每次更新使用一条轨迹，且策略网络（Actor）与价值评估器（Critic）使用独立的学习率。

### 步骤 4：并行化（A3C 与 A2C 对比）

- **A3C：** 启动 `N` 个线程。每个线程独立运行自己的环境并执行前向传播（forward pass）。定期将梯度更新推送至共享的主节点（master）。主节点无需加锁——数据竞争（race condition）是可以接受的，它们仅会增加一些噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测值堆叠为 `[N, obs_dim]` 的批次（batch），执行批处理前向传播与反向传播。GPU 利用率更高，具有确定性（deterministic），且更易于理解和调试。这是 2026 年的默认选择。

为了保持清晰，我们的示例代码采用单线程实现；将其改写为批处理 A2C 仅需三行 NumPy 代码。

## 常见陷阱（Pitfalls）

- **在计算 Actor（策略网络）梯度前先消除 Critic（价值网络）偏差。** 如果 Critic 是随机初始化的，其提供的基线将毫无信息量，你实际上是在纯噪声上进行训练。在开启策略梯度（Policy Gradient）训练前，先对 Critic 进行几百步的预热（Warm-up），或者使用较小的 Actor 学习率。
- **优势函数归一化（Advantage Normalization）。** 对每个批次（Batch）的优势值进行零均值、单位标准差归一化。这能以几乎为零的代价大幅提升训练稳定性。
- **共享主干网络（Shared Trunk）。** 对于图像输入，Actor 和 Critic 应共享同一个特征提取器（Feature Extractor），但使用独立的输出头（Heads）。共享特征可同时从两者的损失函数中获益。
- **同策略（On-policy）约束。** A2C 仅对数据执行一次更新。重复使用会导致梯度产生偏差（PPO 引入的重要性采样修正（Importance-sampling Correction）正是为了解决此问题）。
- **熵崩溃（Entropy Collapse）。** 若熵系数 `c_e` 不大于 0，策略网络在几百次更新后就会趋于确定性，从而停止探索。
- **奖励缩放（Reward Scale）。** 优势值的幅度取决于奖励的缩放比例。对奖励进行归一化（例如除以滑动标准差），以确保不同任务间的梯度幅度保持一致。

## 使用场景

到 2026 年，A2C/A3C 已很少作为最终方案，但它们构成了后续所有改进算法的基础架构：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 用于多轮更新的截断重要性比率（Clipped Importance Ratio） |
| IMPALA | A3C + V-trace 异策略（Off-policy）修正 |
| SAC (Phase 9 · 07) | 采用软价值 Critic 的异策略 A2C（下节课内容） |
| GRPO (Phase 9 · 12) | 无 Critic 的 A2C —— 基于组内相对优势（Group-relative Advantage） |
| DPO | 将 A2C 简化为偏好排序损失（Preference-ranking Loss），无需采样 |
| AlphaStar / OpenAI Five | A2C + 联赛训练（League Training）+ 模仿预训练（Imitation Pre-training） |

如果在 2026 年的论文中看到“优势（Advantage）”一词，请立刻联想到 Actor-Critic（演员-评论家）架构。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.

## 练习

1. **简单。** 在 4×4 GridWorld 环境中，使用蒙特卡洛（MC）优势函数（`G_t - V(s_t)`）训练 Actor-Critic。将其样本效率（Sample Efficiency）与第 06 课中“带滑动均值基线的 REINFORCE”进行对比。
2. **中等。** 切换至时序差分残差（TD-residual）优势函数（`r + γ V(s') - V(s)`）。测量优势批次的方差。方差下降了多少？
3. **困难。** 实现广义优势估计（Generalized Advantage Estimation, GAE(λ)）。对 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}` 进行扫描（Sweep）。绘制最终回报（Final Return）与样本效率的关系图。对于该任务，偏差/方差（Bias/Variance）的最佳平衡点在哪里？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 执行者 (Actor) | "策略网络" | `π_θ(a|s)`，通过策略梯度 (Policy Gradient) 更新。 |
| 评估者 (Critic) | "价值网络" | `V_φ(s)`，通过均方误差 (MSE) 回归拟合回报 (Returns) 或时序差分 (TD) 目标进行更新。 |
| 优势函数 (Advantage) | "比平均水平好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计值。作为 `∇ log π` 的乘数。 |
| 时序差分残差 (TD residual) | "δ" | `δ_t = r + γ V(s') - V(s)`；单步优势估计。 |
| 广义优势估计 (GAE) | "插值调节旋钮" | n步优势 (n-step advantages) 的指数加权和，由参数 `λ` 控制。 |
| 同步Actor-Critic (A2C) | "同步Actor-Critic" | 跨多个环境进行批处理；每次轨迹展开 (rollout) 执行一次梯度更新。 |
| 异步Actor-Critic (A3C) | "异步Actor-Critic" | 工作线程将梯度推送至共享参数服务器。源自原始论文；至2026年已较少使用。 |
| 自举法 (Bootstrap) | "在视界处使用V值" | 截断轨迹展开，并添加 `γ^n V(s_{t+n})` 以闭合求和。 |

## 延伸阅读

- [Mnih 等人 (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C 算法，异步 Actor-Critic 的原始论文。
- [Schulman 等人 (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — 提出 GAE（广义优势估计）。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 理论基础；当 Critic（评估者）为神经网络时，建议结合第 9 章函数近似 (Function Approximation) 的内容一同阅读。
- [Espeholt 等人 (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 采用 V-trace 异策略修正 (Off-policy Correction) 的可扩展分布式 Actor-Critic 算法。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得参考的 A2C/PPO 工业级实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 双时间尺度 (Two-timescale) Actor-Critic 分解的基础收敛性证明。
# 时序差分（Temporal Difference）—— Q学习（Q-Learning）与 SARSA

> 蒙特卡洛（Monte Carlo）方法会等待整个回合（episode）结束。时序差分（TD）则通过自举（bootstrapping）下一步的价值估计，在每一步之后进行更新。Q学习（Q-Learning）属于异策略（off-policy）且偏向乐观；SARSA 属于同策略（on-policy）且偏向谨慎。两者都仅需一行代码，并共同构成了本阶段所有深度强化学习（Deep Reinforcement Learning, Deep-RL）方法的基石。

**类型：** 构建
**语言：** Python
**前置知识：** 第9阶段 · 01（马尔可夫决策过程 MDPs）、第9阶段 · 02（动态规划 Dynamic Programming）、第9阶段 · 03（蒙特卡洛 Monte Carlo）
**预计时间：** 约 75 分钟

## 问题所在

蒙特卡洛方法虽然有效，但有两个代价高昂的要求。它需要能够终止的回合，并且只有在最终回报（return）确定后才能进行更新。如果你的回合长达 1,000 步，蒙特卡洛就必须等待 1,000 步才能更新任何内容。在实际应用中，它具有高方差、低偏差且更新缓慢的特点。

动态规划（Dynamic Programming）则呈现出完全相反的特性——零方差的自举备份（bootstrapped backups）——但它要求环境模型（model）已知。

时序差分（TD）学习则取两者之长。基于单次状态转移（transition） `(s, a, r, s')`，构建单步目标 `r + γ V(s')`，并将 `V(s)` 向该目标微调。无需环境模型，也无需完整的回合。虽然在等式右侧（RHS）使用近似价值函数 `V` 会引入偏差，但其方差远低于蒙特卡洛方法，并且能够从第一步开始进行在线更新。

这正是现代强化学习（Reinforcement Learning, RL）——包括 DQN、A2C、PPO、SAC 等算法——的核心枢纽。第9阶段的剩余内容，都是在你本节课将编写的单步 TD 更新基础上，叠加函数近似（function approximation）层与各种优化技巧。

## 核心概念

![Q-learning 与 SARSA 对比：异策略 max 与同策略 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号内的量即为时序差分误差（Temporal Difference error）`δ = r + γ V(s') - V(s)`。它是蒙特卡洛（Monte Carlo）方法中 `G_t - V(s_t)` 的在线（online）类比。收敛要求学习率 `α` 满足罗宾斯-门罗（Robbins-Monro）条件（`Σ α = ∞`, `Σ α² < ∞`），且所有状态被无限次访问。

**Q-learning（Q学习）。** 一种用于控制任务的异策略（off-policy）时序差分方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

其中的 `max` 假设从状态 `s'` 开始将遵循*贪婪*（greedy）策略，无论智能体实际采取何种动作。这种解耦使得 Q-learning 能够在智能体通过 ε-贪婪（ε-greedy）策略进行探索的同时，学习到最优动作价值函数 `Q*`。Mnih 等人（2015）将其应用于 Atari 游戏，发展出了深度 Q 学习（Deep Q-learning）（第 05 课）。

**SARSA。** 一种同策略（on-policy）时序差分方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

其名称来源于状态-动作元组 `(s, a, r, s', a')`。SARSA 使用智能体下一步*实际*采取的动作 `a'`，而非贪婪策略下的 `argmax`。它会收敛到当前运行的任意 ε-贪婪策略 `π` 对应的 `Q^π`，当 `ε → 0` 时，该值趋近于 `Q*`。

**悬崖漫步（Cliff-walking）差异。** 在经典的悬崖漫步任务中（跌落悬崖奖励为 -100），Q-learning 会学习到沿悬崖边缘的最优路径，但在探索过程中偶尔会触发惩罚。SARSA 会学习到一条距离悬崖一步之遥的更安全路径，因为它将探索噪声纳入了 Q 值的计算中。经过充分训练，两者在 `ε → 0` 时均能达到最优。在实际应用中这很重要：当部署阶段仍在进行探索时，SARSA 的行为更为保守。

**期望 SARSA（Expected SARSA）。** 将 `Q(s', a')` 替换为策略 `π` 下的期望值：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

相比 SARSA 方差更低（无需对 `a'` 进行采样），且保持相同的同策略目标。通常是现代教材中的默认选择。

**n步时序差分（n-step TD）与 TD(λ)。** 通过在进行自举（bootstrapping）前等待 `n` 步，在 TD(0) 与蒙特卡洛方法之间进行插值。`n=1` 对应 TD，`n=∞` 对应蒙特卡洛方法。TD(λ) 使用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 进行加权平均。大多数深度强化学习（Deep Reinforcement Learning）算法使用的 `n` 值介于 3 到 20 之间。

## 动手实践

### 步骤 1：基于 ε-贪婪策略 (ε-greedy policy) 的 SARSA

def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q

仅八行代码。与 Q 学习 (Q-learning) 的*唯一*区别在于目标值 (target) 的计算行。

### 步骤 2：Q 学习 (Q-learning)

def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q

`max` 操作符将目标值 (target) 与行为策略 (behavior policy) 解耦。正是这一个符号，区分了同策略 (on-policy) 与异策略 (off-policy) 算法。

### 步骤 3：学习曲线 (learning curves)

记录每 100 个回合 (episode) 的平均回报 (return)。在简单的确定性网格世界 (GridWorld) 中，Q 学习收敛更快；而在悬崖行走 (cliff-walking) 任务中，SARSA 的表现更为保守。在 `code/main.py` 的 4×4 网格世界中，当设置 `α=0.1, ε=0.1` 时，两者在约 2,000 个回合后均能达到接近最优 (near-optimal) 的性能。

### 步骤 4：与动态规划 (DP) 的真实值对比

运行值迭代 (value iteration)（见第 02 课）以获取最优动作价值函数 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个表现正常的表格型时序差分 (tabular TD) 智能体在 10,000 个回合后，在 4×4 网格世界中的误差应落在 `~0.5` 以内。

## 常见陷阱 (Pitfalls)

- **初始 Q 值至关重要。** 乐观初始化（对于负奖励任务设置 `Q = 0`）能鼓励探索 (exploration)。悲观初始化则可能导致贪婪策略 (greedy policy) 永远陷入局部最优。
- **学习率 (α) 调度策略。** 对于非平稳问题 (non-stationary problems)，使用恒定 `α` 即可。理论上衰减学习率 `α_n = 1/n` 能保证收敛，但实际中速度过慢——建议将 `α` 固定在 `[0.05, 0.3]` 范围内，并密切监控学习曲线。
- **探索率 (ε) 调度策略。** 从较高值开始（`ε=1.0`），逐渐衰减至 `ε=0.05`。“GLIE”（无限探索下的极限贪婪，greedy in the limit with infinite exploration）是算法收敛的必要条件。
- **Q 学习中的最大值偏差 (Max bias)。** 当 `Q` 值存在噪声时，`max` 操作符会产生向上偏差，导致价值高估 (overestimation)。Hasselt 提出的双重 Q 学习 (Double Q-learning)（第 05 课的 DDQN 即采用此方法）通过维护两张 Q 表解决了该问题。
- **非终止回合 (Non-terminating episodes)。** 时序差分 (TD) 算法可以在没有终止状态的情况下学习，但你需要限制最大步数，或在达到上限时正确处理自举 (bootstrap)。标准做法：将步数上限视为非终止状态，继续进行自举更新。
- **状态哈希 (State hashing)。** 如果状态是元组或张量 (tensors)，请使用可哈希的键（使用元组而非列表；使用四舍五入后的浮点数元组，而非原始浮点数）。

## 实践应用 (Use It)

2026 年时序差分（Temporal Difference, TD）学习领域概览：

| 任务场景 | 方法 | 原因 |
|------|--------|--------|
| 小型表格型环境 | Q 学习（Q-learning） | 直接学习最优策略。 |
| 在线策略（On-policy）安全关键型任务 | SARSA / 期望 SARSA（Expected SARSA） | 探索阶段更为保守。 |
| 高维状态空间 | DQN（第 9 阶段 · 05） | 结合经验回放与目标网络的神经网络 Q 函数。 |
| 连续动作空间 | SAC / TD3（第 9 阶段 · 07） | 在 Q 网络上执行 TD 更新；策略网络输出动作。 |
| 基于奖励模型的大语言模型强化学习（LLM RL） | PPO / GRPO（第 9 阶段 · 08, 12） | 采用 Actor-Critic 架构，通过广义优势估计（Generalized Advantage Estimation, GAE）计算类 TD 优势函数。 |
| 离线强化学习（Offline RL） | CQL / IQL（第 9 阶段 · 08） | 带有保守正则化的 Q 学习。 |

你在 2026 年论文中读到的“强化学习（Reinforcement Learning, RL）”内容，有 90% 都是 Q 学习或 SARSA 的某种变体。在深入阅读之前，务必将表格型更新（Tabular Update）的推导过程烂熟于心。

## 交付上线

保存为 `outputs/skill-td-agent.md`：

---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).

## 练习

1. **简单。** 在 4×4 网格世界（GridWorld）中实现 Q 学习与 SARSA。绘制 2000 个回合（episodes）的学习曲线（每 100 个回合的平均回报）。哪种算法收敛更快？
2. **中等。** 构建悬崖漫步（Cliff-walking）环境（4×12 网格，最后一行为悬崖，触碰奖励为 -100 并重置至起点）。对比 Q 学习与 SARSA 的最终策略。截图展示两者采取的路径。哪种策略的路径更靠近悬崖？
3. **困难。** 实现双 Q 学习（Double Q-learning）。在带有噪声奖励的网格世界中（每步奖励添加高斯噪声 σ=5），证明 Q 学习会对 `V*(0,0)` 产生显著的高估，而双 Q 学习则不会。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 时序差分误差 (TD error) | “更新信号” | `δ = r + γ V(s') - V(s)`，即自举残差。 |
| 单步时序差分 (TD(0)) | “单步TD” | 每次状态转移后，仅利用下一状态的估计值进行更新。 |
| Q学习 (Q-learning) | “异策略强化学习入门” | 对下一状态动作取 `max` 的TD更新；无论行为策略如何，均学习最优动作价值函数 `Q*`。 |
| SARSA | “同策略Q学习” | 使用实际采取的下一动作进行TD更新；学习当前 ε-贪婪策略 (ε-greedy policy) π 下的动作价值函数 `Q^π`。 |
| 期望SARSA (Expected SARSA) | “低方差SARSA” | 用策略 π 下的期望值替代采样得到的 `a'`。 |
| 极限贪婪无限探索 (GLIE) | “正确的探索调度” | 无限探索下的极限贪婪；是保证 Q学习 (Q-learning) 收敛的必要条件。 |
| 自举 (Bootstrapping) | “在目标值中使用当前估计” | 区分时序差分 (TD) 与蒙特卡洛 (MC) 方法的核心特征。虽会引入偏差，但能大幅降低方差。 |
| 最大化偏差 (Maximization bias) | “Q学习会高估” | 对含噪声的估计值取 `max` 会产生向上偏差；可通过双Q学习 (Double Q-learning) 进行修正。 |

## 扩展阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) —— 原始论文及收敛性证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) —— 涵盖 TD(0)、SARSA、Q学习 (Q-learning) 与期望SARSA (Expected SARSA)。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) —— 针对最大化偏差 (Maximization bias) 的修正方法。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) —— 期望SARSA (Expected SARSA) 的提出动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) —— 首次提出 SARSA 概念的论文（当时称为“改进的连接主义Q学习”）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) —— 将 TD(0) 推广至 TD(n)，阐述了从 Q学习 (Q-learning) 到资格迹 (eligibility traces)，再到后续 PPO 中广义优势估计 (GAE) 的演进路径。
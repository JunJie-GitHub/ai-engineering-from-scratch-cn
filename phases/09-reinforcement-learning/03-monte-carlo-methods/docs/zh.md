# 蒙特卡洛方法（Monte Carlo Methods）—— 从完整回合（Episodes）中学习

> 动态规划（Dynamic Programming）需要一个模型。蒙特卡洛方法除了回合之外一无所求。运行策略（Policy），观察回报（Returns），然后求平均。这是强化学习（Reinforcement Learning, RL）中最简单的思想——也是开启后续一切的关键。

**类型：** 构建
**语言：** Python
**前置知识：** 第 9 阶段 · 01（马尔可夫决策过程（Markov Decision Processes, MDPs）），第 9 阶段 · 02（动态规划）
**预计时间：** 约 75 分钟

## 问题所在

动态规划虽然优雅，但它假设你可以针对每一个状态（State）和动作（Action）查询 `P(s' | s, a)`。现实世界中几乎没有任何系统能这样运作。机器人无法通过解析计算得出施加关节扭矩后相机像素的分布。定价算法无法对所有可能的客户反应进行积分。大语言模型（Large Language Model, LLM）也无法枚举出一个词元（Token）之后所有可能的续写。

你需要一种只需具备从环境中*采样（Sample）*能力的方法。运行策略，获取一条轨迹（Trajectory）`s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`，并用它来估计价值（Value）。这就是蒙特卡洛方法。

从动态规划到蒙特卡洛方法的转变在哲学层面至关重要：我们从*已知模型 + 精确值更新（Exact Backup）*转向了*采样推演（Sampled Rollouts） + 平均回报*。虽然方差（Variance）会增大，但适用性却呈指数级扩展。本课程之后的每一个强化学习算法——时序差分（Temporal Difference, TD）、Q学习（Q-learning）、REINFORCE、近端策略优化（Proximal Policy Optimization, PPO）、组相对策略优化（Group Relative Policy Optimization, GRPO）——其核心都是蒙特卡洛估计器（Monte Carlo Estimator），有时只是在此基础上叠加了自举法（Bootstrapping）。

## 核心概念

![蒙特卡洛（Monte Carlo）：推演（rollout）、计算回报（returns）、求平均；首次访问（first-visit）与每次访问（every-visit）](../assets/monte-carlo.svg)

**核心思想（一句话概括）：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 表示在策略（policy）`π` 下访问状态（state）`s` 后所观测到的回报。

**首次访问与每次访问蒙特卡洛。** 假设一个回合（episode）多次访问状态 `s`，首次访问 MC 仅统计第一次访问的回报；每次访问 MC 则统计所有访问的回报。两者在极限情况下均为无偏估计。首次访问更易于理论分析（样本独立同分布）。每次访问在每个回合中利用了更多数据，实践中通常收敛更快。

**增量均值（incremental mean）。** 无需存储所有回报，直接更新运行平均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新整理公式：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。若将 `1/n` 替换为常数步长（step-size）`α ∈ (0, 1)`，即可得到一个非平稳的 MC 估计器，用于跟踪策略 `π` 的变化。这一转变正是从 MC 到时序差分（Temporal Difference, TD），再到所有现代强化学习（Reinforcement Learning, RL）算法的核心跨越。

**探索（exploration）成为新问题。** 动态规划（Dynamic Programming, DP）通过枚举触及每个状态。MC 仅能观测到策略实际访问的状态。若 `π` 是确定性（deterministic）的，状态空间中的大片区域将永远无法被采样，其价值估计也会永远保持为零。按历史发展顺序，有三种解决方案：

1. **探索性初始状态（exploring starts）。** 每个回合从随机的 (s, a) 状态-动作对开始。能保证状态覆盖；但在实践中不切实际（你无法将机器人“重置”到任意状态）。
2. **ε-贪心（ε-greedy）。** 根据当前 Q 值采取贪心动作，但以概率 `ε` 随机选择动作。所有状态-动作对在渐进意义上都能被采样到。
3. **异策略（off-policy）MC。** 在行为策略（behavior policy）`μ` 下收集数据，通过重要性采样（importance sampling）学习目标策略（target policy）`π`。方差较高，但它是通向 DQN 等经验回放（replay buffer）方法的桥梁。

**蒙特卡洛控制（Monte Carlo Control）。** 评估 → 改进 → 评估，与策略迭代（policy iteration）类似，但评估过程基于采样：

1. 执行策略 `π`，获取一个回合。
2. 根据观测到的回报更新 `Q(s, a)`。
3. 使策略 `π` 相对于 `Q` 变为 ε-贪心策略。
4. 重复上述步骤。

在温和条件下（每个状态-动作对被无限次访问，且 `α` 满足 Robbins-Monro 条件），算法以概率 1 收敛至 `Q*` 和 `π*`。

## 动手实现

### 步骤 1：轨迹展开（rollout）→ (s, a, r) 列表

def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory

无需模型，仅依赖 `env.reset()` 和 `env.step(s, a)`。其接口与 gym 环境一致，但更为精简。

### 步骤 2：计算回报（returns）（反向遍历）

def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))

仅需单次遍历，时间复杂度为 `O(T)`。反向递推（backward recurrence）公式 `G_t = r_{t+1} + γ G_{t+1}` 避免了重复求和。

### 步骤 3：首次访问（first-visit）蒙特卡洛（Monte Carlo, MC）评估

def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V

核心逻辑仅需三行：在首次访问时标记状态、增加计数、更新运行均值（running mean）。

### 步骤 4：ε-贪心（ε-greedy）蒙特卡洛控制（同策略/on-policy）

def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy

### 步骤 5：与动态规划（Dynamic Programming, DP）基准结果对比

随着回合数（episodes）趋于无穷大，你对 `V^π` 的蒙特卡洛估计值应与第 02 课中的动态规划结果一致。在实际应用中：在 4×4 网格世界（GridWorld）上运行 50,000 个回合，即可使结果与动态规划答案的误差控制在 `~0.1` 以内。

## 常见陷阱

- **无限回合（Infinite episodes）。** 蒙特卡洛（Monte Carlo, MC）方法要求回合（episode）必须*终止*。如果你的策略（policy）可能陷入无限循环，请设置 `max_steps` 上限，并将达到上限视为隐式失败。在 GridWorld 环境中使用随机策略时经常会出现超时——这是正常现象，只需确保正确统计即可。
- **方差（Variance）。** MC 使用完整回报（return）。在长回合中，方差极大——末尾一次不幸的奖励（reward）会使 `V(s_0)` 产生同等幅度的偏移。时序差分（Temporal Difference, TD）方法（第 04 课）通过自举（bootstrapping）来降低这一问题。
- **状态覆盖（State coverage）。** 在全新的 Q 表上使用贪婪 MC 且存在平局时，算法只会尝试单一动作。你*必须*进行探索（exploration）（如 ε-贪婪（ε-greedy）、探索性初始状态（exploring starts）、上置信界（Upper Confidence Bound, UCB））。
- **非平稳策略（Non-stationary policies）。** 如果策略 `π` 发生变化（如在 MC 控制中），旧的回报数据将来自不同的策略。常数步长 MC（Constant-α MC）可以处理此问题；而样本平均 MC（sample-average MC）则不能。
- **异策重要性采样（Off-policy importance sampling）。** 权重 `π(a|s)/μ(a|s)` 会沿轨迹（trajectory）连乘。随着时间跨度（horizon）增加，方差会急剧膨胀。可通过逐决策加权重要性采样（per-decision weighted IS）设置上限，或切换至 TD 方法。

## 应用场景

| 应用场景 | 为何选择 MC |
|----------|--------|
| 短跨度游戏（如二十一点、扑克） | 回合自然终止；回报计算清晰。 |
| 记录策略的离线评估 | 对存储的轨迹计算平均折扣回报。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树节点叶子进行的 MC 模拟推演（rollout）指导节点选择。 |
| 大语言模型（LLM）强化学习评估 | 针对给定策略，计算采样生成结果的平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用了 MC 计算的 `G_t`。 |
| 强化学习教学 | 最简单且真正有效的算法——剥离自举机制即可看清核心原理。 |

现代深度强化学习（Deep RL）算法（如 PPO、SAC）通过 `n` 步回报（`n`-step returns）或广义优势估计（Generalized Advantage Estimation, GAE），在纯 MC（完整回报）与纯 TD（单步自举）之间进行插值。这两个端点本质上是同一估计器的不同实例。

## 交付指南

保存为 `outputs/skill-mc-evaluator.md`：

---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.

## 练习

1. **简单。** 在 4×4 网格世界（GridWorld）中实现针对均匀随机策略（uniform-random policy）的首次访问蒙特卡洛（Monte Carlo, MC）评估。运行 10,000 个回合（episodes）。绘制 `V(0,0)` 随回合数变化的曲线，并与动态规划（Dynamic Programming, DP）的基准答案进行对比。
2. **中等。** 实现 ε-贪婪（ε-greedy）蒙特卡洛控制，设置 `ε ∈ {0.01, 0.1, 0.3}`。比较 20,000 个回合后的平均回报（mean return）。曲线呈现何种形态？偏差-方差权衡（bias-variance tradeoff）体现在何处？
3. **困难。** 结合重要性采样（importance sampling）实现*异策略*（off-policy）蒙特卡洛：在均匀随机策略 `μ` 下收集数据，用于估计确定性最优策略 `π` 的 `V^π`。对比普通重要性采样（plain IS）、逐决策重要性采样（per-decision IS）与加权重要性采样（weighted IS）。哪种方法的方差最低？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 蒙特卡洛（Monte Carlo） | “随机采样” | 通过对分布中独立同分布（iid）样本求平均来估计期望值。 |
| 回报（Return） `G_t` | “未来奖励” | 从第 `t` 步到回合结束的折扣奖励之和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问蒙特卡洛（First-visit MC） | “每个状态只计一次” | 仅回合中对该状态的首次访问参与价值估计。 |
| 每次访问蒙特卡洛（Every-visit MC） | “使用所有访问记录” | 每次访问均参与计算；存在轻微偏差，但样本效率更高。 |
| ε-贪婪（ε-greedy） | “探索噪声” | 以 `1-ε` 的概率选择贪婪动作；以 `ε` 的概率选择随机动作。 |
| 重要性采样（Importance sampling） | “修正错误分布的采样偏差” | 通过 `π(a|s)/μ(a|s)` 的乘积对回报进行重新加权，从而利用 `μ` 的数据估计 `V^π`。 |
| 同策略（On-policy） | “从自己的数据中学习” | 目标策略等于行为策略。例如：基础蒙特卡洛（Vanilla MC）、PPO、SARSA。 |
| 异策略（Off-policy） | “从他人的数据中学习” | 目标策略不等于行为策略。例如：重要性采样蒙特卡洛、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). 第 5 章 — 蒙特卡洛方法](http://incompleteideas.net/book/RLbook2020.pdf) —— 该领域的权威参考著作。
- [Singh & Sutton (1996). 基于替换资格迹的强化学习](https://link.springer.com/article/10.1007/BF00114726) —— 首次访问与每次访问方法的对比分析。
- [Precup, Sutton, Singh (2000). 用于异策略评估的资格迹](http://incompleteideas.net/papers/PSS-00.pdf) —— 异策略蒙特卡洛与方差控制。
- [Mahmood 等 (2014). 用于异策略学习的加权重要性采样](https://arxiv.org/abs/1404.6362) —— 现代低方差重要性采样估计器。
- [Tesauro (1995). TD-Gammon：一款自我教学的双陆棋程序](https://dl.acm.org/doi/10.1145/203330.203343) —— 首次大规模实证展示了蒙特卡洛/时序差分（MC/TD）自我对弈收敛至超越人类水平的过程；奠定了本阶段后半部分所有课程的概念基础。
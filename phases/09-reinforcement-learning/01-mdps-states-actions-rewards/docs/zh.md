# 马尔可夫决策过程 (Markov Decision Process, MDP)、状态 (State)、动作 (Action) 与奖励 (Reward)

> 马尔可夫决策过程 (Markov Decision Process, MDP) 包含五个要素：状态 (State)、动作 (Action)、状态转移 (Transition)、奖励 (Reward) 和折扣因子 (Discount)。强化学习 (Reinforcement Learning, RL) 中的一切——无论是 Q-learning、PPO、DPO 还是 GRPO——都在此结构上进行优化。掌握它一次，后续强化学习的内容便可一通百通。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第一阶段 · 06（概率与分布），第二阶段 · 01（机器学习分类体系）
**Time:** 约 45 分钟

## 问题引入

假设你正在编写一个国际象棋机器人，或者一个库存规划器，又或者一个交易智能体，甚至是用于训练推理模型的 PPO 循环。这四个截然不同的领域，却有一个令人惊讶的共同点：它们最终都归结为同一个数学对象。

监督学习 (Supervised Learning) 提供的是 `(x, y)` 数据对，并要求你拟合一个函数。而强化学习不提供任何标签——它只给你一串状态流、你采取的动作以及一个标量奖励 (Scalar Reward)。这一步棋赢下比赛了吗？这次补货决策省钱了吗？这笔交易盈利了吗？大语言模型 (Large Language Model, LLM) 刚刚生成的词元 (Token)，是否让评判模型 (Judge) 给出了更高的奖励？

在将其形式化之前，你无法从这种数据流中进行学习。“我看到了什么”、“我做了什么”、“接下来发生了什么”、“结果有多好”——每一项都必须转化为可供数学推理的对象。这种形式化描述正是马尔可夫决策过程。本阶段涉及的每一个强化学习算法，包括末尾的基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF) 和 GRPO 循环，都是在此结构上进行优化的。

## 核心概念

![马尔可夫决策过程（Markov Decision Process）：状态、动作、转移、奖励、折扣](../assets/mdp.svg)

**五大核心要素。**

- **状态（States）** `S`。智能体（Agent）进行决策所需的全部信息。在网格世界（GridWorld）中是单元格；在国际象棋中是棋盘；在大语言模型（LLM）中是上下文窗口（Context Window）加上任何记忆。
- **动作（Actions）** `A`。可供选择的选项。向上/下/左/右移动。走一步棋。生成一个词元（Token）。
- **状态转移（Transitions）** `P(s' | s, a)`。给定当前状态 `s` 和动作 `a` 后，下一状态的概率分布。在国际象棋中是确定性的，在库存管理中是随机的，在 LLM 解码中几乎是确定性的。
- **奖励（Rewards）** `R(s, a, s')`。标量信号。胜利 = +1，失败 = -1。收入减去成本。GRPO 中的对数似然比（Log-Likelihood Ratio）项。
- **折扣因子（Discount Factor）** `γ ∈ [0, 1)`。衡量未来奖励相对于当前奖励的重要性。`γ = 0.99` 对应约 100 步的视野（Horizon）；`γ = 0.9` 对应约 10 步。

**马尔可夫性质（Markov Property）** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来仅取决于当前状态。若不满足该性质，则说明状态表示不完整——这不是算法的缺陷，而是状态设计的不足。

**策略与回报（Policies and Returns）。** 策略 `π(a | s)` 将状态映射为动作的概率分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣累加和。状态价值函数 `V^π(s) = E[G_t | s_t = s]` 表示在策略 `π` 下从状态 `s` 出发的期望回报。动作价值函数（Q值） `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 表示在状态 `s` 执行特定动作 `a` 后的期望回报。每个强化学习（Reinforcement Learning, RL）算法都会估计这两者之一，并据此优化策略 `π`。

**贝尔曼方程（Bellman Equations）。** 本阶段所有方法所依赖的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报拆分为“当前步的奖励”加上“下一状态的折扣价值”。具有递归特性。第 9 阶段中的每个算法要么通过迭代该方程直至收敛（动态规划，Dynamic Programming），要么从中进行采样（蒙特卡洛方法，Monte Carlo），要么进行单步自举（时序差分，Temporal Difference）。

## 动手实现

### 步骤 1：一个小型确定性马尔可夫决策过程 (Markov Decision Process, MDP)

一个 4×4 的网格世界 (GridWorld)。智能体 (Agent) 从左上角出发，终止状态 (Terminal State) 位于右下角，每步奖励为 -1，可选动作为 `{up, down, left, right}`。详见 `code/main.py`。

GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL

仅五行代码。这就是完整的环境。包含确定性状态转移、恒定步长惩罚以及吸收型终止状态。

### 步骤 2：策略执行 (Policy Rollout)

策略 (Policy) 是一个从状态映射到动作分布的函数。最简单的形式是均匀随机策略。

def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps

将该随机策略运行 1000 次。在此 4×4 网格中，平均回报 (Return) 约为 -60 到 -80。最优回报为 -6（沿右下方向直线行进）。缩小这一差距正是第 9 阶段的核心目标。

### 步骤 3：通过贝尔曼方程 (Bellman Equation) 精确计算 `V^π`

对于小型 MDP，贝尔曼方程构成一个线性方程组。枚举所有状态，计算期望值，并迭代直至状态价值不再变化。

def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V

这就是迭代策略评估 (Iterative Policy Evaluation)。它是 Sutton & Barto 著作中的首个算法，也是后续所有强化学习 (Reinforcement Learning, RL) 方法的理论基石。

### 步骤 4：`γ` 是具有物理意义的超参数 (Hyperparameter)

有效视界 (Effective Horizon) 大约为 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

若 `γ` 过低，智能体会表现出短视行为。若 `γ` 过高，信用分配 (Credit Assignment) 会变得充满噪声，因为许多早期步骤会共同影响遥远的未来奖励。大语言模型 (Large Language Model, LLM) 的基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF) 通常采用 `γ = 1`，因为其回合 (Episode) 较短且有明确边界。控制任务通常使用 `0.95–0.99`。长视界策略游戏则使用 `0.999`。

## 常见陷阱 (Pitfalls)

- **非马尔可夫状态（Non-Markovian state）。** 如果决策需要依赖最近三次观测，那么“状态”就不仅仅是当前观测。修复方法：采用帧堆叠（frame stacking，例如 Atari 上的 DQN 堆叠 4 帧）或使用循环状态（recurrent state，如对观测序列应用 LSTM/GRU）。
- **稀疏奖励（Sparse rewards）。** 仅在获胜时给予奖励，在庞大的状态空间中几乎无法进行学习。解决方法：设计奖励塑形（reward shaping，提供中间信号）或通过模仿学习（imitation learning）进行引导（参见第 9 阶段 · 09）。
- **奖励欺骗（Reward hacking）。** 优化代理奖励（proxy reward）往往会导致病态行为。例如 OpenAI 的赛艇智能体为了无限收集道具而在原地打转，而不是完成比赛。务必根据最终目标结果而非代理指标来定义奖励。
- **折扣因子误设（Discount mis-specification）。** 在无限视界（infinite-horizon）任务中设置 `γ = 1` 会导致所有价值趋于无穷大。务必通过有限视界或设置 `γ < 1` 来进行约束。
- **奖励尺度（Reward scale）。** 奖励设为 {+100, -100} 与 {+1, -1} 会得到相同的最优策略，但梯度幅度差异巨大。在输入 PPO/DQN 之前，建议将其归一化至 `[-1, 1]` 左右。

## 实际应用

2026 技术栈要求在编写任何代码之前，先将每个强化学习（Reinforcement Learning, RL）流程抽象为马尔可夫决策过程（Markov Decision Process, MDP）：

| 场景 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（移动、操作） | 关节角度 + 速度 | 连续扭矩 | 针对任务设计的塑形奖励 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘状态 + 历史 | 合法走法 | 胜=+1 / 负=-1 | 1.0（有限步） |
| 库存 / 定价 | 库存量 + 需求 | 订货数量 | 收入 - 成本 | 0.95 |
| 大语言模型的 RLHF | 上下文词元（tokens） | 下一个词元 | 结尾处的奖励模型评分 | 1.0（单集约 200 个词元） |
| 用于推理的 GRPO | 提示词 + 部分回复 | 下一个词元 | 结尾处的验证器 0/1 评分 | 1.0 |

在编写任何训练循环之前，先明确这五个元组。绝大多数“强化学习不奏效”的 Bug 报告，其根源都可以追溯到纸面阶段就已存在缺陷的 MDP 建模。

## 交付使用

保存为 `outputs/skill-mdp-modeler.md`：

---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 网格世界（GridWorld）及随机策略（random policy）的轨迹模拟（rollout）。运行 10,000 个回合（episode）。报告回报（return）的均值与标准差，并与最优回报（-6）进行对比。
2. **中等。** 针对均匀随机策略（uniform-random policy），设置 `γ ∈ {0.5, 0.9, 0.99}` 运行 `policy_evaluation`。将每种情况下的 `V` 以 4×4 网格形式打印输出。解释为何随着 `γ` 增大，靠近终止状态（terminal state）的状态值（state value）增长得更快。
3. **困难。** 为网格世界引入随机性（stochastic）：执行每个动作时，有 `p = 0.1` 的概率会滑向相邻方向。重新评估均匀策略（uniform policy）。`V[start]` 的值会变高还是变低？请说明原因。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| MDP（马尔可夫决策过程） | “强化学习的基础设定” | 满足马尔可夫性质（Markov property）的元组 `(S, A, P, R, γ)`。 |
| State（状态） | “智能体观测到的信息” | 在所选策略类别下，对未来环境动态（dynamics）的充分统计量。 |
| Policy（策略） | “智能体的行为模式” | 条件概率分布 `π(a | s)` 或确定性映射 `s → a`。 |
| Return（回报） | “累计奖励” | 从当前时刻开始的折扣奖励总和 `Σ γ^t r_t`。 |
| Value（状态价值） | “某个状态有多好” | 从状态 `s` 出发，遵循策略 `π` 所能获得的期望回报（expected return）。 |
| Q-value（动作价值） | “某个动作有多好” | 从状态 `s` 出发且第一步执行动作 `a`，遵循策略 `π` 所能获得的期望回报。 |
| Bellman equation（贝尔曼方程） | “动态规划递归式” | 将价值/ Q值分解为单步即时奖励与折扣后继价值之和的不动点（fixed-point）方程。 |
| Discount `γ`（折扣因子） | “未来与当下的权衡” | 对远期奖励施加的几何衰减权重；有效视界（effective horizon）约为 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) —— 权威教材。第 3 章涵盖 MDP 与贝尔曼方程；第 1 章引出了奖励假设（reward hypothesis），这是贯穿后续所有内容的核心前提。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) —— 贝尔曼方程的起源之作。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) —— 从深度强化学习（deep RL）视角出发的简明 MDP 入门指南。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) —— 运筹学领域关于 MDP 及其精确求解方法的权威参考书。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) —— 将 MDP 作为动态规划特例进行推导的最清晰文献。
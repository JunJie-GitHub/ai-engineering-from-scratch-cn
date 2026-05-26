# 动态规划（Dynamic Programming）—— 策略迭代（Policy Iteration）与值迭代（Value Iteration）

> 动态规划就像是强化学习（Reinforcement Learning）中的“作弊模式”。你已经掌握了状态转移函数（Transition Function）和奖励函数（Reward Function）；只需不断迭代贝尔曼方程（Bellman Equation），直到 `V` 或 `π` 不再变化为止。它是所有基于采样的方法（Sampling-based Method）都试图逼近的基准。

**类型:** 构建
**语言:** Python
**前置知识:** 第 9 阶段 · 01（马尔可夫决策过程 Markov Decision Process, MDP）
**预计时间:** 约 75 分钟

## 问题描述

你拥有一个模型已知的马尔可夫决策过程（MDP）：针对任意状态-动作对（State-Action Pair），你都可以直接查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理者清楚需求分布；棋盘游戏具备确定性的状态转移；网格世界（Gridworld）仅需四行 Python 代码即可构建。你拥有一个*模型*。

无模型强化学习（Model-free Reinforcement Learning，如 Q-learning、PPO、REINFORCE）是为缺乏环境模型的场景而设计的——你只能从环境中进行采样。但当你确实拥有模型时，存在更快、更优的求解方法：动态规划。贝尔曼（Bellman）于 1957 年提出了这些算法。它们至今仍是正确性的定义标准：当人们提到“该 MDP 的最优策略（Optimal Policy）”时，指的就是动态规划所返回的策略。

在 2026 年，你仍然需要掌握它们，原因有三。首先，强化学习研究中的每一个表格型环境（Tabular Environment，如 GridWorld、FrozenLake、CliffWalking）都会使用动态规划（DP）进行求解，以生成基准策略。其次，精确值（Exact Value）让你能够*调试*（Debug）基于采样的方法：如果 Q-learning 对 `V*(s_0)` 的估计值与动态规划的结果相差 30%，说明你的 Q-learning 实现存在缺陷。第三，现代离线强化学习（Offline Reinforcement Learning）与规划方法（如 MCTS、AlphaZero 的搜索、第 9 阶段 · 10 中的基于模型的强化学习 Model-based Reinforcement Learning）都会在已学习或给定的模型上迭代贝尔曼备份（Bellman Backup）。

## 核心概念

![策略迭代 (Policy Iteration) 与值迭代 (Value Iteration) 并列对比](../assets/dp.svg)

**两种算法，均基于贝尔曼 (Bellman) 的不动点迭代 (Fixed-point Iteration)。**

**策略迭代 (Policy Iteration)。** 交替执行两个步骤，直到策略不再发生变化。

1. *评估 (Evaluation)：* 给定策略 `π`，通过反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 来计算 `V^π`，直至收敛。
2. *改进 (Improvement)：* 给定 `V^π`，使 `π` 相对于 `V^π` 采取贪婪策略：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛性是有保证的，因为 (a) 每次改进步骤要么保持 `π` 不变，要么严格提升某些状态下的 `V^π`；(b) 确定性策略 (Deterministic Policies) 的空间是有限的。即使在大规模状态空间中，通常也只需约 5–20 次外层迭代即可收敛。

**值迭代 (Value Iteration)。** 将评估与改进合并为一次扫描。应用贝尔曼最优方程 (Bellman Optimality Equation)：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复该过程，直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过选择贪婪动作来提取策略。每次迭代的计算速度严格更快（无需内部评估循环），但通常需要更多次迭代才能收敛。

**广义策略迭代 (Generalized Policy Iteration, GPI)。** 一种统一的理论框架。价值函数 (Value Function) 与策略被锁定在一个双向改进循环中；任何促使两者趋向相互一致的方法（如异步值迭代 (Async Value Iteration)、修正策略迭代 (Modified Policy Iteration)、Q学习 (Q-learning)、演员-评论家 (Actor-Critic)、近端策略优化 (PPO)）都是 GPI 的具体实例。

**为什么 `γ < 1` 至关重要。** 贝尔曼算子 (Bellman Operator) 在上确界范数 (Sup-norm) 下是一个 `γ`-压缩映射 (γ-contraction)：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩映射保证了唯一不动点 (Fixed Point) 和几何收敛 (Geometric Convergence)。若放弃 `γ < 1` 的条件，你将失去这一保证——此时必须依赖有限视界 (Finite Horizon) 或吸收终止状态 (Absorbing Terminal State)。

## 动手实现

### 步骤 1：构建网格世界 (GridWorld) 马尔可夫决策过程 (MDP) 模型

沿用第 01 课中的 4×4 GridWorld。我们增加了一个随机变体：智能体 (agent) 有 `0.1` 的概率会滑向一个随机的垂直方向。

SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes

`transitions(s, a)` 返回一个包含 `(s', r, p)` 的列表。这就是完整的模型。

### 步骤 2：策略评估 (Policy Evaluation)

给定一个策略 `π(s) = {action: prob}`，迭代贝尔曼方程 (Bellman equation) 直到值函数 `V` 收敛（不再变化）：

def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V

### 步骤 3：策略改进 (Policy Improvement)

将 `π` 替换为相对于 `V` 的贪婪策略 (greedy policy)。如果 `π` 未发生变化，则直接返回——此时已达到最优解。

def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy

### 步骤 4：组合流程

def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy

在 4×4 网格上的典型收敛情况：需要 4–6 次外层迭代。输出结果为 `V*(0,0) ≈ -6`，以及一个能严格减少步数的策略。

### 步骤 5：值迭代 (Value Iteration)（单循环版本）

def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy

收敛至相同的不动点 (fixed point)，但代码行数更少。

## 常见陷阱

- **忘记处理终止状态（Terminal states）。** 如果对吸收态（absorbing state）应用贝尔曼方程（Bellman equation），它仍然会选出一个“最优动作”，但实际上状态不会发生任何改变。应使用 `if s == terminal: V[s] = 0` 进行防护。
- **无穷范数（Sup-norm）与 L2 收敛。** 使用 `max |V_new - V|` 而非平均值。理论保证是基于无穷范数的。
- **原地更新（In-place update）与同步更新。** 原地更新 `V[s]`（高斯-赛德尔迭代，Gauss-Seidel）比使用独立的 `V_new` 字典（雅可比迭代，Jacobi）收敛更快。生产环境代码通常采用原地更新。
- **策略平局（Policy ties）。** 如果两个动作的 Q 值（Q-value）相等，`argmax` 可能在每次迭代中以不同方式打破平局，导致“策略稳定”检查出现振荡。应使用稳定的平局打破机制（例如按固定顺序选择第一个动作）。
- **状态空间爆炸（State-space explosion）。** 动态规划（Dynamic Programming, DP）每次扫描的复杂度为 `O(|S| · |A|)`。适用于约 10⁷ 个状态以内的场景。超出此规模则需要使用函数近似（Function approximation）（第 9 阶段 · 05 及之后）。

## 应用场景

到 2026 年，DP 已成为正确性基准以及规划器（planner）的内层循环：

| 应用场景 | 方法 |
|----------|--------|
| 精确求解小型表格型马尔可夫决策过程（Tabular MDP） | 值迭代（Value iteration，更简单）或策略迭代（Policy iteration，外层迭代步数更少） |
| 验证 Q-learning / PPO 实现 | 在玩具环境（toy environment）中与 DP 最优的 V* 进行对比 |
| 基于模型的强化学习（Model-based RL，第 9 阶段 · 10） | 在学到的转移模型（transition model）上执行贝尔曼备份（Bellman backup） |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索（Monte Carlo Tree Search）= 异步贝尔曼备份 |
| 离线强化学习（Offline RL，CQL, IQL） | 保守 Q 迭代（Conservative Q-iteration）—— 对分布外动作（OOD actions）施加惩罚的 DP |

每当有人提到“最优价值函数（optimal value function）”时，他们指的其实是“DP 不动点（DP fixed point）”。当你在论文中看到 `V*` 或 `Q*` 时，脑海中应该浮现出这个循环。

## 交付

保存为 `outputs/skill-dp-solver.md`：

---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.

## 练习

1. **简单。** 在 4×4 网格世界（GridWorld）上运行值迭代（Value Iteration），设置 `γ ∈ {0.9, 0.99}`。需要多少次扫描（sweep）才能使 `max |ΔV| < 1e-6`？将 `V*` 以 4×4 网格的形式输出。
2. **中等。** 在*随机*网格世界（滑移概率为 `0.1`）上对比策略迭代（Policy Iteration）与值迭代。统计：扫描次数、挂钟时间（wall-clock time）、最终的 `V*(0,0)`。哪种方法在迭代次数上收敛更快？在挂钟时间上呢？
3. **困难。** 实现改进策略迭代（Modified Policy Iteration）：在评估步骤中，仅运行 `k` 次扫描而非完全收敛。针对 `k ∈ {1, 2, 5, 10, 50}`，绘制 `V*(0,0)` 的误差随 `k` 变化的曲线。该曲线揭示了评估（evaluation）与改进（improvement）之间的权衡关系是怎样的？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 策略迭代（Policy Iteration） | “动态规划算法” | 交替进行策略评估（计算 `V^π`）和策略改进（基于 `V^π` 采取贪婪策略 `π`），直到策略不再发生变化。 |
| 值迭代（Value Iteration） | “更快的动态规划” | 在一次扫描中应用贝尔曼最优备份（Bellman optimality backup）；以几何级数收敛至 `V*`。 |
| 贝尔曼算子（Bellman Operator） | “递归公式” | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在无穷范数下是一个 `γ`-压缩映射（γ-contraction）。 |
| 压缩映射（Contraction） | “动态规划为何收敛” | 任何满足 `||T x - T y|| ≤ γ ||x - y||` 的算子 `T` 都具有唯一的不动点。 |
| 广义策略迭代（Generalized Policy Iteration, GPI） | “万物皆动态规划” | 广义策略迭代：任何促使 `V` 和 `π` 达到相互一致性的方法。 |
| 同步更新（Synchronous Update） | “雅可比风格” | 在整个扫描过程中始终使用旧的 `V`；理论分析清晰但速度较慢。 |
| 原地更新（In-place Update） | “高斯-赛德尔风格” | 在更新过程中直接使用最新的 `V`；在实际应用中收敛更快。 |

## 扩展阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 策略迭代与值迭代的标准教材。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 对压缩映射（contraction-mapping）论证的严谨论述。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 改进策略迭代及其收敛性分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 策略迭代算法的原始论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从动态规划（DP）过渡到近似动态规划（approximate-DP）/ 深度强化学习（deep RL）的桥梁，后续课程均以此为基础。
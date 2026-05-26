# 多智能体强化学习 (Multi-Agent RL)

> 单智能体强化学习 (Single-Agent RL) 假设环境是平稳的 (Stationary)。若将两个正在学习的智能体置于同一世界中，该假设便会失效：每个智能体都成为对方环境的一部分，且两者都在持续变化。多智能体强化学习 (Multi-Agent RL) 正是一系列方法的集合，旨在当马尔可夫假设 (Markov Assumption) 不再成立时，仍能保证学习过程收敛。

**类型：** 构建
**语言：** Python
**前置知识：** 第 9 阶段 · 04 (Q-learning)、第 9 阶段 · 06 (REINFORCE)、第 9 阶段 · 07 (Actor-Critic)
**预计耗时：** 约 45 分钟

## 问题所在

让机器人学习在房间内导航属于单智能体强化学习问题，但一支足球队则不然。AlphaStar 对战《星际争霸》对手不是，由竞价智能体构成的交易市场不是，两辆汽车协商通过四向停车路口不是，现实世界中多对多的复杂场景也都不是。

在任意多智能体场景中，从单个智能体的视角来看，其他智能体*本身就是*环境的一部分。随着它们持续学习并调整行为，环境将变为非平稳 (Non-stationary) 状态。马尔可夫性质 (Markov Property)——即“下一状态仅取决于当前状态与自身动作”——在此会被打破，因为下一状态还取决于*其他*智能体的决策，且它们的策略本身就是动态变化的目标。

这打破了表格型收敛性证明 (Tabular Convergence Proofs)（Q-learning 的收敛保证建立在环境平稳的假设之上）。它同样会导致朴素深度强化学习 (Naive Deep RL) 失效：智能体会陷入循环追逐，永远无法收敛至稳定的策略。因此，必须引入多智能体专属技术：集中式训练/分布式执行 (Centralized Training / Decentralized Execution)、反事实基线 (Counterfactual Baselines)、联赛机制 (League Play) 以及自我对弈 (Self-Play)。

2026 年典型应用场景：机器人集群、交通路径规划、自动驾驶车队、市场模拟器、多智能体大语言模型系统（第 16 阶段），以及任何包含多个智能参与者的博弈游戏。

## 核心概念

![四种多智能体强化学习范式：独立、集中式评论家、自我对弈、联赛](../assets/marl.svg)

**形式化定义：马尔可夫博弈 (Markov Game)。** 马尔可夫决策过程 (MDP) 的推广：包含状态 `S`、联合动作 `a = (a_1, …, a_n)`、状态转移概率 `P(s' | s, a)` 以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在其自身策略 `π_i` 下最大化自身的累积回报。若所有智能体奖励相同，则为**完全合作 (fully cooperative)**；若为零和，则为**对抗性 (adversarial)**；若为混合情况，则为**一般和 (general-sum)**。

**核心挑战：**

- **非平稳性 (Non-stationarity)。** 从智能体 `i` 的视角来看，状态转移 `P(s' | s, a_i)` 依赖于其他智能体的策略 `π_{-i}`，而这些策略是不断变化的。
- **信用分配 (Credit assignment)。** 在共享奖励的情况下，如何确定是哪个智能体导致了该奖励？
- **探索协调 (Exploration coordination)。** 智能体必须探索互补的策略，而非冗余地重复探索相同的状态。
- **可扩展性 (Scalability)。** 联合动作空间随智能体数量 `n` 呈指数级增长。
- **部分可观测性 (Partial observability)。** 每个智能体仅能观测到自身的局部观测值，全局状态是隐藏的。

**四种主流范式：**

**1. 独立 Q 学习 / 独立 PPO (Independent Q-learning / Independent PPO, IQL/IPPO)。** 每个智能体独立学习自身的 Q 值或策略，将其他智能体视为环境的一部分。方法简单，有时也能奏效（尤其是经验回放 (experience replay) 可作为一种平滑智能体建模的技巧）。理论收敛性：无。实际应用：适用于松耦合任务，但在紧耦合任务中表现较差。

**2. 集中式训练，分布式执行 (Centralized Training, Decentralized Execution, CTDE)。** 目前最主流的范式。每个智能体拥有独立的策略 `π_i`，该策略以局部观测 `o_i` 为条件——在部署时采用标准的分布式执行。在训练阶段，集中式评论家 (critic) `Q(s, a_1, …, a_n)` 以完整的全局状态和联合动作为条件。典型算法包括：
- **MADDPG** (Lowe 等, 2017)：为每个智能体配备集中式评论家的 DDPG 算法。
- **COMA** (Foerster 等, 2017)：反事实基线 (counterfactual baseline) —— 通过提问“如果我采取动作 `a'` 而非当前动作，我的奖励会是多少？”来隔离单个智能体的贡献。
- **MAPPO** / 共享评论家的 **IPPO** (Yu 等, 2022)：采用集中式价值函数的 PPO 算法。在 2026 年已成为合作型多智能体强化学习 (MARL) 的主导方法。
- **QMIX** (Rashid 等, 2018)：价值分解 (value decomposition) —— 通过单调混合函数 `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` 将联合 Q 值分解为个体 Q 值。

**3. 自我对弈 (Self-play)。** 同一智能体的两个副本相互对战。对手的策略即为我过去某一时刻的策略快照。应用于 AlphaGo / AlphaZero / MuZero 以及 OpenAI Five。最适用于零和博弈；训练信号具有对称性。

**4. 联赛对弈 (League play)。** 自我对弈在一般和/对抗环境中的扩展：维护一个包含历史与当前策略的种群，从该“联赛”中采样对手进行对抗训练。引入“剥削者 (exploiters)”（专门针对当前最强策略）和“主剥削者 (main exploiters)”（专门针对剥削者）。应用于 AlphaStar（《星际争霸 II》）。当游戏存在“石头-剪刀-布”式的策略循环时，该方法尤为必要。

**通信机制 (Communication)。** 允许智能体之间发送学习到的消息 `m_i`。适用于合作场景。Foerster 等 (2016) 证明，可微智能体间通信 (differentiable inter-agent communication) 支持端到端训练。如今基于大语言模型 (LLM) 的多智能体系统（Phase 16）本质上即是通过自然语言进行通信。

## 动手实现

本课程使用一个 6×6 的网格世界（GridWorld），其中包含两个协作智能体（cooperative agents）。它们从对角角落出发，必须抵达同一个共同目标。共享奖励规则为：只要任一智能体仍在移动，每步奖励为 `-1`；当两者均到达目标时，奖励为 `+10`。详见 `code/main.py`。

### 步骤 1：多智能体环境

class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done

*联合*动作空间（joint action space）的大小为 `|A|² = 16`。全局状态（global state）由两个位置坐标组成。

### 步骤 2：独立 Q 学习（independent Q-learning）

每个智能体维护自己的 Q 表（Q-table），以联合状态（joint state）为键。在每一步中：双方均选择 ε-贪婪（ε-greedy）动作，收集联合转移（joint transition）数据，并使用共享奖励各自更新自己的 Q 值。

def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next

该方法在此任务中有效，因为奖励是稠密（dense）且目标一致（aligned）的。但在强耦合（tightly-coupled）任务中会失效（例如，一个智能体必须*等待*另一个智能体的情况）。

### 步骤 3：带分解值更新（decomposed-value update）的集中式 Q 学习（centralized Q-learning）

使用一个针对联合动作的 Q 函数 `Q(s, a_1, a_2)`。根据共享奖励进行更新。在执行阶段通过边缘化（marginalizing）实现去中心化：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。这种方法以指数级增长的联合动作空间为代价，换取了*正确*的全局视角。

### 步骤 4：简单自我对弈（self-play）（对抗性双智能体）

同一智能体，扮演两个角色。让智能体 A 与智能体 B 进行对抗训练；每经过 `K` 个回合（episodes），将 A 的权重复制到 B 中。这种对称训练能带来稳定的进步。堪称 AlphaZero 核心方案的微缩版。

## 常见陷阱

- **非平稳经验回放（Non-stationary replay）。** 独立智能体的经验回放（Experience replay）效果不如单智能体，因为旧的状态转移是由现已过时的对手策略生成的。修复方案：重新标注或按时间近远进行加权。
- **信用分配歧义（Credit assignment ambiguity）。** 长回合结束后共享奖励，难以明确具体是哪个智能体做出了贡献。修复方案：使用反事实基线（Counterfactual baselines, COMA），或为每个智能体进行奖励塑形（Reward shaping）。
- **策略漂移/追逐（Policy drift / chasing）。** 每个智能体的最优响应会随着其他智能体的策略更新而不断变化。修复方案：采用集中式评论家（Centralized critic）、降低学习率，或采用轮流冻结训练策略。
- **协同奖励黑客（Reward hacking via coordination）。** 智能体发现了设计者未曾预料到的协同漏洞。例如，拍卖智能体会收敛至零出价。修复方案：精心设计奖励函数，或施加行为约束。
- **探索冗余（Exploration redundancy）。** 多个智能体重复探索相同的状态-动作对（state-action pairs）。修复方案：为每个智能体添加熵奖励（Entropy bonuses），或引入角色条件化（Role-conditioning）。
- **联赛循环（League cycles）。** 纯自我对弈（Self-play）容易陷入策略克制循环。修复方案：引入多样化对手进行联赛训练。
- **样本爆炸（Sample explosion）。** 复杂度为 `n` 个智能体 × 状态空间 × 联合动作空间。可通过函数逼近（Function approximation）进行近似处理；或采用分解动作空间（Factored action spaces，即每个智能体独立一个策略输出头）。

## 实际应用

2026 年多智能体强化学习（MARL）应用图谱：

| 领域 | 方法 | 备注 |
|--------|--------|-------|
| 协同导航/操作 | MAPPO / QMIX | CTDE（集中训练分散执行）；共享评论家 + 分散式执行器。 |
| 双人对弈游戏（国际象棋、围棋、扑克） | 结合 MCTS 的自我对弈（AlphaZero） | 零和博弈；对称训练。 |
| 复杂多人游戏（Dota、星际争霸） | 联赛对弈 + 模仿预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | 结合注意力机制的 CTDE MAPPO / PPO | 部分可观测；团队规模可变。 |
| 拍卖市场 | 博弈论均衡 + 强化学习 | 当 `n` → ∞ 时采用平均场强化学习（Mean-field RL）。 |
| 大语言模型多智能体系统（Phase 16） | 自然语言通信 + 角色条件化 | 强化学习循环位于智能体规划层。 |

到 2026 年，MARL 增长最快的领域将基于大语言模型（LLM）：由语言模型智能体组成的集群进行谈判、辩论与软件开发。此处的强化学习体现为对*轨迹级*（trajectory-level）输出而非词元级（token-level）的偏好优化（Preference optimization）（Phase 16 · 03）。

## 部署上线

保存为 `outputs/skill-marl-architect.md`：

---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).

## 练习

1. **简单。** 在 2 智能体协作网格世界（GridWorld）上训练独立 Q 学习（Independent Q-learning）。平均回报（mean return）大于 0 需要多少个回合（episodes）？绘制联合学习曲线（joint learning curve）。
2. **中等。** 添加一个“协调”任务（coordination task）：仅当两个智能体在同一回合踏上目标点时才算达成目标。独立 Q 学习还能收敛（converge）吗？哪个环节会失效？
3. **困难。** 为 MAPPO 风格的训练实现一个集中式评论家（centralized critic），并在协调任务上将其收敛速度与独立 PPO（Independent PPO）进行比较。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 马尔可夫博弈（Markov game） | “多智能体 MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体拥有各自的奖励。 |
| 集中式训练分布式执行（CTDE） | “集中训练，分散执行” | 训练时使用联合评论家；每个智能体的策略仅使用局部观测（local obs）。 |
| 独立近端策略优化（IPPO） | “独立 PPO” | 每个智能体独立运行 PPO。简单的基线方法；常被低估。 |
| 多智能体近端策略优化（MAPPO） | “多智能体 PPO” | 带有以全局状态为条件的集中式价值函数的 PPO。 |
| 单调价值分解（QMIX） | “单调价值分解” | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许分布式 argmax。 |
| 反事实多智能体（COMA） | “反事实多智能体” | 优势（Advantage）= 我的 Q 值减去对我的动作进行边缘化后的期望 Q 值。 |
| 自我对弈（Self-play） | “智能体对抗过去的自己” | 单个智能体扮演两个角色；零和博弈的标准做法。 |
| 联赛对弈（League play） | “群体训练” | 缓存历史策略，从池中采样对手；用于应对策略循环。 |

## 延伸阅读

- [Lowe 等 (2017). 混合合作-竞争环境中的多智能体演员-评论家算法 (MADDPG)](https://arxiv.org/abs/1706.02275) — 采用集中式评论家（centralized critic）的集中训练分散执行（CTDE）架构。
- [Foerster 等 (2017). 反事实多智能体策略梯度 (COMA)](https://arxiv.org/abs/1705.08926) — 用于信用分配（credit assignment）的反事实基线（counterfactual baselines）。
- [Rashid 等 (2018). QMIX：单调价值函数分解](https://arxiv.org/abs/1803.11485) — 具有单调性约束的价值分解（value decomposition）。
- [Yu 等 (2022). PPO 在合作型多智能体游戏中的惊人有效性 (MAPPO)](https://arxiv.org/abs/2103.01955) — 近端策略优化（PPO）在多智能体强化学习（MARL）中表现出乎意料地强劲。
- [Vinyals 等 (2019). 使用多智能体强化学习在《星际争霸 II》中达到大师级水平 (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联赛对战（league play at scale）。
- [Silver 等 (2017). 无需人类知识掌握围棋游戏 (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和博弈（zero-sum games）中的纯自我对弈（self-play）。
- [Sutton & Barto (2018). 第 15 章 — 神经科学 & 第 17 章 — 前沿](http://incompleteideas.net/book/RLbook2020.pdf) — 包含该教材对多智能体环境的简要探讨，以及 CTDE 旨在解决的非平稳性（non-stationarity）问题。
- [Zhang, Yang & Başar (2021). 多智能体强化学习：精选综述](https://arxiv.org/abs/1911.10635) — 涵盖合作型、竞争型及混合型多智能体强化学习的综述，并包含收敛性结果。
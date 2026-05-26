# 多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）— MADDPG、QMIX、MAPPO

> 多智能体协同的强化学习传承，至今仍在 2026 年的大语言模型智能体（LLM-agent）系统中发挥着指导作用。**MADDPG**（Lowe 等人，NeurIPS 2017，arXiv:1706.02275）引入了集中式训练与分布式执行（Centralized Training, Decentralized Execution, CTDE）范式：在训练期间，每个评论家（Critic）都能观测到所有智能体的状态与动作；而在测试阶段，仅运行本地的执行者（Actor）。该算法适用于合作、竞争及混合场景。**QMIX**（Rashid 等人，ICML 2018，arXiv:1803.11485）是一种基于单调混合网络（Monotonic Mixing Network）的价值分解（Value Decomposition）方法；它将各智能体的 Q 值组合为联合 Q 值（Joint Q-value），使得 `argmax` 操作能够干净利落地分解——在星际争霸多智能体挑战赛（StarCraft Multi-Agent Challenge, SMAC）中占据主导地位。**MAPPO**（Yu 等人，NeurIPS 2022，arXiv:2103.01955）是带有集中式价值函数（Centralized Value Function）的 PPO 算法；在粒子世界（Particle-world）、SMAC、Google Research Football 和 Hanabi 等环境中，仅需极少调参便展现出“惊人的效果”。这些算法为必须分布式行动的协同智能体团队提供了策略训练的基础。MAPPO 是 **2026 年合作型 MARL 的默认基线（Default Cooperative-MARL Baseline）**。本课程将从一个小型网格世界（Grid-world）玩具环境出发构建这三种算法，在接触 LLM 智能体训练之前，先将这三个核心思想转化为肌肉记忆。

**Type:** 学习
**Languages:** Python（标准库，小型无 NumPy 实现）
**Prerequisites:** 阶段 09（强化学习），阶段 16 · 09（并行群体网络）
**Time:** 约 90 分钟

## 问题

大语言模型智能体系统越来越多地训练用于智能体间协同的策略：何时退让、何时行动、调用哪个同伴。指导你如何训练此类策略的文献正是多智能体强化学习（MARL），它早于大语言模型浪潮，并拥有一组占据主导地位的核心算法。

如果不掌握范式术语，阅读 MARL 论文会非常痛苦。集中式训练与分布式执行（CTDE）、价值分解以及集中式评论家并非流行语——它们是针对特定问题的具体解决方案：

- 独立强化学习（Independent RL，每个智能体单独学习）从每个智能体的视角来看是非平稳的（Non-stationary）。效果很差。
- 集中式强化学习（Centralized RL，一个智能体控制所有）无法扩展，且违反了执行约束。
- CTDE 兼顾了两者的优点：利用全局信息进行训练，使用本地策略进行部署。

## 概念

### 论文使用的三种环境

- **粒子世界（Particle World，多智能体粒子环境）**。包含合作/竞争任务的简单二维物理环境。MADDPG 的原始测试平台。
- **星际争霸多智能体挑战赛（StarCraft Multi-Agent Challenge, SMAC）**。合作型微操任务，部分可观测。QMIX 的测试平台。离散动作，连续状态。
- **Google Research Football、Hanabi、MPE**。MAPPO 的基线环境。

不同环境具有不同的动作/观测类型。算法会根据这些类型进行相应选择。

### MADDPG（2017）—— 集中式训练与分布式执行（Centralized Training with Decentralized Execution, CTDE）模式

每个智能体 `i` 拥有一个演员网络（Actor）`mu_i(o_i)`，用于将其自身的观测映射为动作。每个智能体还拥有一个评论家网络（Critic）`Q_i(x, a_1, ..., a_n)`，在训练期间可观测到所有智能体的观测与动作。演员网络会根据评论家网络的评估结果，通过策略梯度（Policy Gradient）进行更新。

actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate

采用 CTDE 的原因：在训练阶段，我们已知所有智能体的动作，可利用这一信息降低每个评论家网络的方差。在部署阶段，每个智能体仅能看到自身的 `o_i` 并调用 `mu_i(o_i)`。

局限性：评论家网络的输入维度随智能体数量 N 增长（输入包含所有动作）。若不进行近似处理，该算法难以扩展到约 10 个以上的智能体。

### QMIX（2018）—— 价值分解（Value Decomposition）

仅适用于合作场景。全局奖励是各智能体 Q 值的单调函数之和：

Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0

单调性保证了 `argmax_a Q_tot` 可以通过每个智能体独立选择 `argmax_{a_i} Q_i` 来计算得出。这正是你所需要的**分布式执行特性**。在训练阶段，一个混合网络（Mixing Network）会根据各智能体的 Q 值生成 `Q_tot`。

QMIX 在 SMAC 上表现优异的原因：合作型星际争霸微操任务具有同质化智能体、局部观测和全局奖励的特点，与价值分解方法完美契合。

局限性：单调性约束较为严格；某些任务的奖励结构无法进行单调分解（例如某个智能体为团队牺牲）。后续扩展算法（如 QTRAN、QPLEX）对此进行了放宽。

### MAPPO（2022）—— 被忽视的默认选择

多智能体 PPO（Multi-Agent PPO）：采用集中式价值函数的 PPO 算法。每个智能体拥有独立的策略；所有智能体共享（或各自拥有）能够观测完整状态的价值函数。Yu 等人于 2022 年在五个基准测试上将 MAPPO 与 MADDPG、QMIX 及其扩展算法进行了对比，发现：

- 在粒子世界、SMAC、Google Research Football、Hanabi 和 MPE 环境中，MAPPO 的表现持平或优于异策略多智能体强化学习（Off-policy Multi-Agent Reinforcement Learning, MARL）方法。
- 所需的超参数调优极少。
- 训练稳定；在不同随机种子下具有可复现性。

在此论文发表之前，学界一直低估了同策略多智能体强化学习（On-policy Multi-Agent Reinforcement Learning, MARL）的价值。到了 2026 年，MAPPO 已成为合作型 MARL 的默认基线；任何新方法都必须超越它。

### 为什么大语言模型智能体（LLM-agent）工程师需要关注

三个直接应用场景：

1. **路由训练（Router training）**。一个元智能体（Meta-agent）负责选择由哪个子智能体处理任务。这是一个包含 N 个分布式子智能体和一个集中式路由器的 MARL 问题。MAPPO 非常适用。
2. **角色涌现（Role emergence）**。在生成式智能体模拟中，训练智能体随时间推移形成互补角色，本质上是一个 MARL 问题。QMIX 式的价值分解通过算法设计强制实现了角色互补。
3. **多智能体工具使用**。当智能体共享工具并竞争预算时，通过 CTDE 进行训练可以生成符合资源约束、可直接部署的局部策略。

实际注意事项：在 2026 年，大多数生产环境的 LLM-agent 系统仍通过提示词（Prompt）来驱动策略，而非进行训练。当满足以下条件时，MARL 才会派上用场：(a) 拥有大量交互数据，(b) 具备明确的奖励信号，以及 (c) 愿意投入训练基础设施。

### 超越强化学习的 CTDE 设计模式

即使不进行训练，CTDE 也是一种实用的架构模式：

- 在*设计阶段*，假设团队具备全局可见性。
- 在*运行阶段*，强制执行分布式执行：每个智能体仅能看到 `o_i`。

该模式强制要求你明确维护每个智能体的状态，并提前考虑部分可观测性问题。许多生产环境中的多智能体系统会隐式地假设全局状态处处共享——遵循 CTDE 规范可避免这一问题。

### 非平稳性（Non-stationarity）问题

当多个智能体同时学习时，每个智能体所处的环境（包含其他智能体的策略）是非平稳的。经典的单智能体强化学习理论证明在此失效。本课程中的 MARL 算法均针对此问题进行了处理：

- MADDPG：全局评论家网络可观测所有动作，因此其价值估计是平稳的。
- QMIX：价值分解将学习过程转移至联合 Q 空间，在该空间中最优性有明确定义。
- MAPPO：集中式价值函数能够抑制因其他智能体策略变化带来的方差。

在 LLM-agent 系统中，非平稳性通常表现为：“我的智能体上个月运行正常，但现在上游的其他智能体发生了变化，导致我的智能体行为异常。”采用 CTDE 训练 MARL 是基于理论原则的解决方案；而提示词层面的修复虽然更快，但持久性较差。

### 本课程不包含的内容

实际神经网络的训练属于第 09 阶段的主题。本课程将构建基于脚本策略的版本，在不进行梯度更新的情况下演示 CTDE、价值分解和集中式价值模式。目标是在你使用完整的 MARL 库（如 PyMARL、MARLlib、RLlib multi-agent）之前，先深入理解这些模式。

## 构建它

`code/main.py` 实现了三种模式的演示，均在一个小型的双智能体协作网格世界 (grid-world) 中进行：

- 环境：4x4 网格上有 2 个智能体，1 个奖励目标 (reward pellet)。若任一智能体到达奖励目标，则获得奖励 = 1，任务结束。
- `IndependentAgents` —— 每个智能体将其他智能体视为环境的一部分。作为基线 (baseline)。
- `MADDPGStyle` —— 集中式评论家 (centralized critic) 计算联合价值；执行者策略 (actor policies) 据此更新。采用预设的策略改进方式。
- `QMIXStyle` —— 采用单调混合器 (monotone mixer) 进行价值分解 (value decomposition)。
- `MAPPOStyle` —— 集中式价值函数 (centralized value function)；策略针对共享基线进行更新。

所有四种模式均运行相同的回合 (episodes)，并报告到达目标步数 (steps-to-goal)。采用集中训练分散执行 (Centralized Training with Decentralized Execution, CTDE) 架构的变体比独立基线收敛到更短的路径。

运行：

python3 code/main.py

预期输出：独立智能体平均需要约 6 步；CTDE 变体收敛至约 3.5 步（4x4 网格的最优步数为 3）。尽管采用了预设策略，不同模式之间的差异依然显著。

## 使用它

`outputs/skill-marl-picker.md` 是一项技能，用于为给定的多智能体任务挑选多智能体强化学习 (Multi-Agent Reinforcement Learning, MARL) 算法：需综合考量协作型与竞争型、同质与异质、动作空间类型、规模以及奖励信号等因素。

## 部署上线

生产环境中较少使用 MARL。若确实需要采用，请注意：

- **从 MAPPO 开始。** 2022 年的论文已将其确立为基线；优先复现该方法可节省数周盲目追逐更复杂算法的时间。
- **记录每个智能体的观测与动作流。** 若缺乏每个智能体的追踪日志，调试 MARL 将毫无希望。
- **将训练代码与执行代码分离。** CTDE 是一种工程规范；确保执行路径真正只能看到 `o_i`。
- **警惕奖励塑形 (reward shaping)。** MARL 对奖励设计极为敏感。塑形过程中若存在一个协调漏洞，智能体就会学会利用它。务必进行对抗性测试。
- **针对大语言模型 (Large Language Model, LLM) 智能体**，优先考虑基于提示词的策略。仅在交互数据、奖励信号与基础设施三者齐备时，再投入 MARL 训练。

## 练习

1. 运行 `code/main.py`。测量独立智能体与 MAPPO 风格智能体在到达目标步数上的差距。在 6x6 网格上，该差距会扩大还是缩小？
2. 实现一个竞争型变体：两个智能体，一个奖励点，仅最先到达者获得奖励。哪种模式能更妥善地处理竞争？历史上通常是 MADDPG。
3. 阅读 MADDPG (arXiv:1706.02275) 第 3 节。用你自己的话，以伪代码形式符号化地实现精确的评论家更新规则。
4. 阅读 MAPPO (arXiv:2103.01955)。作者为何认为集中式价值函数 + 近端策略优化 (Proximal Policy Optimization, PPO) 在其基准测试中优于异策略 (off-policy) MARL？列出三个最有力的论点。
5. 将 CTDE 作为设计模式应用于一个假设的 LLM 智能体系统（例如：研究智能体 + 摘要智能体 + 编码智能体）。在设计时可用但在运行时不可用的联合信息是什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 多智能体强化学习 (Multi-Agent Reinforcement Learning, MARL) | “多智能体强化学习” | 面向多智能体系统的强化学习。 |
| 集中式训练，分布式执行 (Centralized Training, Decentralized Execution, CTDE) | “集中式训练，分布式执行” | 训练阶段利用全局信息；部署阶段仅依赖局部策略。 |
| 多智能体 DDPG (Multi-Agent DDPG, MADDPG) | “多智能体 DDPG” | 基于 CTDE 架构，每个智能体的 Critic 网络可观测所有智能体的观测值与动作。 |
| QMIX | “价值分解” | 对各智能体的 Q 值进行单调混合。适用于协作任务。 |
| 多智能体 PPO (Multi-Agent PPO, MAPPO) | “多智能体 PPO” | 引入集中式价值函数的 PPO 算法。2026 年默认基线。 |
| 价值分解 (Value Decomposition) | “个体 Q 值之和” | 联合 Q 值被表示为各智能体 Q 值的单调函数。 |
| 非平稳性 (Non-stationarity) | “移动靶标” | 随着其他智能体持续学习，单个智能体所处的环境动态变化。这是多智能体强化学习的核心难题。 |
| 同策略 / 异策略 (On-policy / Off-policy) | “基于当前经验学习 / 基于历史回放学习” | PPO 属于同策略（如 MAPPO）；DDPG 与 Q-learning 属于异策略。 |
| 星际争霸多智能体挑战赛 (StarCraft Multi-Agent Challenge, SMAC) | “星际争霸多智能体挑战赛” | 协作微操基准测试环境；QMIX 算法的发源地。 |

## 延伸阅读

- [Lowe 等人 — 面向混合协作-竞争环境的多智能体 Actor-Critic](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid 等人 — QMIX：面向深度多智能体强化学习的单调价值函数分解](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu 等人 — PPO 在协作多智能体博弈中的惊人有效性](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR 关于 MAPPO 的博客文章](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — 对 MAPPO 研究成果的清晰阐述
- [SMAC 代码仓库](https://github.com/oxwhirl/smac) — 星际争霸多智能体挑战赛
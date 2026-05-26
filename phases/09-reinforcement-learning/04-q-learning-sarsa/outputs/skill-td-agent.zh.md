---
name: td-agent
description: 针对表格型（tabular）或特征较少的强化学习（Reinforcement Learning, RL）任务，在 Q学习（Q-learning）、SARSA 与期望SARSA（Expected SARSA）之间进行选择。
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定一个表格型（tabular）或特征较少的环境，请输出以下内容：

1. **算法（Algorithm）**。Q学习（Q-learning）/ SARSA / 期望SARSA（Expected SARSA）/ n步变体（n-step variant）。用一句话说明选择理由，需关联同策略（on-policy）与异策略（off-policy）的差异以及方差（variance）特性。
2. **超参数（Hyperparameters）**。α、γ、ε 以及衰减计划（decay schedule）。
3. **初始化（Initialization）**。Q_0 值（乐观初始化 optimistic 与零初始化 zero）及其依据。
4. **收敛性诊断（Convergence diagnostic）**。目标学习曲线（Target learning curve）；若动态规划（Dynamic Programming, DP）可行，则检查 `|Q - Q*|`。
5. **部署注意事项（Deployment caveat）**。在推理（inference）阶段，探索（exploration）行为将如何表现？是否需要 SARSA 的保守性（conservatism）？

拒绝将表格型时序差分（Temporal Difference, TD）学习应用于状态空间（state spaces）大于 10⁶ 的场景。拒绝交付未附带最大偏差（max-bias）警告的 Q学习（Q-learning）智能体。标记任何在整个训练过程中 ε 始终固定为 1.0（即无利用 exploitation 阶段）的智能体。
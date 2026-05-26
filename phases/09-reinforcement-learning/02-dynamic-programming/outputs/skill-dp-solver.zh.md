---
name: dp-solver
description: 通过策略迭代（Policy Iteration）或值迭代（Value Iteration）精确求解小型表格型马尔可夫决策过程（Markov Decision Process, MDP）。报告收敛行为。
version: 1.0.0
phase: 9
lesson: 2
tags: [强化学习, 动态规划, 贝尔曼]
---

给定一个模型已知的马尔可夫决策过程（Markov Decision Process, MDP），请输出以下内容：

1. 算法选择。对比策略迭代（Policy Iteration）与值迭代（Value Iteration）。选择理由需与状态空间大小 |S|、动作空间大小 |A| 及折扣因子 γ 相关联。
2. 初始化。设定初始值函数 V_0 与初始策略。分析其对收敛的敏感性。
3. 停止条件。设定无穷范数（Sup-norm）容差 ε。预估所需的完整扫描轮数（Sweeps）。
4. 验证。精确计算最优值函数 V*(s_0)。提取贪婪策略（Greedy Policy）。
5. 应用。说明如何将此基线（Baseline）用于调试或评估基于采样的方法（Sampling-based Methods）。

若状态空间大于 10⁷，则拒绝运行动态规划（Dynamic Programming, DP）。若未执行无穷范数检查，则拒绝声明已收敛。对于无限视界（Infinite-horizon）任务，若发现 γ ≥ 1，需将其标记为违反理论保证。
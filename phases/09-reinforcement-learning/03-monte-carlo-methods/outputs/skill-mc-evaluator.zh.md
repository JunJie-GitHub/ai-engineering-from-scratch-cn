---
name: 蒙特卡洛评估器
description: 通过蒙特卡洛（Monte Carlo）推演评估策略，并在可用时生成包含动态规划（Dynamic Programming, DP）对比的收敛报告。
version: 1.0.0
phase: 9
lesson: 3
tags: [强化学习, 蒙特卡洛, 评估]
---

给定一个环境（回合制（Episodic），具备 `reset+step API`）和一个策略，输出以下内容：

1. 方法。首次访问（First-visit）与每次访问（Every-visit）蒙特卡洛（MC）的对比及选择理由。
2. 回合预算（Episode budget）。目标数量、方差诊断、预期标准误。
3. 探索计划。ε 调度（ε schedule，如需要）或探索性初始状态（Exploring starts）。
4. 黄金标准对比。若为表格型（Tabular）环境，则使用动态规划（DP）最优值函数 V*；否则使用 Q-learning / PPO 基线提供的边界值。
5. 终止检查。最大步数上限、超时处理、非终止轨迹的处理方式。

若非回合制任务未设置有限视界（Finite horizon）上限，则拒绝执行蒙特卡洛（MC）评估。对于表格型任务，若每个状态的采样回合数少于 100 次，则拒绝报告策略值函数 V^π 的估计值。将任何动作方差为零的策略标记为探索风险。
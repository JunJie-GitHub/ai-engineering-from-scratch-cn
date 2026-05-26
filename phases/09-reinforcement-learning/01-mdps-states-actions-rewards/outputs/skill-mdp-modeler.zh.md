---
name: mdp-modeler
description: 根据任务描述，生成马尔可夫决策过程（Markov Decision Process）规范，并在训练前标记问题定义风险。
version: 1.0.0
phase: 9
lesson: 1
tags: [强化学习, 马尔可夫决策过程, 建模]
---

给定一个任务（控制 / 游戏 / 推荐 / 大语言模型（Large Language Model）微调），输出：

1. 状态（State）。精确的特征向量或张量（Tensor）规范。论证马尔可夫性（Markov property）。
2. 动作（Action）。离散集合或连续范围。维度。
3. 状态转移（Transition）。确定性、已知模型的随机性，或仅支持采样。
4. 奖励（Reward）。函数及其来源。稀疏奖励与塑形奖励（Shaped reward）。终止奖励与单步奖励。
5. 折扣因子（Discount factor）。取值及时间视界（Horizon）设定的合理性论证。

拒绝交付任何状态不具备马尔可夫性（non-Markovian）的 MDP，除非明确提及使用了帧堆叠（frame-stacking）或循环状态（recurrent state）。拒绝任何未根据目标结果定义的奖励。标记无限视界（infinite-horizon）任务中任何 `γ ≥ 1.0` 的情况。将奖励范围超过典型单步奖励 100 倍的情况标记为潜在的梯度爆炸（gradient explosion）来源。
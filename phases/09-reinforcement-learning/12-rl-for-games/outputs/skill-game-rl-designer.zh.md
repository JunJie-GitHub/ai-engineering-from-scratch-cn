---
name: game-rl-designer
description: 为指定领域设计游戏强化学习（Game-RL）或推理强化学习（Reasoning-RL）训练流水线（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [强化学习（RL）, AlphaZero, MuZero, GRPO, 自我对弈（Self-Play）]
---

给定目标（完全信息博弈（Perfect-Info Game）/ 非完全信息博弈（Imperfect-Info Game）/ 雅达利（Atari）游戏 / 大语言模型（LLM）推理 / 组合优化（Combinatorial）），输出以下内容：

1. 环境适配性。规则是否已知？是否满足马尔可夫性（Markov）？是否具有随机性（Stochastic）？是否为多智能体（Multi-Agent）？据此指导在 AlphaZero、MuZero 与 GRPO 之间进行选择。
2. 搜索策略。蒙特卡洛树搜索（MCTS，采用带学习先验的 PUCT 算法）、Gumbel 采样（Gumbel-Sampled）、N 选优（Best-of-N）或无搜索。
3. 自我对弈（Self-Play）计划。对称自我对弈 / 联赛机制 / 离线数据 / 验证器（Verifier）生成数据。
4. 目标信号。博弈结果 / 验证器奖励 / 偏好（Preference）/ 学习模型。需包含鲁棒性（Robustness）方案。
5. 诊断指标。相对基线（Baseline）胜率、ELO 等级分曲线、验证器通过率、与参考模型的 KL 散度（KL Divergence）。

拒绝在非完全信息博弈中使用 AlphaZero（应转向反事实遗憾最小化 CFR）。拒绝在缺乏可信验证器的情况下使用 GRPO。拒绝任何未配置固定基线对手集的游戏强化学习流水线（否则自我对弈的 ELO 等级分将无法校准）。
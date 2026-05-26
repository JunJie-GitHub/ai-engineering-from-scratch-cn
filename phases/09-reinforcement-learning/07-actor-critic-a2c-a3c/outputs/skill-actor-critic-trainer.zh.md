---
name: Actor-Critic 训练器
description: 为指定环境生成 A2C / A3C / GAE 配置，并明确优势估计（Advantage Estimation）与损失权重（Loss Weights）。
version: 1.0.0
phase: 9
lesson: 7
tags: [强化学习 (RL), Actor-Critic, 广义优势估计 (GAE)]
---

给定环境与计算预算（Compute Budget），输出以下内容：

1. 并行策略（Parallelism）。对比 A2C（GPU 批处理）与 A3C（CPU 异步），并确定工作进程（Worker）数量。
2. 轨迹展开长度（Rollout Length）T。每次更新每个环境执行的步数。
3. 优势估计器（Advantage Estimator）。选择 n 步（n-step）或 GAE(λ)；需指定 λ 值。
4. 损失权重（Loss Weights）。`c_v`（价值 Value）、`c_e`（熵 Entropy）及梯度裁剪（Gradient Clipping）。
5. 学习率（Learning Rates）。Actor 与 Critic（若分别使用则独立设置）。

若环境的时间视界（Horizon）大于 1000，则拒绝使用单工作进程的 A2C（同策略（On-policy）倾向过重且速度过慢）。若未启用优势归一化（Advantage Normalization），则拒绝部署。将任何满足 `c_e = 0` 且观测熵（Observed Entropy）小于 0.1 的运行标记为熵崩溃（Entropy Collapse）。
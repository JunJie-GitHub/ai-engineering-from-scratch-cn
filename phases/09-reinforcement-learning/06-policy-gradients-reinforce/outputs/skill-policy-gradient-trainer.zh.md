---
name: 策略梯度训练器 (Policy Gradient Trainer)
description: 为指定任务生成 REINFORCE / Actor-Critic / PPO 训练配置，并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [强化学习 (RL), 策略梯度 (Policy Gradient), REINFORCE]
---

给定环境（离散/连续动作、时间视界 (Horizon)、奖励统计 (Reward Stats)），输出以下内容：

1. 策略头 (Policy Head)。采用 Softmax（离散）或高斯分布 (Gaussian)（连续），并注明参数数量。
2. 基线 (Baseline)。无（基础版 (Vanilla)）、滑动平均 (Running Mean)、学习得到的 `V̂(s)` 或 A2C 评论家 (A2C Critic)。
3. 方差控制 (Variance Controls)。默认启用累积回报 (Reward-to-go)、回报归一化 (Return Normalization) 及梯度裁剪值 (Gradient Clip Value)。
4. 熵奖励 (Entropy Bonus)。系数 β 及其衰减计划 (Decay Schedule)。
5. 批次大小 (Batch Size)。每次更新的回合数；同策略 (On-policy) 数据新鲜度约束 (Data Freshness Contract)。

若时间视界 > 500 步，则拒绝使用无基线 REINFORCE (REINFORCE-no-baseline)。拒绝为连续动作控制 (Continuous-action Control) 任务使用 Softmax 策略头。将任何 `β = 0` 且观测到策略熵 < 0.1 的运行标记为熵坍缩 (Entropy Collapse)。
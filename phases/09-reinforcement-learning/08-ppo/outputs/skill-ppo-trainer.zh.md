---
name: ppo-trainer
description: 为给定环境生成近端策略优化 (PPO) 训练配置与诊断计划。
version: 1.0.0
phase: 9
lesson: 8
tags: [强化学习 (RL), 近端策略优化 (PPO), 策略梯度 (Policy Gradient)]
---

给定环境与训练预算，输出以下内容：

1. 轨迹展开规模 (Rollout size)。`N` 个环境 × `T` 步。
2. 更新计划 (Update schedule)。`K` 个轮次 (epochs)、小批量大小 (minibatch size)、学习率调度 (LR schedule)。
3. 代理目标参数 (Surrogate params)。`ε`（裁剪阈值）、`c_v`、`c_e`，开启优势函数归一化 (advantage normalization)。
4. 优势函数 (Advantage)。采用广义优势估计 GAE(`λ`)，并明确指定折扣因子 `γ` 与平滑系数 `λ`。
5. 诊断计划 (Diagnostics plan)。设置 KL 散度 (KL divergence)、裁剪比例 (clip fraction)、解释方差 (explained variance) 的阈值及告警机制。

拒绝 `K > 30` 或 `ε > 0.3` 的配置（信任域 (trust region) 不安全）。拒绝任何未开启优势函数归一化或未进行 KL 散度/裁剪比例监控的 PPO 运行。若裁剪比例持续高于 0.4，则标记为策略漂移 (drift)。
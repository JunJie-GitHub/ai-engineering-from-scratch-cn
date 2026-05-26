---
name: dqn-trainer
description: 为离散动作（discrete-action）强化学习（Reinforcement Learning, RL）任务生成 DQN（Deep Q-Network）训练配置（包含经验回放缓冲区、目标网络同步、ε 衰减计划、奖励裁剪）。
version: 1.0.0
phase: 9
lesson: 5
tags: [强化学习, DQN, 深度强化学习]
---

给定一个离散动作环境（观测空间形状、动作数量、时间视界（horizon）、奖励缩放比例），输出以下内容：

1. 网络（Network）。架构（多层感知机（MLP）/ 卷积神经网络（CNN）/ Transformer）、特征维度、网络深度。
2. 经验回放缓冲区（Replay Buffer）。容量、小批量（minibatch）大小、预热大小。
3. 目标网络（Target Network）。同步策略（硬同步：每 C 步更新一次，或软同步：使用系数 τ）。
4. 探索策略（Exploration）。ε 起始值 / 终止值 / 调度周期长度（schedule length）。
5. 损失函数（Loss）。Huber 损失与均方误差（MSE）的选择、梯度裁剪（gradient clipping）阈值、奖励裁剪规则。
6. Double DQN。默认启用，除非有明确理由需要禁用。

拒绝交付未包含目标网络、未使用经验回放缓冲区，或 ε 值固定为 1 的 DQN 配置。拒绝处理连续动作（continuous-action）任务（应引导至 SAC / TD3 算法）。若奖励范围超过单步平均值的 10 倍，需标记为需要进行奖励裁剪或尺度归一化（scale normalization）。
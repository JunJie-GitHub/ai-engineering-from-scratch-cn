---
name: rlhf-architect
description: 为语言模型 (Language Model) 设计基于人类反馈的强化学习 (RLHF) / 直接偏好优化 (DPO) / 群组相对策略优化 (GRPO) 对齐 (Alignment) 流水线，涵盖奖励模型 (Reward Model)、KL 散度 (KL Divergence) 及数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [强化学习 (RL), RLHF, 对齐 (Alignment), 大语言模型 (LLM)]
---

给定基础语言模型 (Language Model)、目标行为（对齐 / 推理 / 拒绝 / 智能体）以及偏好或验证器预算，输出以下内容：

1. 阶段。监督微调 (Supervised Fine-Tuning)？奖励模型 (Reward Model)？直接偏好优化 (Direct Preference Optimization)？群组相对策略优化 (Group Relative Policy Optimization)？需提供选择理由。
2. 偏好或验证器来源。人工标注、AI 反馈、基于规则、单元测试通过，或奖励蒸馏 (Reward Distillation)。
3. KL 策略。固定 β 值、自适应 β 值，或 DPO（隐式 KL 约束）。
4. 诊断指标。平均 KL 散度、奖励稳定性、过优化 (Over-optimization) 防护（保留集人工评估 (Holdout Human Evaluation)）。
5. 安全门控。红队测试集 (Red-team Set)、拒绝率、安全性奖励模型需与有用性奖励模型分离。

严禁在未配置 KL 监控器 (KL Monitor) 的情况下部署 RLHF-PPO。严禁使用参数量小于目标策略 (Target Policy) 的奖励模型。严禁仅依赖长度奖励 (Length-only Rewards)。若任何流水线未预留盲测人工评估集 (Blind Human Evaluation Set)，则标记为缺乏过优化防护。
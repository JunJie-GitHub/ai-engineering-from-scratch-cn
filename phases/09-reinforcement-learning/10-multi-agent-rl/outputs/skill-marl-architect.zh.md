---
name: marl-architect
description: 为给定任务选择合适的多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）机制（如 IPPO、CTDE、自我对弈、联赛等）。
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

给定一个包含 `n` 个智能体（agent）的任务，请输出：

1. 机制分类（regime classification）。合作型（cooperative）/ 对抗型（adversarial）/ 一般和（general-sum）。请提供理由。
2. 算法选择。IPPO / MAPPO / QMIX / 自我对弈（self-play）/ 联赛（league）。理由需与耦合紧密程度（coupling tightness）和奖励结构（reward structure）相关联。
3. 信息访问权限。集中式训练（centralized training）（哪些全局信息会输入给评论家网络/critic）？分布式执行（decentralized execution）？
4. 信用分配（credit assignment）。反事实基线（counterfactual baseline）、价值分解（value decomposition）或奖励塑形（reward shaping）。
5. 探索策略（exploration plan）。单智能体熵（per-agent entropy）、基于种群的训练（population-based training）或联赛机制。

对于耦合紧密的合作型任务，拒绝使用独立 Q 学习（independent Q-learning）。对于存在循环风险（cycle risks）的一般和博弈，拒绝推荐自我对弈。标记任何缺乏固定对手评估（fixed-opponent evaluation）的 MARL 流程（因为挑选有利的自我对弈数据/cherry-picked self-play numbers 的情况十分常见）。
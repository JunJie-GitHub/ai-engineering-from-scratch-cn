---
name: marl-picker
description: 为指定的多智能体任务选择多智能体强化学习（MARL）算法（MADDPG、QMIX、MAPPO、IQL 或其扩展）。需综合考虑合作与竞争关系、动作空间类型、智能体异构性、奖励结构以及任务规模。
version: 1.0.0
phase: 16
lesson: 20
tags: [多智能体, MARL, MADDPG, QMIX, MAPPO, CTDE]
---

根据多智能体任务描述，选择合适的 MARL 算法。

输出内容：

1. **任务分类（Task taxonomy）。** 完全合作（共享奖励）、完全竞争（零和博弈）、混合博弈或一般和博弈。智能体数量。同质（Homogeneous）与异构（Heterogeneous）。
2. **可观测性（Observability）。** 完全可观测（每个智能体均可获取全局状态）、部分可观测（每个智能体仅能获取自身观测）或支持通信。
3. **动作空间（Action space）。** 离散型（类似 Atari 游戏、SMAC 环境）或连续型（粒子世界、MuJoCo 环境）。该特性直接影响算法选择。
4. **奖励结构（Reward structure）。** 稠密奖励（每步提供塑形奖励）与稀疏奖励（仅在终止时提供）。稠密奖励使 MAPPO 更具实用性；稀疏奖励则需要信用分配（Credit assignment）机制的辅助（如 QMIX 的价值分解（Value decomposition））。
5. **算法推荐（Algorithm recommendation）。** 根据 Yu 等人（2022）的研究，首选 MAPPO 作为基线模型。在以下情况可切换至：
   - **QMIX**：适用于合作型 + 同质智能体 + 需要强稀疏奖励信用分配的场景
   - **MADDPG**：适用于混合型（合作与竞争并存）+ 连续动作空间的场景
   - **扩展算法（QTRAN、QPLEX、FACMAC）**：当单调性约束（Monotonicity constraint）过于严格时
6. **训练基础设施（Training infrastructure）。** 是否具备以下条件：充足的交互数据、计算预算、奖励塑形（Reward shaping）专业知识、稳定性预算（每个实验 5-10 个随机种子）？若不具备，建议为大语言模型（LLM）智能体采用提示词级策略（Prompt-level policies）。
7. **部署契约（Deployment contract）。** 集中训练分散执行（CTDE）：在部署阶段，每个智能体仅能获取局部观测。需明确定义该契约，以确保运行时代码严格遵守。

硬性拒绝项：

- 首次运行即选择非 MAPPO 基线。MAPPO 是 2026 年的标准基线，请从此开始。
- 在混合合作-竞争任务中使用 QMIX。价值分解假设单调聚合（Monotone aggregation）。
- 为缺乏交互数据或奖励信号的大语言模型智能体系统推荐 MARL 训练。在数据充足之前，提示词级策略的表现将更优。
- 训练时未记录每个智能体的观测与动作。这将导致无法进行调试。

拒绝规则：

- 若任务的交互数据少于约 1000 个回合（Episodes），建议采用提示词级策略或监督微调（Supervised fine-tuning）。
- 若任务是非马尔可夫的（Non-Markovian，需要记忆机制），但推荐方案未包含循环评论家网络（Recurrent critics），需明确指出该缺陷。
- 若任务为一般和竞争型（存在多个均衡点），仅靠 MARL 无法收敛至单一均衡；建议引入机制设计（Mechanism design）或均衡选择（Equilibrium selection）方法。

输出格式：一份单页简报。以一句话推荐开头（“采用带集中式价值函数的 MAPPO 基线；每个智能体使用离散型 Actor；部署时采用 CTDE 架构；每个实验使用 5 个随机种子。”），随后依次列出上述七个部分。最后附上从训练到部署的流水线：数据收集、模型训练、性能评估、策略推演（Rollout）。
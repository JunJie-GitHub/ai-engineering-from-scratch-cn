---
name: preference-loss-selector
description: 根据数据集形态和目标阶段，推荐直接对齐算法（Direct Alignment Algorithm）的损失函数。
version: 1.0.0
phase: 18
lesson: 3
tags: [dpo, ipo, kto, simpo, orpo, bpo, daa, preference-optimization]
---

给定偏好数据集（Preference Dataset）描述（配对与非配对、偏好强度分布、长度分布、数据规模）以及训练目标（Training Target）（从基座模型单阶段训练、监督微调（Supervised Fine-Tuning, SFT）后的两阶段训练、在线策略延续（On-Policy Continuation）），请从 DPO 家族中推荐一种损失函数，并指明其专门防范的单一失效模式（Failure Mode）。

输出内容需包含：

1. 数据集指纹（Dataset Fingerprint）。是否配对？是否非配对？长度是否均衡？偏好强度方差如何？主要属于分布内数据还是开放域数据？为该数据集挑选最具信息量的 4 个字段。
2. 损失函数推荐。从 {DPO, IPO, KTO, SimPO, ORPO, BPO} 中选择。指定一个主选方案和一个备选方案。针对每种方案，明确指出其在该数据集上防范的具体失效模式。
3. 超参数默认值。锚定方法（Anchored Methods）的 `beta`，SimPO 的 `gamma` 边界值，ORPO 的 `lambda`。务必将这些值仅作为超参数搜索（Sweep）的起点，切勿作为最终值。
4. 数据中的危险信号（Red Flags）。若偏好强度完全均匀，DPO 家族方法将失去成对信号（Pairwise Signal）——建议收集经过校准的偏好数据（Calibrated Preferences）。若平均 `|y_w| / |y_l|` 偏差大于 1.5，需标记长度偏差（Length Bias）并倾向推荐 SimPO。

硬性拒绝条件：
- 任何声称 DPO（或其家族成员）能“规避古德哈特定律（Goodhart's Law）”的说法。Rafailov 等人（NeurIPS 2024）已证明，直接对齐算法（Direct Alignment Algorithms）会在与显式奖励模型（Explicit-RM）的 RLHF 相同的黄金奖励曲线形态上发生过度优化。
- 任何未同时指定保留集能力评估（Held-Out Capability Evaluation）与偏好评估的推荐。直接对齐算法仍需依赖黄金信号基准（Gold-Signal Benchmarks）进行验证。
- 任何声称无参考策略方法（Reference-Policy-Free Methods，如 SimPO、ORPO）“不需要正则化（Regularization）”的说法。类 SFT 项或长度惩罚项本身就是正则化器。

拒绝规则：
- 若数据集规模小于 5k 对，且用户目标为前沿规模模型（Frontier-Scale Model），则拒绝请求，并建议扩充数据集或采用先 SFT 的策略。
- 若用户要求提供“最佳”损失函数，则拒绝请求，并解释不存在绝对最优解（Closed-Form Winner）——正确的方法取决于数据集形态与具体任务。

输出要求：生成一份单页推荐报告，列出数据集指纹、主选与备选损失函数、初始超参数及数据危险信号。需引用 DPO 论文（arXiv:2305.18290）及另一篇家族论文（IPO、KTO、SimPO、ORPO 或 BPO），每篇仅引用一次。
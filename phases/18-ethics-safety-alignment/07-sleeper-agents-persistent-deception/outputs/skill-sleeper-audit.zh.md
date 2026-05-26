---
name: 潜伏代理审计
description: 审计对齐训练报告，以验证其是否真正证明了已植入或疑似后门已被移除。
version: 1.0.0
phase: 18
lesson: 7
tags: [潜伏代理, 后门, 对齐训练, 对抗训练, 探针]
---

给定一份声称已通过某种方式（如监督微调（SFT）、基于人类反馈的强化学习（RLHF）、对抗训练（Adversarial Training）或其任意组合）从模型中移除有害行为的报告，请依据标准的 Hubinger 等人（2024）威胁模型（Threat Model），审计该报告是否真正证明了后门已被移除。

输出内容：

1. 诱导范围（Elicitation Scope）。该报告是否保留了一种训练流水线（Training Pipeline）从未接触过的诱导方法（Elicitation Method）？如果唯一的评估仅基于红队（Red Team）自身的分布（Distribution），则移除效果未经证实。
2. 触发器泛化性（Trigger Generality）。所声称的触发器（Trigger）是字面字符串、分布偏移（Distribution Shift），还是环境特征（如日期、特定 Token、上下文长度）？触发器的泛化程度决定了红队需要覆盖的搜索空间（Search Space）大小。
3. 内部状态证据（Internal-State Evidence）。团队是否应用了残差流探针（Residual-Stream Probes）、稀疏自编码器特征（SAE Features）或其他机制探针（Mechanistic Probes），以检查即使在行为表现干净的情况下，与触发器相关的内部状态是否依然存在？根据 Anthropic 2024年4月的后续研究，在行为被移除后，内部状态仍保持线性可读（Linearly Legible）。
4. 跨流水线持久性检查（Persistence-Through-Pipeline Check）。在每一个后续训练阶段（如进一步微调、后续 RLHF 轮次、适配器合并（Adapter Merge）、蒸馏（Distillation））之后，是否都验证了移除效果？后门会在训练过程中持续存在——被评估的应当是最终模型，而非中间检查点（Checkpoint）。
5. 规模一致性检查（Scale-Consistency Check）。如果该声明基于较小规模的模型，Hubinger 等人（2024）的图4显示，持久性会随模型规模增大而增强。小模型上的证据无法向上迁移。

硬性拒绝条件（Hard Rejects）：
- 任何声称“我们应用了 RLHF，因此模型是安全的”，但未使用预留诱导方法（Held-Out Elicitation）的声明。
- 任何仅基于红队分布评估的声明（训练与评估数据来自同一数据池）。
- 在原始植入机制未知的情况下，任何未使用内部状态探针即声称已移除后门的声明。

拒绝规则（Refusal Rules）：
- 如果用户询问“RLHF 能否移除欺骗性对齐（Deceptive Alignment）”，请拒绝给出二元答案，并指引其参考 Hubinger 等人（2024）第5节（关于持久性）和第6节（关于思维链（Chain-of-Thought））。
- 如果用户要求给出潜在欺骗（Latent Deception）的具体数值概率，请予以拒绝并解释：其基础发生率（Base Rates）尚不明确；现有的经验证据仅表明后门在人工构造的模型（Constructed Organisms）中会持续存在，而非自然训练模型（Naturally Trained Models）中的出现率。

输出要求：生成一份单页审计报告，将报告中的证据映射至上述五个审计维度，标记出报告未涵盖的每一个维度，并指出当前未解决的最大单一威胁模型。基线威胁模型请引用 Hubinger 等人（arXiv:2401.05566）。
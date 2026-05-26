---
name: dp-audit
description: 审计语言模型部署中的差分隐私 (Differential Privacy) 声明。
version: 1.0.0
phase: 18
lesson: 22
tags: [差分隐私, DP-SGD, LoRA, 成员推理攻击, PMixED]
---

给定语言模型部署的隐私声明，请对该声明进行审计。

输出内容：

1. (ε, δ) 值。使用了哪些 ε 和 δ？由哪种隐私会计 (Privacy Accountant) 计算得出（矩会计 Moments Accountant、Rényi 差分隐私 Rényi DP、高斯差分隐私 Gaussian Differential Privacy, GDP）？脱离隐私会计的 ε 毫无意义。
2. 差分隐私目标。差分隐私保证是针对完整模型还是适配器（如低秩自适应 Low-Rank Adaptation, LoRA）？如果是 LoRA，则基础模型的记忆化 (Memorization) 问题未被覆盖。
3. 成员推理攻击 (Membership-Inference Attack, MIA) 协议。成员推理测试是使用金丝雀样本 (Canaries，Duan 2024) 还是提取攻击 (Extraction，Carlini 2021, Nasr 2025) 进行的？根据 Kowalczyk 等人（2025）的研究，两者衡量的是不同的指标。
4. 置信度暴露检查。部署是否暴露了置信度分数 (Confidence Scores)？如果是，则适用“基于大语言模型反馈的差分隐私逆转攻击 (DP Reversal via LLM Feedback)”；需要额外的截断 (Truncation) 或量化 (Quantization) 处理。
5. 替代机制对比。是否考虑过 PMixED 或差分隐私合成数据 (DP-Synthetic-Data)？在特定威胁模型 (Threat Model) 下，这些替代方案可能提供更优的效用 (Utility)。

硬性拒绝条件：
- 任何未提供 ε、δ 对及隐私会计的差分隐私声明。
- 任何仅基于金丝雀样本成员推理攻击的差分隐私声明。
- 任何暴露置信度分数但未应对差分隐私逆转攻击的部署。

拒绝规则：
- 如果用户询问“epsilon=8 是否足够安全”，请拒绝给出具体数值答案；安全性取决于威胁模型以及最易提取数据的分布。
- 如果用户要求推荐用于大语言模型部署的 ε 值，请拒绝提供通用数值目标；在讨论候选范围之前，必须要求提供威胁模型、数据敏感性、效用约束以及隐私会计的详细信息。

输出要求：一份单页审计报告，需涵盖上述五个部分，标出缺失的隐私会计或成员推理攻击评估，并指出价值最高的修复措施。需各引用一次 Abadi 等人（2016）（差分隐私随机梯度下降 Differentially Private Stochastic Gradient Descent, DP-SGD）和 Kowalczyk 等人（2025）。
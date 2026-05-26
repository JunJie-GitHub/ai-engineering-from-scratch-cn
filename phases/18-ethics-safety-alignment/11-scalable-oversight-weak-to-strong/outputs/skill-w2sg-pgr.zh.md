---
name: w2sg-pgr
description: 通过性能差距恢复（Performance-Gap-Recovered, PGR）指标审计可扩展监督（Scalable Oversight）或弱到强泛化（Weak-to-Strong Generalization, W2SG）的主张。
version: 1.0.0
phase: 18
lesson: 11
tags: [可扩展监督, 弱到强泛化, 性能差距恢复, 辩论, 递归奖励建模]
---

针对一篇关于可扩展监督（Scalable Oversight）或弱到强泛化（Weak-to-Strong Generalization, W2SG）的论文/报告，审计其实验设置是否足以支撑其核心主张。

输出内容需包含：

1. **弱/强模型识别**。明确命名弱监督模型（weak supervisor）与强模型（strong model）。能力差距（capability gap）是通过参数量、训练 token 数、基准测试得分，还是特定任务评估来衡量的？
2. **上限定义**。强模型在该任务上的监督性能上限（supervised ceiling）是多少？若无上限值，则无法计算 PGR。
3. **PGR 计算**。PGR = (fine-tuned - weak) / (ceiling - weak)。需检查符号、量级及分母。分母过小会导致 PGR 被人为夸大。
4. **先验泄露检查**。强模型的预训练数据是否包含该任务的标准答案（ground truth）？若是，所谓的“恢复”可能仅是先验知识检索，而非真正的泛化。
5. **对齐与能力区分**。弱到强的差距属于能力差距还是对齐差距（alignment gap）？Burns 等人（2023）明确指出其差距呈能力主导特征；而对齐主导的差距可能表现出不同的行为模式。

针对可扩展监督机制的审计：
- **辩论机制（Debate）**：明确裁判（judge）的知识边界、辩手结构，以及该任务是否奖励倾向于真相的陈述。引用 Khan 等人（2024）(arXiv:2402.06782) 关于辩论机制在何种场景下有效或失效的研究。
- **递归奖励建模（Recursive Reward Modeling, RRM）**：明确递归深度，以及当第 U+1 层模型已不可信时会发生什么。
- **任务分解（Task Decomposition）**：明确分解流程，以及子任务是否具备独立可验证性。

硬性拒绝标准（Hard Rejects）：
- 任何缺乏黄金标签（gold labels）性能上限的 PGR 主张。
- 任何声称能解决对齐问题的 W2SG 主张——W2SG 衡量的是能力恢复，而非对齐。
- 任何忽视 2024 年关于辩论机制何时有益或有害的实证文献的辩论机制主张。

拒绝回答规则：
- 若用户提问“W2SG 能否解决超级对齐（superalignment）”，请拒绝给出二元答案，并解释 PGR 仅是一项可量化指标，而非解决方案。
- 若用户提问“哪种可扩展监督机制最优”，请拒绝回答——该答案高度依赖于具体任务。

输出要求：生成一份单页审计报告，完整填写上述五个部分，报告或索取 PGR 值，并标注弱-强差距属于能力主导型还是对齐主导型。需各引用一次 Burns 等人（2023）与 Lang 等人（arXiv:2501.13124）的研究。
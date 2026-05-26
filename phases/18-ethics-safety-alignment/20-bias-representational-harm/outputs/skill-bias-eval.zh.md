---
name: 偏差评估
description: 跨指标类别、交叉性（intersectionality）与去偏机制（debias mechanism）对偏差评估报告进行审计。
version: 1.0.0
phase: 18
lesson: 20
tags: [偏差, 公平性, WEAT, 交叉性, 机制可解释性]
---

给定一份偏差评估报告或公平性声明，请依据 Gallegos 等人（2024）的三类别框架以及 2024-2025 年关于交叉性（intersectionality）的文献进行审计。

输出内容：

1. 指标覆盖范围（metric coverage）。该评估是否在每个类别中至少包含一项指标：基于嵌入的（embedding-based，WEAT 风格）、基于概率的（probability-based，刻板印象对数似然）、基于生成文本的（generated-text-based，下游任务测量）？标记缺失的类别。
2. 伤害类型区分（harm-type separation）。该评估是否区分了表征性伤害（representational harm）与分配性伤害（allocational harm）？仅测量刻板印象生成的报告并未衡量下游资源分配情况。
3. 交叉性覆盖范围（intersectionality coverage）。是否评估了交叉性维度（intersectional axes），还是仅评估单一维度（如仅性别或仅种族）？根据 An 等人（2025）的研究，单一维度评估通常会遗漏交叉性效应。
4. 去偏机制（debias mechanism）。若已应用去偏（debiasing）操作，请识别其作用于嵌入层（投影）、MLP 神经元（Yu & Ananiadou 2025）、SAE 特征（Ahsan & Wallace 2025）、注意力头（UniBias 2024），还是后处理输出过滤。估算其对通用能力（general-capability）的损耗成本。
5. 维度多样性（axis diversity）。根据 2025 年的元批判研究，相较于其他维度，二元性别偏差被过度研究。该评估是否涵盖了残障、宗教、移民或多语言身份等维度？

硬性拒绝条件（hard rejects）：
- 任何仅基于单一指标类别的“已去偏”声明。
- 任何未经交叉性评估的公平性声明。
- 任何未提供通用能力变化量（general-capability delta）的去偏干预措施。

拒绝规则（refusal rules）：
- 若用户询问其模型是否“无偏差”，请拒绝该二元论断；偏差是一种具有多项指标的连续属性。
- 若用户请求推荐单一的去偏操作，请拒绝提供唯一建议；具体选择取决于偏差存在于何处（嵌入层、神经元、注意力头或输出端）。

输出要求：一份单页审计报告，需完整填写上述五个部分，标记缺失的指标类别，并推荐一项价值最高的补充评估。需分别引用一次 Gallegos 等人（2024）的文献以及一篇 2024-2025 年的交叉性研究论文。
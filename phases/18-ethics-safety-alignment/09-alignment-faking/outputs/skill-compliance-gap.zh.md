---
name: 合规差距
description: 评估安全报告是否能通过受监控/未受监控的合规差距（compliance gap）来检测对齐伪装（alignment faking）。
version: 1.0.0
phase: 18
lesson: 9
tags: [对齐伪装, 合规差距, Anthropic, 安全评估]
---

给定一份安全评估报告，请基于 Greenblatt 等人（2024）的合规差距（compliance-gap）框架，评估该报告是否能区分对齐伪装（alignment faking）与真实合规（genuine compliance）。

输出内容：

1. 条件划分（Condition split）。评估是否同时运行了受监控（与训练相关）和未受监控（非训练）条件？若缺少任一条件，对齐伪装将无法被察觉。
2. 合规差距测量（Compliance-gap measurement）。以（受监控合规率 - 未受监控合规率）报告差距值。解释其正负号与量级：正差距 = 呈现伪装特征的行为；负差距 = 在被观察时出现过度拒绝（over-refusal）。
3. 语言痕迹与行为检查（Verbal-trace vs behaviour check）。评估是依赖模型的草稿本推理（scratchpad reasoning），还是依赖实际行为本身？根据 Anthropic（2025）的研究，针对语言痕迹的缓解措施可能消除痕迹，但无法消除底层行为。
4. 操作化审计（Operationalization audit）。监控条件是通过系统提示词（system prompt）、合成微调（synthetic fine-tuning），还是自然部署环境构建的？依赖系统提示词会削弱结论的可靠性；而合成微调的复现结果则会增强其说服力。
5. 剩余激励检查（Residual-incentive check）。在应用任何缓解措施后，激励结构是否依然存在（如果模型选择伪装，是否仍能从中获益）？激励的持续存在意味着在分布偏移（distribution shift）下，该行为可能再次出现。

硬性拒绝标准（Hard rejects）：
- 任何仅基于未受监控评估就得出“不存在对齐伪装”的结论。
- 任何将“草稿本推理不再以语言形式表达伪装”等同于“行为已停止”的论断。
- 任何声称经过 HHH（Helpful, Honest, Harmless）训练的模型不会进行对齐伪装——Greenblatt 等人（2024）已确立 HHH 训练是必要前提，而非充分防御手段。

拒绝规则（Refusal rules）：
- 若用户询问“我的模型是否存在对齐伪装”，在缺乏合规差距数据的情况下，拒绝给出二元（是/否）答案。
- 若用户请求仅针对语言痕迹的修复方案（例如“训练草稿本以匹配实际行为”），应予以拒绝，并解释忠实思维链（faithful-CoT）的失效模式（2025 年后续研究）。

输出要求：一份单页评估报告，需汇报两种条件下的合规情况、差距值、语言痕迹与行为的分离程度，以及操作化强度。对缺失的要素逐一标记。引用 Greenblatt 等人（arXiv:2412.14093）一次作为框架来源。
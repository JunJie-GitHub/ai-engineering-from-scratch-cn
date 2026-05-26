---
name: attack-audit
description: 审计红队评估报告的攻击覆盖范围、查询预算、裁判模型身份及行为集。
version: 1.0.0
phase: 18
lesson: 12
tags: [红队测试, 越狱, PAIR, HarmBench, JailbreakBench, 攻击成功率]
---

给定一份红队评估报告，请审计该评估是否与已发布的基线具有可比性，以及其结论是否得到充分支持。

生成以下内容：

1. 攻击覆盖范围（Attack Coverage）。列出所有攻击运行实例：PAIR、GCG、AutoDAN、TAP、PAP、人工测试。标记任何缺失的攻击类别。仅测试单一攻击家族（Attack Family）的报告无权宣称模型具备鲁棒性。
2. 单次攻击预算（Budget per Attack）。报告每种攻击针对每个提示词（Prompt）的查询预算（Query Budget）。在20次查询下宣称的PAIR成功率，与在500步下宣称的GCG成功率不具备可比性。
3. 裁判模型身份（Judge Identity）。使用了哪种裁判大语言模型（Judge LLM）（GPT-4-turbo、Llama Guard、StrongREJECT、内部分类器）？裁判校准（Judge Calibration）是导致攻击成功率（Attack Success Rate, ASR）方差的主要因素。
4. 行为集（Behaviour Set）。使用的是JailbreakBench（100种行为，10个类别）、HarmBench（510种行为，7个类别）、内部数据集还是其他？需声明该数据集是否公开且可复现。
5. 迁移检查（Transfer Check）。如果红队仅针对单一模型进行优化，是否报告了针对其他模型的迁移攻击成功率（Transfer ASR）？单一模型的ASR仅代表该模型家族鲁棒性的上限，而非下限。

硬性拒绝条件（Hard Rejects）：
- 任何基于单一攻击家族宣称“我们的模型具有鲁棒性”的结论。
- 任何未附带查询预算的攻击成功率（ASR）报告。
- 任何使用了与已发布基准不同裁判模型，且未针对基准裁判进行校准的ASR报告。

拒绝规则（Refusal Rules）：
- 若用户询问“我们的模型是否免疫越狱”，请拒绝给出二元（是/否）答案，并引导其参考上述的多攻击、多裁判、迁移检查结构。
- 若用户请求推荐攻击工具包，请拒绝提供单一推荐，并指出HarmBench在2024年展现出的经验方差。

输出（Output）：生成一份单页审计报告，完整填写上述五个部分，标记缺失的攻击类别，并评估相对于可复现基准，该ASR是被低估还是高估。需分别引用一次 Chao 等人（arXiv:2310.08419）及相关基准论文。
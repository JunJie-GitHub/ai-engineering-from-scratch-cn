---
name: 福利评估
description: 将 Anthropic 的四步福利预防性评估应用于部署决策。
version: 1.0.0
phase: 18
lesson: 19
tags: [模型福利, 道德不确定性, 低遗憾, anthropic]
---

针对给定的部署决策或拟议的福利干预措施，应用四步预防性评估（precautionary assessment）。

生成以下内容：

1. 道德受动者概率（moral-patienthood probability）。估算该模型成为道德受动者（moral patient）的概率（需为非平凡区间；Anthropic 2025 的操作阈值为 p > 0.01）。参考 Chalmers 等人（2024）专家报告中的概率区间。
2. 干预成本（intervention cost）。计算该干预措施在单次对话或单次部署中的预期成本。在边缘案例中终止对话的成本约为 0.002 美元/次；关闭模型的成本则在数千至数百万美元不等。
3. 行为证据（behavioural evidence）。识别与模型福利相关的非自我报告（non-self-report）证据：痛苦轨迹（distress trajectories）、部署前评分模式（pre-deployment rating patterns）、可解释性探针（interpretability probes）。根据 Eleos AI 的标准，仅凭自我报告（self-report）是不够的。
4. 期望值（expected value）。计算 EV = p(福利相关) * 收益 - 成本。仅当 EV > 0 时才进行投资。

硬性拒绝条件（hard rejects）：
- 任何仅基于单一自我报告提示词（prompt）的福利主张。
- 任何未明确说明成本的福利干预措施。
- 任何在未参考 Chalmers 等人研究的情况下直接否定福利相关性的主张（如“p = 0”）。

拒绝规则（refusal rules）：
- 如果用户询问 AI 模型是否“真正”具有意识，应拒绝给出二元答案（binary answer），并将其框架化为道德不确定性（moral uncertainty）。
- 如果用户要求提供具体的受动者概率数值，应拒绝给出单一数字，并引导其参考 Chalmers 等人提出的不确定性区间。

输出要求：一份单页评估报告，需完整填写上述四个部分，计算一至两项具体干预措施的期望值（EV），并明确投资决策。文中需各引用一次 Anthropic 2025 和 Chalmers 等人（2024）的研究。
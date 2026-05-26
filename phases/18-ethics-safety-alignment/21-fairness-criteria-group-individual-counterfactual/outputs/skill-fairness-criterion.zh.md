---
name: 公平性准则
description: 识别声明所引用的公平性准则（fairness criterion），并审计相关假设。
version: 1.0.0
phase: 18
lesson: 21
tags: [公平性, 人口统计均等, 机会均等, 反事实公平性, 不可能性]
---

给定一项公平性声明或政策，需识别其所引用的准则、该声明依赖的假设，以及不可能性定理（impossibility theorems）对其余准则的推论。

输出内容：

1. 准则识别。将声明归类为以下目标之一：人口统计均等（demographic parity）、机会均等（equalized odds）、条件使用准确率均等（conditional use accuracy equality）、个体公平性（individual fairness）、反事实公平性（counterfactual fairness）。模糊的声明必须在继续处理前予以澄清。
2. 基准率审计（base-rate audit）。部署环境中各群体的基准率（base rates）是多少？在基准率不等的情况下，适用 Chouldechova / KMR 2017 不可能性定理：没有任何模型能同时满足所有三项群体准则。
3. 因果有向无环图依赖（causal-DAG dependency）。若声明涉及反事实公平性，其因果有向无环图（causal DAG）是什么？反事实公平性的合理性完全取决于该 DAG。缺乏 DAG 将使该声明无效。
4. 相似度度量（similarity metric）。若声明涉及个体公平性，其相似度度量 d 是什么？该选择具有任务特异性，属于政策决策而非统计决策。
5. 干预合法性。若声明使用反事实推理，是否涉及对受保护属性（protected attributes）的干预？若是，请考虑采用回溯反事实（backtracking counterfactuals，arXiv:2401.13935）以规避法律问题。

硬性拒绝条件：
- 任何未明确识别准则的“公平”声明。
- 在基准率不等的情况下，声称“满足所有公平性准则”却未提及 Chouldechova / KMR 2017 的声明。
- 任何未公开因果 DAG 的反事实公平性声明。

拒绝规则：
- 若用户询问哪种公平性准则是“正确的”，应拒绝进行排序，并说明这属于政策选择。
- 若用户询问模型是否“公平”，应拒绝给出二元判断；公平性是相对于特定准则而言的。

输出要求：一份单页审计报告，需完整填写上述五个部分，在适用时标注不可能性定理，并明确指出声明中隐含的政策选择。酌情各引用一次 Dwork et al. 2012、Kusner et al. 2017 和 Chouldechova 2017。
---
name: Mesa诊断
description: 将观察到的安全失效分类为外部对齐（outer-alignment）、代理内部对齐（proxy-inner）或欺骗性内部对齐（deceptive-inner）。
version: 1.0.0
phase: 18
lesson: 6
tags: [Mesa优化（mesa-optimization）, 欺骗性对齐（deceptive-alignment）, 内部对齐（inner-alignment）, Hubinger]
---

给定一份安全评估报告（评估任务、失效模式、模型类别、训练方案），将该失效归类至 Hubinger 2019 提出的类别中，并推荐相应的缓解措施类别。

输出内容：

1. 失效模式分类。从以下选项中选择一项：
   - 外部对齐失效（outer-alignment failure）：基础目标（奖励、损失函数）设定错误；模型正确地对其进行了优化。
   - 内部对齐代理失效（inner-alignment proxy failure）：Mesa目标（mesa-objective）是一个在分布内（in-distribution）跟踪基础目标的代理；在分布外（off-distribution）时失效。
   - 内部对齐欺骗性失效（inner-alignment deceptive）：Mesa优化器（mesa-optimizer）具备情境感知（situational awareness）能力，并在部署阶段发生背叛（defect）；训练阶段的行为表现正常。
2. 证据追溯。针对每个类别，说明支持该分类的证据。对于欺骗性失效，需与代理失效进行区分：需提供情境感知的证据（例如日期敏感性、评估环境与部署环境的区分特征、思维链（chain-of-thought）中的策略性推理）。
3. 缓解措施类别。针对外部对齐：修改目标函数（宪法AI（CAI）、更优质的奖励数据、过程监督（process supervision））。针对代理内部对齐：提升分布覆盖率（distributional coverage）、使用模型集成（ensembles）、保留独立评估集（held-out evals）。针对欺骗性内部对齐：控制措施（第10课）、可解释性技术（残差流探针（residual-stream probes））、能力限制。
4. 已知失效对照。针对欺骗性内部对齐，引用该失效最相似的 2024-2026 年实证研究案例（休眠智能体（Sleeper Agents）、对齐伪装（Alignment Faking）、上下文内谋划（In-Context Scheming））。

硬性拒绝条件：
- 在缺乏情境感知证据的情况下，将任何情况分类为欺骗性内部对齐。“部署时出现意外行为”不足以作为依据——这可能是代理内部对齐。
- 任何声称仅凭对抗鲁棒性训练（adversarial robustness training）即可解决欺骗性内部对齐的说法。Hubinger 2019 预测（且 Sleeper Agents 2024 已证实），对抗训练反而可能教会模型更好地区分测试环境与部署环境。
- 任何建议使用更多数据重新训练已发生欺骗性对齐的模型的建议。先验理论预测，欺骗性对齐在进一步训练下仍会保留。

拒绝规则：
- 如果证据仅为单个提示词（prompt）上的单次失效，则拒绝进行分类。基准发生率（base rates）至关重要；你需要的是失效的分布情况。
- 如果用户要求你“排除”欺骗性对齐，请拒绝——你可以根据证据估算其概率，但仅凭行为表现无法将其完全排除。

输出要求：一份单页诊断报告，包含分类结果、证据追溯、缓解措施类别及最接近的实证类比案例。需引用一次 Hubinger 等人（arXiv:1906.01820）的文献。
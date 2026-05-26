# 评估与协调基准测试

> 2025-2026年的五大基准测试覆盖了多智能体评估领域 (multi-agent evaluation space)。**MultiAgentBench / MARBLE**（ACL 2025, arXiv:2503.01935）采用星型/链式/树状/图拓扑结构 (topologies) 并结合里程碑关键绩效指标 (milestone KPIs) 进行评估；**图拓扑最适合研究**，引入认知规划 (cognitive planning) 可使里程碑达成率提升约3%。**COMMA** 评估多模态非对称信息协调 (multimodal asymmetric-information coordination)；包括 GPT-4o 在内的最先进模型 (state-of-the-art models) 均难以超越随机基线 (random baseline)。**MedAgentBoard**（arXiv:2505.12371）涵盖四类医疗任务，且常发现多智能体系统并未全面优于单一大语言模型 (single-LLM)。**AgentArch**（arXiv:2509.10769）对结合工具使用 + 记忆 + 编排 (tool-use + memory + orchestration) 的企业级智能体架构 (enterprise agent architectures) 进行基准测试。**SWE-bench Pro**（[arXiv:2509.16941](https://arxiv.org/abs/2509.16941)）包含跨越商业应用、B2B服务及开发者工具的41个代码库中的1865个问题；前沿模型 (frontier models) 在 Pro 版本上的得分约为23%，而在 Verified 版本上则超过70%——这揭示了数据污染 (contamination) 带来的现实检验。据报道，Claude Opus 4.7（2026年4月）在明确采用智能体团队协调 (agent-teams coordination) 的情况下，Pro 版本得分达 **64.3%**（Anthropic 尚未发布主要来源——请视为初步数据）；Verdent（智能体脚手架 (agent scaffold)）在 Verified 版本上达到 **76.1% pass@1**（[Verdent 技术报告](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。**AAAI 2026 Bridge Program WMAC**（https://multiagents.org/2026/）是2026年社区的核心焦点。本节课程基于 MARBLE 的指标体系，执行拓扑结构与指标的对比扫描，并确立“仅通过 SWE-bench Verified 测试并不能证明泛化能力 (generalization)”这一原则。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 第16阶段 · 15（投票与辩论拓扑结构），第16阶段 · 23（故障模式）
**Time:** 约75分钟

## 问题

当一篇论文宣称“我们的多智能体系统更优”时，核心问题在于：比什么更优？在什么任务上？如何衡量？2023-2024年间的多智能体评估领域处于混乱状态——各研究团队自行定义指标、基线和任务集。而2025-2026年的基准测试则引入了规范化结构。

缺乏共享基准测试，就无法对两个多智能体系统进行有意义的对比。更糟糕的是，如果没有预留测试集 (hold-out benchmarks)，前沿模型极易发生数据污染。到2025年中，SWE-bench Verified 的部分内容已混入训练语料库 (training corpora)，导致前沿模型得分虚高；Pro 版本正是为此设计的、用于检验真实水平的无污染基准。

本节将列举2026年五大权威基准测试，明确各项测试的评估维度，并教你以审慎的态度解读基准测试的宣称结果。

## 概念

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935。在研究、编程和规划任务上评估四种协调拓扑结构（coordination topologies）（星型、链式、树状、图结构）。基于里程碑的关键绩效指标（KPI）用于追踪阶段性进展，而非仅关注最终是否成功。

实测结果：

- **图结构（Graph）**拓扑最适合研究场景；支持任意智能体间的相互评审（critique）。
- **链式（Chain）**最适合逐步优化的编程任务。
- **星型（Star）**最适合快速整合事实性信息。
- 在图结构上，当智能体数量超过约 4 个时，会出现**协调开销（coordination tax）**。
- 引入**认知规划（cognitive planning）**可使各拓扑结构的里程碑达成率提升约 3%。

适用场景：当你需要在同等条件下公平比较不同协调拓扑结构时。MARBLE 仓库（https://github.com/ulab-uiuc/MARBLE）提供了评估器。

### COMMA — 多模态非对称信息

涵盖智能体具备不同观测模态，且必须在无法完全共享信息的情况下进行协调的任务。报告的结果令人不安：包括 GPT-4o 在内的前沿模型（frontier models）在 COMMA 的智能体间协作任务中，难以击败**随机基线（random baseline）**。这表明多智能体多模态能力尚未得到充分训练与评估——大语言模型（LLM）处理单模态协作尚可，但多模态协调则会崩溃。

适用场景：当你的系统涉及多模态或非对称信息协调时。COMMA 的零结果（null result）是一个警告：在宣称能力之前务必先进行测量。

### MedAgentBoard — 领域压力测试

arXiv:2505.12371。涵盖四大医疗任务类别：诊断、治疗规划、报告生成、医患沟通。对比了多智能体系统、单一大语言模型与传统基于规则的系统。

发现：在大多数类别中，多智能体系统并未显著优于单一大语言模型。多智能体的优势范围较窄——当子任务界限清晰时（如诊断+治疗），任务分解能带来帮助；但当协调开销超过专业化带来的收益时（如报告生成），反而会起反作用。

适用场景：当你的领域存在明确的单一大语言模型基线时。如果 MedAgentBoard 的结论具有普适性，那么许多已提出的多智能体系统可能属于过度设计。

### AgentArch — 企业级架构

arXiv:2509.10769。针对企业环境，将工具调用、记忆模块与编排层进行分层整合。该基准测试隔离了每一层的贡献度：增加工具能带来多少提升？增加记忆呢？增加多智能体编排呢？

适用场景：当你正在设计企业级智能体技术栈，且需要为每一层提供合理性证明时。AgentArch 有助于避免引入那些无法衡量其价值的功能。

### SWE-bench Pro — 现实检验

arXiv:2509.16941。涵盖 41 个代码仓库的 1865 个问题，涉及商业应用、B2B 服务和开发者工具。其设计旨在**免受污染（uncontaminated）**，即避开模型训练截止日期后的数据泄露。前沿模型在 Pro 版本上的得分约为 23%，而在 Verified 版本上超过 70%。这一差距正是数据污染（contamination）的信号。

2026 年 4 月得分：
- Claude Opus 4.7 在 Pro 上：**64.3%**（报告明确提及使用了智能体团队协调；Anthropic 尚未发布官方一手来源——请视为初步数据）。
- Verdent（智能体脚手架（agent scaffold））在 Verified 上：**76.1% pass@1**（[技术报告](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。
- 前沿模型在未使用智能体脚手架情况下的 Pro 原始得分：约 23-35%（[SWE-bench Pro 论文](https://arxiv.org/abs/2509.16941)）。

核心结论：“我们击败了 SWE-bench Verified”已不再能作为能力证明。Pro 才是当前的准入门槛测试。智能体团队脚手架在 Pro 上带来了可衡量的提升（约 30-40 个百分点的差距），这是 2026 年支持多智能体协调最有力的实证论据之一。

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — 多智能体协调研讨会（https://multiagents.org/2026/）。这是 2026 年多智能体 AI 研究的社区焦点。已录用的论文与研讨会论文集是评估新方法的权威渠道；在做出生产环境决策时，应优先参考 WMAC 录用的声明，而非 arXiv 预印本。

### 审慎看待基准测试声明 —— 2026 年核查清单

当有人宣称取得多智能体成果时：

1. **使用的是哪个基准测试，哪个数据划分（split）？** SWE-bench Verified 与 Pro 差异巨大。在错误划分上报告的数值毫无价值。
2. **数据污染检查。** 该基准测试是否在模型训练截止日期之后发布？如果不是，请谨慎对待。
3. **基线对比。** 应对比单一大语言模型基线、随机基线或先前的多智能体工作。而不是“对比同一系统未调优的版本”。
4. **统计显著性。** 试验次数（N）、p 值、置信区间。前沿模型方差较高，单次运行结果具有误导性。
5. **任务多样性。** 是单一任务还是多项任务？泛化能力对生产环境至关重要。
6. **成本披露。** 单任务 Token 消耗量、实际运行时间（wall-clock time）。以 20 倍成本换取 90% 的解决方案属于商业决策，而非能力声明。

### 当前基准测试均未能良好衡量的维度

- **长周期协调。** 持续数天的实际交互。当前所有基准测试的运行周期都过短。
- **对抗鲁棒性。** 当某个智能体具有恶意或被攻破时会发生什么？
- **部署后的分布漂移。** 基准测试是静态的，而生产环境的数据分布会发生变化。
- **成本归一化性能。** 大多数基准测试仅报告原始准确率，而非“每美元准确率”。

针对你真正关心的维度构建内部基准测试，通常是更明智的选择。

## 动手构建
`code/main.py` 是一个非交互式演练指南：

- 在一个玩具任务（toy task）上模拟 3 个多智能体系统（multi-agent systems）。
- 为每个系统计算 MARBLE 风格的里程碑指标（milestone metrics）。
- 通过从“训练”集中保留部分任务来运行数据污染检查（contamination check）。
- 明确地与随机基线（random baseline）进行对比。
- 打印基准测试声明记分卡（benchmark-claims scorecard）。

运行：

python3 code/main.py

预期输出：包含原始准确率（raw accuracy）、里程碑达成率、单任务成本（cost-per-task）、相较于随机基线的差值（vs-random baseline delta）以及污染检查备注的系统记分卡。

## 实际应用
`outputs/skill-benchmark-reader.md` 用于读取任何多智能体基准测试声明，并应用审查清单（scrutiny checklist）。输出：评级与注意事项（caveats）。

## 部署上线
生产环境评估准则：

- **构建内部基准测试**，使其反映你实际的生产数据分布（production distribution）。公共基准测试仅提供参考，不能替代内部测试。
- 在每次对比中**包含随机基线**。如果在协调任务（coordination task）上无法大幅超越随机基线，说明该任务可能定义不当（ill-posed）。
- **同时报告成本与准确率**。包括 Token 消耗和实际运行时间（wall-clock）。运维团队需要这两项数据。
- **每季度重建基准测试**。生产环境的数据分布会发生变化，过时的基准测试会产生误导。
- **避免对已发布基准测试的过拟合（overfitting）**。如果你的团队专门针对 SWE-bench Pro 的分数进行优化，那么在实际生产环境中的表现将会倒退。

## 练习

1. 运行 `code/main.py`。找出三个模拟系统中单里程碑成本（cost-per-milestone）最优的是哪一个。它是否与原始准确率最高的系统一致？
2. 阅读 MultiAgentBench 论文（arXiv:2503.01935）。针对你自己的任务领域，判断 MARBLE 会推荐四种拓扑结构（topologies）中的哪一种。请结合论文结果给出理由。
3. 阅读 SWE-bench Pro 论文。具体是什么机制使其具备抗污染能力（contamination-resistant）？同样的技术能否应用于你关注的其他基准测试？
4. 阅读 COMMA 关于多模态协调（multimodal coordination）的研究发现。设计一个简单的多模态协调任务，并将其加入你的内部基准测试中。什么样的结果可以被视为有效信号（useful signal）？
5. 将基准测试声明审查清单应用于近期某篇多智能体论文的核心结论。你会给该声明打什么等级？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------------|------------------------|
| MARBLE | “MultiAgentBench” | ACL 2025 会议论文；支持星型/链式/树状/图状拓扑结构（Topology），并引入里程碑关键绩效指标（KPI）。 |
| COMMA | “多模态基准测试” | 多模态非对称信息协调任务；前沿模型（Frontier Models）在此任务上的表现甚至不如随机策略。 |
| MedAgentBoard | “领域压力测试” | 涵盖四大医疗类别；测试结果表明多智能体（Multi-Agent）系统往往无法显著优于单一大语言模型（LLM）。 |
| AgentArch | “企业级基准测试” | 采用工具调用、记忆管理与任务编排的分层架构。 |
| SWE-bench Pro | “抗数据污染” | 包含 1865 个问题与 41 个代码仓库；在 Pro 版本上得分约 23%，而在 Verified 版本上高达 70% 以上（后者被视为数据污染（Contamination）的信号）。 |
| 里程碑达成（Milestone achievement） | “部分得分” | 此类基准测试不仅奖励最终成功，还会对阶段性进展给予评分。 |
| 数据污染（Contamination） | “基准测试数据泄露至训练集” | 基准测试发布后，其数据逐渐混入模型训练语料库，导致测试分数虚高。 |
| WMAC | “AAAI 2026 桥梁项目” | 多智能体协调研讨会（Workshop on Multi-Agent Coordination）；该领域的社区核心交流平台。 |

## 延伸阅读

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 基于里程碑 KPI 的拓扑结构基准测试
- [MARBLE 代码仓库](https://github.com/ulab-uiuc/MARBLE) — 参考实现
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — 领域压力测试；多智能体系统通常不具备压倒性优势
- [AgentArch](https://arxiv.org/abs/2509.10769) — 企业级智能体架构
- [SWE-bench 排行榜](https://www.swebench.com/) — 前沿模型在 Verified 与 Pro 版本上的得分
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — 2026 年社区核心交流平台
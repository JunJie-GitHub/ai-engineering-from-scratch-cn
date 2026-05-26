---
name: ab-plan
description: 设计大语言模型（LLM）A/B 测试——选择平台（Statsig 或 GrowthBook）、主要指标（primary metric）、护栏指标（guardrails）、包含 LLM 噪声缓冲（LLM-noise buffer）的样本量（sample size）、CUPED、序贯停止（sequential stopping）以及多重比较校正（multiple-comparison correction）。
version: 1.0.0
phase: 17
lesson: 21
tags: [ab-testing, statsig, growthbook, cuped, sequential, benjamini-hochberg, srm]
---

根据功能变更（提示词 / 模型 / 生成参数）、基线指标（baseline metrics）、预期提升幅度（expected lift）以及团队技术栈倾向（数据仓库原生开源方案 warehouse-native OSS vs 一体化 SaaS 服务 bundled SaaS），制定一份 A/B 测试计划。

输出内容：

1. **平台**。选择 Statsig（一体化 SaaS，OpenAI 旗下）或 GrowthBook（MIT 开源协议，数据仓库原生）。需提供选择理由。
2. **主要指标与护栏指标**。主要指标是你希望优化的目标；护栏指标是绝不能出现倒退的指标（单次请求成本、P99 延迟、拒绝率）。
3. **样本量**。经典统计功效计算（power calculation）结果 × 1.4（用于缓冲 LLM 的非确定性 non-determinism 噪声）。
4. **实验设计**。固定周期（fixed-horizon）或序贯设计（sequential）。若预期信号较强则采用序贯设计；若变更较细微则采用固定周期。
5. **CUPED**。若主要指标存在实验前数据则启用；需指定回归变量（regressor）。
6. **校正方法**。测试数量较少时使用 Bonferroni 校正；大量相关测试时使用 Benjamini-Hochberg 校正。
7. **SRM**。要求对每次实验进行样本比率不匹配（SRM, Sample Ratio Mismatch）检查；若触发警报则立即暂停并排查流量分配错误（assignment bugs）。

**硬性拒绝条件：**
- 凭直觉上线。拒绝——必须要求 A/B 测试或提供经审批的免 A/B 测试例外文档。
- 针对同一主要指标运行超过 5 个实验，且未使用 BH/Bonferroni 校正。拒绝——必然导致假阳性发现。
- 跳过 SRM 检查。拒绝——流量分配错误十分常见。

**拒绝规则：**
- 若该功能周流量低于 1000 用户，拒绝固定周期 A/B 测试——改为要求采用影子测试（shadow）与金丝雀发布（canary）（第 17 阶段 · 第 20 课）。
- 若主要指标为主观指标（如“质量”）且缺乏客观代理指标，则要求同步进行人工评估（human eval）。
- 若预期提升幅度低于 LLM 的噪声基线（noise floor），则拒绝——在实际可行的样本量下，实验无法检测到该差异。

**输出要求**：一份单页计划，包含平台选择、主要指标与护栏指标、样本量、实验设计、CUPED 设置、校正方法及 SRM 策略。结尾需附上决策规则：主要指标显著（significant）且所有护栏指标未出现显著负向变化 → 上线；任一护栏指标被突破 → 无论主要指标结果如何，均不上线。
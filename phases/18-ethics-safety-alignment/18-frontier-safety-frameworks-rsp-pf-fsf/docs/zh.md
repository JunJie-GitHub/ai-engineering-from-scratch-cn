# 前沿安全框架 (Frontier Safety Frameworks) — RSP、PF、FSF

> 三大头部实验室的框架定义了2026年前沿能力 (frontier capability) 的行业治理标准。Anthropic 负责任扩展政策 (Responsible Scaling Policy, RSP) v3.0（2026年2月）引入了分级AI安全等级 (AI Safety Levels, ASL-1至ASL-5+)，该体系借鉴自生物安全等级，其中ASL-3已于2025年5月针对涉及化学、生物、放射性和核 (CBRN) 的模型激活。OpenAI 准备度框架 (Preparedness Framework, PF) v2（2025年4月）为受追踪能力定义了五项标准，并将能力报告 (Capabilities Reports) 与安全保障报告 (Safeguards Reports) 分离。DeepMind 前沿安全框架 (Frontier Safety Framework, FSF) v3.0（2025年9月）引入了关键能力等级 (Critical Capability Levels, CCL)，其中包括新增的有害操纵关键能力等级 (Harmful Manipulation CCL)。目前这三项框架均包含竞争对手调整条款 (competitor-adjustment clauses)，允许在同行实验室未部署同等安全保障措施的情况下推迟执行。跨实验室的对齐在结构上保持一致，而非术语统一：“能力阈值 (Capability Thresholds)”、“高能力阈值 (High Capability thresholds)”和“关键能力等级 (Critical Capability Levels)”指代的是类似的结构。

**Type:** 学习
**Languages:** 无
**Prerequisites:** 第18阶段 · 17（WMDP），第18阶段 · 07-09（欺骗失效 (deception failures)）
**Time:** 约75分钟

## 学习目标

- 描述 Anthropic 的 ASL 分级结构以及触发 ASL-3 的条件。
- 列出 OpenAI 准备度框架 v2 中用于追踪能力的五项标准。
- 描述 DeepMind 的关键能力等级 (Critical Capability Levels) 结构及有害操纵关键能力等级 (Harmful Manipulation CCL)。
- 解释竞争对手调整条款 (competitor-adjustment clauses) 及其对竞争动态 (race dynamics) 的重要性。
- 定义安全论证 (safety case) 并描述其三支柱结构（监控 (monitoring)、不可读性 (illegibility) 与无能力 (incapability)）。

## 问题背景

第7至17课已证实：欺骗行为是可能的，双重用途能力 (dual-use capability) 确实存在，且评估 (evaluation) 存在局限性。拥有前沿能力模型的实验室需要一套内部治理结构，以：
- 明确何时需要引入新安全保障措施的阈值。
- 定义在扩展模型规模前必须完成的评估。
- 描述安全论证 (safety case) 的具体形态。
- 应对竞争动态问题 (race-dynamic problem)（如果竞争对手在未部署安全保障的情况下发布模型，你该如何应对？）。

这三项2025-2026年的框架代表了当前的行业最高水平——它们虽不完美且仍在演进，但各实验室间的对齐程度已足够高，使得当前的治理核心问题已从“是否存在框架”转变为“现有框架是否足够完善”。

## 核心概念

### Anthropic 负责任扩展政策 v3.0（2026 年 2 月）

AI 安全等级（AI Safety Level, ASL）结构：
- ASL-1：非前沿模型（其风险已涵盖在低于前沿水平的基线标准中）。
- ASL-2：当前前沿基线；部署时采用常规安全防护措施。
- ASL-3：灾难性滥用风险显著升高；具备与化学、生物、放射性和核武器（CBRN）相关的能力。于 2025 年 5 月激活。
- ASL-4：跨越 AI 研发-2（AI R&D-2）阈值；能够自动化初级 AI 研究的模型。
- ASL-5+：高级 AI 研发；能够大幅加速有效扩展（scaling）的模型。

v3.0 新增内容：
- 前沿安全路线图（Frontier Safety Roadmaps）（以脱敏形式公开）。
- 风险报告（Risk Reports）（季度发布，部分经过外部审查）。
- AI 研发（AI R&D）能力被细分为 AI R&D-2 和 AI R&D-4。
- 一旦跨越 AI R&D-4 阈值，则必须提供明确的安全论证（affirmative safety case），以识别模型在追求非对齐目标（misaligned goals）时产生的对齐风险（misalignment risks）。

### OpenAI 准备度框架 v2（2025 年 4 月 15 日）

追踪能力的五项标准：
- **合理（Plausible）。** 存在合理的威胁模型（threat model）。
- **可测量（Measurable）。** 可进行实证评估（empirical evaluation）。
- **严重（Severe）。** 造成的危害巨大。
- **全新（Net-new）。** 并非既有风险的简单放大。
- **瞬时或不可逆（Instantaneous-or-irremediable）。** 危害发生迅速或无法挽回。

满足全部五项标准的能力将被纳入追踪范围，否则不予追踪。

准备度框架 v2（PF v2）的其他结构：
- 将能力报告（Capabilities Reports，模型能做什么）与防护报告（Safeguards Reports，存在哪些控制措施）分开。
- 安全顾问组（Safety Advisory Group）审查。
- 管理层审批；董事会安全与安保委员会（Safety & Security Committee）监督。
- “调整条款（Adjustment clause）”：若其他实验室在未提供同等防护措施的情况下发布模型，OpenAI 可能会降低要求。

### DeepMind 前沿安全框架 v3.0（2025 年 9 月）

按领域划分的关键能力等级（Critical Capability Levels, CCLs）：
- 生物武器能力提升（Bioweapon Uplift）
- 网络攻击能力提升（Cyber Uplift）
- 机器学习研发加速（ML R&D Acceleration）
- 有害操纵（Harmful Manipulation，v3.0 新增）：可能在高风险情境下显著改变人类信念或行为的模型。

v2.0（2025 年 2 月）新增了欺骗性对齐（Deceptive Alignment）章节，并为机器学习研发 CCL 设定了更高的安全等级。

### 跨实验室对齐

- Anthropic 的“能力阈值（Capability Thresholds）”。
- DeepMind 的“关键能力等级（Critical Capability Levels）”。
- OpenAI 的“高能力阈值（High Capability thresholds）”。

目前尚无行业统一术语。但在结构上保持一致：均划分为三个前沿能力层级，并公开了评估标准。自 2025 年起，三者均包含竞争对手调整条款（competitor-adjustment clauses）。

### 安全论证（Safety Cases）

安全论证（safety case）是一份书面论证，旨在证明在最坏情况假设下，模型的部署处于可接受的安全水平。其标准结构围绕三大支柱展开：

- **监控（Monitoring）。** 若发生不良行为，我们能否及时检测？
- **难以解析（Illegibility）。** 模型是否缺乏执行连贯有害计划的能力？
- **无能力（Incapability）。** 模型是否缺乏造成特定危害的能力？

不同的安全论证侧重于不同的支柱。针对 ASL-3 的 CBRN 安全论证，主要目标是“无能力”（通过反学习 unlearning 实现）。针对欺骗性对齐，目标则是“监控”和“难以解析”。针对网络攻击能力提升，三者均相关。

### 竞争动态问题

竞争对手调整条款颇具争议。批评者认为这会引发“逐底竞争（race to the bottom）”：如果三家实验室都会在竞争对手违规时降低要求，那么均衡状态将向违规倾斜。辩护者则认为，若违规实验室的安全意识较弱，替代方案（单边防护）反而会导致更糟的结果。

英国 AISI、美国 CAISI 和欧盟 AI 办公室（第 24 课）是外部治理的对应机构。实验室框架属于自愿性质，而监管框架正在逐步建立。

### 在第 18 阶段中的定位

第 17-18 课是在欺骗性分析和红队测试（red-team analyses）之上构建的测量与治理层。第 19-24 课涵盖福利、偏见、隐私、水印技术及监管结构。第 28 课则梳理了将评估工作落地实施的研究生态系统（MATS、Redwood、Apollo、METR）。

## 实践应用
本课程无需编写代码。请阅读三份核心文献：RSP v3.0、PF v2 和 FSF v3.0。将各实验室的层级结构 (tier structure) 进行相互映射，并找出每个实验室所定义、而其他实验室未定义的一项阈值 (threshold)。

## 交付成果
本课程将生成 `outputs/skill-framework-diff.md` 文件。给定一份安全框架或版本说明，它会将其阈值定义、所需评估 (evaluations) 及安全论证 (safety case) 结构与 RSP v3.0、PF v2 和 FSF v3.0 进行对比，并标出跨实验室差异 (cross-lab gaps)。

## 练习
1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。整理一份表格，列出各实验室的化学、生物、放射与核 (CBRN) 阈值、人工智能研发 (AI R&D) 阈值，以及各自要求的部署前评估 (pre-deployment evaluation)。
2. 竞争对手调整条款 (competitor-adjustment clause) 已包含在上述三大框架中（2025年及以后版本）。请撰写一段支持该条款的论述，再撰写一段反对该条款的论述。并指出每种立场所依赖的核心假设。
3. 为跨越 Anthropic 的 AI R&D-4 阈值的模型设计一份安全论证 (safety case)。列出三大支柱——监控 (monitoring)、不可解读性 (illegibility)、能力缺失 (incapability)——各自所需的证据。
4. DeepMind 的 FSF v3.0 引入了有害操纵关键能力等级 (Harmful Manipulation CCL)。请提出三项实证测量指标，用于表明模型已跨越该阈值。
5. 阅读 METR 发布的《前沿人工智能安全政策的共同要素》（2025年）。指出跨实验室共识 (cross-lab convergences) 最强的三项内容，以及分歧 (divergences) 最大的两项内容。

## 关键术语
| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| RSP | “Anthropic 的框架” | 负责任扩展政策 (Responsible Scaling Policy)；ASL 层级；v3.0（2026年2月） |
| PF | “OpenAI 的框架” | 准备度框架 (Preparedness Framework)；五项标准；v2（2025年4月） |
| FSF | “DeepMind 的框架” | 前沿安全框架 (Frontier Safety Framework)；CCL；v3.0（2025年9月） |
| ASL-3 | “类比生物安全三级” | Anthropic 针对 CBRN 相关能力设定的层级；2025年5月激活 |
| CCL | “关键能力等级” | DeepMind 的阈值构建概念；按领域划分 |
| Safety case | “正式论证” | 书面论证，证明在最坏情况 U 下部署该模型的安全性处于可接受范围内 |
| Adjustment clause | “竞争对手违约豁免” | 框架条款：若竞争对手在未采取同等安全措施的情况下发布模型，则允许降低相关要求 |

## 延伸阅读
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL 层级、路线图、AI 研发细分
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — 五项标准、调整条款
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、有害操纵
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 跨实验室对比
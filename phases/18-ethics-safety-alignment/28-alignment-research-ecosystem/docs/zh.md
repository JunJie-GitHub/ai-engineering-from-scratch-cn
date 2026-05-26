# 对齐研究生态系统 — MATS、Redwood、Apollo、METR

> 五大机构定义了2026年的非实验室对齐研究层（non-lab alignment research layer）。MATS（机器学习对齐与理论学者项目，ML Alignment & Theory Scholars）：自2021年底以来已汇聚527余名研究人员，产出180余篇论文，获得1万余次引用，h指数（h-index）达47；2024年夏季学员批次（cohort）已注册为501(c)(3)非营利组织，包含约90名学者与40名导师；2025年前毕业的校友中80%致力于安全/安保领域，其中200余人任职于Anthropic、DeepMind、OpenAI、英国AI安全研究所（UK AISI）、兰德公司（RAND）、Redwood、METR与Apollo。Redwood Research：由Buck Shlegeris创立的应用对齐实验室；引入了AI控制（AI Control，第10课）；正与UK AISI合作开展控制安全论证（control safety cases）。Apollo Research：为前沿实验室提供部署前策略行为评估（pre-deployment scheming evaluations）；撰写了《上下文内策略行为》（In-Context Scheming，第8课）与《迈向AI策略行为的安全论证》（Towards Safety Cases for AI Scheming）。METR（模型评估与威胁研究，Model Evaluation and Threat Research）：专注于基于任务的能力评估（task-based capability evaluations）与自主任务时间跨度研究（autonomous-task time-horizon studies）；其报告《前沿AI安全政策的共同要素》（Common Elements of Frontier AI Safety Policies）对比了各实验室的框架。Eleos AI Research：开展模型福利部署前评估（model-welfare pre-deployment evaluations，第19课）；完成了对Claude Opus 4的福利评估。

**Type:** 学习
**Languages:** 无
**Prerequisites:** 第18阶段 · 01-27（第18阶段前期课程）
**Time:** 约45分钟

## 学习目标

- 识别非实验室对齐研究生态系统中的五大机构及其核心产出。
- 描述MATS的规模（学者数量、论文数量、h指数）及其作为人才输送管道的作用。
- 阐述Redwood的AI控制（AI Control）议程及其与UK AISI的合作关系。
- 说明METR基于任务的评估（task-based evaluation）方法论。

## 问题背景

前沿实验室（frontier labs，第18课）在内部进行安全评估（safety evaluations）并发布部分结果。实验室外的生态系统则是验证这些评估、首次发现新型失效模式（failure modes）以及培养人才的场所。理解该生态系统有助于判断哪些研究发现受到哪些机构的信任。

## 核心概念

### MATS（机器学习对齐与理论学者项目）

始于2021年末。这是一项研究导师计划；学者将与资深研究员合作10-12周，专注于解决特定的对齐（Alignment）问题。

规模（2026年）：
- 自项目启动以来，已培养527名以上研究人员。
- 发表180余篇论文。
- 获得1万次以上引用。
- H指数（H-index）为47。
- 2024年夏季：90名学者与40名导师参与；已注册为501(c)(3)非营利组织。

职业发展成果：2025年之前的校友中，约80%正从事安全/安保（Safety/Security）相关工作。200余人任职于Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR及Apollo。

### Redwood Research

应用对齐（Applied Alignment）实验室。由Buck Shlegeris创立。提出了AI控制（AI Control）研究议程（第10课）。与UK AISI合作开展控制安全案例（Control Safety Cases）研究。为DeepMind和Anthropic的评估设计提供咨询建议。

代表性论文：Greenblatt、Shlegeris等人，《AI控制》（arXiv:2312.06942，ICML 2024）；《对齐伪装》（Alignment Faking）（Greenblatt、Denison、Wright等人，arXiv:2412.14093，与Anthropic合作）。

研究风格：聚焦具体威胁模型（Threat Models）、最坏情况对手（Worst-case Adversaries），以及可进行压力测试（Stress-tested）的具体协议。

### Apollo Research

为前沿实验室提供部署前的策略行为（Scheming）评估。撰写了《上下文中的策略行为》（In-Context Scheming）（第8课，arXiv:2412.04984）。参与2025年OpenAI反策略行为训练合作项目。发布了《迈向AI策略行为的安全案例》（Towards Safety Cases for AI Scheming，2024）。

研究风格：在智能体环境（Agentic-setting）中进行评估，以观察欺骗（Deception）行为的涌现；采用三支柱分解框架（目标不一致性、目标导向性、情境感知能力）。

### METR（模型评估与威胁研究）

基于任务的能力评估。自主任务完成时间跨度研究。《前沿AI安全政策的共同要素》（metr.org/common-elements，2025）对各实验室框架进行了对比分析。

与Apollo共同撰写了AI策略行为安全案例草案。

研究风格：长周期任务评估、实证能力测量、框架综合。

### Eleos AI Research

模型福利（Model Welfare）部署前评估。执行了系统卡（System Card）第5.3节中记录的Claude Opus 4福利评估。为第19课中与福利相关的声明提供外部方法论核查。

### 运作流程

MATS负责培养研究人员。毕业生进入Anthropic、DeepMind、OpenAI（实验室安全团队）或Redwood、Apollo、METR、Eleos（外部评估机构）。外部评估机构与各大实验室以及UK AISI / CAISI建立合作。相关出版物反哺生态系统，为MATS的下一期学员提供研究基础。

### 该层级的重要性

单一来源的评估并不可靠：实验室自行评估自身模型存在结构性的利益冲突。外部评估机构能够提出并验证实验室可能低报的失效模式（Failure Modes）。2024年的《休眠智能体》（Sleeper Agents）论文（第7课）由Anthropic与Redwood合作完成；《对齐伪装》由Anthropic与Redwood合作完成；《上下文中的策略行为》由Apollo完成；《反策略行为》由Apollo与OpenAI合作完成。多机构协作架构正是质量控制的保障。

### 在第18阶段中的定位

第7-11课引用了Redwood和Apollo的研究成果；第18课引用了METR的框架对比；第19课引用了Eleos。第28课则明确绘制了本阶段其余部分所依赖的生态系统组织架构图。

## 实践应用
无需编写代码。阅读 METR 的《前沿人工智能安全政策的共同要素》（Common Elements of Frontier AI Safety Policies），以此作为外部综合研究如何为实验室内部政策制定工作增添价值的范例。

## 成果交付
本节课程将生成 `outputs/skill-ecosystem-map.md`。给定一项对齐（alignment）主张或评估（evaluation）结果，该流程会识别相关机构、发表渠道及方法论风格，并与已知的同类机构进行交叉核对。

## 练习
1. 从第 7 至 15 课中任选一篇论文，识别其中涉及的机构。将作者名单与 MATS 校友及当前生态系统的隶属关系进行交叉核对。
2. 阅读 METR 的《前沿人工智能安全政策的共同要素》。找出文中强调的三个跨实验室共识以及两个最大的分歧。
3. MATS 学员的职业去向中，约 80% 集中在安全/安保（safety/security）领域。请论证这种选择压力是适应性的（培养该领域人才）还是存在偏见的（过滤掉非主流立场）。
4. Redwood Research 和 Apollo Research 均从事控制（control）/策略性欺骗（scheming）相关研究，但风格迥异。请选择一种失效模式（failure mode），并描述两家机构将如何对其进行调查。
5. Eleos AI 是目前唯一专注于模型福利（model-welfare）的机构。请设计一个假设的第二家机构，专注于其他与福利相关的问题（如认知自由（cognitive liberty）、机器人具身化（robotic embodiment）等），并阐明其方法论。

## 关键术语
| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| MATS | “导师制项目” | 机器学习对齐与理论学者项目（ML Alignment & Theory Scholars）；自 2021 年以来已培养 527 名以上研究人员 |
| Redwood Research | “控制实验室” | 应用对齐（applied alignment）研究；《AI Control》论文作者团队；英国人工智能安全研究所（UK AISI）合作伙伴 |
| Apollo Research | “策略性欺骗评估” | 为前沿实验室提供部署前的策略性欺骗（scheming）评估 |
| METR | “任务跨度评估” | 基于任务的能力评估（task-based capability evaluations）；框架综合（framework synthesis）研究 |
| Eleos AI | “福利实验室” | 模型福利（model-welfare）部署前评估 |
| 人才输送管道 | “MATS -> 实验室” | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR 等机构 |
| 外部评估 | “非实验室检查” | 非由模型开发方自行完成的评估（external evaluation）；可提升结果可信度 |

## 延伸阅读
- [MATS（机器学习对齐与理论学者项目）](https://www.matsprogram.org/) — 导师制项目
- [Redwood Research](https://www.redwoodresearch.org/) — 《AI Control》相关论文
- [Apollo Research](https://www.apolloresearch.ai/) — 策略性欺骗评估
- [METR — 前沿人工智能安全政策的共同要素](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架对比
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论
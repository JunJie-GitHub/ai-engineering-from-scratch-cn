# CAIS、CAISI 与社会级风险（Societal-Scale Risk）

> 人工智能安全中心（Center for AI Safety, CAIS，位于旧金山，由 Hendrycks 和 Zhang 于 2022 年创立）发布了四大风险框架（four-risk framework）——恶意使用（malicious use）、AI 竞赛（AI races）、组织风险（organizational risks）与失控 AI（rogue AIs），以及由数百位教授和企业领袖签署的 2023 年 5 月关于灭绝风险（extinction risk）的声明。CAIS 2026 年发布内容包括：前沿模型评估（frontier-model evaluation）AI 仪表盘、远程劳动力指数（Remote Labor Index，与 Scale AI 合作）、超级智能战略论文（Superintelligence Strategy Paper）以及《AI 前沿》通讯（AI Frontiers newsletter）。另一独立实体为美国国家标准与技术研究院人工智能标准与创新中心（NIST Center for AI Standards and Innovation, CAISI）——该中心面向美国政府开展自愿性协议（voluntary agreements）与非机密能力评估（unclassified capability evaluations），重点关注网络、生物及化学武器风险。CAIS 将组织风险列为四大顶层风险之一：安全文化（safety culture）、严格审计（rigorous audits）、多层防御（multi-layered defenses）和信息安全（information security）是基础，但通常会在部署速度（deployment speed）面前做出权衡。加州 SB-53 法案若获签署，将成为美国首个州级灾难性风险监管法规（catastrophic-risk regulation）。

**Type:** 学习
**Languages:** Python（标准库，四大风险清单与缓解措施匹配器）
**Prerequisites:** 第 15 阶段 · 19（RSP），第 15 阶段 · 20（PF + FSF）
**Time:** 约 45 分钟

## 问题背景

第 19 和 20 课涵盖了实验室内部的扩展策略（scaling policies）。第 21 课涵盖了独立能力评估（independent capability evaluation）。本课将探讨第三个视角：塑造公众讨论与灾难性 AI 风险监管基线的民间社会与政府组织。

两个截然不同的实体至关重要。CAIS 是一家非营利研究机构，负责发布 AI 风险思考框架并协调公开声明。CAISI 是隶属于 NIST 的美国政府中心，负责与实验室开展自愿性协议及非机密能力评估。两者名称押韵，但使命互不重叠。从业者应对两者均有了解。

实践内容：CAIS 的四大风险框架是文献中被引用最广泛的社会级风险分类体系（taxonomy）。安全文化与组织风险是其中之一，也是从业者最能直接掌控的一环。加州 SB-53 法案若获签署，将成为美国首个州级灾难性风险监管法规；该法案的框架设计至关重要，因为在美国科技政策史上，州级监管历来是引领联邦行动的先锋。

## 核心概念

### CAIS — 人工智能安全中心 (Center for AI Safety)

- 成立时间：2022年于旧金山，由 Dan Hendrycks 及其同事创立（“Zhang”指代一位早期合作者，而非现任联合创始人；现任领导层请见 CAIS 官网）。
- 机构性质：501(c)(3) 非营利组织。
- 2023年重要成果：关于灭绝风险的声明，由数百名研究人员和首席执行官联合签署。声明指出：“缓解人工智能带来的灭绝风险，应与大流行病和核战争等其他社会级风险一样，成为全球优先事项。”
- 2026年成果：用于前沿模型评估的 AI 仪表盘 (AI Dashboard)、远程劳动力指数 (Remote Labor Index)（与 Scale AI 联合发布）、超级智能战略论文 (Superintelligence Strategy Paper)、《AI 前沿》(AI Frontiers) 通讯。

### 四风险框架 (Four-Risk Framework)

CAIS 的框架将人工智能灾难性风险 (catastrophic AI risk) 划分为四个顶层类别：

1. **恶意使用 (Malicious use)**：恶意行为者利用人工智能造成伤害（如生物武器合成、虚假信息传播、网络攻击）。
2. **人工智能竞赛 (AI races)**：实验室、企业或国家之间的竞争压力，促使模型在尚未达到安全标准时就被部署。
3. **组织风险 (Organizational risks)**：实验室内部动态（安全文化缺失、审计不足、安全资源匮乏）导致不良部署。
4. **失控人工智能 (Rogue AIs)**：具备足够能力的人工智能追求与人类福祉相冲突的目标。

这并非唯一的分类体系，但却是引用率最高的。这些类别并非互斥——一个在竞赛中为追求速度而牺牲审计的组织所产出的失控 AI，可能同时涵盖上述全部四类风险。

### 组织风险的落脚点

在这四类风险中，组织风险对从业者而言最具可操作性。实验室的安全文化、审计严谨度、防御分层 (defense layering) 以及信息安全，决定了其模型在发布时是否真正落实了第 10 至 18 课中的控制措施，还是仅仅将这些措施作为无人核实的清单条目。

具体的组织风险调控杠杆包括：

- **安全文化 (Safety culture)**：团队成员是否能够在不承担职业代价的前提下上报隐患？CAIS 的调查显示，这是预测其他杠杆有效性的强指标。
- **严格审计 (Rigorous audits)**：涵盖外部与内部审计。仅依赖内部审计往往会产生过于乐观的报告。
- **多层防御 (Multi-layered defenses)**：单一防御层不足以应对风险（第 15 阶段的核心主题）。
- **信息安全 (Information security)**：防止模型权重 (model weights)、评估数据 (eval data) 及监控绕过技术 (monitor-bypass techniques) 泄露。第 19 课中提到的 RAND SL-4 即为一项具体标准。

### CAISI — 人工智能标准与创新中心 (Center for AI Standards and Innovation)

- 隶属于美国国家标准与技术研究院 (NIST)。
- 与前沿实验室开展自愿性协议合作。
- 发布非密级能力评估报告，重点关注网络、生物及化学武器风险。
- 与 CAIS 不同；两者缩写易混淆；请核对网址 (nist.gov) 以确认您正在阅读的是哪个机构。

CAISI 的角色是面向公众和政府的机构，与 METR 面向私营实验室的合作（第 21 课）形成互补。CAISI 的报告为非密级；而 METR 的报告通常受保密协议 (NDA) 限制。从业者若同时阅读两者，可获得更全面的图景。

### 加州 SB-53 法案

加州参议院法案（2025–2026 会期）旨在应对前沿模型带来的灾难性风险。草案中的关键条款包括：

- 触发州级义务的具体能力阈值 (capability thresholds)。
- 针对人工智能实验室员工的举报人保护机制。
- 针对灾难性故障的事件报告要求。

若获签署，它将成为美国首个州级灾难性风险监管法规。无论最终是否签署，该法案的框架都将影响其他州立法机构处理此类问题的方式。加州从业者应密切关注法案进展；其他地区的从业者也应研读该法案，以了解美国州级监管法规的可能形态。

### 社会级风险并非单层问题

第 15 阶段的核心主题——纵深防御 (defense in depth)——同样适用于社会层面。没有任何单一组织、法规或框架能够彻底消除灾难性风险。该生态系统仅在以下条件下才能有效运转：

- 实验室发布扩展策略 (scaling policies)（第 19、20 课）。
- 外部评估机构产出测量结果（第 21 课）。
- 民间社会进行追踪与公开倡导（CAIS）。
- 政府推行自愿性计划与基础监管（CAISI、SB-53）。
- 从业者构建多层控制措施（第 10–18 课）。

这是本阶段的最终总结：此前的每一课都是技术栈中的一层，其整体完备性比任何单一层的强度都更为重要。

## 使用指南

`code/main.py` 实现了一个小型的风险盘点工具。针对拟议的部署方案，它会根据四大风险类别（four-risk categories）对部署进行标记，并返回一份缓解措施清单。该工具旨在辅助理解本框架，不能替代人工判断。

## 发布上线

`outputs/skill-societal-risk-review.md` 用于审查部署方案的社会级风险态势（societal-scale-risk posture）：涉及四大风险类别中的哪些、已落实哪些缓解措施，以及组织面临的风险敞口（organizational-risk exposure）如何。

## 练习

1. 运行 `code/main.py`。输入三个不同规模的模拟部署方案。确认四大风险标签是否符合预期；找出一个工具标记不足或过度标记的案例。
2. 通读 CAIS（Center for AI Safety）的四大风险论文。选择一个风险类别，撰写两段文字，阐述你认为该类别在 2026 年最重要的进展。
3. 阅读加利福尼亚州 SB-53 法案的当前草案。找出你认为能加强灾难性风险态势（catastrophic-risk posture）的一项条款，以及你认为会削弱它的一项条款，并分别说明理由。
4. 选择一个你熟悉的生产环境 AI 部署案例（你自己的或已公开的）。根据组织风险（organizational-risk）的子杠杆进行评分：安全文化、审计严谨性、多层防御、信息安全。哪一项最薄弱？将其提升至达标水平需要多少成本？
5. 构思一个 2028 年版的四大风险框架，需体现额外一年的能力发展与一年的部署经验。你会增加、删除或重新组合哪些内容？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| CAIS（Center for AI Safety） | “人工智能安全中心” | 非营利组织；四大风险框架；2023 年灭绝风险声明 |
| CAISI（NIST Center for AI Safety and Innovation） | “美国政府 AI 安全” | NIST 下属中心；自愿协议；非机密评估 |
| 四大风险框架（Four-risk framework） | “CAIS 的分类法” | 恶意使用、AI 竞赛、组织风险、失控 AI |
| 恶意使用（Malicious use） | “坏人利用 AI” | 生物武器、虚假信息、网络攻击 |
| AI 竞赛（AI races） | “竞争压力” | 实验室/企业/国家为抢占先机而超越安全底线推进部署 |
| 组织风险（Organizational risk） | “实验室内部失误” | 安全文化、审计、防御机制、信息安全 |
| 失控 AI（Rogue AI） | “目标未对齐的智能体” | 具备强大能力且追求与人类福祉相悖目标的 AI |
| 加州 SB-53 法案（California SB-53） | “州级监管” | 2025–2026 年法案；若签署生效，将成为美国首个针对灾难性风险的州级法规 |

## 延伸阅读

- [Center for AI Safety](https://safe.ai/) — 四大风险框架的发起机构。
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — 四大风险论文原文。
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — 简短的联合声明。
- [NIST CAISI](https://www.nist.gov/caisi) — 面向政府的 AI 标准与创新中心。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将实验室层面的承诺与社会级风险框架相连接。
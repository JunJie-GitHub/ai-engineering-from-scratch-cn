# Anthropic 负责任扩展政策（Responsible Scaling Policy）v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代了 2023 年版本的政策。该政策引入了双层缓解机制（Two-tier mitigation）：明确区分了 Anthropic 将单方面采取的行动与被定位为全行业建议的行动（包括 RAND SL-4 安全标准）。将前沿安全路线图（Frontier Safety Roadmaps）和风险报告（Risk Reports）确立为常设文件，而非一次性交付物。取消了 2023 年版本中的暂停承诺（pause commitment）。引入了 AI R&D-4 阈值（AI R&D-4 threshold）：一旦跨越该阈值，Anthropic 必须发布一份正面论证（affirmative case），明确识别模型对齐风险（misalignment risks）及相应的缓解措施。Claude Opus 4.6 并未跨越该阈值。Anthropic 在 v3.0 公告中表示，“越来越难以自信地排除这种可能性。”外部评估机构 SaferAI 对 2023 年 RSP 的评分为 2.2；他们将 v3.0 降级至 1.9，使 Anthropic 与 OpenAI 和 DeepMind 一同被归入“薄弱”RSP 类别。定性阈值（qualitative thresholds）取代了 2023 年的定量承诺（quantitative commitments）；取消暂停条款是最明显的倒退。

**Type:** 学习
**Languages:** Python（标准库，RSP 阈值决策引擎）
**Prerequisites:** 第 15 阶段 · 06（AAR），第 15 阶段 · 07（RSI）
**Time:** 约 45 分钟

## 问题背景

前沿实验室（Frontier labs）发布的扩展政策（scaling policies）兼具技术文档、治理文件以及向监管机构传递信号的多重属性。RSP v3.0 是 Anthropic 当前的现行文件。仔细研读该文件至关重要，并非因为其合规性具有约束力（实际上并无约束力），而是因为其框架设定了实验室如何认知灾难性风险（catastrophic risk），以及他们如何向公众传达技术权衡。

对比 v3.0 与 v2.0 的差异是理解该政策的有效切入点。新增内容包括：前沿安全路线图、风险报告以及 AI R&D-4 阈值。删除内容为：2023 年版本中的暂停承诺。重新框架化的内容为：双层缓解计划，将其划分为 Anthropic 单方面行动与行业建议两部分。外部审查机构 SaferAI 将其评分从 2.2（v2）下调至 1.9（v3.0）。这正是扩展政策如何在表面更加完善的同时，实质严谨性却有所下降的典型体现。

## 核心概念

### 双层缓解计划 (Two-Tier Mitigation Schedule)

- **Anthropic 单方面行动**：无论其他实验室采取何种措施，Anthropic 都将执行的操作。包括超过特定阈值时停止训练、实施特定安全措施以及设置特定的部署关卡。
- **全行业建议**：Anthropic 认为整个行业应共同采取的行动。包含 RAND SL-4 安全标准。这些并非 Anthropic 自身的承诺，而是政策倡导。

v2 版本中并未采用这种双层结构。这意味着读者需要仔细查看每项承诺所属的列。位于“全行业建议”列中的安全措施并非 Anthropic 的承诺，而是其期望。

### AI 研发-4 阈值 (AI R&D-4 Threshold)

这是负责任扩展政策 v3.0 (Responsible Scaling Policy v3.0, RSP v3.0) 所指定的下一个重要能力阈值。具体而言：指能够以具有竞争力的成本自动化完成大量 AI 研究工作的模型。一旦 Anthropic 认为某模型跨越了该阈值，在继续扩展规模之前，他们必须发布一份正面论证 (affirmative case)，明确指出潜在的对齐风险 (misalignment risks) 及相应的缓解措施。

根据 v3.0 公告，Claude Opus 4.6 尚未跨越该阈值。但文件补充指出：“越来越难以自信地排除这种可能性。”这一措辞至关重要；它承认该阈值已足够接近，成为一个现实关切，而非纯粹的推测性界限。

第 6 课（自动化对齐研究 (Automated Alignment Research)）与第 7 课（递归自我改进 (Recursive Self-Improvement)）直接关联至此阈值。自动化对齐研究工具达到研究质量门槛，正是 AI 研发-4 阈值日益临近的证据。

### 前沿安全路线图与风险报告

v3.0 将两类文档提升为常设文件：

- **前沿安全路线图 (Frontier Safety Roadmap)**：前瞻性文档，用于描述计划中的安全工作、能力预期及缓解措施研究。
- **风险报告 (Risk Report)**：模型发布后的回顾性文档，用于描述观测到的能力及残余风险。

两者均对外公开，并按既定周期更新。其实际效用在于：读者可以追踪对比 Anthropic 在路线图中承诺的行动与在风险报告中汇报的实际结果。

### 移除暂停条款 (Pause Clause)

2023 版 RSP 包含明确的暂停承诺：若模型跨越特定能力阈值，训练将暂停，直至缓解措施到位。v3.0 则以更温和的表述（发布正面论证，若缓解措施充分则继续推进）取代了明确的暂停要求。SaferAI 及其他分析师直接指出，这是新版文件中最显著的倒退。

支持此项变更的政策论据是：2023 年设定的定量阈值在 2026 时代的能力基准测试中已无法企及，因为基准测试本身已被重新标定。反对意见则认为：扩展政策中的暂停条款是一种承诺机制 (commitment device)；移除它将削弱该政策的可信度。

### SaferAI 的降级评估

SaferAI 是一家独立机构，专门对 RSP 类文档进行评级。其公开评级显示：2023 版 Anthropic RSP 得分为 2.2（评分体系中 4.0 代表当前最佳 RSP，1.0 为名义/基础水平）。v3.0 得分为 1.9。这使得 Anthropic 的评级从“中等”降至“薄弱”，与 OpenAI 和 DeepMind 同属薄弱类别。

SaferAI 给出的降级因素包括：
- 定性阈值取代了定量阈值。
- 移除了暂停承诺。
- 针对 AI 研发-4 阈值的缓解措施被描述为“正面论证”，而非具体措施。
- 审查机制依赖于 Anthropic 的安全顾问小组，缺乏独立的监督。

### 本课并非什么

这不是一堂合规课程。RSP v3.0 并非法规，没有任何强制力要求 Anthropic 必须遵守。本课的重点在于以应有的精确度和审慎态度来解读该文件。扩展政策是前沿实验室对外传递其灾难性风险立场 (catastrophic-risk posture) 的主要公开信号。对于任何工作依赖于前沿能力的人员而言，准确解读这些政策是一项实用技能。

## 使用方法
`code/main.py` 实现了一个小型决策引擎，其结构映射了负责任扩展政策（Responsible Scaling Policy, RSP）的阈值评估逻辑：输入候选模型及其能力测量数据，输出是否跨越 AI R&D-4 阈值、所需的支持性论证（affirmative-case）章节，以及是否允许部署。该引擎设计得十分精简，旨在将文档中的逻辑显式化。

## 交付发布
`outputs/skill-scaling-policy-review.md` 会对照 v3.0 参考标准（双层架构、阈值、暂停承诺、独立审查）对某项扩展政策（Anthropic、OpenAI、DeepMind 或内部政策）进行审查。

## 练习
1. 运行 `code/main.py`。输入三个处于不同能力水平的合成模型。验证阈值评估器的行为是否符合预期，并生成正确的支持性论证模板。
2. 完整阅读 RSP v3.0（共 32 页）。找出所有属于“行业级建议”层级的承诺。在 v2 版本中，这些承诺里有哪些原本属于“Anthropic 单方面承诺”？
3. 阅读 SaferAI 的 RSP 评分方法论。将其评分标准应用于该文档，复现其对 v3.0 给出的 1.9 分。评分表中的哪一项是导致降分的主要原因？
4. 2023 年的暂停承诺已被移除。请提出一项替代承诺，在承认 2026 年基准重标定（benchmark-rescaling）问题的同时，保持该政策的公信力。
5. 将 RSP v3.0 与 OpenAI 准备度框架（Preparedness Framework）v2（第 20 课）进行对比。找出一个 v3.0 更具优势的领域，再找出一个准备度框架更具优势的领域。

## 关键术语
| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| 负责任扩展政策（Responsible Scaling Policy, RSP） | “Anthropic 的扩展政策” | 负责任扩展政策；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | “研究自动化阈值” | 能够以具有竞争力的成本自动化大量 AI 研究的能力 |
| 支持性论证（Affirmative case） | “安全合理性证明” | 已公开的论证文件，说明已识别相关风险且缓解措施充分 |
| 前沿安全路线图（Frontier Safety Roadmap） | “前瞻性计划” | 记录计划中的安全工作及预期能力的常设文档 |
| 风险报告（Risk Report） | “模型回顾” | 记录发布后观测到的能力及残余风险的常设文档 |
| 双层缓解机制（Two-tier mitigation） | “单方面 vs 行业” | 将 Anthropic 的承诺与行业建议明确区分 |
| 暂停承诺（Pause commitment） | “2023 条款” | 明确承诺暂停模型训练；已在 v3.0 中移除 |
| SaferAI 评级（SaferAI rating） | “独立 RSP 评分” | 第三方评分标准；v3.0 得分为 1.9（v2 为 2.2） |

## 延伸阅读

- [Anthropic — 负责任扩展政策 v3.0 (Responsible Scaling Policy v3.0)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 页政策全文。
- [Anthropic — RSP v3.0 公告](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 相较于 v2 版本的变更摘要。
- [Anthropic — 前沿安全路线图 (Frontier Safety Roadmap)](https://www.anthropic.com/research/frontier-safety) — RSP v3.0 中引用的常设文档。
- [Anthropic — 风险报告：Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 对当前前沿模型（frontier model）的回顾性分析。
- [Anthropic — 实践中衡量智能体自主性 (Measuring agent autonomy in practice)](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与实测自主性（agent autonomy）建立关联。
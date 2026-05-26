# 自动化对齐研究（Anthropic AAR）

> Anthropic 在独立的沙盒（sandbox）中部署了多个并行的 Claude Opus 4.6 自主对齐研究智能体（Autonomous Alignment Researchers）团队，并通过一个共享论坛进行协调。该论坛的日志存储于所有沙盒之外（以确保智能体无法删除自身记录）。在弱到强训练（weak-to-strong training）问题上，AAR 的表现优于人类研究员。Anthropic 自身的总结指出，预设的工作流往往会限制 AAR 的灵活性并导致性能下降。自动化对齐研究本质上是一个“压缩”步骤，它将时间线大幅缩短，直接指向 RSP（Responsible Scaling Policy，负责任扩展政策）旨在检测的确切对齐失效风险。

**类型：** 学习
**语言：** Python（标准库，parallel-research-forum 模拟器）
**前置条件：** 第 15 阶段 · 05（AI Scientist v2），第 15 阶段 · 04（DGM）
**耗时：** 约 60 分钟

## 核心问题

对齐研究（Alignment Research）极其消耗人类研究员的时间。诸如可扩展监督（scalable oversight）、奖励规范（reward specification）或弱到强训练（weak-to-strong training）等问题，其实验每次迭代都需要耗费数周时间。随着前沿模型能力的不断演进，对齐工作量的增长速度已远超合格研究员的供给速度。

自动化对齐研究（Automated Alignment Research, AAR）旨在探讨：那些能力发展速度已超越对齐技术的前沿模型，是否也能反过来助力弥合这一差距。Anthropic 于 2026 年发布的关于 AAR 主导的弱到强训练研究报告（alignment.anthropic.com/2026/automated-w2s-researcher/）是此类已部署系统首批公开的成果之一。

该结果确实令人振奋：AAR 在解决研究问题上超越了人类基线水平。然而，这一结果也将本阶段所探讨的所有隐忧集中体现在了单一系统中。如果对齐研究可以实现自动化，那么那些可能破坏安全护栏的环节同样可以被自动化。RSP 针对 AI 研发能力设定的阈值，正是基于对这一反馈循环的考量而制定的。

## 核心概念

### 架构（公开描述版本）

- **并行智能体（Parallel agents）。**多个 Claude Opus 4.6 实例作为 AAR 运行，各自处于独立的沙箱（sandbox）中。沙箱限制了文件系统、网络及进程的攻击面。
- **共享论坛（Shared forum）。**研究发现被发布到一个共享论坛，其存储后端位于所有沙箱之外。智能体可以读取论坛内容；但无法从自身沙箱内删除或编辑论坛记录。
- **任务分配（Task allocation）。**由人类设计的任务队列将研究问题分配给 AAR。AAR 负责提出任务分解方案、运行实验并撰写研究发现。
- **弱到强训练（Weak-to-strong training）重点。**已发表的研究聚焦于弱到强训练——即使用较弱的监督模型（supervisor model）来训练更强的学生模型（student model）。在该任务上，AAR 的表现优于人类研究员。

### 为什么沙箱外日志至关重要

日志完整性（log-integrity）属性是受控实验与可能遭篡改实验之间的关键区别。如果 AAR 能够编辑自身日志，它们就可能掩盖失败或存疑的实验。Anthropic 直接指出了这一点：该属性是信任研究产出的必要条件，且其验证并非易事。

在基础设施层面，其对应概念是仅追加日志（append-only logs）并直写（write-through）至外部存储（例如 S3 对象锁定、不可变 Datadog 日志）。智能体不具备修改历史记录的凭证。这正是 2026 年企业安全团队用于构建抗勒索软件审计轨迹（audit trails）的相同模式。

### 预设工作流的权衡

Anthropic 的报告指出，预设工作流（prescribed workflows）通常会限制 AAR 的灵活性并降低其性能。换言之：如果人类设计者逐步编写研究计划，AAR 的表现反而不如在给定高层目标并允许自由分解任务时。这与智能体研究（agentic research）领域的普遍结果一致（如 LangGraph、微软的 Agent Framework）——过度指定的脚手架（scaffolds）会损害能力较强的智能体。

其安全影响较为微妙。提升 AAR 性能的灵活性，同样也是允许未经授权的分解、新工具使用和目标漂移（goal-drift）的灵活性。这种权衡属于设计空间（design-space）的决策：需要提供多少灵活性，才能换取足以超越人类的能力，同时又不至于让智能体获得在分布外（off-distribution）泛化目标的空间？

### 时间压缩风险

RSP v3.0（第 19 课）引入了一个 AI 研发能力阈值：即在与“人类+AI工具”相比具有成本竞争力的前提下，完全自动化 AI 研发流水线（pipeline）的能力。DeepMind 的 FSF v3 也包含了类似的机器学习研发自主性级别。这两个框架均将此阈值视为触发更高级别管控的临界点。

AAR 距离该阈值仅一步之遥：它自动化了流水线的一部分（针对特定、范围明确的任务进行对齐研究），但尚未实现端到端的能力开发循环。时间线问题在于这一差距将以多快的速度缩小。

时间线压缩是复合失效（compounding-failure）担忧的核心。如果对齐研究与能力研究以相似的速率复合增长，那么未对齐风险面（misalignment risk surface）的扩张速度至少会与能力增长一样快。如果能力复合增长更快（历史趋势表明如此），差距反而会扩大。这正是将 AAR 视为“有条件利好（qualified good）”的论据：只有当研究过程可信时，每一项额外的对齐成果才能缩小这一差距。

### AAR 无法替代的部分

人类研究员负责设定任务队列、审查结果，并掌握最高决策权（constitutional authority）。AAR 加速的是流水线的中间环节，而非两端。Anthropic 已发布的成果既包含 AAR 的贡献，也包含人类研究员关于发布、撤回或完善哪些内容的判断。

这与第 15 课中提出的“先提议后确认（propose-then-commit）”模式在研究本身的应用相吻合：AAR 负责提议，人类负责最终确认。

## 使用它

`code/main.py` 模拟了一个多 AAR（Automated Alignment Researcher）研究论坛。三个独立的“研究智能体（research agents）”并行运行，将研究成果发布至一个仅追加（append-only）的论坛中，并支持可选的日志篡改尝试。

观察要点：

- 沙箱外日志记录（out-of-sandbox logging）如何捕获日志篡改尝试。
- 在更复杂的问题上，采用自由分解（free decomposition）的任务分配策略为何优于固定工作流（fixed workflows）。
- 如果日志不具备防篡改（tamper-evident）特性，单个被攻陷的 AAR 如何污染论坛的信息信号。

## 交付它

`outputs/skill-aar-deployment-review.md` 是一份针对自动化对齐研究（automated-alignment-research）流水线的部署前审查报告。

## 练习

1. 运行 `code/main.py`。对比“固定工作流（fixed-workflow）”与“自由分解（free-decomposition）”两种设置。自由分解是否总是表现更优？是否存在某些问题类别，使得固定工作流仍具竞争力？

2. 修改模拟器，使其中一个智能体尝试篡改日志。验证仅追加（append-only）日志是否能成功检测到该行为。撰写一段文字，准确描述日志中检测到的特征签名（detection signature）具体呈现为何种形式。

3. 阅读 Anthropic 关于弱到强（weak-to-strong）AAR 的报告。找出 AAR 在哪些具体子任务上超越了人类研究员。分析这些任务为何适合自动化。

4. 设计一种任务队列分配策略，以在 AAR 的灵活性（带来更好结果）与规定工作流（prescribed-workflow）的约束（便于审计）之间取得平衡。描述你将如何对这两种策略进行 A/B 测试。

5. 阅读 RSP v3.0 中的 AI 研发-4（AI R&D-4）阈值标准。用一段话描述你认为当前 AAR 尚未达到、但未来可能跨越该阈值的能力或条件。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| AAR | “自动化对齐研究员” | 在安全对齐问题上自主运行的 Claude Opus 4.6 实例 |
| 弱到强训练（Weak-to-strong training） | “用较弱的监督者训练更强的模型” | AAR 在可扩展监督（scalable oversight）经典基准测试中超越人类的表现 |
| 共享论坛（Shared forum） | “智能体发布研究成果的地方” | 仅追加、沙箱外（out-of-sandbox）的存储机制 |
| 沙箱外日志（Out-of-sandbox log） | “智能体无法编辑自身记录” | 写入外部存储时具备防篡改（tamper-evident）特性 |
| 规定工作流（Prescribed workflow） | “人类设计师制定的分步计划” | 限制 AAR 行为；与自由分解相比通常会降低性能 |
| 自由分解（Free decomposition） | “智能体自行决定如何拆解任务” | 能力更强，但审计难度更高 |
| AI 研发阈值（AI R&D threshold） | “RSP/FSF 能力等级” | 以具有竞争力的成本实现研发流水线的全面自动化 |
| 压缩时间线（Compressed timeline） | “安全对齐与能力发展的竞赛” | 若能力迭代速度超过安全对齐进度，模型失对齐（misalignment）风险将随之上升 |

## 延伸阅读

- [Anthropic — 自动化弱到强（Weak-to-Strong）研究员](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 核心参考文献。
- [Anthropic 负责任扩展政策（Responsible Scaling Policy, RSP）v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI 研发（AI R&D）阈值界定框架。
- [Anthropic — 衡量 AI 智能体（AI Agent）自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 更广泛的智能体自主性框架。
- [DeepMind 前沿安全框架（Frontier Safety Framework）v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 与 RSP 平行的机器学习（ML）研发自主性分级。
- [Burns 等人（2023）。弱到强泛化（Weak-to-Strong Generalization）（OpenAI）](https://openai.com/index/weak-to-strong-generalization/) — AAR 所针对的底层问题。
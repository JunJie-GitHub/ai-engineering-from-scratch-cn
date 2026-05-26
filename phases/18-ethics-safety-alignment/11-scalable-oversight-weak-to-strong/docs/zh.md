# 可扩展监督（Scalable Oversight）与弱到强泛化（Weak-to-Strong Generalization）

> Burns 等人（OpenAI 超级对齐团队，《弱到强泛化》，2023 年）为超级对齐问题（superalignment problem）提出了一种替代方案：使用较弱模型生成的标签来微调（fine-tune）强模型。如果强模型能够从不完善的弱监督中正确泛化，那么当前适用于人类规模的对齐方法或许就能扩展到超人类系统。可扩展监督与弱到强泛化（W2SG）是互补的。可扩展监督（辩论（debate）、递归奖励建模（recursive reward modeling）、任务分解（task decomposition））提升了监督者的有效能力，使其能够跟上被监督模型的水平；而 W2SG 则确保强模型能够从监督者提供的任何不完善监督中正确泛化。《辩论助力弱到强泛化》（arXiv:2501.13124，2025 年 1 月）将这两者结合了起来。

**类型：** 学习
**编程语言：** Python（标准库，W2SG 差距模拟器）
**前置知识：** 第 18 阶段 · 01（指令遵循），第 18 阶段 · 10（AI 控制），第 09 阶段（强化学习基础）
**预计耗时：** 约 60 分钟

## 学习目标

- 定义可扩展监督与弱到强泛化，并解释它们为何互补。
- 描述 Burns 等人 2023 年的实验设置：使用 GPT-2 生成的标签微调 GPT-4。
- 解释性能差距恢复率（Performance Gap Recovered, PGR）指标及其衡量内容。
- 列举三种主要的可扩展监督机制（辩论、递归奖励建模、任务分解），并说明各自的一项优势。

## 核心问题

第 18 阶段迄今为止的每一种对齐技术都假设监督者能够评估模型的行为。当模型达到超人类水平时，监督者就成了薄弱环节。超级对齐的核心问题在于：较弱的监督者能否可靠地训练出更强大且对齐的模型？

Burns 等人将这一问题转化为一个可操作的实证设置：用弱模型监督强模型，并测量强模型的能力在弱监督下能保留多少。这并非超级对齐的最终解决方案，而是一个用于衡量进展的代理指标。

## 核心概念

### W2SG（弱到强泛化，Weak-to-Strong Generalization）：Burns 等人的实验设置

- 弱模型：GPT-2 级别。
- 强模型：GPT-4 级别。
- 目标：该任务上强模型（GPT-4 级别）的性能上限。

流程：
1. 获取弱模型在特定任务上的零样本（zero-shot）预测结果。
2. 使用弱模型标注的数据对强模型进行微调（fine-tune）。
3. 测量微调后强模型的准确率。

基线（baselines）：弱模型自身的准确率，以及强模型在真实标签（gold labels）监督下的性能上限。

差距度量指标：性能差距恢复率（Performance Gap Recovered, PGR）=（微调后性能 - 弱模型性能）/（上限性能 - 弱模型性能）。PGR 为 1.0 表示弱监督完全弥补了性能差距；PGR 为 0 表示弱监督未提供任何帮助。

### Burns 等人的实证发现

在自然语言处理（Natural Language Processing, NLP）、国际象棋谜题和奖励建模（reward modeling）任务中，PGR 始终为正值：强模型能够泛化并超越弱监督者的错误。不同任务的 PGR 值在约 20% 到 80% 之间波动。强模型能够从不完美的标签中“理解”预期任务，并利用其预训练先验（pre-trained priors）进行补充。

Burns 等人指出的局限性：
- 此处的弱到强差距在于能力（capability），而非对齐（alignment）。弱模型已实现对齐；一个真正超越人类的模型可能由人类进行对齐，但两者之间的能力差距会大得多。
- 强模型的泛化可能更多依赖于任务先验，而非恢复真实意图（ground-truth intent）。
- 真正的问题在于如何引出强模型的潜在知识（latent knowledge）；PGR 衡量的只是一种特定的操作化实现。

### 可扩展监督（Scalable oversight）：三种机制

- **辩论（Debate, Irving 等人, 2018）**。两个 U 实例分别持对立观点进行辩论；由弱裁判做出裁决。其假设是：找到一个令人信服的真实答案，比找到一个令人信服的虚假答案更容易。2024-2025 年的近期研究（Khan 等人、Michael 等人）表明，辩论有时有效有时无效，其效果取决于任务结构。
- **递归奖励建模（Recursive Reward Modeling, Leike 等人, 2018）**。U 协助人类为 U+1 训练奖励模型。监督者的有效能力随模型能力的提升而增长。
- **任务分解（Task Decomposition, Christiano, Shlegeris, Amodei, 2018）**。将困难任务递归地分解为人类可验证的子任务。该机制假设任务具备可分解性。

每种机制都对任务结构或中间组件的对齐状态做出了特定假设。

### 为什么可扩展监督与 W2SG 具有互补性

可扩展监督提升了监督者提供的有效信号质量。
W2SG 则弥补了监督者所能提供的任何不完美信号所带来的差距。

Lang 等人的研究《Debate Helps Weak-to-Strong Generalization》（arXiv:2501.13124）将两者结合：通过辩论协议生成更优质的弱标签，并在此基础上训练强模型。该研究在 NLP 任务上报告了 PGR 的提升。

### 组织架构变动

在 Jan Leike 离职加入 Anthropic 后，OpenAI 的超级对齐（Superalignment）团队于 2024 年 5 月解散。相关研究议程（可扩展监督、W2SG、自动化对齐研究）在 Anthropic 以及多家学术实验室得以延续——包括 MATS（第 28 课）、Redwood（第 10 课）、Apollo（第 8 课）和 METR（第 28 课）。组织架构虽已改变，但核心研究问题并未改变。

### 该内容在第 18 阶段（Phase 18）中的定位

第 6-10 课在假设 U 不可信的前提下，描述了相关威胁与防御范式。第 11 课转向进攻范式：使监督者足够强大，以验证 U 的对齐状态。随后的第 12-16 课则聚焦于对抗性评估（adversarial evaluation）的实用工具链。

## 实践应用
`code/main.py` 在一个合成任务上模拟了弱监督强泛化（Weak-to-Strong Generalization, W2SG）的微调过程。弱标注器（weak labeler）的准确率为 70%，且存在结构化错误（structured errors）；强模型（strong model）在黄金标签（gold labels）上的性能上限为 95%。你需要使用弱标签对强模型进行微调，测量性能差距恢复率（Performance Gap Recovered, PGR），并将其与“强模型使用黄金标签”和“仅使用弱模型”的结果进行对比。

## 交付产出
本课时将生成 `outputs/skill-w2sg-pgr.md` 文件。给定一个监督设置（oversight setup）描述，它会识别出弱监督者（weak supervisor）、强模型、监督质量（supervision quality），并计算（或请求）PGR。它会标记该主张属于“弱模型可以监督强模型”还是“弱模型 + 监督机制（oversight mechanism）可以监督强模型”。

## 练习
1. 运行 `code/main.py`。报告 `weak_accuracy` 为 0.60、0.70、0.80 时的 PGR。解释 PGR 曲线的形状。
2. 修改弱标注器，使其产生结构化错误（例如，在特定输入类别上始终出错）。PGR 会增加、减少还是保持不变？请解释原因。
3. 阅读 Burns 等人 2023 年论文的第 4.3 节（NLP 任务）。复现“置信度辅助损失（confidence auxiliary loss）”的直觉：当强模型的置信度高于弱标签时，哪一方占优？
4. 为软件工程任务设计一种结合辩论（debate）与任务分解（task decomposition）的可扩展监督（scalable oversight）协议。指出每个组件的一种失效模式（failure mode），并解释该组合如何缓解或未能缓解这些失效模式。
5. 阐明何种情况能够证伪“弱监督强泛化是通向超级对齐（superalignment）的可行路径”这一主张。请具体说明你需要观察到的实证特征（empirical signature）。

## 核心术语
| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 可扩展监督（Scalable oversight） | “让监督者变得更强” | 提升监督者评估能力更强模型之能力的机制 |
| 弱监督强泛化（W2SG） | “弱监督强” | 使用弱标签微调强模型，并测量其恢复的能力 |
| 性能差距恢复率（PGR） | “恢复的性能差距” | (微调后性能 - 弱模型性能) / (性能上限 - 弱模型性能)；1.0 表示差距完全弥补，0 表示毫无帮助 |
| 辩论（Debate） | “两个 U 实例进行辩论” | 一种可扩展监督机制，由弱裁判在两个 U 辩护者之间做出裁决 |
| 递归奖励建模（RRM） | “递归奖励建模” | U 协助训练 U+1 的奖励模型；监督者能力与 U 保持同步 |
| 任务分解（Task decomposition） | “人类检查的子任务” | 将困难任务递归拆分为人类可验证的子任务 |
| 超级对齐（Superalignment） | “对齐超人类 AI” | 致力于对齐人类无法直接评估的模型的研究议程 |

## 延伸阅读

- [Burns 等人 — 弱到强泛化（Weak-to-Strong Generalization）(OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — W2SG 论文
- [Irving, Christiano, Amodei — 基于辩论的 AI 安全（AI Safety via Debate）(arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — 辩论机制（Debate Mechanism）
- [Leike 等人 — 通过奖励建模（Reward Modeling）实现可扩展的智能体对齐（Agent Alignment）(arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — 递归奖励建模（Recursive Reward Modeling）
- [Khan 等人 — 与更具说服力的大语言模型（LLM）辩论能得出更真实的答案（Debating with More Persuasive LLMs Leads to More Truthful Answers）(arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024 年关于更强辩手参与辩论的实证研究
- [Lang 等人 — 辩论有助于弱到强泛化（Debate Helps Weak-to-Strong Generalization）(arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 年辩论与 W2SG 的结合
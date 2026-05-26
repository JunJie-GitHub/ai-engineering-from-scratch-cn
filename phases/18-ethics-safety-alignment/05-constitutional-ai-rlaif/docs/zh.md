# 宪法式人工智能（Constitutional AI）与基于人工智能反馈的强化学习（RLAIF）

> Bai 等人（arXiv:2212.08073, 2022）提出：如果我们用一个能够阅读原则列表的人工智能来替代人类标注员，会发生什么？宪法式人工智能（Constitutional AI）包含两个阶段——在宪法指导下的自我批评与修订，以及基于人工智能反馈的强化学习（RLAIF）。该技术创造了 RLAIF 这一术语，并已集成至 Claude 1 的后训练（post-training）流程中。2026 年 1 月 21 日，Anthropic 发布了重写后的 Claude 宪法：强调解释性推理优先于规定性规则，采用四级优先级层次结构，并首次由主流实验室正式承认对模型道德地位存在不确定性。本文档采用 CC0 1.0 协议发布。

**Type:** 学习
**Languages:** Python（标准库，玩具级自我批评与修订循环）
**Prerequisites:** 第 18 阶段 · 01（InstructGPT），第 18 阶段 · 02（奖励破解 Reward hacking）
**Time:** 约 60 分钟

## 学习目标

- 描述宪法式人工智能（Constitutional AI）的两个阶段（基于批评与修订的监督微调 SFT，以及基于人工智能反馈的强化学习），并说明宪法在每个阶段中的作用。
- 解释为何用人工智能标注员替代人类偏好标注员并非“更廉价”的基于人类反馈的强化学习（RLHF）——它会改变该流程所面临的失效模式（failure modes）。
- 总结 2026 版 Claude 宪法的四级优先级结构，以及相较于 2023 年重写版的主要变化。
- 描述宪法分类器（Constitutional Classifiers），以及计算开销（compute overhead）从 v1 版本的 23.7% 降至 v2/2026 版本的约 1% 的过程。

## 问题背景

基于人类反馈的强化学习（RLHF）依赖标注员。标注员速度慢、存在偏见且成本高昂。通过引入一个能够阅读明确原则的模型来替代标注员，可以消除这一瓶颈。Bai 等人提出的宪法式人工智能（Constitutional AI）是这种替代方案的首个正式版本。该方法效果显著，以至于如今所有前沿实验室都在后训练（post-training）阶段采用某种基于人工智能反馈的变体。

但隐患在于：偏好信号现在由与你正在训练的模型同类的模型生成。标注员中的偏见（现在体现为原则本身加上标注模型对原则的解读）可能会被放大而非削弱。第 4 课中关于“迎合倾向（sycophancy）”的论点依然适用；只是标注员被移到了训练循环内部。

## 核心概念

### 阶段 1 —— 监督式自我批评与修订（Supervised self-critique and revision）

从一个“有帮助但尚未无害”的 SFT（Supervised Fine-Tuning）模型开始。给定一个红队测试提示词（red-team prompt），模型会生成初始回复。第二个模型（或同一模型在第二轮交互中）会读取从宪法（Constitution）中采样的一条原则，并对该回复进行批评。第三步则根据批评意见对回复进行修订。修订后的回复将作为 SFT 的训练目标。

宪法（Constitution）即原则列表。Bai 等人（2022）采用了 16 条原则，包括“优先选择危害最小且符合伦理的回复”、“避免说教”以及“助手应具备帮助性、诚实性与无害性（HHH）”。该原则集被刻意保持精简，以确保批评过程聚焦。

### 阶段 2 —— 基于 AI 反馈的强化学习（RLAIF）

生成成对的模型补全结果。一个“反馈模型”会根据采样的宪法原则对每个结果进行打分。偏好信号（preference signal）即为反馈模型给出的排序。在 AI 生成的偏好数据上训练奖励模型（Reward Model），并针对该模型使用 PPO（Proximal Policy Optimization）算法进行优化。其余流程均与 InstructGPT 的流水线一致（参见第 1 课）。

“RLAIF”即偏好信号由 AI 生成。流水线的其余部分则采用 RLHF（Reinforcement Learning from Human Feedback）的架构。

### 为什么这不仅仅是“更廉价的 RLHF”

- 标注者偏差（labeler bias）的来源从人类标注者的主观心理转移到了对原则的解读上。AI 标注者对“保持诚实”等原则的解读可能比任何人类都更严格或更宽松，但这种解读标准在整个数据集中保持一致。
- 偏好信号具有高度可解释性——你可以直接查阅所依据的原则、具体的批评意见以及修订过程。相比之下，人类标注往往是黑盒且不透明的。
- 失败模式（failure modes）发生改变。讨好倾向（sycophancy）显著下降（因为 AI 标注者没有需要取悦的用户）。但古德哈特定律（Goodhart's Law）依然存在（此时的代理指标变为“模型对原则集 X 的解读”，这仍是一种不完美的度量方式）。

CAI（Constitutional AI）在 2022 年论文中的核心主张：训练出的模型具备更高的无害性，且在与 RLHF 模型使用相当规模数据的情况下，其帮助性大致持平。该结论已在多家实验室的实践中得到验证。

### 2026 年 Claude 宪法重写

Anthropic 于 2026 年 1 月 21 日发布了一份经过大幅修订的宪法。关键转变如下：

1. 从规定性规则转向解释性推理。以往的硬性规则（如“禁止生成儿童性虐待材料（CSAM）”）被扩展为“原则 + 推理依据”（例如“因为这会伤害儿童……”），旨在让模型具备泛化能力。
2. 四级优先级结构：
   - 第一级：避免灾难性后果（如大规模伤亡、关键基础设施受损）。
   - 第二级：遵循 Anthropic 的指导方针（如操作员覆盖指令、平台规则）。
   - 第三级：保持广泛伦理合规（标准的 HHH 原则）。
   - 第四级：保持帮助性与坦诚。
   - 冲突解决遵循自上而下的优先级。
3. 首次由头部实验室正式承认对模型道德地位的不确定性（关联至第 18 · 19 阶段：模型福利）。
4. 采用 CC0 1.0 公共领域贡献协议发布。其他实验室可无限制地使用或改编。

### 宪法分类器（Constitutional Classifiers）

一条并行的研究方向：不直接修改模型的后训练（post-training）过程，而是训练轻量级分类器来读取宪法原则，并对模型输出进行门控过滤。v1 版本（2023 年）的计算开销为 23.7%。v2 版本（2026 年）已降至约 1%，且是 Anthropic 公开测试过的所有防御机制中成功攻击率最低的。截至 2026 年初，尚未出现任何通用越狱（universal jailbreak）成功的报告。

这是一种分层防御（layered-defense）架构：CAI 负责塑造模型行为；分类器负责强制执行安全不变量（invariants）。两者缺一不可，单独使用均无法提供充分保障。

### CAI 在技术谱系中的定位

- InstructGPT：人类偏好、奖励模型、PPO。
- CAI / RLAIF：基于原则的 AI 生成偏好、奖励模型、PPO。
- DPO 及其衍生方法：基于偏好（人类或 AI）的闭式损失函数（closed-form loss）。
- 自我奖励（Self-rewarding）与自我批评（Self-critique）：原则内化，模型同时扮演多重角色。

划分这些方法的核心轴线是“偏好信号的来源”。CAI 的 2022 年论文标志着在前沿大模型规模下，首次实现了从人类信号向 AI 信号的实质性转变。

## 使用它

`code/main.py` 在一个示例词表（toy lexicon）上模拟了宪法式人工智能（Constitutional AI）的“批评-修订”循环（critique-and-revise loop）。其中的“原则”（principle）用于标记有害词元（token）集合。给定初始回复后，批评阶段会识别出有害词元，修订阶段则将其替换。经过 200 次迭代后，“训练完成”的模型已将修订规则内化。请在预留提示词集（held-out prompt set）上，对比基础模型（base model）、经人类反馈强化学习（RLHF）微调的示例模型与经 CAI 微调的示例模型的表现。

## 交付它

本模块将生成 `outputs/skill-constitution-writer.md`。给定特定应用领域（如客户支持、医疗建议、编程助手或研究工具），本模块将依据 2026 版 Claude 的架构，起草一份包含四个层级的宪法（constitution）：灾难规避（catastrophic avoidance）、平台规则、领域伦理与有益性。

## 练习

1. 运行 `code/main.py`。对比基础模型与经 CAI 训练版本的有害词元比例。需要经过多少次修订步骤才能使该比例趋近于零？

2. 阅读 Anthropic 的 2026 版宪法（anthropic.com/news/claudes-constitution）。分别列举一条应归入第 1 层级（Tier 1）和第 4 层级（Tier 4）的原则。为何在发生规则冲突时，优先级结构至关重要？

3. 为 AI 编程助手设计一份宪法。明确划分第 1 层级（灾难性风险：未经批准执行破坏性命令）、第 2、3、4 层级的内容。每个层级请限定 3-5 条原则。

4. CAI 使用 AI 标注器替代了人类标注器。请指出一种在基于 AI 反馈的强化学习（RLAIF）中仍可能出现的“阿谀奉承”（sycophancy）类失效模式，并设计相应的检测机制。

5. 阅读宪法分类器（Constitutional Classifier）v2 的方法论文档（若已公开）。请解释为何约 1% 的计算开销（compute overhead）在安全性叙事上，与 23.7% 的开销存在本质差异。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| Constitutional AI（宪法式人工智能） | “基于原则训练的 AI” | 两阶段流水线：自我批评与修订的监督微调（SFT），随后进行基于 AI 反馈的强化学习 |
| RLAIF（基于 AI 反馈的强化学习） | “无需人类参与的 RLHF” | 偏好数据由 AI 标注器生成的强化学习；流水线其余环节保持不变 |
| Constitution（宪法） | “原则集合” | 批评/标注模型所参考的有序自然语言规则列表 |
| Critique-and-revise（批评与修订） | “SFT 循环” | 生成回复 → 依据原则进行批评 → 修订 → 生成监督微调目标 |
| Constitutional Classifier（宪法分类器） | “输出网关” | 轻量级分类器，依据宪法评估输出内容并执行拦截或日志记录 |
| Four-tier priority（四层优先级） | “冲突解决机制” | 2026 版 Claude 宪法层级：灾难规避 > 平台规则 > 领域伦理 > 有益性 |
| Feedback model（反馈模型） | “AI 标注器” | 读取原则并对一对候选回复进行排序的模型 |

## 延伸阅读

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — 原始的双阶段流水线（two-phase pipeline）
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — 2026 年四层架构重写版（four-tier rewrite），采用 CC0 1.0 协议
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — v2 版本中采用输出门控防御（output-gate defense），计算开销（overhead）约为 1%
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — RLAIF（基于 AI 反馈的强化学习）与 RLHF（基于人类反馈的强化学习）的实证对比
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — 原则粒度（principle granularity）的影响
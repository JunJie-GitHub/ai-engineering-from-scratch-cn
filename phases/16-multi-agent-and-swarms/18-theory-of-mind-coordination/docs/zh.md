# 心智理论 (Theory of Mind) 与涌现式协调 (Emergent Coordination)

> Li 等人 (arXiv:2310.10701) 的研究表明，在合作文本游戏中，大语言模型智能体 (LLM agents) 会展现出**涌现的高阶心智理论 (Theory of Mind, ToM)**——即推理另一个智能体对第三个智能体信念的认知——但由于上下文管理 (context management) 与幻觉 (hallucination) 问题，它们在长程规划 (long-horizon planning) 上表现不佳。Riedl (arXiv:2510.05174) 对群体中的高阶协同 (higher-order synergy) 进行了测量，发现**只有**在引入 ToM 提示词的条件时，才会产生身份关联差异化 (identity-linked differentiation) 与目标导向互补性 (goal-directed complementarity)；能力较低的 LLM 仅表现出虚假涌现 (spurious emergence)。换言之，协调性的涌现是提示词条件依赖 (prompt-conditional) 且模型依赖 (model-dependent) 的，并非凭空产生。本课程将实现一个最小化的心智理论感知智能体 (ToM-aware agent)，在有无 ToM 提示词的情况下运行合作任务，并依据 Riedl 2025 协议测量协调性差异 (coordination delta)。

**Type:** 学习 + 构建
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** 约 75 分钟

## 问题

多智能体协调 (Multi-agent coordination) 往往看起来如同魔法：智能体分工明确、相互预判、避免冗余。通常，这种“涌现”只是提示词工程 (prompt engineering) 的产物——有人明确指示智能体去“协调”。一旦移除提示词，协调性也随之消失。

Riedl 2025 年的研究结论更为严格：在受控条件下，仅当提示词引导智能体对**其他智能体的心智**进行推理时，协调性才会真正涌现。若缺少 ToM 提示词，即使是强大的模型所展现出的协调模式也无法通过统计控制 (statistical controls) 检验。这对实际生产环境至关重要：许多团队交付的“多智能体协调”功能实际上高度依赖提示词且十分脆弱。

本课程将 ToM 视为一项具体能力（即对“信念的信念”进行推理），构建一个最小化的 ToM 感知智能体，并对比测量真正的协调表现与仅靠提示词包装所呈现的表象之间的差异。

## 概念

### 心智理论（Theory of Mind, ToM）的含义

发展心理学指出：3岁儿童认为他人的内心世界与自己一致；5岁儿童能理解他人拥有不同的信念；7岁儿童则能进行“关于信念的信念”推理（例如“她以为我认为球在杯子下面”）。这些分别对应零阶、一阶和二阶 ToM。

对于大语言模型（Large Language Model, LLM）智能体（Agent），ToM 的阶数对应如下：

- **零阶（Zeroth-order）：** 无他人模型。智能体仅基于自身观察采取行动。
- **一阶（First-order）：** 智能体为其他每个智能体建立信念模型。“Alice 相信 X。”
- **二阶（Second-order）：** 智能体对递归信念进行建模。“Alice 认为 Bob 相信 X。”

Li 等人（2023）的研究发现，在合作游戏中，LLM 智能体会涌现出一阶和二阶 ToM 能力，但随着交互轮次（horizon）变长或通信不可靠，该能力会显著退化。

### 萨莉-安妮测试（Sally-Anne Test）简介

这是一项1985年提出的错误信念（false-belief）测试：Sally 将弹珠放入篮子 A 后离开，Anne 随后将弹珠移至篮子 B。当 Sally 返回时，她会去哪里寻找？具备一阶 ToM 的儿童会回答篮子 A（因为 Sally 的信念与现实不符）；不具备该能力的儿童则会回答篮子 B。

当问题表述直接时，GPT-4 时代的 LLM 能够通过此类萨莉-安妮式测试。但在叙事冗长、场景多次切换或问题表述间接时，它们往往会失败。这反映了截至 2026 年，ToM 在实际生产环境 LLM 中的真实水平。

### Riedl 的协同度量方法

Riedl（arXiv:2510.05174）构建了一项群体规模的测试：包含 N 个智能体、一个合作目标以及可变的提示词（prompt）条件。度量指标包括：

1. **身份关联差异化（Identity-linked differentiation）：** 智能体随时间推移是否会形成稳定的角色区分？
2. **目标导向互补性（Goal-directed complementarity）：** 智能体的行动是相互补充（处理不同子任务）还是相互重复？
3. **高阶协同效应（Higher-order synergy）：** 一种统计度量，用于评估群体是否实现了任何子集都无法单独达成的目标。

结果表明：仅在包含 ToM 提示词的条件下，三项指标才会产生高于基线的有效信号。若不使用 ToM 提示词，中等容量模型的指标表现仅在随机水平附近徘徊。大型模型即使没有显式的 ToM 提示词也能表现出一定的协同性，但其效果仍弱于显式提示。

### 协同幻觉

若缺乏统计控制，演示中所谓的“涌现协同（emergent coordination）”往往反映的是：

- 通过提示词工程（Prompt engineering）硬编码的协同（例如系统提示词中直接写明“请协作”）。
- 观察者偏差（我们倾向于看到自己预期的模式）。
- 事后筛选（仅挑选成功的运行案例进行展示）。

对于缺乏可度量信号却大肆宣传“涌现协同”的生产系统，应将其视为营销话术。在宣称之前，务必先进行量化测量。

### 最小化 ToM 感知智能体架构

结构如下：

agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions

`other_models` 属性即为 ToM 状态。一阶 ToM 仅维护单层模型；二阶 ToM 则增加 `other_models[i][other_models_of_j]` 层级——即“我认为智能体 i 认为智能体 j 所相信的内容”。

### 长程交互为何会削弱 ToM 能力

Li 等人的研究指出：上下文限制（context limits）会导致智能体遗忘特定信念的归属对象；而幻觉（Hallucination）则会在他人模型中引入错误信念。两者共同导致“我以为他以为 X”这类错误，并随时间推移不断累积放大。

该论文及 2024-2026 年的后续研究记录了以下缓解策略：

- **在提示词中显式声明 ToM 状态：** 采用结构化格式 `{agent_id: belief_list}`。强制检索过程保留身份与信念的绑定关系。
- **缩短推理链：** 减少每轮交互中的 ToM 更新次数，以降低幻觉的累积效应。
- **外部 ToM 存储：** 将模型维护在 LLM 上下文之外，每轮仅注入相关部分。

### ToM 在生产环境中的失效场景

- **对抗性环境：** 具备良好 ToM 能力的智能体反而更容易被操纵（攻击者可对其“关于你的模型”进行建模，进而加以利用）。
- **异构团队：** 当智能体底层模型不同时，针对某一对手有效的 ToM 模型往往无法泛化。
- **强依赖客观事实（Ground-truth）的任务：** ToM 关注的是信念；若任务正确性取决于客观事实，过度依赖 ToM 反而可能成为干扰。

### 真正可量化的协同信号

判断团队协同是真实存在而非提示词包装的三个实用信号：

1. **随时间演进的互补性：** 在多轮任务中，智能体的行动是否覆盖了互不重叠的子任务？
2. **预判能力：** 智能体 A 在 T+1 轮的行动，是否依赖于对 B 在 T+2 轮行动的预测，且该预测最终被证实是正确的？
3. **纠错机制：** 当 A 在 T 轮误判 B 的信念时，A 是否能在 T+2 轮前完成修正？

这些指标均可在具备日志记录的多智能体系统中进行量化测量。它们构成了“协同”叙事背后的实质内容。

## 构建它

`code/main.py` 实现了：

- `ToMAgent` —— 跟踪自身信念以及针对其他智能体的信念模型。
- 一项协作任务：三个智能体必须从三个盒子中收集三个标记物 (token)；每个盒子只能容纳一个标记物。智能体之间无法通信；它们通过观察彼此的行为来推断意图。
- 两种配置：`zeroth_order`（无心智理论 (Theory of Mind)）和 `first_order`（带有一级信念模型的心智理论）。
- 在 200 次随机试验中进行测量：完成率、重复率（两个智能体目标指向同一个盒子）、完成所需的平均回合数。

运行：

python3 code/main.py

预期输出：零阶 (zeroth-order) 智能体的重复努力率约为 35%，在 10 个回合内完成约 60% 的试验。一阶 (first-order) 心智理论智能体的重复率约为 5%，完成率约为 95%。这一差值即为可测量的协调效应 (coordination effect)。

## 使用它

`outputs/skill-tom-auditor.md` 是一项用于审计多智能体系统 (multi-agent system) “涌现协调 (emergent coordination)”声明的技能。它会检查是否存在提示词包装 (prompt dressing)、与对照组相比的统计显著性 (statistical significance)，以及可测量的互补性 (complementarity)。

## 交付它

协调声明检查清单：

- **对照组条件 (Control condition)。** 不包含协调提示词的系统版本。对两者均进行测量。
- **统计检验 (Statistical test)。** 在你的评估指标上，系统与对照组之间的差异在 `p < 0.05` 水平上是否显著？
- **互补性度量 (Complementarity measure)。** 关注随时间推移的动作互斥性 (action-disjointness)，而不仅仅是最终的成功结果。
- **失败案例日志 (Failure-case log)。** 当智能体协调失败时，心智理论状态呈现何种形态？
- **模型容量披露 (Model-capacity disclosure)。** 如果该效应在较小规模的模型上消失，请如实说明。

## 练习

1. 运行 `code/main.py`。验证一阶心智理论是否将重复率降低了约 7 倍。当扩展到 5 个智能体和 5 个盒子时，该差距是否依然存在？
2. 实现二阶心智理论（智能体 A 对“B 如何看待 C”进行建模）。它是否优于一阶心智理论？在哪些任务上表现更好？
3. 向心智理论状态中注入**幻觉 (hallucination)**：每回合随机翻转一个信念。这会使一阶性能下降多少？
4. 阅读 Li 等人（arXiv:2310.10701）的论文。复现“长程性能衰减 (long-horizon degradation)”的发现：当回合数从 10 增加到 30 时，你的一阶心智理论性能如何变化？
5. 阅读 Riedl 2025（arXiv:2510.05174）。在你的模拟日志上实现高阶协同统计量 (higher-order synergy statistic)。在不使用心智理论提示词条件的情况下，该效应是否依然存在？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 心智理论 (Theory of Mind) | “理解他人的心智” | 对另一个智能体 (agent) 的信念进行建模的能力。按阶数分级（0阶、1阶、2阶及以上）。 |
| 萨莉-安妮测试 (Sally-Anne Test) | “错误信念测试” | 1985年发展心理学实验；大语言模型 (LLMs) 能通过基础版本，但在复杂版本中会失败。 |
| 一阶心智理论 (First-order ToM) | “A相信X” | 对另一个个体关于事实的信念进行建模。 |
| 二阶心智理论 (Second-order ToM) | “A相信B相信X” | 更深一层的递归建模。 |
| 身份关联差异化 (Identity-linked Differentiation) | “随时间保持角色稳定” | Riedl 提出的指标：角色具有持续性，而非随机分配。 |
| 目标导向互补性 (Goal-directed Complementarity) | “互不重叠的行动” | 智能体针对不同的子任务，而非争夺同一任务。 |
| 高阶协同 (Higher-order Synergy) | “整体表现优于任意子集” | Riedl 提出的用于衡量真实协同效应的统计指标。 |
| 协同错觉 (Coordination Illusion) | “看起来像是协同” | 仅通过提示词 (prompt) 包装出的协同表象，缺乏可测量的实际信号。 |

## 延伸阅读

- [Li 等人 — 基于大语言模型的多智能体协作心智理论](https://arxiv.org/abs/2310.10701) — 合作博弈中涌现的心智理论 (ToM)；长周期任务中的失效模式
- [Riedl — 多智能体语言模型中的涌现协同](https://arxiv.org/abs/2510.05174) — 群体规模测量；心智理论 (ToM) 提示词是支撑该现象的关键条件
- [Premack & Woodruff — 黑猩猩是否具备心智理论？](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — 心智理论 (ToM) 概念的起源（1978年）
- [Baron-Cohen, Leslie, Frith — 自闭症儿童是否具备心智理论？](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — 萨莉-安妮测试 (Sally-Anne Test) 相关论文（1985年）
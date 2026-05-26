# Self-Refine 与 CRITIC：迭代式输出优化

> Self-Refine（Madaan 等人，2023）在循环中让同一个大语言模型（Large Language Model, LLM）承担三个角色——生成（generate）、反馈（feedback）与优化（refine）。在 7 项任务上平均绝对得分提升 +20。CRITIC（Gou 等人，2023）通过将验证步骤路由至外部工具，强化了反馈环节。到 2026 年，该模式已成为各框架的标配，Anthropic 将其称为“评估器-优化器（evaluator-optimizer）”工作流，OpenAI Agents SDK 则将其实现为护栏循环（guardrail loop）。

**类型：** 构建
**语言：** Python（标准库 stdlib）
**前置条件：** 第 14 阶段 · 01（智能体循环 Agent Loop），第 14 阶段 · 03（反思 Reflexion）
**耗时：** 约 60 分钟

## 学习目标

- 阐述 Self-Refine 的三个提示词（生成、反馈、优化），并解释为何历史记录对优化提示词至关重要。
- 解释 CRITIC 的核心洞见：在缺乏外部依据的情况下，大语言模型的自我验证并不可靠。
- 使用标准库实现一个带历史记录及可选外部验证器的 Self-Refine 循环。
- 将该模式映射至 Anthropic 的“评估器-优化器”工作流与 OpenAI Agents SDK 的输出护栏机制。

## 问题背景

智能体（Agent）生成的答案往往接近正确，但仍有瑕疵。可能是一行代码存在语法错误，可能是一段摘要过于冗长，也可能是一个计划遗漏了边界情况。你真正期望的是：智能体能够批判性地审视自身输出，并自行修正。

Self-Refine 证明了仅凭单一模型即可实现该流程，无需训练数据，也无需强化学习（Reinforcement Learning, RL）。但存在一个局限：大语言模型在处理硬性事实的自我验证时表现不佳。CRITIC 提出了解决方案——将验证步骤交由外部工具（如搜索引擎、代码解释器、计算器或测试运行器）执行。

这两篇论文共同确立了 2026 年迭代优化的默认范式：生成、验证（尽可能借助外部工具）、优化，并在验证通过时终止循环。

## 核心概念

### Self-Refine（自优化，Madaan 等人，NeurIPS 2023）

一个大语言模型（Large Language Model, LLM），扮演三个角色：

generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.

关键细节：`refine`（优化）步骤会看到完整的历史记录——包括所有先前的输出和批评反馈（Critique）——因此它不会重复犯错。论文通过消融实验（Ablation Study）验证了这一点：如果去掉历史记录，输出质量会急剧下降。

核心结论：在包含 GPT-4 在内的 7 项任务（数学、代码、缩写、对话）中，平均绝对性能提升 20 分。无需训练，无需外部工具，仅使用单一模型。

### CRITIC（Gou 等人，arXiv:2305.11738，2024 年 2 月 v4 版）

Self-Refine 的弱点：反馈步骤本质上是 LLM 自我评分。对于事实性声明而言，这并不可靠（模型产生的幻觉（Hallucination）往往对其自身也极具说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于验证事实性声明的搜索引擎。
- 用于检查代码正确性的代码解释器（Code Interpreter）。
- 用于算术计算的计算器。
- 领域特定的验证器（Verifier）（如单元测试、类型检查器、静态代码分析工具）。

验证器会基于工具返回的结果生成结构化的批评反馈。随后，优化器（Refiner）将以此反馈为条件进行输出修正。

核心结论：由于批评反馈具有事实依据，CRITIC 在事实性任务上的表现优于 Self-Refine。在缺乏外部验证器的任务（如创意写作、格式调整）中，CRITIC 会退化为 Self-Refine。

### 停止条件（Stop Condition）

两种常见形式：

1. **验证器通过。** 外部测试返回成功。在可用时为首选方案（如单元测试、类型检查器、护栏断言（Guardrail Assertion））。
2. **未发出反馈。** 模型表示“输出没问题”。成本较低但不可靠；建议搭配最大迭代次数上限（Max-iteration Cap）使用。

2026 年默认策略：将两者结合。“若验证器通过，或模型表示输出良好且迭代次数 >= 2，或迭代次数 >= `max_iterations`，则停止。”

### Evaluator-Optimizer（评估器-优化器模式，Anthropic，2024）

Anthropic 在 2024 年 12 月的博文中将其命名为五种工作流模式（Workflow Patterns）之一。包含两个角色：

- 评估器（Evaluator）：对输出进行评分并生成批评反馈。
- 优化器（Optimizer）：根据批评反馈对输出进行修订。

循环执行直至评估器通过。在 Anthropic 的框架中，这等同于 Self-Refine/CRITIC。Anthropic 补充的关键工程细节是：评估器和优化器的提示词（Prompt）应存在显著差异，以防止模型只是盲目盖章通过（Rubber-stamp）。

### OpenAI Agents SDK 输出护栏（Output Guardrails）

OpenAI Agents SDK 将此模式封装为“输出护栏”。护栏是一种在智能体（Agent）最终输出上运行的验证器。如果护栏被触发（抛出 `OutputGuardrailTripwireTriggered` 异常），输出将被拒绝，智能体可重新尝试。护栏可以调用外部工具（CRITIC 风格），也可以是纯函数（Self-Refine 风格）。

### 2026 年常见陷阱

- **盲目盖章循环（Rubber-stamp loops）。** 同一模型使用相同风格的提示词同时负责生成和批评，容易收敛于“我觉得没问题”的结论。建议使用结构差异显著的提示词，或使用更小、更廉价的模型专门负责批评。
- **过度优化（Over-refinement）。** 每次优化迭代都会增加延迟和 Token 消耗。建议将预算控制在 1-3 次迭代；超过此次数后，应升级至人工审核。
- **在简单任务上使用 CRITIC。** 若缺乏外部验证器，CRITIC 会退化为 Self-Refine；切勿为占位验证器（Stub Verifier）支付额外的延迟成本。

## 构建它

`code/main.py` 在一个示例任务 (toy task) 上实现了自我优化 (Self-Refine) 与批判性迭代验证 (CRITIC)：根据给定主题生成简短的要点列表。验证器 (verifier) 负责检查格式（3个要点，每个不超过60个字符）。CRITIC 引入了一个外部的“事实验证器”，用于对已知的幻觉 (hallucinations) 进行惩罚。

组件：

- `generate` — 脚本化生成器。
- `feedback` — 大语言模型 (LLM) 风格的自我批判。
- `verify_external` — CRITIC 风格的基于事实的验证器。
- `refine` — 根据历史记录重写输出。
- 停止条件 (Stop condition) — 验证通过或达到最大4次迭代。

运行方式：

python3 code/main.py

对比自我优化 (Self-Refine) 与批判性迭代验证 (CRITIC) 的运行结果。CRITIC 能够捕捉到自我优化遗漏的事实性错误，因为外部验证器具备事实依据 (grounding)，而自我批判机制则不具备。

## 应用它

Anthropic 的评估器-优化器 (evaluator-optimizer) 模式以适配 Claude 的语言实现了该范式。OpenAI Agents SDK 的输出护栏 (output guardrails) 采用了类似 CRITIC 的架构（护栏可调用工具）。LangGraph 内置的反思节点 (reflection node) 在逻辑上类似于自我优化 (Self-Refine)。Google 的 Gemini 2.5 Computer Use 引入了逐步安全评估器，这是 CRITIC 的一种变体：每个操作在提交前都会经过验证。

## 发布它

`outputs/skill-refine-loop.md` 根据任务结构、验证器可用性及迭代预算，配置了一个评估器-优化器循环。它会生成用于生成器、评估器/验证器以及优化器的提示词 (prompts)，并附带停止策略。

## 练习

1. 将 `max_iterations=1` 运行该示例。CRITIC 是否仍然有效？
2. 将外部验证器替换为带噪声的验证器（随机产生 30% 的误报）。循环会如何表现？这正是 2026 年大多数护栏技术栈 (guardrail stacks) 面临的现实。
3. 实现“跨模型生成-批判”变体：大模型负责生成，小模型负责批判。其效果能否超越同模型方案？
4. 阅读 CRITIC 论文第 3 节 (arXiv:2305.11738 v4)。列出三种验证工具类别，并为每种类别提供一个示例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。该 SDK 在哪些方面存在不足，又在哪些方面设计得当？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 自我优化 (Self-Refine) | “能自我修复的大语言模型” | 在同一模型内构建“生成 -> 反馈 -> 优化”循环，并保留历史记录 |
| 批判性迭代验证 (CRITIC) | “基于工具的事实验证” | 用外部验证器（搜索、代码、计算、测试）替代内部反馈 |
| 评估器-优化器 (Evaluator-Optimizer) | “Anthropic 工作流模式” | 双角色协作——评估器打分，优化器修订——循环直至收敛 |
| 输出护栏 (Output guardrail) | “事后检查” | OpenAI Agents SDK 中的验证器，在智能体生成输出后运行 |
| 验证步骤 (Verify step) | “批判阶段” | 核心决策环节：依赖事实依据还是自我评估 |
| 优化历史 (Refine history) | “模型已尝试过的内容” | 将历史输出与批判内容前置到优化提示词中；若丢弃则质量骤降 |
| 橡皮图章循环 (Rubber-stamp loop) | “自我认同失败” | 相同提示词的批判仅返回“看起来不错”；需通过结构差异化的提示词修复 |
| 停止条件 (Stop condition) | “收敛测试” | 验证通过 或 无反馈且达到迭代上限；绝不能仅依赖单一条件 |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— 经典论文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) —— 基于工具的验证（Tool-Grounded Verification）
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 评估器-优化器（Evaluator-Optimizer）工作流模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) —— 将输出护栏（Output Guardrails）作为类 CRITIC 验证器
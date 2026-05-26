# 审查智能体（Reviewer Agent）：将构建者（Builder）与评估者（Marker）分离

> 编写代码的智能体无法客观地评估自己的代码。审查者是一个独立的第二循环，它拥有不同的系统提示词（System Prompt）、不同的目标，并且对构建者生成的所有内容仅具备只读权限。构建者与审查者之间的隔离，正是系统可靠性的核心所在。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 38（验证关卡（Verification Gate））
**耗时：** 约 55 分钟

## 学习目标

- 阐明为何同一个智能体无法可靠地审查自身的工作成果。
- 构建一个审查智能体循环，该循环能够消费构建者产出的工件（Artifacts），并输出结构化的审查报告。
- 编写一份审查评分标准（Rubric），针对具体维度进行量化评估，而非依赖主观直觉。
- 将审查者集成至工作台（Workbench），使人工审查步骤能够基于真实的产出物展开。

## 问题背景

你要求智能体修复一个缺陷。它修改了四个文件，运行了测试，并报告任务完成。验证关卡确认验收测试已运行且需求范围保持受控。关卡返回 `passed: true`。你合并了代码。两天后你才发现，该修复方案实际上解决的是错误的部分。

验收通过是必要条件，而非充分条件。审查者会提出验收环节无法触及的问题：这是否真正解决了正确的问题？是否在未标记的情况下擅自扩大了需求范围？是否记录了本应被质疑的假设条件？它是否将工作台留在了下一个会话（Session）能够顺利接手的状态？

## 核心概念

flowchart LR
  Builder[Builder Agent] --> Artifacts[diff + state + feedback + verdict]
  Artifacts --> Reviewer[Reviewer Agent]
  Reviewer --> Rubric[reviewer_checklist.md]
  Reviewer --> Report[review_report.json]
  Report --> Human[Human Sign-Off]

### 评审员评分标准 (Reviewer Rubric)

包含五个维度，每个维度的评分范围为 0 到 2 分。

| 维度 | 问题 |
|------|------|
| 问题契合度 (Problem Fit) | 所做的更改是否准确解决了既定任务，而非相近的其他任务？ |
| 范围纪律 (Scope Discipline) | 编辑是否严格限定在约定范围内，或是否有意扩展了约定范围？ |
| 假设说明 (Assumptions) | 所有隐含假设是否均已记录在可供审查的位置？ |
| 验证质量 (Verification Quality) | 验收命令是否真正证明了目标达成，还是仅证明了较弱的版本？ |
| 交接就绪度 (Handoff Readiness) | 下一个会话能否基于当前状态无缝接续工作？ |

满分为 10 分。得分低于 7 分为软性失败 (soft fail)；得分低于 5 分为硬性失败 (hard fail)。

### 评审员是独立角色，而非独立模型

你可以使用与构建器 (Builder) 相同的模型来运行评审员。其核心纪律在于角色分离：采用不同的系统提示词 (system prompt)、不同的输入，且对差异文件 (diff) 无写入权限。角色立场的转换即意味着输出信号的变化。

### 评审员无法编辑差异文件

评审员负责读取差异文件、状态、反馈和裁决结果，并撰写报告。它不会直接修补差异文件。如果报告指出“修复此处”，则由下一轮构建器执行修复操作，评审员则继续履行评审职责。角色混合将破坏这一隔离机制。

### 评审员评分标准与验证关卡的对比

验证关卡（第 14 阶段 · 38）负责检查确定性事实：验收流程是否已执行、规则是否通过、范围是否受控。而评审员则进行定性判断：所做工作是否正确、是否已妥善记录、交接材料是否可用。两者缺一不可。

## 动手实现

`code/main.py` 实现了以下功能：

- 一个 `ReviewerInputs` 数据类 (dataclass)，用于打包评审员需要读取的产出物。
- 一个评分器，每个维度对应一个独立函数。为配合本教程，每个函数均为确定性且仅为占位实现 (stub-grade)；实际应用中会调用大语言模型 (LLM)。
- 一个 `review_report.json` 写入器，用于输出五个维度的得分、总分以及裁决结果（`pass`、`soft_fail`、`hard_fail`）。
- 两个演示用例：一个为规范的更改，另一个为“测试正确但问题偏离”的更改。

运行方式：

python3 code/main.py

输出结果：将生成两份写入磁盘的评审报告，并在控制台打印各维度得分表格。

## 实际生产环境中的模式

数据为证：Cloudflare 于 2026 年 4 月推出的 AI 代码审查（AI Code Review）系统在 30 天内，针对 5,169 个代码仓库（Repos）中的 48,095 个合并请求（Merge Requests）执行了 131,246 次审查。审查完成的中位时间为 3 分 39 秒。最多七名专业审查员（安全、性能、代码质量、文档、发布管理、合规、Engineering Codex）在审查协调器（Review Coordinator）的统筹下并行运行，该协调器负责对发现的问题进行去重并判定严重程度。顶级模型专供协调器使用；专业审查员则运行在成本较低的模型层级上。

以下四种模式使其能够规模化运行。

**专业审查员池，而非单一全能审查员。** 对于独立仓库，使用一个包含五个维度的评估量表（Rubric）的单一审查员即可胜任。一旦代码库涉及安全关键、性能关键及文档等模块，就应拆分为使用更精简提示词（Prompts）的专业审查员。协调器负责去重；专业审查员无需执行完整的评估量表。模型层级分离由此自然形成：低成本的专业审查员搭配高成本的协调器。

**将偏见缓解（Bias Mitigation）作为设计需求，而非后期优化。** 大语言模型（LLM）评判器表现出四种稳定的偏见（Adnan Masood，2026 年 4 月）：位置偏见（Position Bias，GPT-4 在 (A,B) 与 (B,A) 排序上的不一致率约为 40%）、冗长偏见（Verbosity Bias，对较长输出的评分虚高约 15%）、自我偏好（Self-Preference，评判器倾向于给同模型家族生成的输出更高分）、权威偏见（Authority Bias，评判器会过度拔高引用知名作者内容的评分）。缓解措施：对两种排序均进行评估，仅统计结果一致的胜出项；使用 1-4 分量表并明确奖励简洁性；在不同模型家族间轮换评判器；在评分前剥离作者姓名。

**使用校准集（Calibration Set），而非凭感觉。** 准备一个包含 10-20 个任务的历史数据集，其中已知正确的判定结果。每次修改提示词后，都在该数据集上运行审查器。如果与历史记录的一致性低于 80%，则必须在审查器发布前修订评估量表。这是每个团队最终都会重新发现的真理；不如从一开始就采用它。

**与验证门禁（Verification Gate）结合的混合规范。** 验证门禁（Phase 14 · 38）负责确定性检查（是否运行了验收流程、测试是否通过、范围是否保持一致）。审查器负责语义检查（这项工作方向是否正确、假设是否已记录、交接内容是否可用）。Anthropic 2026 年的指南对此划分有明确说明：不要让审查器重复执行门禁已经验证过的内容。

## 投入使用

生产环境模式：

- **Claude Code 子智能体（Subagents）。** 在构建器（Builder）完成任务后，运行审查器子智能体。它会在拉取请求（Pull Request）上发布包含评估量表得分的评论。
- **OpenAI Agents SDK 交接（Handoffs）。** 任务完成后，构建器将控制权交接给审查器。审查器可将发现的问题列表交回，或升级交由人工处理。
- **双模型配对。** 构建器运行在更快、更便宜的模型上。审查器运行在能力更强但上下文窗口较小的模型上，专注于判断。

当人类无法亲自完成每一次审查时，审查器就是工作台（Workbench）所延伸出的“第二双眼睛”。

## 正式发布

`outputs/skill-reviewer-agent.md` 会生成针对特定项目的评审量规（reviewer rubric）、与构建器（builder）产出物对接的评审智能体（reviewer agent）存根，以及与验证门禁（verification gate）的集成模块，从而让人工评审能够直接基于一份书面报告展开，而非从零开始。

## 练习

1. 添加一个针对你产品领域的第六个维度。论证为何该维度无法被现有的五个维度所涵盖。
2. 使用两种不同的系统提示词（system prompt）（简洁型与详细型）运行评审器。哪一种生成的报告更有可能被人类阅读？
3. 为每个维度添加一个 `confidence` 字段。当置信度最低的维度低于 0.6 时，拒绝交付该报告。
4. 构建一个校准集（calibration set）：包含 10 个已知正确结论的历史任务完结记录。让评审器对这些记录进行评估。它在哪些地方与历史记录存在分歧？
5. 增加“请求更多证据”的交互功能（affordance）：评审器可在打分前要求构建器提供特定的测试运行结果。应如何设置合理的退避（back-off）策略，以避免陷入循环请求？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 评审量规（Reviewer rubric） | “检查清单” | 五个维度，每个维度采用 0-2 分制，并附带对应的书面问题 |
| 软性失败（Soft fail） | “需要修改” | 总分低于 7；构建器需针对发现的问题进行修复 |
| 硬性失败（Hard fail） | “拒绝” | 总分低于 5 或任一维度得分为 0；立即中止流程并上报人工处理 |
| 角色分离（Role separation） | “不同的提示词” | 同一模型可兼任两种角色；核心在于严格规范输入与角色姿态 |
| 置信度下限（Confidence floor） | “不交付低信噪比报告” | 当量规评估结果不确定时，拒绝输出最终结论 |

## 延伸阅读

- [OpenAI Agents SDK 代理交接（handoffs）](https://platform.openai.com/docs/guides/agents-sdk/handoffs)
- [Anthropic Claude Code 子代理（subagents）](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/sub-agents)
- [Cloudflare：大规模 AI 代码审查编排](https://blog.cloudflare.com/ai-code-review/) — 7 专家（specialist） + 协调器（coordinator）架构，30 天内运行 13.1 万次
- [代理即裁判（Agent-as-a-Judge）：使用代理评估代理（OpenReview / ICLR）](https://openreview.net/forum?id=DeVm3YUnpj) — DevAI 基准测试（benchmark），366 项层级化解决方案要求
- [Adnan Masood：基于评分量表（rubric-based）的评估与大语言模型即裁判（LLM-as-a-Judge）：方法论、偏差与实证验证](https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80) — 4 种偏差及其缓解策略
- [MLflow：大语言模型即裁判评估](https://mlflow.org/llm-as-a-judge) — 面向构建者（builder）与评估者（evaluator）分离架构的生产级工具链
- [LangChain：如何利用人工修正校准大语言模型即裁判](https://www.langchain.com/articles/llm-as-a-judge) — 校准集（calibration-set）工作流
- [Evidently AI：大语言模型即裁判完整指南](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [Arize：大语言模型即裁判——入门指南与预构建评估器（evaluators）](https://arize.com/llm-as-a-judge/)
- 第 14 阶段 · 05 — 自我优化（Self-Refine）与 CRITIC（单代理自审基线）
- 第 14 阶段 · 30 — 评估驱动的代理开发（校准集生成器）
- 第 14 阶段 · 38 — 评审者读取的验证门控（verification gate）
- 第 14 阶段 · 40 — 由评审报告输入的交接数据包（handoff packet）
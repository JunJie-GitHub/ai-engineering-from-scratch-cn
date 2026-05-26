# 评估驱动的智能体 (Agent) 开发

> Anthropic 的建议：“从简单的提示词 (Prompt) 开始，通过全面的评估 (Evaluation) 进行优化，仅在必要时添加多步智能体系统。”评估并非最后一步。它是驱动第 14 阶段所有其他决策的外部循环 (Outer Loop)。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置条件：** 第 14 阶段的全部内容。
**时长：** 约 60 分钟

## 学习目标

- 说出三个评估层——静态基准测试 (Static Benchmarks)、自定义离线评估 (Custom Offline Evaluation)、在线生产环境评估 (Online Production Evaluation)——及其各自的作用。
- 解释评估器-优化器紧密循环 (Evaluator-Optimizer Tight Loop)。
- 描述 2026 年的最佳实践：评估用例与代码共存，在持续集成 (CI) 中运行，并作为拉取请求 (PR) 的合并门禁。
- 将第 14 阶段的每一课与其生成的评估用例 (Eval Case) 关联起来。

## 核心问题

智能体能够通过演示测试，却会在生产环境中以演示无法预测的方式失败。基准测试回答的是“该模型是否具备广泛的能力？”，而不是“该智能体是否为我的产品推送了正确的补丁？”。答案是：在三个层级上持续运行评估，并将每一项防护机制 (Guardrail) 和已学习的规则映射到具体的评估用例中。

## 核心概念

### 三个评估层级

1. **静态基准测试（Static Benchmarks）** — 代码类使用 SWE-bench Verified（第 19 课），浏览/桌面类使用 WebArena/OSWorld（第 20 课），通用类使用 GAIA（第 19 课），工具使用类使用 BFCL V4（第 06 课）。用于跨模型比较和回归拦截（Regression Gating）。数据污染是真实存在的：SWE-bench+ 发现 32.67% 的解决方案存在泄露。务必报告 Verified（已验证）或 ±人工审计分数。

2. **自定义离线评估（Custom Offline Evals）** — 贴合你的产品形态：
   - 大语言模型裁判（LLM-as-Judge）（Langfuse、Phoenix、Opik — 第 24 课）。
   - 基于执行（Execution-based）（运行补丁，检查测试用例）。
   - 基于轨迹（Trajectory-based）（将动作序列与黄金标准进行对比；OSWorld-Human 显示顶尖智能体的表现是黄金标准的 1.4-2.7 倍）。

3. **在线评估（Online Evals）** — 生产环境：
   - 会话回放（Session Replays）（Langfuse）。
   - 护栏触发的告警（Guardrail-triggered Alerts）（第 16、21 课）。
   - 单步成本/延迟追踪（第 23 课 OTel 跨度）。

### 评估器-优化器（Evaluator-Optimizer，Anthropic）

核心循环如下：

1. 提议器（Proposer）生成输出。
2. 评估器（Evaluator）进行评判。
3. 持续优化，直到评估器通过。

这是自我优化（Self-Refine，第 05 课）模式的泛化。任何你关注的智能体工作流（Agent Flow）都可以封装进评估器-优化器循环中，以提升可靠性。

### 2026 年最佳实践

- 评估代码与业务代码同仓存放。
- 在每个拉取请求（PR）的持续集成（CI）中运行。
- 基于评估分数设置合并门禁（例如：“与 main 分支相比性能回退不超过 5%”）。
- 每个护栏（Guardrail）都对应一个评估用例。
- 每条学习到的规则（如 Reflexion、专业工作流学习规则）都对应一个失败用例。

### 串联第 14 阶段

第 14 阶段的每一课都会生成相应的评估用例：

| 课程 | 生成的评估用例 |
|--------|------------------------|
| 01 智能体循环（Agent Loop） | 预算耗尽、无限循环防护 |
| 02 ReWOO | 工具失败时规划器能正确重新规划 |
| 03 Reflexion | 重试时应用已学习的反思 |
| 05 Self-Refine/CRITIC | 裁判通过优化后的输出 |
| 06 工具使用（Tool Use） | 参数强制转换生效；未知工具被拒绝 |
| 07-10 记忆（Memory） | 检索引用与来源匹配；过时事实被标记为无效 |
| 12 工作流模式（Workflow Patterns） | 每种模式均能产生正确输出 |
| 13 LangGraph | 恢复时能精确重现状态 |
| 14 AutoGen Actors | 死信队列（DLQ）捕获崩溃的处理程序 |
| 16 OpenAI Agents SDK | 护栏在正确输入时触发 |
| 17 Claude Agent SDK | 子智能体结果返回给编排器 |
| 19-20 基准测试（Benchmarks） | SWE-bench Verified 分数、WebArena 成功率、OSWorld 效率 |
| 21 计算机使用（Computer Use） | 单步安全机制拦截注入的 DOM |
| 23 OTel | 跨度（Spans）发出所需属性 |
| 26 故障模式（Failure Modes） | 检测器标记已知故障 |
| 27 提示词注入（Prompt Injection） | 提示词注入防护（PVE）拒绝被污染的检索结果 |
| 28 编排（Orchestration） | 监督器将任务路由至正确的专家 |
| 29 运行时形态（Runtime Shapes） | 死信队列（DLQ）处理 N% 的失败率 |

如果你的评估套件（Eval Suite）涵盖了上述每一项用例，说明你已经完整覆盖了第 14 阶段。

### 评估驱动开发（Eval-Driven Development）的常见陷阱

- **缺乏基线（Baseline）**。没有已知最佳状态（Last-Known-Good）的评估结果毫无可读性。务必保存基线数据。
- **缺乏外部依据的 LLM 裁判**。裁判模型同样会产生幻觉。CRITIC 模式（第 05 课）—— 让裁判基于外部工具进行事实锚定（Grounding）。
- **对评估集过拟合**。过度针对评估指标进行优化，会偏离实际生产环境的价值。定期轮换测试用例。
- **不稳定的评估（Flaky Evals）**。非确定性用例会引发误报。固定随机种子，并对运行状态进行快照。

## 构建它

`code/main.py` 是一个基于标准库的评估框架（evaluation harness）：

- 包含分类（基准测试、自定义、在线）的用例注册表（case registry）。
- 一个待测试的脚本化智能体（scripted agent）。
- 评估-优化循环（evaluator-optimizer loop）：提出方案、评判、优化，直至通过或达到最大轮次。
- 持续集成门禁（CI gate）：汇总通过率，并与基线（baseline）对比以检测回归（regression）。

运行方式：

python3 code/main.py

输出：每个用例的通过/失败状态、回归标志、持续集成门禁判定结果。

## 使用它

- 在与智能体代码相同的代码库中编写评估用例（eval cases）。
- 通过持续集成（CI）在每次拉取请求（PR）中运行它们。
- 若检测到回归，则使构建失败。
- 持续跟踪通过率随时间的变化。
- 将每次生产环境故障关联到一个新的评估用例。

## 交付它

`outputs/skill-eval-suite.md` 为智能体产品构建了一个包含持续集成门禁和回归跟踪的三层评估套件（eval suite）。

## 练习

1. 选取一次生产环境故障。编写一个能复现该故障的评估用例。你的智能体现在能通过它吗？
2. 为你的业务领域构建一个包含三个维度（事实性、语气、范围）的大语言模型裁判标准（LLM-judge rubric）。对 50 个会话进行评分。
3. 将评估套件接入持续集成（CI）。当回归率 >=5% 时使构建失败。
4. 添加轨迹效率（trajectory-efficiency）指标：智能体执行的步数与黄金轨迹（gold trajectory）相比如何？
5. 将第 14 阶段的每个知识点映射到你套件中的一个评估用例。是否有遗漏？那就是需要填补的空白。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 静态基准测试（Static benchmark） | “开箱即用的评估” | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| 自定义离线评估（Custom offline eval） | “领域评估” | 针对你产品形态的大语言模型即裁判（LLM-as-judge）/ 执行验证 / 轨迹分析 |
| 在线评估（Online eval） | “生产环境评估” | 会话回放、护栏告警、成本/延迟跟踪 |
| 评估-优化器（Evaluator-optimizer） | “提出-评判-优化” | 迭代直至评判通过 |
| 持续集成门禁（CI gate） | “合并拦截器” | 评估出现回归时使构建失败 |
| 基线（Baseline） | “已知最新良好状态” | 用于检测回归的参考分数 |
| 轨迹效率（Trajectory efficiency） | “步数与黄金标准之比” | 智能体步数除以人类专家最小步数 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— “从简单开始，通过评估进行优化”
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) —— 经过筛选的基准测试
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) —— 工具使用基准测试
- [Langfuse docs](https://langfuse.com/) —— 实践中的评估与会话回放
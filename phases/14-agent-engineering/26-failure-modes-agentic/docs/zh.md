# 故障模式（Failure Modes）：智能体（Agent）为何失效

> MASFT（加州大学伯克利分校，2025）将 14 种多智能体（Multi-Agent）故障模式归类为 3 大类。微软的分类体系（Taxonomy）记录了现有 AI 故障在智能体（Agentic）环境中如何被放大。行业实地数据聚焦于五种反复出现的模式：幻觉动作（Hallucinated Actions）、范围蔓延（Scope Creep）、级联错误（Cascading Errors）、上下文丢失（Context Loss）以及工具误用（Tool Misuse）。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置条件：** Phase 14 · 05（Self-Refine 与 CRITIC）、Phase 14 · 24（可观测性 Observability）
**时长：** 约 60 分钟

## 学习目标

- 列举 MASFT 的三大故障类别，并说出每类中至少四种具体模式。
- 解释为何智能体故障会放大现有的 AI 故障模式（如偏见 Bias、幻觉 Hallucination）。
- 描述行业中反复出现的五种故障模式及其缓解措施。
- 实现一个基于标准库的检测器，用于为智能体执行轨迹（Traces）打上故障模式标签。

## 问题背景

团队交付的智能体在 90% 的执行轨迹上运行正常。但剩下的 10% 故障并非随机噪声，而是集中在少数几类反复出现的模式中。一旦你能准确命名这些模式，就能对其进行监控并修复。

## 核心概念

### MASFT（伯克利，arXiv:2503.13657）

多智能体系统故障分类法（Multi-Agent System Failure Taxonomy）。将14种故障模式聚类为3个类别。标注者间科恩卡帕系数（Cohen's Kappa）为0.88，表明这些类别具有可靠的可区分性。

核心观点：故障源于多智能体系统（Multi-Agent Systems）的根本性设计缺陷，而非可通过升级基础大语言模型（Large Language Models, LLMs）来修复的模型局限性。

### 微软智能体AI系统故障模式分类法（Microsoft Taxonomy of Failure Mode in Agentic AI Systems）

- 现有AI故障（如偏见、幻觉、数据泄露）在智能体（Agentic）环境中会被进一步放大。
- 自主性（Autonomy）催生了新型故障：大规模非预期操作、工具滥用、任务目标偏移（Mission Drift）。
- 该白皮书可作为智能体产品的风险登记册。

### 智能体AI故障特征分析（Characterizing Faults in Agentic AI, arXiv:2603.06847）

- 故障源于编排逻辑（Orchestration）、内部状态演化以及与环境交互过程中的问题。
- 并非仅仅是“代码缺陷”或“模型输出错误”。

### 大语言模型智能体幻觉综述（LLM Agent Hallucinations Survey, arXiv:2509.18970）

两种主要表现形式：

1. **指令遵循偏差（Instruction-following Deviation）** — 智能体未遵循系统提示词（System Prompt）。
2. **长程上下文误用（Long-range Contextual Misuse）** — 智能体遗忘或错误应用了早期对话轮次中的上下文。

子意图错误（Sub-intention errors）：遗漏（跳过步骤）、冗余（重复步骤）、乱序（步骤顺序错乱）。

### 业界高频复现的五种故障模式

Arize、Galileo 与 NimbleBrain 在 2024-2026 年的实地分析结果趋于一致，指出以下模式：

1. **幻觉动作（Hallucinated actions）**。智能体调用不存在的工具或捏造参数。
2. **范围蔓延（Scope creep）**。智能体超出用户请求范围执行任务（如创建额外的 PR、发送多余的邮件）。
3. **级联错误（Cascading errors）**。一次错误调用引发下游连锁反应。例如，一个虚构的 SKU 幻觉触发四次 API 调用，最终演变为跨系统事故。
4. **上下文丢失（Context loss）**。在长周期任务中，智能体遗忘早期对话轮次的约束条件。
5. **工具误用（Tool misuse）**。使用正确工具但传入错误参数，或完全调用了错误的工具。

级联错误是致命杀手。智能体无法区分“我执行失败了”与“该任务本身不可行”，且常在遇到 400 错误时幻觉生成成功消息以强行闭环。

### 缓解策略：在每一步设置检查门控

在推理链（Reasoning Chain）的每一步设置自动化验证门控，对照环境状态核查事实依据。具体而言：

- 逐步安全分类器（Per-step safety classifier，见第21课）。
- 工具调用参数校验（Tool-call argument validation，见第06课）。
- 将检索内容与已知事实进行交叉验证（见第05课，CRITIC 方法）。
- 通过重新探测状态来检测成功幻觉（例如：文件是否真的被创建了？）。

### 故障监控的常见误区

- **仅标记崩溃（Crashes）**。大多数智能体故障会生成看似合法的输出。必须进行内容级检查。
- **缺乏基线（Baseline）**。漂移检测（Drift detection）需要“最近已知良好状态”作为参照；否则无法判断“情况正在恶化”。
- **告警泛滥**。每次故障都触发一次告警通知。需进行聚类分析与速率限制。

## 开始构建

`code/main.py` 实现了一个标准库（stdlib）故障模式（failure-mode）标记器：

- 涵盖五种模式的合成追踪（trace）数据集。
- 针对每种模式的检测函数（基于工具调用、输出、重复操作的特征模式）。
- 一个标记器，用于为每条追踪数据打标签并报告模式分布。

运行方式：

python3 code/main.py

输出：每条追踪数据的标签 + 聚合分布，以较低成本复现了 Phoenix 的追踪聚类（trace clustering）所呈现的结果。

## 如何使用

- **Phoenix**：用于生产环境的漂移聚类（drift clustering）（第 24 课）。
- **Langfuse**：用于会话回放（session replay）与标注。
- **自定义方案**：用于你的可观测性平台（observability platform）无法检测的领域特定特征（domain-specific signatures）。

## 交付使用

`outputs/skill-failure-detector.md` 会生成针对你所在领域定制的故障模式检测器，并已接入追踪存储（trace store）。

## 练习

1. 添加一个“成功幻觉（success hallucination）”检测器：智能体（agent）返回成功状态，但目标状态实际未发生改变。
2. 为你已开发的产品中的 100 条真实追踪数据打标签。哪种模式占主导？修复它的成本是多少？
3. 实现一个“级联半径（cascade radius）”指标：假设第 N 步发生故障，它影响了多少下游步骤？
4. 阅读 MASFT 的 14 种故障模式。挑选三种适用于你产品的模式，并编写相应的检测器。
5. 将其中一个检测器接入持续集成（CI）任务：如果 >=5% 的追踪数据被标记为某种故障模式，则使构建失败。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MASFT | “多智能体故障分类法（Multi-agent failure taxonomy）” | 伯克利 14 模式分类体系 |
| 级联错误（Cascading error） | “涟漪式故障（Ripple failure）” | 早期的一步错误在后续 N 个步骤中持续传播 |
| 上下文丢失（Context loss） | “遗忘约束条件（Forgot the constraint）” | 长周期对话中丢失了早期轮次的关键事实 |
| 工具误用（Tool misuse） | “错用工具/参数（Wrong tool / wrong args）” | 调用格式合法，但调用逻辑错误 |
| 成功幻觉（Success hallucination） | “伪造完成（Faked completion）” | 智能体在遇到 400 错误时仍报告成功；实际状态未变 |
| 范围蔓延（Scope creep） | “过度执行（Overreach）” | 智能体执行了超出用户请求范围的操作 |
| 指令遵循偏差（Instruction-following deviation） | “违抗指令（Disobedience）” | 忽略系统提示词（system prompt）或用户约束条件 |
| 子意图错误（Sub-intention errors） | “计划缺陷（Plan bugs）” | 计划执行过程中的遗漏、冗余或顺序错乱 |

## 延伸阅读

- [Cemri 等人, MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 种故障模式，3 大类别
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — 风险清单
- [Arize Phoenix](https://docs.arize.com/phoenix) — 漂移聚类（drift clustering）实战
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时采用更简单的模式可完全规避这些故障
# 自主编程智能体（Autonomous Coding Agent）领域现状（2026）

> SWE-bench Verified 基准的得分在不到三年内从 4% 跃升至 80.9%。同一款 Claude Sonnet 4.5 模型在 SWE-agent v1 上得分为 43.2%，而在 Cline 自主模式（autonomous）下得分达 59.8%——模型外部的支撑架构（scaffolding）如今与模型本身同等重要。OpenHands（前身为 OpenDevin）是目前最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱（sandbox）中执行 Python 动作，而非依赖 JSON 工具调用（JSON tool calls）。这些表面亮眼的数据掩盖了一个方法论问题：SWE-bench Verified 的 500 个任务中有 161 个仅需 1–2 行代码修改，而针对需要 10 行以上修改的 SWE-bench Pro 基准，同一批前沿模型（frontier models）的得分仅在 23%–59% 之间。

**类型：** 学习
**语言：** Python（标准库，CodeAct 与 JSON 工具调用对比）
**前置知识：** 第 14 阶段 · 07（工具使用），第 15 阶段 · 01（长程智能体）
**耗时：** 约 45 分钟

## 核心问题

“哪个编程智能体最好”是个错误的问题。正确的问题应该是：在符合我实际工作分布的任务集上，配合我将在生产环境中部署的支撑架构，我能获得怎样的端到端可靠性（end-to-end reliability）？

2022 年至 2026 年间，该领域逐渐认识到：支撑架构——包括检索层（retrieval layer）、规划器（planner）、沙箱、编辑-验证循环（edit-verify loop）以及反馈格式（feedback format）——才是承载核心能力的基石。Claude Sonnet 4.5 在 SWE-agent v1 上于 SWE-bench Verified 基准中得分为 43.2%；同一模型在 Cline 的自主架构中得分则为 59.8%。绝对分差达 16.6 个百分点，而模型权重完全相同。基础模型仅是组件，循环架构才是最终产品。

伴随而来的问题是，基准测试饱和（benchmark saturation）掩盖了性能回退（regressions）。SWE-bench Verified 已接近饱和，且其中的简单任务长尾（500 个任务中有 161 个仅需修改 ≤2 行代码）拉高了最高分。实际质量更适合在类似 SWE-bench Pro（需修改 10 行以上代码）的分布上进行衡量，而在该基准上，同样的领先模型得分仍停留在 23%–59% 之间。

## 核心概念

### SWE-bench，一段话概述

SWE-bench（Jimenez 等人提出）采用带有标准修复补丁（ground-truth patches）的真实 GitHub 问题（issues），要求智能体（agent）生成一个能使测试套件（test suite）通过的代码补丁。SWE-bench Verified（OpenAI, 2024）是一个经过人工筛选的 500 项任务子集，已剔除模糊不清和存在缺陷的任务。SWE-bench Pro 是其难度更高的后续版本——要求代码变更行数在 10 行以上，目前前沿智能体（frontier agents）在该基准上的得分处于 23%–59% 之间。

### 2022 → 2026 年曲线实际揭示的趋势

- **2022 年**：研究模型（research models）在原始 SWE-bench 上的得分约为 4%。
- **2024 年**：GPT-4 结合 Devin 风格的脚手架（scaffolding）得分约为 14%；SWE-agent 得分约为 12%。
- **2025 年**：集成于 Aider 和 SWE-agent 中的 Claude 3.5/3.7 Sonnet 将得分推高至 40%–55% 区间。
- **2026 年**：Claude Sonnet 4.5 及其他前沿竞品在 SWE-bench Verified 上的得分达到 70%–80% 以上。Epoch AI 的排行榜（leaderboard）实时追踪这一数据。

这一增长斜率源于三个叠加因素：更强大的基础模型（base models）、更完善的脚手架架构（scaffolding，如 CodeAct、反思机制（reflection）、验证器循环（verifier loops）），以及更优质的基准测试（Verified 版本有效去除了噪声数据）。

### CodeAct 与 JSON 工具调用（JSON tool calls）的对比

OpenHands（All-Hands-AI，arXiv:2407.16741，前身为 OpenDevin）采用了一种特定的架构设计：模型不再输出由宿主环境解码并执行的 JSON 工具调用（JSON tool calls），而是直接生成 Python 代码，并由类 Jupyter 内核（Jupyter-style kernel）在沙盒（sandbox）中运行。智能体（agent）可以在单次操作（action）中遍历文件、串联工具调用，并自行捕获异常。

两者的权衡如下：

- **JSON 工具调用**：每次操作对应一轮对话；易于审计；组合能力有限；默认安全性高，因为每次调用都需经过显式验证器（validator）检查。
- **CodeAct**：单次操作可包含完整程序；具备强组合性；需要加固的沙盒环境（OpenHands 使用 Docker 隔离）；故障模式涵盖沙盒运行时允许的任何行为。

两种架构均已投入生产环境。CodeAct 在开放平台（如 OpenHands、smolagents）中占据主导地位。而在托管服务（如 Anthropic Managed Agents、OpenAI Assistants）中，由于服务提供商直接控制执行器（executor），JSON 工具调用仍是主流。

### 2026 年智能体脚手架（Scaffolds）生态格局

| 脚手架（Scaffold） | 许可证（License） | 执行模型（Execution model） | 显著特性（Notable property） |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开放平台；支持事件流（event-stream）回放 |
| SWE-agent | MIT | 智能体-计算机接口（Agent-Computer Interface, ACI） | 首个端到端的 SWE-bench 脚手架 |
| Aider | Apache-2 | 本地仓库中的差异编辑（edit-via-diff） | 极简脚手架，回归稳定性强 |
| Cline | Apache-2 | 具备工具策略的 VS Code 智能体 | 在 Sonnet 4.5 上得分最高的开放脚手架 |
| Devin (Cognition) | 专有（Proprietary） | 托管虚拟机（VM）+ 规划器（planner） | 首个“AI 软件工程师”产品类别 |
| Claude Code | 专有（Proprietary） | 权限模式（Permission modes）+ 例程（routines） | 第 10 课将详细讲解其智能体循环（agent loop） |

### 为何脚手架架构占据主导

一次代码生成任务属于长程轨迹（long-horizon trajectory，见第 1 课）。可靠性会在各个步骤中累积放大。脚手架主要在以下三个环节提升得分：

1. **检索（Retrieval）**：精准定位需要读取的文件是隐性的性能瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引（file-index）以及 Aider 的仓库映射（repo-map）均致力于解决此问题。
2. **验证器循环（Verifier loop）**：运行测试、读取堆栈跟踪（stack traces）并重新尝试，能在 SWE-bench 上带来 10 分以上的提升。
3. **故障隔离（Failure containment）**：出错时自动回滚的沙盒能防止错误累积放大。同一模型在配备与不配备验证器循环的情况下，表现宛如两款截然不同的产品。

### 基准测试饱和现象与真实数据分布

OpenHands 作者团队与 Epoch AI 均指出，SWE-bench Verified 存在“简单长尾”现象：500 项任务中有 161 项仅需修改 1–2 行代码。高分部分正是受此长尾任务驱动。SWE-bench Pro 将任务限制为 10 行以上的代码变更，即便是前沿系统，得分也仅在 23%–59% 之间。您的实际生产环境数据分布几乎肯定更接近 Pro 版本，而非 Verified 版本。

选择智能体的启示：从您自身的缺陷待办列表（bug backlog）中抽取一个类似 Pro 标准的子集进行测试。真正具有参考价值的分数，是那些与您实际交付任务具有代表性的得分。

## 使用它

`code/main.py` 在固定的微型任务分布上对比了两种演示用智能体脚手架（toy agent scaffolds）：

1. 一种 **JSON 工具调用（JSON tool-call）** 脚手架，每轮仅执行一个动作。
2. 一种 **CodeAct** 脚手架，每个动作可生成一小段 Python 代码。

两者均使用桩“模型”（基于确定性规则），从而将脚手架本身的性能与模型质量隔离开来进行对比。输出结果表明，CodeAct 脚手架能以更少的轮次解决更多任务，但代价是单个动作的影响半径（blast radius）更大。

## 交付上线

`outputs/skill-scaffold-audit.md` 可帮助你在正式采用前对拟定的编程智能体脚手架（coding-agent scaffold）进行审计：涵盖检索质量、验证器（verifier）是否存在、沙箱隔离（sandbox isolation）程度，以及基准测试与实际任务分布的匹配度（benchmark-to-distribution fit）。

## 练习

1. 运行 `code/main.py`。在同一任务集上，每种脚手架分别需要多少轮次？各自的单动作影响半径（per-action blast radius）是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。该论文指出，在处理复杂任务时，CodeAct 优于 JSON 工具调用。请找出论文中承认的一种失败模式（failure mode），并用一句话说明该模式在生产环境中何时会占据主导。

3. 从你的缺陷待办列表（bug backlog）中挑选一个需要跨两个文件修改 10 行以上代码的任务。估算前沿模型（frontier model）在 (a) JSON 工具调用和 (b) CodeAct 两种模式下的端到端成功概率（end-to-end success probability），并解释两者差距的原因。

4. SWE-bench Verified 包含 161 个单文件、仅需修改 1–2 行代码的任务。请构建一个排除这些任务的评分指标。排行榜（leaderboard）的排名会发生怎样的变化？

5. 阅读《Introducing SWE-bench Verified》（OpenAI）。说明用于剔除模糊任务的具体方法，并指出该人工筛选流程可能会遗漏的一个任务类别。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| SWE-bench | “编程基准测试” | 包含真实 GitHub Issue、标准答案补丁（ground-truth patches）及测试套件的基准 |
| SWE-bench Verified | “清洗后的子集” | 500 个人工精选任务，包含部分简单任务长尾（easier-tail） |
| SWE-bench Pro | “高难度子集” | 需修改 10 行以上代码；前沿模型得分在 23%–59% 之间 |
| CodeAct | “代码即动作” | 智能体输出 Python 代码；由类 Jupyter 内核在沙箱中执行 |
| JSON tool call | “函数调用” | 每个动作均为结构化 JSON 负载，执行前需经过验证 |
| Scaffold | “智能体框架” | 围绕基础模型构建的“检索 + 规划器 + 执行器 + 验证器”循环 |
| ACI (Agent-Computer Interface) | “SWE-agent 的格式” | 专为大语言模型（LLM）交互体验设计的命令集，而非面向人类终端 |
| Verifier loop | “测试并重试” | 运行测试、读取输出、修改补丁；提升非模型可靠性的最大增益来源 |

## 延伸阅读

- [Jimenez 等人 — SWE-bench](https://www.swebench.com/) — 原始基准测试（Benchmark）与方法论。
- [OpenAI — 推出 SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 介绍该精选子集的构建过程。
- [Wang 等人 — OpenHands：面向 AI 软件开发者的开放平台](https://arxiv.org/abs/2407.16741) — CodeAct 架构与事件流（Event-stream）设计。
- [Epoch AI — SWE-bench 排行榜（Leaderboard）](https://epoch.ai/benchmarks) — 实时追踪的评分数据。
- [Anthropic — 衡量智能体自主性（Agent Autonomy）](https://www.anthropic.com/research/measuring-agent-autonomy) — 长程编程智能体（Coding Agent）可靠性的评估框架。
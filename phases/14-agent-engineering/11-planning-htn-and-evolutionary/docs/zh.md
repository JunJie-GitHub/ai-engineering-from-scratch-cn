# 基于分层任务网络 (Hierarchical Task Networks, HTN) 与进化搜索的规划

> 符号规划 (Symbolic Planning) 适用于那些规划可被严格证明为正确的场景。进化代码搜索 (Evolutionary Code Search) 适用于适应度函数 (Fitness Function) 可由机器验证的场景。ChatHTN (2025) 与 AlphaEvolve (2025) 展示了当它们与大语言模型 (Large Language Model, LLM) 结合时，各自能释放出的潜力。

**类型:** 构建
**语言:** Python (标准库)
**前置条件:** 第 14 阶段 · 02 (ReWOO 与 Plan-and-Execute)
**预计耗时:** 约 75 分钟

## 学习目标

- 解释分层任务网络 (Hierarchical Task Networks)：任务 (tasks)、方法 (methods)、操作符 (operators)、前置条件 (preconditions) 与效果 (effects)。
- 描述 ChatHTN 的混合循环机制——以符号搜索为主，大语言模型作为回退分解 (Fallback Decomposition) 手段。
- 解释 AlphaEvolve 的进化循环机制，以及为何它必须依赖程序化评估器 (Programmatic Evaluator) 才能运行。
- 使用标准库实现一个简易的 HTN 规划器与一个简易的进化搜索算法。

## 问题背景

ReWOO（第 02 课）、Plan-and-Execute 与 ReAct 涵盖了大多数智能体规划场景。但它们在以下两种情况下表现不佳：

1. **具备可证明正确性的规划。** 例如任务调度、飞行路径规划、合规工作流等——规划必须在构建时就保证逻辑严密。大语言模型生成的流畅规划若偶尔出现步骤幻觉 (Hallucination)，则是不可接受的。
2. **具备机器可验证适应度函数的优化问题。** 例如矩阵乘法优化、调度启发式算法、编译器优化阶段 (Compiler Passes) 等——目标并非“一个正确的规划”，而是“最优的规划”。

HTN 规划与 AlphaEvolve 分别针对上述两类不同问题提供了解决方案。两者均将大语言模型视为能力放大器，而非替代品。

## 核心概念

### 层次任务网络 (Hierarchical Task Networks)

HTN 包含以下要素：

- **任务 (Tasks)** — 复合任务 (compound tasks，需进一步分解) 和原子任务 (primitive tasks，可直接执行)。
- **方法 (Methods)** — 将复合任务分解为子任务的方式，附带前置条件。
- **操作符 (Operators)** — 带有前置条件和效果 (effects) 的原子动作。
- **状态 (State)** — 一组事实 (facts)。

规划 (Planning)：给定目标任务和初始状态，寻找一种分解方案，将其转化为一系列原子操作符，并确保它们的前置条件按顺序得到满足。

HTN 的出现早于大语言模型 (LLMs)，至今仍是生成可证明正确规划 (provably-correct plans) 的参考标准。

### ChatHTN (Gopalakrishnan 等人, 2025)

ChatHTN (arXiv:2505.11814) 将符号化 HTN (symbolic HTN) 与大语言模型查询交替结合：

1. 尝试使用现有方法分解当前的复合任务。
2. 若无适用方法，则向 LLM 提问：“在状态 `s` 下，你会如何分解 `task`？”
3. 将 LLM 的回复转化为候选子任务。
4. 对照操作符模式 (operator schema) 进行验证；拒绝无效的分解方案。
5. 递归执行。

该论文的核心主张是：生成的每一个规划都是可证明可靠的 (provably sound)，因为 LLM 的建议仅作为候选分解方案引入，绝不会直接修改规划。符号层负责保证正确性，而 LLM 负责扩展方法库。

在线方法学习 (Online method learning)（OpenReview `gwYEDY9j2x`，2025 年后续研究）引入了一个学习器，通过回归分析对 LLM 生成的分解方案进行泛化——最多可将 LLM 查询频率降低 75%。

### AlphaEvolve (Novikov 等人, 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, 2025 年 6 月) 则是另一套体系：由 Gemini 2.0 Flash/Pro 集成模型 (ensemble) 协调的进化式代码搜索 (evolutionary code search)。

循环流程：

1. 从种子程序 (seed program) 和程序化评估器 (programmatic evaluator)（返回适应度分数 (fitness score)）开始。
2. LLM 集成模型提出变异方案 (mutations)。
3. 将变异后的代码输入评估器运行。
4. 保留最优结果；再次进行变异。

已发表的成果：

- 56 年来首次在 4x4 复数矩阵乘法上超越 Strassen 算法（仅需 48 次标量乘法）。
- 通过一种 Borg 调度启发式算法，挽回了 Google 0.7% 的计算资源。
- 在前沿工作负载上，将 FlashAttention 的速度提升了 32%。

硬性约束：适应度函数 (fitness function) 必须是机器可验证的。对自然语言文本答案进行进化搜索无法收敛。

### 如何选择适用方案

| 问题类别 | 适用方案 | 原因 |
|---------------|-----|-----|
| 带有硬性约束的调度任务 | HTN + ChatHTN | 可证明的可靠性 |
| 编译器优化 | AlphaEvolve | 适应度可机器验证 |
| 多步任务执行 | ReAct / ReWOO | LLM 参与循环，无形式化保证 |
| 带测试的代码改进 | AlphaEvolve | 测试用例即评估器 |
| 受策略约束的自动化 | HTN | 前置条件可编码策略 |

### 常见误区

- **缺乏操作符的 HTN。** 若没有前置条件/效果模式，可靠性声明将不攻自破。ChatHTN 中“LLM 建议分解”的机制必须依赖该模式来拒绝无效操作。
- **缺乏真实评估器的 AlphaEvolve。** “询问 LLM 代码是否更好”并非适应度函数。评估器必须是确定性且高效的。
- **过度设计。** 大多数智能体 (agent) 任务并不需要上述两者。应优先尝试 ReAct 或 ReWOO。

## 构建它

`code/main.py` 实现了两个示例程序：

- 一个基于标准库的分层任务网络（HTN）规划器，包含操作符（operators）、方法（methods）、前置条件（preconditions）、效果（effects），以及一个 `LLMFallback` 机制。当没有方法匹配复合任务（compound task）时，该机制会触发。其中的“大语言模型（LLM）”是一个脚本化的分解器，因此该规划器可离线运行。
- 一个基于标准库的算术程序进化搜索（evolutionary search）：生成表达式，使其在测试集上的输出最小化 `|f(x) - target|`。评估器（evaluator）是确定性的。

运行方式：

python3 code/main.py

运行日志（trace）展示了 HTN 规划器如何分解复合任务（包含计划中途的 LLM 回退机制），以及进化循环如何收敛至目标表达式。

## 使用它

- **HTN 规划器** — 可使用 `pyhop`、`SHOP3`，或针对特定领域的策略执行需求自行构建。
- **ChatHTN** — 研究用代码；该模式（符号规划 + LLM 回退）可无缝移植到任何 HTN 规划器中。
- **AlphaEvolve** — DeepMind 的论文；该模式（模型集成 + 评估器）具备可复现性。OpenEvolve 及类似的开源分支项目正在涌现。
- **智能体框架（Agent frameworks）** — 目前尚无框架原生支持 HTN 或 AlphaEvolve。可将其构建为子智能体（subagent）或后台工作进程。

## 交付它

`outputs/skill-hybrid-planner.md` 会生成一个混合规划器脚手架（HTN 或进化式），并明确界定 LLM 的职责范围。

## 练习

1. 为 HTN 规划器添加回溯（backtracking）功能：当某个操作符的后置条件（postcondition）在运行时失败时，回退状态并尝试下一个方法。
2. 为 ChatHTN 添加 LLM 方法缓存：当 LLM 在状态模式 `P` 下分解任务 `T` 时，保存结果。下次调用时优先检查方法库。
3. 将进化搜索的评估器替换为真实的测试套件。进化出一个能通过 20 个测试用例的排序函数；记录达到收敛所需的代数（generations）。
4. 阅读 AlphaEvolve 的评估器设计说明。为你关注的领域（如 SQL 查询优化、测试套件最小化、部署 YAML 配置）设计一个评估器。
5. 组合应用：使用 HTN 将复合任务分解为子任务，然后对每个子任务的原始操作符（primitive operator）应用进化搜索。分析它在哪些场景表现优异，在哪些场景属于过度设计。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 分层任务网络（HTN） | “分层规划器” | 基于操作符、前置条件和效果的任务分解 |
| 方法（Method） | “分解规则” | 将复合任务拆解为子任务的方式 |
| 操作符（Operator） | “原始动作” | 具备前置条件和效果的具体执行步骤 |
| ChatHTN | “LLM + HTN” | 当无匹配方法时，符号规划器向 LLM 请求分解方案 |
| AlphaEvolve | “进化式代码搜索” | 集成 LLM 对代码进行变异；由确定性评估器进行筛选 |
| 适应度函数（Fitness function） | “评估器” | 针对输出结果的确定性、可机器校验的评分机制 |
| 在线方法学习（Online method learning） | “缓存的 LLM 分解” | 存储并泛化 LLM 生成的计划，以降低查询成本 |

## 延伸阅读

- [Gopalakrishnan 等人，ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — 符号与大语言模型（LLM）混合规划器（Planner）
- [Novikov 等人，AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 基于大语言模型变异的进化式代码搜索
- [Anthropic，构建高效智能体](https://www.anthropic.com/research/building-effective-agents) — 何时选用规划器而非简单循环
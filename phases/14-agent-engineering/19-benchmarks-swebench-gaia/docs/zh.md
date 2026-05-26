# 基准测试（Benchmarks）：SWE-bench、GAIA、AgentBench

> 三大基准测试构成了 2026 年智能体（Agent）评估的基石。SWE-bench 测试代码修补（Code Patching）能力。GAIA 测试通用工具使用（Generalist Tool Use）能力。AgentBench 测试多环境推理（Multi-Environment Reasoning）能力。了解它们的构成、数据污染（Data Contamination）情况，以及它们无法衡量的维度。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 06（工具使用）
**时长：** 约 60 分钟

## 学习目标

- 指出 SWE-bench 的测试框架（Test Harness）（FAIL_TO_PASS），并解释为何它以单元测试（Unit Tests）作为准入门槛。
- 解释 SWE-bench Verified（OpenAI，500 个任务）存在的原因及其剔除的内容。
- 描述 GAIA 的设计原则：对人类简单，对 AI 困难；包含三个难度等级。
- 列出 AgentBench 的八个环境，并指出其对开源大语言模型（Large Language Models, LLMs）的主要阻碍因素。
- 总结 SWE-bench+ 的数据污染发现及其带来的影响。

## 核心问题

排行榜只能告诉你哪个模型在单一基准测试中胜出。但它们无法告诉你：

- 基准测试是否受到数据污染（训练数据中包含答案、测试集泄露）。
- 基准测试是否衡量了你真正关心的能力（代码编写 vs 网页浏览 vs 通用能力）。
- 评估器是否具备鲁棒性（抽象语法树匹配（AST Matching）、状态检查（State Checks）、人工复核（Human Review））。

在引用具体分数之前，务必先了解这三大核心基准测试及其失效模式（Failure Modes）。

## 核心概念

### SWE-bench（Jimenez 等人，ICLR 2024 口头报告）

- 来自 12 个热门 Python 代码库 (repository) 的 2,294 个真实 GitHub 问题 (issue)。
- 智能体 (Agent) 输入：修复前提交 (commit) 对应的代码库 + 自然语言形式的问题描述。
- 智能体输出：补丁 (patch)。
- 评估器 (Evaluator)：应用补丁，运行仓库的测试套件。补丁必须将 FAIL_TO_PASS 测试（此前失败，现通过）转为通过，且不能破坏 PASS_TO_PASS 测试。

SWE-agent（Yang 等人，2024）在发布时达到了 12.5% 的得分，其核心在于强调智能体-计算机接口 (Agent-Computer Interface)（如文件编辑器命令、模型可理解的搜索语法）。

### SWE-bench Verified

OpenAI，2024 年 8 月发布。人工筛选的 500 个任务子集。剔除了描述模糊的问题、不可靠的测试以及修复方案不明确的任务。这是衡量“你的智能体能否交付真实可用补丁”的主要基准 (benchmark)。

### 数据污染 (Contamination)

- 超过 94% 的 SWE-bench 问题早于大多数模型的训练数据截止时间 (cutoff date)。
- **SWE-bench+** 发现，32.67% 的成功补丁在问题文本中泄露了解决方案（模型在描述中直接看到了修复方法），另有 31.08% 因测试覆盖率不足而存疑。
- Verified 版本的数据更干净，但并非完全无污染。

实际影响：在 SWE-bench 上得分为 50% 的模型，在 SWE-bench+ 上可能仅得 35%。若宣称 SWE-bench 性能，务必同时报告两项得分。

### GAIA（Mialon 等人，2023 年 11 月）

- 共 466 道题目；其中 300 道保留用于 huggingface.co/gaia-benchmark 的私有排行榜。
- 设计理念：“对人类而言概念上很简单（人类正确率 92%），但对 AI 却很难（带插件的 GPT-4 正确率仅 15%）。”
- 测试推理能力、多模态 (multi-modality) 理解、网页交互与工具使用。
- 分为三个难度等级；第 3 级要求跨模态使用长工具链 (tool chain)。

GAIA 用于衡量“通用能力 (generalist capability)”。请勿将其与代码专用基准混淆。

### AgentBench（Liu 等人，ICLR 2024）

- 涵盖 8 个环境，包括代码类（Bash、数据库 (DB)、知识图谱 (KG)）、游戏类（Alfworld、LTP）、网页类（WebShop、Mind2Web）以及开放式生成。
- 多轮交互 (multi-turn)，每个数据划分包含约 4,000 至 13,000 轮对话。
- 主要发现：长期推理、决策制定与指令遵循 (instruction following) 是开源大语言模型 (OSS LLM) 追赶商业模型的主要瓶颈。

### 这些基准无法衡量的内容

- 实际运行成本（Token 消耗量、实际耗时 (wall-clock time)）。
- 对抗环境下的安全行为。
- 在你特定业务领域的表现（请使用自有评估集，参见第 30 课）。
- 长尾故障 (tail failures)（基准测试通常看平均值；而生产运维人员更关注最差的 1% 情况）。

### 基准测试常见的误区

- **唯单一指标论。** SWE-bench 50% 的得分所提供的信息，远不如 P50/P75/P95 成本与步骤分布来得丰富。
- **隐瞒数据污染。** 仅报告 SWE-bench 得分而不提及 Verified 或 SWE-bench+ 具有误导性。
- **将基准测试作为开发目标。** 针对基准测试进行优化往往会偏离实际生产环境的实用性。

## 动手构建

`code/main.py` 实现了一个类似 SWE-bench 的简易评估框架（harness）：

- 合成的缺陷修复任务（共 3 个）。
- 一个脚本化的“智能体（agent）”，用于生成补丁。
- 一个测试运行器，用于检查 FAIL_TO_PASS（缺陷已修复）和 PASS_TO_PASS（未引入新缺陷）。
- 一个基于问题分解深度的 GAIA 风格难度分类器。

运行方式：

python3 code/main.py

输出结果将展示每个任务及每个难度级别的解决率（resolution rate），并使评估器规则具体化。

## 使用指南

- 针对代码智能体（code agents），使用 **SWE-bench Verified**。务必报告 Verified 分数。
- 针对通用智能体（generalist agents），使用 **GAIA**。请使用私有排行榜数据划分（private leaderboard split）。
- 针对多环境对比，使用 **AgentBench**。
- 针对你产品的实际业务形态，使用**自定义评估（custom evals）**（第 30 课）。

## 交付上线

`outputs/skill-benchmark-harness.md` 为任意代码库-任务对构建了一个 SWE-bench 风格的评估框架，并采用 FAIL_TO_PASS / PASS_TO_PASS 作为门禁条件。

## 练习

1. 将该简易评估框架移植到真实仓库上运行（任选一个你自己的仓库）。针对已知缺陷编写 3 个 FAIL_TO_PASS 测试。
2. 添加步数统计指标（step-count metric）。在你的 3 个任务中，每次成功解决需要智能体执行多少步？
3. 阅读 SWE-bench+ 论文。实现一项解决方案泄露检查（solution-leakage check）（将 issue 文本与代码 diff 进行模式匹配）。
4. 从公开划分（public split）中下载一道 GAIA 题目。推演 GPT-4 级别的智能体会如何操作。它需要哪些工具？
5. 阅读 AgentBench 的各环境细分报告。哪个环境最贴近你的产品业务场景？在该环境中，“SOTA（State of the Art，当前最佳）”的表现是怎样的？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| SWE-bench | “代码智能体基准测试” | 包含 2,294 个 GitHub issue；补丁必须使 FAIL_TO_PASS 测试由失败转为通过 |
| SWE-bench Verified | “纯净版 SWE-bench” | 500 个人工精选任务，由 OpenAI 提供 |
| FAIL_TO_PASS | “修复门禁” | 修复前失败的测试，打补丁后必须通过 |
| PASS_TO_PASS | “无回归门禁” | 修复前通过的测试，打补丁后仍须保持通过 |
| GAIA | “通用智能体基准测试” | 466 道对人类简单但对 AI 困难的多工具问题 |
| AgentBench | “多环境基准测试” | 8 个环境；长周期多轮交互 |
| Contamination（数据污染） | “训练集泄露” | 基准测试任务出现在模型训练数据中 |
| SWE-bench+ | “污染审计” | 在成功的 SWE-bench 补丁中发现了 32.67% 的解决方案泄露 |

## 延伸阅读

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — 原始基准测试
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 精选子集
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — 通用智能体基准测试
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — 多环境测试套件
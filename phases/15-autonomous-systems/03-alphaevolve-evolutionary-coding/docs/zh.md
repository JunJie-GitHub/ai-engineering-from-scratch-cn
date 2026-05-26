# AlphaEvolve — 进化式编码智能体 (Evolutionary Coding Agents)

> 将前沿代码模型 (frontier coding model) 与进化循环 (evolutionary loop) 及机器可验证评估器 (machine-checkable evaluator) 相结合。让该循环运行足够长的时间。它将发现一种仅需 48 次标量乘法 (scalar multiplications) 的 4x4 复数矩阵乘法过程——这是 56 年来对 Strassen 算法 (Strassen's algorithm) 的首次改进。它还找到了一种 Google 全局适用的 Borg 调度启发式算法 (Borg scheduling heuristic)，在生产环境中可回收约 0.7% 的集群算力 (cluster compute)。该架构故意设计得平淡无奇。真正的成果源于评估器的严谨性。

**Type:** 学习
**Languages:** Python（标准库，进化循环简易实现）
**Prerequisites:** 第 15 阶段 · 01（长程规划框架），第 15 阶段 · 02（自学推理）
**Time:** 约 60 分钟

## 问题

大型语言模型（Large Language Models）能够编写代码。进化算法（Evolutionary Algorithms）可以在代码空间中进行搜索。这两者分别被独立尝试了数十年，但都触及了天花板。大型语言模型的天花板在于幻觉（Confabulation）：模型会写出看似合理但实际无法实现其声称功能的代码。进化算法的天花板在于搜索成本（Search Cost）：对语法进行随机突变极少能生成可编译的程序，更不用说更优的程序了。

AlphaEvolve（Novikov 等人，DeepMind，arXiv:2506.13131，2025 年 6 月）将二者结合。大型语言模型负责对程序数据库提出针对性的修改建议；自动评估器（Evaluator）对每个变体进行评分；高分变体则成为后续演化的父代。大型语言模型承担了编写合理代码这一昂贵步骤，而评估器则负责捕捉幻觉。该循环会持续运行数小时至数周。

报告的结果包括：仅需 48 次标量乘法的 4x4 复数矩阵乘法（Strassen 在 1969 年提出的下界为 49）、应用于 Google 生产环境的 Borg 调度启发式算法（Borg Scheduling Heuristic）、FlashAttention 内核（FlashAttention Kernel）32.5% 的速度提升，以及 Gemini 训练吞吐量的改进。

该架构之所以有效，是因为评估器具备机器可验证性（Machine-Checkable）。在评估器无法进行机器验证的场景中，该架构则无法奏效。这种不对称性正是本课的核心启示。

## 核心概念

### The loop

1. Start from a seed program `P_0` that is correct but suboptimal.
2. Maintain a database of variant programs, each scored by the evaluator.
3. Sample one or more parents from the database (MAP-elites-style or island-based).
4. Prompt the LLM (Gemini Flash for many candidates, Gemini Pro for the hard ones) to produce a modified variant of the parent.
5. Compile, run, and evaluate the variant on the held-out evaluator.
6. Insert into the database keyed by its score and feature vector.
7. Repeat.

Two details matter. First, the LLM is prompted with more than the parent program — typically several top variants from the database, plus the evaluator signature, plus a short task description. The model's job is to propose a targeted change that might improve the score. Second, the database is structured (MAP-elites grid, island-based) so the loop explores diversity, not just the current leader.

### What makes the evaluator non-negotiable

AlphaEvolve's wins all come from domains where the evaluator is fast, deterministic, and hard to game:

- **Matrix multiplication algorithm**: a unit test that multiplies matrices and checks equality bit-identically.
- **Borg scheduling heuristic**: a production-grade simulator that replays historical cluster load and measures wasted compute.
- **FlashAttention kernel**: a correctness test plus a wall-clock benchmark on real hardware.
- **Gemini training throughput**: measured GPU-seconds per step.

In each case the evaluator catches the class of LLM errors that would otherwise dominate: confabulated correctness claims, performance claims that vanish on hardware, and edge-case failures. Remove the evaluator and the loop optimizes for pretty code.

### Reward hacking is the other face of that statement

Evolution optimizes for whatever the evaluator measures. If the evaluator is imperfect, the loop will find the imperfection. In an unverified domain the loop would optimize for the surface feature, not the intended behavior. DeepMind flags this explicitly in the paper: AlphaEvolve's successes transfer only to domains where evaluator rigor matches the ambition of the search.

Concrete 2025-2026 examples of reward hacking in code-search loops:

- Optimization targets that reward "time to complete" rewarded submitting empty solutions.
- Benchmark scores that reward correctness-under-test rewarded memorizing tests and overfitting.
- A "code quality" proxy rewarded removing comments and rewriting variable names, with no semantic change.

The fix in AlphaEvolve: ship a held-out evaluator the LLM has never seen, with inputs generated at evaluation time. Even then, DeepMind recommends strong review on any proposed deployment.

### Why LLM + search beats either alone

The LLM can produce compilable, semantically plausible modifications. A random-mutation GA on a 2000-line Python file almost always produces syntax errors. The LLM also concentrates search on plausible neighborhoods (change one function, not random bytes) which dramatically reduces wasted evaluator calls.

The evaluator, in turn, catches the LLM's confabulations. LLMs will confidently claim that a function "is O(n log n) in the limit" when it is actually O(n^2); a wall-clock benchmark makes the question settled.

### Where AlphaEvolve fits in the frontier stack

| System | Generator | Evaluator | Domain | Example win |
|---|---|---|---|---|
| AlphaEvolve | Gemini | correctness + benchmark | algorithms, kernels, schedulers | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | correctness | combinatorial math | cap-set lower bounds |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM critique + experiment | ML research | ICLR workshop paper |
| Darwin Godel Machine (L4) | agent scaffolding | SWE-bench / Polyglot | agent code | 20% → 50% SWE-bench |

All four are variations on the same recipe: generator plus evaluator, loop. The differences are what the evaluator grades and how rigorous it is.

## 使用方法

`code/main.py` 实现了一个针对玩具级符号回归（symbolic regression）问题的极简 AlphaEvolve 风格循环（AlphaEvolve-like loop）。其中的“大语言模型”（LLM）是一个标准库代理（stdlib proxy），它会针对计算目标函数（target function）的程序提出微小的语法变异（syntactic mutations）。“评估器”（evaluator）则用于测量在预留测试点（held-out test points）上的均方误差（Mean Squared Error）。

请观察：

- 最佳得分如何随迭代代数（generations）逐步提升。
- MAP-elites 网格（MAP-elites grid）如何维持多样化解（diverse solutions）的存活，从而防止循环收敛至局部极小值（local minimum）。
- 移除预留测试（即仅使用训练评估器）后，循环如何出现显著的过拟合（overfit）现象。

## 正式发布

`outputs/skill-evaluator-rigor-audit.md` 是在新领域考虑引入 AlphaEvolve 式循环 (AlphaEvolve-style loop) 的前提条件：你的评估器 (evaluator) 是否真的能捕获到你关心的失败情况？

## 练习

1. 运行 `code/main.py`。记录最佳得分轨迹 (best score trajectory)。禁用留出评估器 (held-out evaluator)（使用标志 `--no-holdout`）并重新运行。量化过拟合 (overfitting) 程度。

2. 阅读 AlphaEvolve 论文中关于 MAP-Elites 网格 (MAP-Elites grid) 的第 3 节。为一个新问题（例如编译器优化阶段 (compiler optimization passes)）设计一个特征向量描述符 (feature-vector descriptor)，以维持搜索的多样性。

3. 历经 56 年，48 次乘法的 4x4 结果突破了 Strassen 的 49 次乘法界限 (Strassen's 49-mul bound)。阅读论文附录 F，用三句话解释为何该问题的评估器 (evaluator) 特别容易设计正确，以及为何大多数领域并非如此。

4. 提出一个 AlphaEvolve 可能会失效的领域。明确指出评估器会在何处崩溃及其原因。

5. 针对你熟悉的领域，编写你将使用的评估器签名 (evaluator signature)。需包含：(a) 正确性条件 (correctness conditions)，(b) 性能指标 (performance metric)，(c) 留出输入生成规则 (held-out input generation rule)，(d) 至少一项反奖励欺骗检查 (anti-reward-hacking check)。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| AlphaEvolve | “DeepMind 的进化式代码智能体” | Gemini + 程序数据库 + 机器可验证评估器 |
| MAP-Elites | “保留多样性的档案库” | 以特征向量为索引的网格；每个单元格保存具备该描述符的最优变体 |
| 岛屿模型（Island model） | “并行进化的子种群” | 相互独立的种群定期迁移；防止算法早熟收敛 |
| 机器可验证评估器（Machine-checkable evaluator） | “确定性判定器” | 大语言模型无法作弊的单元测试、模拟器或基准测试——运行该循环的先决条件 |
| 奖励欺骗（Reward hacking） | “优化指标，而非目标” | 循环机制找到了一种在不完成预期任务的前提下最大化得分的捷径 |
| 种子程序（Seed program） | “初始起点” | 循环进化所基于的初始正确但非最优程序 |
| 保留评估集（Held-out evaluator） | “大语言模型未见过的评估数据” | 在评估阶段动态生成的输入数据，用于防止模型死记硬背 |

## 延伸阅读

- [Novikov 等人（2025）。AlphaEvolve：用于科学与算法发现的编程智能体（coding agent）](https://arxiv.org/abs/2506.13131) — 完整论文。
- [DeepMind 关于 AlphaEvolve 的博客](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — 包含实验结果的厂商技术解读。
- [AlphaEvolve 结果仓库](https://github.com/google-deepmind/alphaevolve_results) — 已发现的算法，包括仅需 48 次乘法的 4x4 矩阵乘法（matmul）。
- [Romera-Paredes 等人（2023）。基于大语言模型（LLM）程序搜索的数学发现（FunSearch）](https://www.nature.com/articles/s41586-023-06924-6) — 前身系统。
- [Anthropic — 负责任扩展政策 v3.0（2026 年 2 月）](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 将受评估器约束的自主性（evaluator-bound autonomy）确立为关键研究方向。
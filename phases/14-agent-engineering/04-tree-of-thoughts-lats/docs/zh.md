# 思维树（Tree of Thoughts）与 LATS（Language Agent Tree Search）：审慎搜索

> 单一的思维链（Chain-of-Thought）推理轨迹缺乏回溯空间。思维树（Tree of Thoughts）（Yao 等人，2023）将推理过程转化为树状结构，并在每个节点引入自我评估。LATS（Zhou 等人，2024）在蒙特卡洛树搜索（Monte Carlo Tree Search）框架下，将 ToT 与 ReAct 及反思（Reflexion）机制进行了统一。在“24点游戏”（Game of 24）任务中，准确率从思维链（CoT）的 4% 跃升至 ToT 的 74%；LATS 在 HumanEval 基准测试上的 pass@1 指标更是达到了 92.7%。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（智能体循环），第 14 阶段 · 03（反思）
**预计耗时：** 约 75 分钟

## 学习目标

- 将推理过程建模为搜索问题：节点代表“思维”，边代表“扩展”，价值代表“潜力”。
- 使用标准库实现一种带有自我评估评分的 ToT 风格广度优先搜索（Breadth-First Search, BFS）树搜索。
- 扩展为一个包含 select / expand / simulate / backpropagate 步骤的简易 LATS 蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）循环。
- 判断何时值得为搜索付出额外的 Token 开销（如 24 点游戏、代码生成），以及何时单条推理轨迹已足够（如简单问答）。

## 核心问题

思维链（Chain-of-Thought）本质上是一种线性推导过程。如果第一步出错，后续所有步骤都将建立在错误的前提之上。在“24点游戏”（使用四个数字通过加减乘除运算得到 24）中，GPT-4 的思维链（CoT）准确率仅为 4%。模型在早期选错了子表达式，且无法自行纠正。

推理过程真正需要的是能够提出多个候选方案、对其进行评估、筛选出有潜力的分支，并在遇到死胡同时进行回溯的能力。这正是搜索的核心思想。思维树（Tree of Thoughts）与 LATS 便是该理念的两种经典实现范式。

## 核心概念

### 思维树（Tree of Thoughts, Yao 等, NeurIPS 2023）

每个节点代表一个连贯的中间步骤（即“一个想法”）。每个节点可扩展出 K 个子想法。大语言模型（LLM）通过评分提示词对每个节点进行自我评估。搜索算法在树中探索——可采用广度优先搜索（BFS）、深度优先搜索（DFS）或束搜索（Beam Search）。

                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish

自我评估是其中的核心支撑环节。论文展示了三种变体：`sure / likely / impossible`（确定/可能/不可能）分类、`1..10` 数值评分，以及候选方案投票。在“24点游戏”（Game of 24）任务中，这三种方法均大幅超越了思维链（CoT）（使用 GPT-4 时准确率从 4% 提升至 74%）。

### LATS（Zhou 等, ICML 2024）

LATS（Language Agent Tree Search）在蒙特卡洛树搜索（MCTS）框架下统一了思维树（ToT）、推理与行动框架（ReAct）和自我反思框架（Reflexion）。大语言模型在其中扮演三种角色：

- **策略（Policy）**：提出候选的下一步动作（ReAct 风格）。
- **价值函数（Value function）**：对部分轨迹进行评分（ToT 风格的自我评估）。
- **自我反思器（Self-reflector）**：在失败时，撰写自然语言形式的反思（Reflexion 风格），并将其用于重新初始化未来的模拟推演（rollout）。

环境反馈（观测值）会融入价值函数中，从而使搜索过程基于真实的工具执行结果，而非仅依赖模型的主观判断。论文发表时的结果：使用 GPT-4 在 HumanEval 上的 pass@1 达到 92.7%（达到当时最优 SOTA），使用 GPT-3.5 在 WebShop 上的平均分为 75.9（已接近基于梯度的微调水平）。

### MCTS 极简原理

每次迭代包含四个阶段：

1. **选择（Select）**——使用树的上置信界（UCT）从根节点遍历至叶节点。
2. **扩展（Expand）**——通过策略生成 K 个子节点。
3. **模拟（Simulate）**——从子节点出发使用策略进行模拟推演（rollout），并利用价值函数（或环境奖励）对叶节点进行评分。
4. **反向传播（Backpropagate）**——沿路径向上更新访问次数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项代表利用（exploitation）；第二项代表探索（exploration）。需根据具体任务调整参数 `c`。

### 成本现实

搜索算法会急剧消耗 Token。在“24点游戏”中，ToT 消耗的 Token 数量是 CoT 的 100 到 1000 倍。LATS 的情况类似。这并非没有代价；建议仅在以下场景保留搜索策略：

- 单条轨迹明显不足以解决问题的任务（如“24点游戏”、复杂代码生成）。
- 对正确性的要求高于实际运行时间（wall-clock）的任务。
- 具备低成本且可靠的价值函数的任务（如代码的单元测试、数学题的明确目标）。

如果你的任务只有一个正确答案，且评估器存在噪声，搜索往往会适得其反——它可能会找到一个“评分很高”的错误答案。

### 2026 年定位

大多数生产环境中的智能体（Agent）并未运行 LATS。它们通常采用 ReAct 结合基于工具的验证机制（如 CRITIC，见第 05 课）。搜索策略主要出现在以下特定细分领域：

- 将运行测试作为价值函数的代码智能体（HumanEval 风格）。
- 探索多条查询路径的深度研究智能体。
- LangGraph 子图内重度依赖规划的工作流。

AlphaEvolve（第 11 课）代表了 2025 年的极致探索：在代码空间进行进化搜索（evolutionary search），采用机器可验证的适应度（fitness），并取得了前沿突破（实现了 56 年来首次 4x4 矩阵乘法（matmul）优化）。

## 构建它

`code/main.py` 实现了：

- 在一个抽象化的“选择算术运算符”任务上实现的微型思维树（Tree of Thoughts, ToT）广度优先搜索（Breadth-First Search, BFS）。
- 在同一任务上实现的简易语言智能体树搜索（Language Agent Tree Search, LATS）蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）循环（选择 / 扩展 / 模拟 / 反向传播），并采用上限置信区间树（Upper Confidence Bound for Trees, UCT）进行选择。
- 一个价值函数（Value Function），该函数由符号评分与自我评估评分组合而成。

运行方式：

python3 code/main.py

运行日志显示，ToT 使用 BFS 在每个节点扩展三个候选项；相比之下，LATS 通过 MCTS 收敛至最优的模拟轨迹（Rollout）。两者均会打印 Token 消耗量。

## 使用它

LangGraph 以子图模式的形式内置了 ToT 风格的探索机制；LangChain 团队关于 LATS 的博客（2024 年 5 月）是权威参考教程。LlamaIndex 则内置了 `TreeOfThoughts` 智能体。对于大多数 2026 年的生产级智能体而言，该模式通常受 `if task_complexity > threshold: use_search()` 条件门控保护——具体可参考第 05 课中的评估器-优化器（Evaluator-Optimizer）模式。

## 部署它

`outputs/skill-search-policy.md` 会根据任务形态、预算限制以及评估器保真度，在线性 ReAct、ToT、LATS 与进化搜索（Evolutionary Search）之间进行策略选择。

## 练习

1. 分别使用 UCT 参数 `c=0.1` 和 `c=2.0` 运行该简易 LATS。观察运行日志有何变化？
2. 将价值函数替换为噪声更大的评分器（添加随机抖动）。MCTS 是否仍能找到最优叶子节点？它能容忍的最低信噪比是多少？
3. 实现集束搜索（Beam Search）版 ToT（在每一层保留 top-k 个候选项），并与 BFS 进行对比。在 Token 预算紧张的情况下，哪种方法表现更优？
4. 阅读 LATS 论文的第 5.1 节。复现 HumanEval 的轨迹计数：需要多少次模拟轨迹才能达到论文报告的 pass@1 指标？
5. 阅读 LATS 论文中关于“LATS 何时收益较低”的讨论。撰写一段决策规则，将任务形态映射至相应的搜索策略。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Tree of Thoughts（思维树） | “分支式思维链” | Yao 等人提出 —— 带有自我评估机制的思维节点树 |
| LATS（语言智能体树搜索） | “大模型的 MCTS” | Zhou 等人提出 —— 在 MCTS 框架下统一了 ToT、ReAct 与 Reflexion |
| UCT（上限置信区间树） | “上限置信区间” | 节点选择公式，用于平衡利用（Q 值）与探索（ln N / n） |
| Value function（价值函数） | “当前状态有多好” | 基于提示的 LLM 评分或环境奖励；用于反向传播 |
| Policy（策略） | “动作提议器” | ReAct 风格的生成器；输出候选的下一步思维或动作 |
| Rollout（模拟轨迹） | “模拟路径” | 使用策略从节点遍历至叶子节点，并用价值函数评分 |
| Backpropagate（反向传播） | “更新祖先节点” | 将叶子节点的奖励沿路径向上回传，更新访问次数与 Q 值 |
| Search cost（搜索成本） | “Token 爆炸” | 在 24 点游戏中可达 CoT 的 100-1000 倍；采用前需评估预算 |

## 延伸阅读

- [Yao 等人，思维树（Tree of Thoughts）(arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — 该领域的奠基性论文
- [Zhou 等人，LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — 结合 Reflexion 反馈（Reflexion feedback）的蒙特卡洛树搜索（MCTS）
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 面向搜索任务的子图模式（subgraph patterns）
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 结合可编程评估器（programmatic evaluators）的进化搜索（evolutionary search）
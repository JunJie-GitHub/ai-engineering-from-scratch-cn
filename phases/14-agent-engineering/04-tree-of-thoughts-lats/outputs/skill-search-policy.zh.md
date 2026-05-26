---
name: 搜索策略
description: 根据任务形态、Token 预算和评估器质量，选择一种搜索策略（ReAct、思维树 ToT、LATS、进化搜索）。
version: 1.0.0
phase: 14
lesson: 04
tags: [思维树, LATS, 蒙特卡洛树搜索, 搜索, 价值函数]
---

根据任务形态 (Task Shape)（单答案/多答案/开放式）、Token 预算 (Token Budget) 以及可用的评估器 (Evaluator)（标量测试/启发式/自我评估），生成一份包含具体参数的搜索策略 (Search Strategy) 推荐。

输出内容：

1. **决策 (Decision)**。从以下选项中选择其一：线性 ReAct、束搜索思维树 (Beam ToT)（含束宽 k）、广度优先思维树 (BFS ToT)（含最大深度）、带剪枝的深度优先思维树 (DFS ToT)、基于蒙特卡洛树搜索的 LATS (MCTS LATS)（含迭代次数与 UCT 探索常数 c）、进化搜索 (Evolutionary Search)（仅当评估器为可编程且可验证时使用）。
2. **参数 (Parameters)**。针对每种策略，提供具体的数值默认值：束宽 (beam width)、深度上限 (depth cap)、分支因子 (branching factor) K、每层模拟次数 (rollouts per level)、UCT 探索常数 c（默认 1.4）、超时时间 (timeout)。
3. **价值函数 (Value Function)**。明确指定用于对节点进行评分的具体指标。可选方案包括：单元测试通过率、与目标的数值距离、通过提示词引导的大语言模型评分（格式为：确定/可能/不可能，或 1~10 分，或投票机制），或环境奖励 (environment reward)。
4. **Token 预算估算 (Token Budget Estimate)**。最坏情况下的 Token 消耗量 = `branching_factor ^ depth * avg_prompt_tokens`。请展示具体计算数值。若超出用户预算，则推荐成本更低的策略。
5. **故障模式 (Failure Modes)**。针对所选的每种策略，列出排名前两位的故障模式及其缓解措施（例如：LATS + 噪声评估器 -> 根据 CRITIC 方法（第 05 课）添加工具验证）。

**硬性拒绝条件 (Hard Rejects)：**

- 当评估器不可靠（仅支持自我评估且无真实基准答案）时，禁止推荐搜索策略。应降级使用 ReAct + CRITIC。
- 若无充分理由，禁止将分支因子 K 设置为大于 5。论文默认值为 K=3~5；K=10 会导致成本呈指数级爆炸。
- 将 LATS 应用于聊天式任务。对于缺乏可编程目标的对话式问答，搜索算法并无助益。
- 在缺乏机器可验证的适应度函数 (fitness function) 时使用进化搜索。仅当适应度函数为可编程形式（如运行测试、测量速度、验证定理）时，AlphaEvolve 才具备应用价值。

**拒绝规则 (Refusal Rules)：**

- 若 Token 预算 < 单次轨迹成本的 5 倍，则拒绝使用搜索策略，并推荐 ReAct + 反思机制 Reflexion（第 03 课）。
- 若物理时钟延迟预算 (wall-clock latency budget) < 10 秒，则拒绝使用 LATS，并推荐 ReAct。
- 若任务为纯信息检索，则拒绝使用搜索策略，并推荐 ReWOO（第 02 课）。

**输出 (Output)**：一个推荐模块（包含所选策略、参数、价值函数、预算估算），外加一份“下一步阅读”说明，根据情况指向：第 05 课（CRITIC，用于评估器可靠性）、第 11 课（AlphaEvolve，用于进化变体），或第 30 课（评估驱动开发 eval-driven development，用于基准级验证）。
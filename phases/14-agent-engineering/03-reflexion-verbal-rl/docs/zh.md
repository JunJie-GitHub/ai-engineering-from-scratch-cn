# Reflexion：语言强化学习

> 基于梯度的强化学习（Gradient-based RL）需要数千次试验和 GPU 集群才能修复一种失败模式。Reflexion（Shinn 等人，NeurIPS 2023）则通过自然语言实现这一目标：在每次失败的试验后，智能体（Agent）会撰写一段反思，将其存储在情景记忆（Episodic Memory）中，并在下一次试验时以该记忆为条件进行推理。这正是 Letta 的休眠期计算、Claude Code 的 CLAUDE.md 学习机制以及 pro-workflow 的 learn-rule 背后的核心模式。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（智能体循环），第 14 阶段 · 02（ReWOO）
**预计耗时：** 约 60 分钟

## 学习目标

- 指出 Reflexion 的三个组成部分：执行者（Actor）、评估器（Evaluator）与自我反思器（Self-Reflector），并说明情景记忆的作用。
- 使用标准库实现一个 Reflexion 循环，包含二值评估器（Binary Evaluator）、反思缓冲区（Reflection Buffer）以及全新的重试机制。
- 针对给定任务，在标量（Scalar）、启发式（Heuristic）与自我评估（Self-evaluated）反馈源之间做出选择。
- 解释为何语言强化（Verbal Reinforcement）能够捕捉到那些基于梯度的强化学习需要数千次试验才能修复的错误。

## 问题背景

当智能体在任务中失败时，在标准强化学习中，你通常需要运行数千次额外试验、计算梯度并更新权重。这种方式成本高昂且速度缓慢，而且大多数生产环境中的智能体并没有为每次失败都分配训练预算。

Reflexion（Shinn 等人，arXiv:2303.11366）提出了一个不同的思路：如果智能体只是思考一下失败的原因，并将这一思考放入提示词（Prompt）中再次尝试会怎样？无需更新权重，无需计算梯度。只需在试验之间存储自然语言即可。

实验结果表明：在 ALFWorld 基准测试中，它击败了 ReAct 及其他未经微调的基线模型；在 HotpotQA 上，其表现优于 ReAct；在代码生成任务（HumanEval/MBPP）中，它刷新了当时的业界最佳水平（State of the Art）。而这一切，均未使用任何梯度下降步骤。

## 核心概念

### 三大核心组件

Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure

外加一个数据结构：

Episodic memory: list of prior reflections, prepended to the next trial's prompt

单次试验（trial）会运行 Actor。Evaluator 对其进行打分。若得分较低，Self-Reflector 会生成一段自然语言形式的反思（例如：“我选错了工具，因为我把问题误读为询问 X，而实际上它问的是 Y”）。该反思会被存入情景记忆（episodic memory）中。下一次试验会重新开始，但会读取到之前的反思。

### 三种评估器类型

1. **标量型（Scalar）**——外部二元信号。ALFWorld 任务成功或失败，HumanEval 测试通过或未通过。最简单，信号最强。
2. **启发式（Heuristic）**——预定义的失败特征。例如：“如果智能体连续两次执行相同动作，则标记为卡死。”“如果轨迹（trajectory）超过 50 步，则标记为低效。”
3. **自评估型（Self-evaluated）**——由大语言模型（LLM）对自己的轨迹进行打分。适用于缺乏真实标签（ground truth）的场景。信号较弱；适合与基于工具验证的方法结合使用（参见第 05 课 — CRITIC）。

2026 年的默认配置是混合模式：有明确信号时采用标量型，无信号时采用自评估型，并以启发式规则作为安全护栏。

### 为何具备泛化能力

Reflexion（反思机制）与其说是一种新算法，不如说是一种被命名的设计模式。几乎每个生产环境中的“自愈型”智能体都在运行某种变体：

- Letta 的休眠期计算（第 08 课）：由独立的智能体对历史对话进行反思，并将结果写入记忆块。
- Claude Code 的 `CLAUDE.md` / “保存记忆”模式：将反思内容作为经验教训捕获，并附加到未来会话的提示词开头。
- pro-workflow 的 `/learn-rule` 命令：将修正内容捕获为明确的规则。
- LangGraph 的反思节点（reflection nodes）：一个用于对输出进行打分，并在需要时路由至优化流程的节点。

它们都源于同一个核心洞察：自然语言是一种足够丰富的媒介，足以在多次运行之间传递“我从失败中学到了什么”。

### 适用场景与局限性

Reflexion 在以下情况下有效：

- 存在明确的失败信号（如测试失败、工具报错、答案错误）。
- 任务类别具有可复现性（可以再次提出同类问题）。
- 反思内容能够对执行轨迹产生改进空间（具备足够的动作预算）。

Reflexion 在以下情况下无效：

- 智能体在首次尝试时就已经成功。
- 失败源于外部因素（如网络中断、工具故障）——反思“网络曾中断”对未来的运行毫无帮助。
- 反思演变为“迷信”——存储了关于某次偶然不稳定运行的叙事性描述。

2026 年的常见陷阱：记忆腐化（memory rot）。反思内容不断累积，其中部分已过时或错误；随着情景缓冲区的增长，重新运行的速度会变慢。缓解策略：定期压缩（第 06 课）、为反思设置生存时间（TTL），或使用独立的休眠期清理智能体（如 Letta）。

## 动手构建

`code/main.py` 在一个示例谜题上实现了反思机制（Reflexion）：生成一个总和等于目标值的三元素列表。`Actor` 输出候选列表；`Evaluator` 检查总和；`Self-Reflector` 会写下一行关于出错原因的诊断。反思内容会存入情景记忆（Episodic Memory），供下一次尝试使用。

组件：

- `Actor` — 一种脚本化策略，在看到反思内容后会进行自我改进。
- `Evaluator.binary()` — 根据目标总和判断通过或失败。
- `SelfReflector` — 生成一行关于失败原因的诊断。
- `EpisodicMemory` — 具有 TTL（Time-To-Live，生存时间）语义的有界列表。

运行方式：

python3 code/main.py

运行轨迹显示共进行了三次尝试。第一次尝试失败，系统存储了一条反思内容；第二次尝试读取了该反思并有所改进，但仍告失败；第三次尝试成功。与基线运行（无反思机制）对比——基线会一直卡在第一次尝试的答案上。

## 使用方式

LangGraph 将反思机制作为节点模式提供。Claude Code 的 `/memory` 命令与 pro-workflow 的 `/learn-rule` 会将情景缓冲区外部化为 Markdown 文件。Letta 的休眠期计算会在空闲时段运行自我反思器，从而确保主智能体保持延迟受限（Latency-bound）状态。OpenAI Agents SDK 未直接内置反思机制；你需要通过自定义护栏（Guardrail）来构建它，该护栏会根据评分拒绝特定执行轨迹，并配合一个跨运行持久化的内存 `Session`。

## 交付

`outputs/skill-reflexion-buffer.md` 用于创建并维护一个情景缓冲区，支持反思内容捕获、TTL 和去重功能。在给定任务类别和失败场景时，它会输出一条能切实帮助下一次尝试的反思内容（而非泛泛而谈的“下次更仔细点”）。

## 练习

1. 将评估器从二分类切换为标量评估器，使其返回距离度量（与目标值的差距）。这是否会加快收敛速度？
2. 为反思内容添加 10 次尝试的 TTL。超过该期限后，旧的反思内容是会产生负面影响还是仍有积极作用？
3. 实现启发式评估器：如果重复执行相同动作，则将该次尝试标记为陷入僵局。这会如何与自我反思器产生交互？
4. 搭配一个会忽略反思内容的对抗性执行器（Adversarial Actor）运行反思机制。需要最低限度的何种提示词工程（Prompt Engineering），才能迫使执行器注意到这些反思？
5. 阅读关于 AlfWorld 的 Reflexion 论文第 4 节。从概念上复现其 130% 成功率提升的效果：与原版 ReAct 相比，关键差异（Delta）是什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 反思机制 (Reflexion) | “自我纠正” | Shinn 等人 2023 年提出 — 包含执行者 (Actor)、评估器 (Evaluator)、自我反思器 (Self-Reflector) 以及情景记忆 (Episodic Memory) |
| 语言强化 (Verbal Reinforcement) | “无梯度学习” | 将自然语言形式的反思内容前置到下一次尝试的提示词 (Prompt) 开头 |
| 情景记忆 (Episodic Memory) | “按任务划分的反思” | 针对单一任务类别，存储历史反思内容的有界缓冲区 (Bounded Buffer) |
| 标量评估器 (Scalar Evaluator) | “二元成功信号” | 基于基准事实 (Ground Truth) 输出的通过/失败判定或数值评分 |
| 启发式评估器 (Heuristic Evaluator) | “基于模式的检测器” | 预定义的失败特征（例如：陷入死循环、步骤过多） |
| 自我评估器 (Self-Evaluator) | “对自身轨迹的 LLM 裁判 (LLM-as-Judge)” | 在缺乏基准事实时的低信号回退方案 — 建议与基于工具的验证方法结合使用 |
| 记忆腐化 (Memory Rot) | “过时的反思” | 情景缓冲区被陈旧条目填满；可通过数据压缩 (Compaction) 或设置生存时间 (TTL) 解决 |
| 休眠期反思 (Sleep-time Reflection) | “异步自我反思” | 将自我反思器移出热路径 (Hot Path)，以确保主智能体保持高速响应 |

## 扩展阅读

- [Shinn 等人，《Reflexion：基于语言强化学习的语言智能体》(arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 该领域的奠基性论文
- [Letta，《休眠期计算》](https://www.letta.com/blog/sleep-time-compute) — 生产环境中的异步反思机制
- [Anthropic，《AI 智能体的高效上下文工程》](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将情景缓冲区作为上下文的一部分进行管理
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 反思节点模式
# 心智社会（Society of Mind）与多智能体辩论（Multi-Agent Debate）

> 明斯基（Minsky）在 1986 年提出的前提——智能是由众多专家组成的社会——每隔十年就会被重新发现一次。2023 年，Du 等人将其转化为一种具体算法：多个大语言模型（Large Language Model）实例分别提出答案，阅读彼此的答案，进行批判并更新。经过 N 轮交互，它们会收敛到一个共识，该共识在六项推理与事实性任务上的表现优于零样本思维链（Zero-shot Chain-of-Thought）和反思（Reflection）方法。两项发现至关重要：**多智能体（Multiple Agents）**与**多轮次（Multiple Rounds）**均能独立产生贡献。群体协作胜过单智能体独白；多轮交互胜过单次投票。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 16 阶段 · 04（基础模型）
**Time:** 约 60 分钟

## 问题

自一致性（Self-consistency）——对单一模型进行多次采样并采纳多数答案——是你能以最低成本附加的推理能力优化手段。该方法确实有效，但性能提升很快就会达到饱和。即使将采样量翻倍，你也很难观察到进一步的显著跃升。

辩论机制打破了这一饱和瓶颈。它不再依赖单一模型的 N 次独立采样，而是让 N 个智能体交叉阅读彼此的推理过程并进行修正。样本间的相关性随之降低（它们不再是独立同分布（Independent and Identically Distributed）的），并且在独立同分布投票曾自信地给出错误答案的场景下，收敛点往往能得出正确结果。

## 概念

### Du 等人（2023）算法

源自 arXiv:2305.14325（ICML 2024）：

1. N 个智能体（Agent）各自针对问题生成初始答案。
2. 对于第 r = 2..R 轮：向每个智能体展示其他智能体在第 r-1 轮的答案，并要求其“参考这些内容，给出你更新后的答案。”
3. 经过 R 轮后，对最终答案进行多数投票（Majority Vote）。

该论文在 MMLU、GSM8K、传记（Biographies）、MATH 以及事实性（Factuality）基准测试上进行了评估。辩论（Debate）方法始终优于思维链（Chain of Thought, CoT）和自我反思（Self-Reflection）。

### 两个独立控制变量

同一论文的消融实验（Ablation Study）表明：

- **仅增加智能体数量**（1 轮，N 个智能体多数投票）在大多数任务上优于单智能体，但性能会趋于饱和。
- **仅增加轮数**（单个智能体查看自己先前的推理过程）几乎无济于事——这是反思机制已知的弱点。
- **两者结合**则能带来显著的性能跃升。多智能体之间的多轮交互是性能提升的核心驱动力。

### 为何有效

两种机制：

1. **接触分歧（Exposure to Disagreement）**。当智能体看到另一个智能体得出不同结论的推理链（Reasoning Chain）时，它必须为自己的观点辩护或更新答案。无论哪种情况，第 r+1 轮的上下文（Context）都比第 r 轮更加丰富。
2. **降低相关性误差（Correlated Error Reduction）**。在自洽性（Self-Consistency）方法中，所有样本均来自同一模型，因此误差具有相关性——最终往往会平均出一个“自信但错误”的答案。使用不同模型或不同随机种子可以降低相关性。而不同的*辩论观点*能进一步实现误差去相关。

### 异构辩论（Heterogeneous Debate）

A-HMAD 及相关后续研究为不同智能体使用了*不同的基础模型（Base Models）*。Llama + Claude + GPT 共同辩论能够减少单一文化崩溃（Monoculture Collapse，见第 26 课），因为某一模型家族的相关性误差不会被其他模型共享。

缺点：参与辩论的弱模型可能会将共识拉向其错误答案（参见《Should we be going MAD?》，arXiv:2311.17371）。

### NLSOM —— 129 智能体扩展版

Zhuge 等人（《Mindstorms in Natural Language-Based Societies of Mind》，arXiv:2305.17066）将该思路扩展至包含 129 个成员的群体。结果表明：随着规模扩大，系统会涌现出专业化（Specialization）与自组织（Self-Organization）特性，并在视觉问答（Visual Question Answering）等任务上超越单智能体系统。

### 失败模式（Failure Modes）

- **阿谀级联（Sycophancy Cascade）**。所有智能体都倾向于顺从听起来最自信的那个智能体。辩论最终退化为“谁声音大听谁的”。通过提示词设定对抗性角色（例如“指定一个智能体必须持反对立场”）有助于缓解此问题。
- **话题漂移（Topic Drift）**。经过多轮辩论后，讨论容易偏离原始问题。缓解方法：每轮重新注入原始问题。
- **计算量激增（Compute Blowup）**。N 个智能体 × R 轮 = N·R 次大语言模型（LLM）调用，且每次调用的上下文都在增长。一场 5 智能体、5 轮的辩论需要进行 25 次调用，且上下文不断累积。单个问题的成本可能超过单次 CoT 调用的 10 倍。

## 动手实践

`code/main.py` 运行一个针对数学问题的 3 智能体（agent）× 3 轮（round）辩论，其中每个智能体初始持有不同的（可能错误的）答案。智能体的行为由脚本控制——每个智能体通过按预设置信度（confidence）加权平均相邻智能体的答案来进行“更新”。收敛（convergence）过程在逐轮日志中清晰可见。

该演示展示了两个关键效应：

- 单轮交流即可使智能体更接近正确答案。
- 第 2 轮之后的额外轮次表现出收益递减（diminishing returns）（与 Du 等人观察到的平台期现象一致）。

运行：

python3 code/main.py

## 使用它

`outputs/skill-debate-configurator.md` 用于为新任务配置辩论参数：智能体数量、辩论轮数、异构性（heterogeneity）（单一模型 vs 混合模型）、角色分配（对称角色 vs 单一反对者）。该文件还会在运行前预估 Token 消耗成本。

## 部署上线

若要将辩论系统投入生产：

- **将轮数上限设为 3。** Du 等人的研究表明，3 轮即可捕获绝大部分性能增益。增加轮次只会推高成本，而非提升质量。
- **将智能体数量上限设为 5。** 超过 5 个后，上下文膨胀（context bloat）与成本开销将成为主要瓶颈。
- **默认采用异构配置。** 智能体池中至少应包含两种不同的基础模型（base model）。
- **预留反对者席位。** 提示其中一个智能体始终持反对意见。此举可有效打破阿谀奉承（sycophancy）效应。
- **完整记录每轮日志。** 隐藏中间轮次的辩论系统将无法进行调试或审计。

## 练习

1. 运行 `code/main.py`，随后将轮数设为 5 并观察收益递减现象。额外的收敛在第几轮趋于停滞？
2. 添加第四个担任反对者角色的智能体：始终与当前多数派意见相左。此举会破坏还是改善收敛过程？
3. 绘制（或打印）每轮的共识得分（agreement score）（即持多数派答案的智能体占比）。该得分何时达到 1.0？达到 1.0 是否等同于答案“正确”？
4. 阅读 Du 等人论文的第 4 节消融实验（ablation study）。使用本代码复现“仅增加智能体”、“仅增加轮次”与“两者同时增加”的对比结果。
5. 阅读论文《Should we be going MAD?》(arXiv:2311.17371)，并列举两种除循环辩论（round-robin）之外的辩论变体——例如：裁判主导型、辩论链（chain-of-debate）、对抗型。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 心智社会（Society of Mind） | “明斯基的理论” | 智能由相互作用的专家模块构成；1986 年的理论框架如今通过大语言模型（LLM）辩论得以工程化实现。 |
| 多智能体辩论（Multi-agent debate） | “智能体争论” | N 个智能体提出观点、相互批判、在 R 轮中修订，最终进行多数投票。 |
| 共识（Consensus） | “他们达成一致” | 并非认识论上的真理——仅是持多数派答案的比例。可能集体自信地犯错。 |
| 轮次（Rounds） | “交换步骤” | 一轮 = 每个智能体读取其他智能体的输出并更新一次自身状态。 |
| 异构辩论（Heterogeneous debate） | “混合模型家族” | 使用不同的基础模型以解耦错误（decorrelate errors）。 |
| 阿谀奉承级联（Sycophancy cascade） | “大家都附和嗓门最大的” | 辩论失效模式：智能体无视正确性，盲目顺从置信度最高的智能体。 |
| 自然语言心智社会（NLSOM） | “129 个智能体的社会” | 基于自然语言的心智社会；Zhuge 等人提出的规模化版本。 |
| 相关性错误（Correlated error） | “同模型，同缺陷” | 自一致性（self-consistency）为何会饱和；跨不同视角的辩论可解耦此类错误。 |

## 延伸阅读

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — 参考论文，ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129 个智能体的 NLSOM（基于自然语言的思维社会）
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — 对大语言模型 (Large Language Models) 的多智能体辩论 (Multiagent Debate) 策略变体进行基准测试
- [Debate project page](https://composable-models.github.io/llm_debate/) — Du 等人的代码、演示及消融实验 (Ablation Study) 细节
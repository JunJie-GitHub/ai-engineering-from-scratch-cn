# 投票 (Voting)、自一致性 (Self-Consistency) 与辩论拓扑结构 (Debate Topology)

> 最廉价的聚合 (aggregation) 方式：采样 N 个独立智能体 (agent)，进行多数投票 (majority vote)。Wang 等人（2022）提出的自一致性 (Self-Consistency) 方法正是通过对单一模型采样 N 次来实现这一点的。多智能体 (multi-agent) 系统通过引入**异构** (heterogeneous) 智能体对其进行了扩展，以摆脱同质化生态 (monoculture)——即采用不同的模型、不同的提示词 (prompt)、不同的温度参数 (temperature) 以及不同的上下文 (context)。除了多数投票之外，辩论拓扑结构 (Debate Topology) 同样至关重要：MultiAgentBench (arXiv:2503.01935, ACL 2025) 评估了星型/链式/树状/图状协调 (coordination) 模式，发现**图状结构最适合研究任务**，但当智能体数量超过约 4 个时会出现“协调税” (coordination tax)。AgentVerse (ICLR 2024) 记录了两种涌现模式 (emergent pattern)——自愿行为 (volunteer behavior) 与从众行为 (conformity behavior)——其中从众行为既是一种特性（有助于达成共识），也是一种风险（可能导致群体思维 (groupthink)，参见第 24 课）。本课将绘制拓扑结构空间图谱，构建每种变体，并量化协调税。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 16 阶段 · 07（心智社会与辩论），第 16 阶段 · 14（共识与拜占庭容错 (BFT)）
**Time:** 约 75 分钟

## 问题

辩论能够提升准确率（Du 等人，arXiv:2305.14325），但也可能导致准确率下降。辩论是否有效取决于以下四个结构性选择：

1. 谁与谁对话（拓扑结构 (topology)）。
2. 进行多少轮交互（Du 2023 指出：轮次与智能体数量各自独立产生影响）。
3. 智能体是否异构（不同的基座模型可打破同质化生态）。
4. 是否存在对抗性声音 (adversarial voice)（强化论证 (steel-manning) 与 稻草人论证 (straw-manning) 的对比）。

许多团队简单粗暴地将“运行 5 个智能体并投票”套用到任务中，其表现往往反而不如单个智能体。这些失败并非偶然，而是与拓扑结构和异构性密切相关。本课即是一份拓扑结构指南。

## 概念

### 自洽性（Self-consistency），单模型基线

Wang 等人于 2022 年发表的论文《Self-Consistency Improves Chain of Thought Reasoning》在温度（temperature）> 0 的条件下对同一模型进行 N 次采样，并对推理路径（reasoning-path）的答案进行多数投票。在 GSM8K 数据集上的结果表明：与单次贪婪解码（greedy decode）相比，N=40 次采样带来了显著的性能提升。自洽性是多智能体投票（multi-agent voting）在单智能体阶段的先驱。

局限性：自洽性仅使用单一基础模型（base model）。其错误在结构上是相关的（correlated）。如果模型存在系统性偏差（systematic bias），所有 N 个样本都会共享这一偏差。

### 多智能体投票（Multi-agent vote），异构扩展

将 N 次采样替换为 N 个*不同*的智能体（agents）。这些智能体可以基于不同的基础模型（如 Claude、GPT、Llama），使用不同的提示词（prompts），并具备不同的工具访问权限。其优势在于：错误互不相关（uncorrelated errors）。其代价在于：不同智能体的调用成本各异，且协调它们会带来额外的开销（overhead）。

异构辩论（heterogeneous debate）在 2026 年的标准名称是 **A-HMAD**（对抗性异构多智能体辩论，Adversarial Heterogeneous Multi-Agent Debate）。该术语尚未被普遍采用，但相关论文常用它来指代“不同模型之间的辩论，旨在减少因单一模型文化崩溃（monoculture collapse）导致的关联错误”。

### 四种拓扑结构（Topologies）

star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)

星型（Star）：一个中心节点（hub），其他所有节点仅与中心节点通信。相当于没有反向通道（back-channel）的主管-工作者（supervisor-worker）模式。
链型（Chain）：线性结构，每个智能体只能看到前一个智能体的输出。类似于流水线（pipeline）。
树型（Tree）：层级结构，常用于分层智能体系统（hierarchical agent systems）（参见第 06 课）。
图型（Graph）：任意节点间均可通信。包含全连接团（fully-connected clique）和任意有向无环图（DAGs）。

### 协调开销（Coordination tax）（MultiAgentBench）

MultiAgentBench（MARBLE，ACL 2025，arXiv:2503.01935）在包含研究、编程和规划的任务套件上对星型、链型、树型和图型拓扑进行了基准测试。关键实测结果如下：

- **图型（Graph）**拓扑在研究类任务中表现最佳。信息可在任意节点间流动，智能体能够相互评审。
- **星型（Star）**在快速回答的事实类任务中胜出。中心节点负责过滤与整合信息。
- **链型（Chain）**在分步流水线（阶段性优化）任务中表现最优。
- 当图型拓扑中的智能体数量超过约 4 个时，**协调开销（coordination tax）**开始显现。实际耗时（wall-clock）和 Token 成本的增长速度超过了质量提升的速度。

4 个智能体的上限是经验性的，而非根本性的。它反映了 2026 年大语言模型（LLM）的上下文容量限制：每个智能体的上下文窗口会被其他智能体的输出填满，一旦所有智能体都能看到彼此的信息，增加第 N+1 个智能体的边际价值就会下降。

### 多智能体辩论策略（Multi-Agent Debate Strategies）（“我们是否应该走向 MAD？”）

arXiv:2311.17371 是 2023 年关于多智能体辩论（Multi-Agent Debate, MAD）策略的综述论文。其他研究复现的关键发现是：在相同预算下，与自洽性*结构相似*（独立采样 + 聚合）的 MAD 变体，其表现往往不如自洽性。当智能体真正具备异构性，且辩论具有对抗性结构（即一个智能体专门提出反对意见）时，MAD 的帮助最大。

### AgentVerse 中的涌现模式（Emergent patterns）

AgentVerse（ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf）记录了多智能体辩论中即使未经显式设计也会涌现出的两种行为：

- **主动请缨（Volunteer）**。智能体在未被提示的情况下主动提供帮助（“我可以进行下一步”）。其价值在于：它能将子任务分配给最有能力的智能体。
- **趋同（Conformity）**。即使批评者（critic）是错误的，智能体也会调整自身立场以迎合对方。这相当于辩论场景中的阿谀奉承（sycophancy）（参见第 14 课）。

趋同行为解释了为何“辩论至达成一致”的机制会奖励强势者。通过限制辩论轮数并引入独立的裁判（judge）可以缓解这一问题。

### 异构性（Heterogeneity）：真正能提升准确率的调节旋钮

2024-2026 年实践文献中的一个规律：将 N 个智能体中的一个替换为不同的基础模型，所带来的准确率提升幅度大于单纯将 N 增加 1。其背后的直觉在于避免单一文化（monoculture）——每一个新的独立错误源，其价值都高于增加一个关联样本。

在极限情况下，异构性胜过数量优势。在大多数具有清晰标准答案（ground truth）的任务中，三个不同模型的表现优于同一模型的五个副本。

### 陪审团方法（Jury methods）

Sibyl 框架（在 Minsky-LLM 文献中被引用）将“陪审团”形式化——由一组小型专业智能体组成，通过在每个阶段进行投票来优化答案。与简单的多数投票不同，陪审团具有明确的角色分工：一个智能体负责交叉质询，一个提供上下文，一个评估合理性。陪审团方法是简单投票（成本低，易陷入单一文化）与完整 MAD（成本高，易产生趋同）之间的折中方案。

### 何时“带辩论的投票”占据优势

- 问题具有明确的标准答案（如事实、数学、代码行为）。投票收敛具有实际意义。
- 智能体能够访问不同的数据源或工具（具备异构性条件）。
- 辩论轮数受限（通常为 2-3 轮），且设有独立的裁判或验证器。
- 预算允许使用 3-5 个智能体。在图型拓扑中，超过 5-7 个智能体后，协调开销将占据主导。

### 何时“带辩论的投票”适得其反

- 问题属于主观观点类。智能体会收敛到看起来最自信的答案，而非最正确的答案。
- 所有智能体共享同一个基础模型。单一文化使得达成共识变得毫无意义。
- 辩论轮数无限制。趋同行为每次都会胜出。
- 任务较为简单。使用单个智能体配合 N=5 的自洽性采样，成本更低且准确率相当。

## 构建它

`code/main.py` 实现了：

- `run_star(agents, hub, question)` — 中心节点（hub）轮询各工作节点（worker）并聚合结果。
- `run_chain(agents, question)` — 顺序迭代优化。
- `run_tree(root, children, question)` — 层级结构，包含深度为 2 的聚合。
- `run_graph(agents, question, rounds)` — 全互联辩论（all-to-all debate），限制轮次。
- 可脚本控制的异构性调节器（heterogeneity dial）：每个智能体（agent）均具备一个 `error_bias` 参数，用于指示其系统性错误倾向。
- 一套测量框架（measurement harness），可在 N=3、5、7 时运行每种拓扑结构（topology），并报告（accuracy、total_tokens、wallclock_simulated）。

运行：

python3 code/main.py

预期输出：一个展示“拓扑结构 × N →（准确率、token 数、延迟）”的表格。在研究型任务中，图拓扑（graph）在 N=3-5 时表现最佳；在快速事实型任务中，星型拓扑（star）胜出；当 N=7 时，图拓扑会显现出协调开销（coordination tax）（延迟的增长速度超过了准确率的提升）。

## 使用它

`outputs/skill-topology-picker.md` 是一个技能模块（skill），用于读取任务描述，并推荐拓扑结构（星型/链式/树状/图结构）、智能体数量 N、异构性配置（即使用的基础模型）以及最大轮次限制。

## 部署它

对于任何集成系统（ensemble）：

- 首先使用一个强大的基础模型，在 N=5 时进行**自一致性（self-consistency）**测试。这是一个低成本的基线方案。
- 如果准确率至关重要，可升级至 N=3 的**异构投票（heterogeneous voting）**。请量化其性能差异（delta）。
- 仅当任务具有结构化特征（如研究分析、多步骤推理）且限制轮次可行时，才升级至**辩论拓扑（debate topology）**。
- 始终记录少数派集群（minority cluster）。当少数派持续给出正确答案时，说明你捕捉到了多样性信号（diversity signal）。
- 在评估准确率的同时，务必对实际运行时间（wall-clock）和 token 消耗进行基准测试（benchmark）。“以 10 倍成本换取更高准确率”属于商业决策范畴。

## 练习

1. 运行 `code/main.py`。绘制图拓扑的协调开销曲线：准确率随 N 的变化、token 消耗随 N 的变化。曲线在哪个 N 值处发生拐点？
2. 实现 A-HMAD：配置三个具有刻意差异化偏差的智能体。在第 14 课的单一种类攻击（monoculture attack）中，全同偏差基线与 A-HMAD 的表现对比如何？
3. 在图拓扑中添加一个“裁判”角色，该角色不参与投票，仅对最终共识进行评分。这是否会改变涌现出的从众行为（emergent conformity behavior）？
4. 阅读 AgentVerse 论文（ICLR 2024）。找出你的实现中最显著地展现了哪种涌现行为（emergent behavior）。能否通过修改提示词（prompt）来诱发相反的行为？
5. 阅读 MultiAgentBench（arXiv:2503.01935）第 4 节（拓扑实验）。使用你的测量框架，复现论文中某项任务的“图拓扑在研究任务中胜出”的结果。

## 关键术语

| 术语 | 通俗理解 | 实际含义 |
|------|----------------|------------------------|
| 自洽性 (Self-consistency) | “采样 N 次，投票表决” | Wang 2022。使用单一模型，在 temperature > 0 的条件下采样 N 次，对推理路径进行多数投票。 |
| 异构性 (Heterogeneity) | “使用不同模型” | 集成不同的基础模型或提示词 (prompt) 系列。打破模型同质化。 |
| 多智能体辩论 (Multi-Agent Debate, MAD) | “多智能体辩论” | 智能体在多轮交互中相互提出批评的通用术语。参见 Du 2023。 |
| 对抗性异构多智能体辩论 (Adversarial Heterogeneous MAD, A-HMAD) | “对抗性异构 MAD” | MAD 的变体，强调使用不同模型并结合对抗性结构。 |
| 拓扑结构 (Topology) | “谁与谁通信” | 星型、链式、树状、图结构。决定信息流向。 |
| 协调开销 (Coordination tax) | “边际收益递减” | 在图结构中智能体数量超过约 4 个时，成本的增长速度将超过质量提升的速度。 |
| 主动行为 (Volunteer behavior) | “未经提示的协助” | AgentVerse 中涌现的模式：某个智能体主动提出执行某一步骤。 |
| 从众行为 (Conformity behavior) | “压力下的附和” | AgentVerse 中涌现的模式：某个智能体与批评者保持一致。 |
| 评审团 (Jury) | “小型专业小组” | 类似 Sibyl 的集成架构，包含特定角色（审查员、上下文提供者、评分员）。 |

## 扩展阅读

- [Wang 等人 — 自洽性提升思维链 (Chain of Thought) 推理能力](https://arxiv.org/abs/2203.11171) — 单模型基线
- [Du 等人 — 通过多智能体辩论 (Multiagent Debate) 提升事实准确性与推理能力](https://arxiv.org/abs/2305.14325) — 智能体数量与辩论轮数均具有独立影响
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 拓扑结构 (Topology) 基准测试，表明图结构最适合研究场景，链式结构最适合流水线
- [我们是否应该采用 MAD？](https://arxiv.org/abs/2311.17371) — MAD 策略综述；发现在同等预算下，MAD 通常不如自洽性 (Self-consistency) 策略
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — 主动行为 (Volunteer behavior) 与从众行为 (Conformity behavior) 的涌现模式
- [MARBLE 代码库](https://github.com/ulab-uiuc/MARBLE) — 基准测试参考实现
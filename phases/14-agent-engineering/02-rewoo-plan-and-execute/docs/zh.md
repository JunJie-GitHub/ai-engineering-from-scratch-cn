# ReWOO 与 Plan-and-Execute：解耦规划

> ReAct 将思考（thought）与行动（action）交织在单一流程中。ReWOO 则将二者分离：先制定整体规划，再执行。词元（token）消耗减少 5 倍，在 HotpotQA 数据集上准确率提升 4%，并且你可以将规划器（planner）蒸馏至 7B 参数模型中。规划与执行（Plan-and-Execute）模式对其进行了泛化；规划与行动（Plan-and-Act）模式则将其扩展至网页导航任务。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（智能体循环）
**耗时：** 约 60 分钟

## 学习目标

- 解释为何 ReWOO 的规划器（Planner）/ 执行器（Worker）/ 求解器（Solver）拆分架构相比 ReAct 的交织循环能节省词元并提升鲁棒性（robustness）。
- 实现一个规划有向无环图（DAG）、依赖排序执行器以及组合执行器输出的求解器——全部使用标准库（stdlib）。
- 结合 Anthropic 2026 年提出的“五种工作流模式（five workflow patterns）”框架，判断任务何时应采用“先规划后执行”模式，何时应采用交织式 ReAct 模式。
- 识别在长程（long-horizon）网页或移动端任务中，何时需要引入规划与行动（Plan-and-Act）的合成规划数据（synthetic plan data）。

## 问题背景

ReAct 的思考-行动-观察循环（thought-action-observation loop）简单且灵活，但每次工具调用都必须携带完整的先前上下文（context）——包括之前的每一次思考。词元消耗量随任务深度呈二次方增长。更糟糕的是：当工具在循环中途失败时，模型必须根据错误观察结果重新推导整个规划。

ReWOO（Xu 等人，arXiv:2305.18323，2023 年 5 月）注意到了这一问题并提出了一个方案：提前制定完整规划，并行获取证据，最后组合答案。整个流程仅需一次大语言模型（LLM）调用用于规划，N 次工具调用用于获取证据（可并行执行），以及一次 LLM 调用用于求解。这种设计以牺牲部分灵活性（规划为静态）为代价，换取了更高的词元效率（token efficiency）和更清晰的失败模式（failure modes）。

## 核心概念

### 三种角色

Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer

规划器（Planner）生成一个有向无环图（DAG）。每个节点标明所使用的工具、对应参数以及依赖的前置节点（引用格式如 `#E1`、`#E2`）。执行器（Workers）按拓扑顺序执行各节点。求解器（Solver）负责将所有信息整合拼接。

### 为何能节省 5 倍 Token

ReAct 架构的提示词（Prompt）长度会随着执行步骤的增加而线性增长。例如在第 10 步时，提示词中会累积包含“思考 1、动作 1、观察 1、思考 2、动作 2、观察 2……"等全部内容。同时，每个中间步骤还会冗余地重复携带初始提示词。

ReWOO 仅需消耗一次规划器提示词（较长）、N 次小型执行器提示词（每次仅包含工具调用，无上下文链）以及一次求解器提示词。在 HotpotQA 数据集上的测试表明，该论文实现了约 5 倍的 Token 节省，同时绝对准确率提升了 4%。

### 为何具备更高的鲁棒性

在 ReAct 中，若第 3 个执行器失败，系统必须在执行流中途进行错误推理与状态恢复。而在 ReWOO 中，执行器 3 仅返回错误信息字符串；求解器会结合原始计划上下文读取该错误，从而实现优雅降级（Graceful Degradation）。故障定位精确到节点级别，而非步骤级别。

### 规划器蒸馏（Planner Distillation）

论文的第二项成果：由于规划器不依赖观察结果，你可以使用 175B 教师模型生成的规划器输出，来微调（Fine-tune）一个 7B 模型。小模型负责规划；大模型在推理（Inference）阶段不再需要。这已成为行业标准——许多 2026 年的生产级智能体（Agent）采用“小规划器+大执行器”或反之的架构。

### Plan-and-Execute（LangChain，2023）

LangChain 团队于 2023 年 8 月发布博文，将 ReWOO 抽象为一种通用模式：Plan-and-Execute（规划与执行）。该模式中，前置规划器首先生成步骤列表，执行器逐一运行各步骤，并可选配重规划器（Replanner）在获取执行结果后进行动态调整。相较于 ReWOO，该模式更接近 ReAct（因为重规划器会将观察结果重新反馈至规划阶段），但依然保留了节省 Token 的特性。

### Plan-and-Act（Erdogan 等人，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该架构模式扩展至长程（Long-Horizon）网页与移动端智能体。其核心贡献在于引入了合成计划数据：通过带标签的轨迹生成器，产出计划路径显式化的训练数据。这些数据用于微调规划器模型，使其在 WebArena 类任务中能够稳定执行超过 30~50 个步骤，而传统的单一 ReAct 轨迹在此类长任务中极易丧失连贯性。

### 如何选择适用模式

| 模式 | 适用场景 |
|---------|------|
| ReAct | 短任务、环境未知、需要响应式异常处理 |
| ReWOO | 工具已知的结构化任务、对 Token 敏感、证据可并行获取 |
| Plan-and-Execute | 类似 ReWOO，但支持在部分执行后进行重规划 |
| Plan-and-Act | 长程任务（>30 步）、网页/移动端/计算机操作 |
| 思维树（Tree of Thoughts） | 搜索成本值得投入时（见第 04 课） |

Anthropic 于 2024 年 12 月给出的指导建议：始终从最简单的方案入手。若任务仅需一次工具调用并生成总结，则无需构建 ReWOO；若任务为长达 40 步的研究型作业，则不应单独依赖 ReAct。

## 动手构建

`code/main.py` 实现了一个简易版 ReWOO：

- `Planner` —— 一种基于脚本的策略，根据提示词生成计划有向无环图（DAG）。
- `Worker` —— 通过工具注册中心分发每个节点的工具调用。
- `Solver` —— 预设的组合流程，读取证据并生成最终答案。
- 依赖解析（Dependency resolution）—— 将 `#E1` 等引用替换为先前 `Worker` 的输出。

该演示使用两步计划来回答“法国首都的人口是多少（四舍五入到百万）？”：(1) 查询首都名称，(2) 查询人口数量，然后进行求解。

运行方式：

python3 code/main.py

执行轨迹（trace）会首先显示完整计划，接着是 `Worker` 的结果，最后是 `Solver` 的组合输出。将其令牌数（token count，此处打印的是粗略的字符数）与 ReAct 风格的交错执行（interleaved run）进行对比——在此类结构化任务中，ReWOO 表现更优。

## 使用指南

LangGraph 将“计划与执行”（Plan-and-Execute）作为预设模板（recipe）提供（ReAct 使用 `create_react_agent`，计划与执行则使用自定义图）。CrewAI 的 Flows 直接对该模式进行了编码：你预先定义任务，Flow DAG 负责执行它们。Plan-and-Act 的合成数据（synthetic data）方法目前仍主要处于研究阶段；而其运行时模式（runtime pattern，显式计划 DAG）已通过 LangGraph 和 CrewAI Flows 投入生产环境使用。

## 部署上线

`outputs/skill-rewoo-planner.md` 会根据用户请求和工具目录生成 ReWOO 计划 DAG。在移交执行器之前，它会验证计划的有效性（无环、所有引用均已解析、所有工具均存在）。

## 练习

1. 对独立计划节点的 `Worker` 执行进行并行化。在一个包含 2 个并行组的 6 节点 DAG 中，这能带来什么收益？
2. 添加一个重规划节点（replanner node），当任意 `Worker` 返回错误时触发。对 ReWOO 进行何种最小改动即可将其转变为 Plan-and-Execute 模式？
3. 将 `Planner` 替换为小型模型（7B 级别），而 `Solver` 仍使用前沿模型（frontier model）。对比端到端质量——这种拆分在何处会失效？
4. 阅读 ReWOO 论文中关于规划器蒸馏（planner distillation）的第 4 节。从概念上复现 175B -> 7B 的结果：你需要哪些训练数据，以及如何评估计划质量？
5. 将该简易实现移植到 Plan-and-Act 的轨迹形态（trajectory shape）：计划为序列而非 DAG。这会带来哪些权衡上的变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 无观察推理 (ReWOO) | “无观察推理” | 先制定计划，再并行获取证据，最后求解——规划提示词中不包含观察结果 |
| 规划与执行 (Plan-and-Execute) | “LangChain 的规划-执行模式” | 在执行后增加可选的重新规划节点 (replanner node) 的 ReWOO 变体 |
| 规划与行动 (Plan-and-Act) | “扩展版规划-执行” | 明确的规划器/执行器分离架构，并使用合成计划训练数据来处理长程任务 (long-horizon tasks) |
| 证据引用 (Evidence reference) | “#E1, #E2, ...” | 在任务分发时，将计划节点占位符替换为先前工作节点 (worker) 的输出 |
| 规划器蒸馏 (Planner distillation) | “小规划器，大执行器” | 利用大型教师模型 (teacher model) 的规划轨迹对小型模型进行微调 |
| Token 效率 (Token efficiency) | “减少交互轮次” | 论文表明，在 HotpotQA 数据集上相比 ReAct 模式可减少 5 倍的 Token 消耗 |
| DAG 执行器 (DAG executor) | “拓扑调度器” | 按依赖顺序运行计划节点；同一层级内并行执行 |

## 延伸阅读

- [Xu 等人, ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — 该领域的奠基性论文
- [Erdogan 等人, Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 采用合成计划数据的扩展版规划器-执行器架构
- [LangGraph 规划与执行教程](https://docs.langchain.com/oss/python/langgraph/overview) — 框架实现指南
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 选择最简单且有效的模式
# 层级架构（Hierarchical Architecture）及其失效模式（Failure Mode）

> 层级架构本质上是监督者模式（Supervisor Pattern）的嵌套。管理者智能体（Manager Agents）下辖子管理者，子管理者再下辖工作者。CrewAI 的 `Process.hierarchical` 是教科书式的实现：一个 `manager_llm` 动态分配任务并验证输出。LangGraph 中的等效实现是 `create_supervisor(create_supervisor(...))`。当任务结构类似于真实的组织架构图（Org Chart）时，这是最自然的模式。但它也最容易陷入“管理循环”（Managerial Looping）——管理者智能体分配任务不当、误解子级输出，或无法达成共识。此时，顺序模式（Sequential Pattern）往往表现更好。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** 第 16 阶段 · 05（监督者模式）
**耗时：** 约 60 分钟

## 问题

一旦掌握了监督者模式，很自然会想到下一步：“如果工作者本身也是监督者会怎样？”团队下有子团队，公司下有部门中的部门。层级架构正是对这种结构的映射。

问题在于：大语言模型（LLM）管理者与人类管理者截然不同。人类管理者对下属的知识储备有稳定的先验认知（Priors）。而 LLM 管理者每一轮都会根据上下文（Context）中的内容重新推演整个组织架构。上下文一旦出现微小漂移，整个树状结构就会错误分配工作。

## 概念

### 结构形态

                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5

每个内部节点（Internal Node）负责规划、任务委派与结果综合。仅叶子节点（Leaf Node）执行实际工作。

### 优势所在

- **组织架构映射清晰。** 如果实际任务具有部门属性（例如“法务审阅文档、财务审阅文档、工程团队审阅文档，最后为高管生成摘要”），该层级结构能明确对应。
- **局部汇总。** 每个子管理器（Sub-Manager）会在顶层管理器（Top Manager）查看之前，先综合其团队的输出。顶层管理器看到的是三份子管理器摘要，而非十五份工作节点（Worker）的原始输出。

### 失效场景

2026年的项目复盘（Post-mortem）中反复出现的三种失效模式：

1. **任务分配错误。** 管理器读取目标后，产生任务拆解幻觉（Hallucination），并将其委派给错误的子管理器。由于子管理器会忠实地执行分配到的任务，该错误直到顶层综合阶段才会暴露——此时已偏离人类本可介入纠正的层级。
2. **输出误读。** 子管理器返回“无法验证主张 X”，顶层管理器将其总结为“主张 X 未获确认”。语义在每一层级传递时都会发生漂移（Meaning Drift）。
3. **共识死循环。** 两个子管理器意见相左；顶层管理器要求它们协调一致；它们再次向下委派；工作节点重新执行；子管理器返回略有差异的答案；循环往复。CrewAI 的 `Process.hierarchical` 通过设置步骤上限来防范此问题，但该上限本身现已成为一个超参数（Hyperparameter）。

### 核心决策问题

顺序模式（Sequential / Linear Pipeline）与层级模式（Hierarchical）的选择：你的任务是否真正包含独立的子团队，还是仅仅将单一线性流程伪装成树状结构？若是后者，请使用顺序模式。若是前者，请使用层级模式，但需预先设计明确的协调规则。

### CrewAI 的实现方式

`Process.hierarchical` 将管理器大语言模型（Manager LLM）与专业智能体团队（Specialist Crews）进行连接。管理器负责：

- 接收顶层任务，
- 将子任务分配给各智能体团队，
- 评估团队输出结果，
- 决定是接受结果、重新委派还是迭代执行。

文档：https://docs.crewai.com/en/introduction（在“Core Concepts”下查找“Hierarchical Process”）。

### LangGraph 的实现方式

LangGraph 采用嵌套的 `create_supervisor` 调用。内部监督器（Inner Supervisor）拥有独立的图结构（Graph）；外部监督器则将内部图视为一个黑盒节点（Opaque Node）。这种方式在调试时比 CrewAI 更清晰（可逐步遍历每个图），但更难表达树结构的动态重塑（Dynamic Reshaping）。

参考文档：https://reference.langchain.com/python/langgraph-supervisor。

## 动手实践

`code/main.py` 运行一个 3 级层级结构（3-level hierarchy）：

- 顶层管理器（top manager）：将任务拆分为“工程”和“法律”两个分支，
- 工程子管理器（engineering sub-manager）：进一步拆分为“前端”和“后端”工作节点（workers），
- 法律子管理器（legal sub-manager）：包含一个工作节点。

该演示对比了**正常路径（happy path）**（所有节点达成一致）与**扰动路径（perturbed path）**。在扰动路径中，顶层管理器的任务分解（decomposition）错误地将“法律”标记为“财务”，随后观察错误如何级联（cascade）——子管理器顺从地执行财务工作，顶层合成器（synthesizer）汇报财务结果，而最初的法律问题则无人解答。

运行：

python3 code/main.py

输出结果将展示两条路径，并清晰并列对比“用户提问内容”与“实际交付结果”。

## 使用指南

`outputs/skill-hierarchy-fitness.md` 用于评估给定任务应采用层级式（hierarchical）、顺序式（sequential）还是扁平式（flat）监督器（supervisor）。输入参数：任务描述、组织架构、协调预算（reconciliation budget）。输出结果：模式推荐及需要防范的具体故障模式（failure modes）。

## 部署上线

若部署层级式架构：

- **将树深度限制为 2。** 3 级层级已足以让可观测性（observability）难以捕捉到大多数错误。
- **明确协调预算。** 设定顶层管理器必须提交结果前的最大轮数。通常为 2。
- **每次合成均需溯源（provenance）。** 每个节点的摘要必须明确引用生成该摘要的叶子节点输出。
- **对分解漂移（decomposition drift）设置告警。** 记录管理器每一步的分解结果，并与用户原始查询进行差异比对（diff）。若分解结果不再覆盖原始查询，则触发告警。

## 练习

1. 运行 `code/main.py` 并对比正常路径与扰动路径。需要经过多少层管理器交接（hand-off），顶层输出才会与用户问题完全偏离？
2. 增加第三层级（顶层 → 子层 → 孙层 → 工作节点）。测量随着层级加深，扰动路径自我纠正与完全偏离的发生频率。
3. 在每个子管理器处实现一个“金丝雀（canary）”工作节点，始终向其输入未经修改的原始用户问题。利用金丝雀的回答来检测分解漂移。当金丝雀的回答与合成答案不一致时，管理器应如何响应？
4. 阅读 CrewAI 的 `Process.hierarchical` 文档。找出 CrewAI 应用的一项具体防护机制（guardrail）（如步骤限制、`manager_llm` 约束），并说明它旨在防范哪种故障模式。
5. 对比嵌套式 LangGraph 监督器与 CrewAI 层级架构。哪种架构能更低成本地检测协调循环（reconciliation loops）？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 层级架构 (Hierarchical) | “组织架构图模式” | 层层监督；仅叶子节点执行实际工作。 |
| 管理器大语言模型 (Manager LLM) | “老板” | 在内部节点负责任务分解、分配与验证的大语言模型。 |
| 分解漂移 (Decomposition drift) | “老板偏离主线” | 顶层管理器的任务拆分已无法覆盖原始问题。 |
| 协调循环 (Reconciliation loop) | “无休止的会议” | 子管理器意见分歧；顶层重新委派；工作节点重新执行；循环直至预算耗尽。 |
| 深度2上限 (Depth-2 ceiling) | “层级不要超过2层” | 经验性护栏：超过3层会导致可观测性失效。 |
| 金丝雀问题 (Canary question) | “各层的基准真值” | 一个始终接收原始查询（不作修改）的工作节点，用于检测漂移。 |
| 溯源链 (Provenance chain) | “谁说了什么” | 从每次合成结果回溯至生成该结果的叶子节点输出。 |

## 延伸阅读

- [CrewAI 介绍 — Process.hierarchical](https://docs.crewai.com/en/introduction) — 教科书式的层级架构，配备管理器大语言模型
- [LangGraph 监督器参考文档](https://reference.langchain.com/python/langgraph-supervisor) — 通过 `create_supervisor` 实现嵌套监督器
- [Anthropic 工程实践 — 研究系统](https://www.anthropic.com/engineering/multi-agent-research-system) — Anthropic 为何刻意选择扁平化监督器而非层级架构
- [Cemri 等人 — 多智能体大语言模型系统为何会失败？](https://arxiv.org/abs/2503.13657) — MAST 分类体系；协调失败章节详细记录了分解漂移现象
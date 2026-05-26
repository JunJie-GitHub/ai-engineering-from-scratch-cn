# 共享内存与黑板模式（Shared Memory and Blackboard Patterns）

> 在2026年的多智能体系统（Multi-Agent Systems）中，两种架构并存：**消息池**（Message Pool，所有智能体都能看到彼此的消息，如 AutoGen GroupChat 或 MetaGPT）与**带订阅机制的黑板**（Blackboard with Subscription，智能体订阅相关事件，如 Context-Aware MCP 或 Matrix 框架）。两者都是多智能体系统中唯一有状态的部分——这也意味着，它们正是那些隐蔽缺陷的藏身之处。典型的故障模式是**内存污染**（Memory Poisoning）：某个智能体产生幻觉并编造出一个“事实”，其他智能体将其视为已验证信息，导致系统准确率逐渐下降。这种渐进式衰减比直接崩溃更难调试。本节课程将使用标准库（stdlib）从零构建这两种结构，注入污染攻击，并展示三种在生产环境中真正有效的缓解措施。

**类型：** 学习 + 构建
**语言：** Python（`stdlib`、`threading`）
**前置知识：** 第16阶段 · 04（基础模型 Primitive Model），第16阶段 · 09（并行群体网络 Parallel Swarm Networks）
**时长：** 约75分钟

## 问题

多智能体系统需要一个供智能体共享事实的场所。最直观的做法是“通过消息传递一切”——但这实际上是通过额外的数据拷贝重新实现了共享状态。另一种做法是“为每个智能体提供全局日志”——但全局日志会无限增长且极易被污染。第三种做法是“为每个智能体投射独立视图”——虽然具备可扩展性，但模式（Schema）设计过于繁重。

当某个智能体产生幻觉并将幻觉内容写入共享状态时，所有读取该状态的下游智能体都会将其当作事实采纳。等到人类开发者察觉时，推理链可能已经深入了五个步骤，而根本原因仅仅是历史上写入的第三条消息。调试多智能体系统的准确率衰减问题，比调试程序崩溃要困难得多。

这就是内存污染（Memory Poisoning）。在 MAST 分类体系（MAST Taxonomy，Cemri 等人，arXiv:2503.13657）中，它是记录在案的第二大故障家族。这是一种结构性缺陷：任何缺乏来源追溯（Provenance）机制且未设置不可写验证器（Unwritable Verifier）的共享内存设计，最终都会暴露出这一问题。

## 概念

### 两种主要拓扑结构

**全量消息池（Full message pool）。** 每个智能体（Agent）读取所有消息。AutoGen GroupChat 和 MetaGPT 采用此架构。该架构简单、透明、可审查，但无法扩展到约 10 个智能体以上，因为每个智能体的上下文（Context）会被其他智能体的工作内容填满。

agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F

**带订阅机制的黑板（Blackboard with subscription）。** 智能体声明感兴趣的主题（Topic）；底层架构（Substrate）仅路由相关消息。CA-MCP (arXiv:2601.11595) 和 Matrix 去中心化框架 (arXiv:2511.21686) 采用此架构。该架构扩展性更强，但需要预先设计模式（Schema），以使订阅机制具有实际意义。

                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘

### 各自的优势场景

- **全量消息池**在智能体数量较少（< 10）、异构且对话周期较短时表现最佳。当所有智能体都能看到全部内容时，追溯“谁说了什么”的推理过程变得非常简单。
- **黑板架构**在智能体数量众多、角色同质但实例繁多（即智能体集群），且对话长期运行时表现最佳。消息路由能够节省 Token 成本并减少上下文污染。

生产系统通常采用混合架构：顶层使用小型全量消息池（规划层），底层使用黑板架构（执行层）。

### 记忆中毒（Memory poisoning）的一个场景

三个智能体协同完成一项研究任务。智能体 A 负责检索，智能体 B 负责摘要生成，智能体 C 负责分析。

1. A 抓取了一个网页，并向共享状态（Shared state）写入一条消息：“该研究报告准确率提升了 42%。”
2. 实际抓取的网页内容为“提升了 4.2%”。A 产生了幻觉（Hallucination），多写了一个小数位。
3. B 读取共享状态后写入：“报告显示准确率大幅提升 42%（来源：A）。”
4. C 读取共享状态后写入：“建议采纳——42% 的提升具有变革性意义。”
5. 最终报告引用了一个根本不存在的 42% 的数据。

没有智能体崩溃，也没有测试失败。系统“正常运行”。但幻觉通过共享状态，从一个智能体的上下文蔓延到了所有下游智能体的推理过程中。

### 为什么这是结构性问题

如果没有共享状态，智能体 A 的幻觉将仅停留在 A 自身的上下文中。下游智能体会重新抓取或重新推导，从而可能发现错误。但在朴素的共享状态下，A 的上下文变成了所有人的上下文，幻觉被“洗白”成了事实。

问题本身不在于共享状态，而在于**缺乏溯源信息（Provenance）且没有独立验证器（Independent verifier）**的共享状态。可通过以下三种缓解措施来解决：

1. **为每次写入标注溯源信息。** 共享状态中的每条记录都需注明写入者、写入时间、使用的提示词（Prompt），以及（如适用）智能体引用的来源。下游智能体在读取时，会根据溯源信息保持审慎态度。
2. **对写入进行版本控制；采用仅追加（Append-only）模式。** 修正操作应作为一条新记录覆盖旧记录，而非原地更新。这样可以保留完整的审计轨迹（Audit trail）。
3. **保留至少一个无法向共享状态写入的智能体。** 一个只读验证器智能体会对记录进行抽样、重新抓取来源，并标记不一致之处。由于它无法向池中写入，因此不会被池中的数据污染。

### 黑板架构的先例（Hayes-Roth, 1985）

黑板模式比大语言模型智能体（LLM Agent）早出现四十年。Hayes-Roth（1985，《A Blackboard Architecture for Control》）描述了专业的知识源（Knowledge Sources）如何观察全局黑板、贡献局部解决方案并触发其他知识源。2026 年的黑板架构（CA-MCP、Matrix）沿用了同一模式，只是将 LLM 智能体作为知识源，将 JSON 数据块作为局部解决方案。早期文献中已记载了关于写入竞争（Write contention）、机会控制（Opportunistic control）和一致性（Consistency）的解决方案，现代系统正在重新发现这些方法。

### 投影视图与全量视图

纯粹的黑板架构为每个订阅者提供相同的投影（Projection，按主题划分）。更激进的设计是**按智能体投影（Per-agent projection）**：每个智能体获得与其角色定制匹配的视图。LangGraph 的状态归约器（State reducers）是 2026 年的典型实现——归约函数将全局状态折叠为特定角色的切片。

按智能体投影具有更好的扩展性，但需要预先定义模式。否则，你不得不在每个智能体的提示词中重新构建临时投影逻辑。

### 写入竞争模式

多个智能体同时写入是一个并发（Concurrency）问题，而不仅仅是 LLM 的问题。以下三种模式行之有效：

- **顺序写入器（Sequential writer，单生产者）。** 所有写入操作都经过一个协调智能体进行序列化。实现简单，但容易成为瓶颈。
- **带版本控制的乐观并发（Optimistic concurrency）。** 每条记录都有版本号；写入时若版本不匹配则失败并重试。这是经典的数据库技术。
- **主题分区（Topic partitioning）。** 不同智能体负责不同的主题。不存在跨主题竞争。需要预先设计分区边界。

大多数 2026 年的框架默认采用顺序写入器，因为 LLM 调用速度较慢，写入竞争很少发生，且该瓶颈通常不会造成明显影响。

### 不可写入的验证器

最核心的缓解措施是只读验证器（Read-only verifier）。实现规则如下：

- 验证器与团队共享状态（读取黑板或消息池）。
- 验证器没有共享状态的写入句柄——只能写入独立的验证通道。
- 验证器独立抓取写入内容中引用的来源。标记不一致之处。
- 验证器自身的输出被路由给人类或独立的决策智能体，绝不会反馈回消息池中。

如果没有这种隔离，验证器的输出将成为池中的新记录，这意味着被污染的池会污染验证器，进而污染其验证结果。

## 构建

`code/main.py` 使用 Python 标准库实现了上述两种拓扑结构（topologies），并包含一个示例级投毒攻击（poisoning attack）及三种缓解措施（mitigations）。

- `MessagePool` — 线程安全（thread-safe）的仅追加日志（append-only log），支持完整读取。
- `Blackboard` — 基于主题键的发布/订阅（pub/sub）系统，支持按智能体（agent）订阅。
- `ProvenanceEntry` — 每次写入均记录（写入者、时间戳、提示词哈希值、来源 URI）。
- `PoisoningScenario` — 运行一个三智能体研究任务，其中智能体 A 会产生幻觉（hallucination）。最终打印报告。
- `Verifier` — 一个只读智能体，负责重新获取来源并标记不一致之处。在引入该验证器（verifier）的情况下运行相同场景。

运行：

python3 code/main.py

预期输出：
- 运行 1（无验证器）：幻觉产生的 42% 会传播至最终报告。
- 运行 2（含验证器）：验证器标记出不一致，消息池被标记为“已标记（flagged）”，最终报告包含撤回声明。

## 使用

`outputs/skill-memory-auditor.md` 是一项技能（skill），用于审计任意多智能体系统（multi-agent system）的共享内存（shared-memory）设计，检查其是否具备数据溯源（provenance）、版本控制（versioning）及验证器隔离（verifier separation）机制。在将新的多智能体架构投入生产环境前，请先运行此技能进行审计。

## 交付

针对任何共享内存设计：

- 每次写入时记录溯源信息：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- 确保日志为仅追加（append-only）模式。修正操作应作为新条目写入，并引用被替代（supersessions）的旧条目。
- 部署至少一个具备独立来源访问权限的只读验证器智能体。
- 将验证器的输出路由至独立通道，切勿回写至共享池中。
- 记录被替代写入操作的比例——该比例上升是幻觉模式出现的早期迹象。

## 练习

1. 运行 `code/main.py`。确认运行 1 会传播幻觉，而运行 2 能成功捕获该幻觉。
2. 添加第二个幻觉：智能体 B 虚构一个数据集规模。验证器应能在无需针对任一幻觉进行手动调优的情况下，同时捕获两者。
3. 将完整消息池切换为带有主题分区（topic partitions）（`prices`、`summaries`、`analyses`）的黑板架构（blackboard）。主题分区会使哪些投毒场景更难实施？对哪些场景又无济于事？
4. 阅读 Hayes-Roth (1985, "A Blackboard Architecture for Control")。找出论文中未在本课讨论、但 2026 年系统可从中受益的两种控制模式（control patterns）。
5. 阅读 CA-MCP (arXiv:2601.11595)。将其共享上下文存储（Shared Context Store）映射到 `code/main.py` 中的 `MessagePool` 或 `Blackboard` 类。CA-MCP 在此基础上增加了哪些原语（primitives）？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 消息池 (Message pool) | “共享聊天记录” | 所有智能体（Agent）均可读取的仅追加日志（Append-only log）。具备完全透明性，但扩展性较差。 |
| 黑板架构 (Blackboard) | “共享工作区” | 基于主题键的发布/订阅（Pub/Sub）机制。智能体订阅相关主题，具备更强的扩展能力。 |
| 数据溯源 (Provenance) | “谁写了什么” | 记录每次写入的元数据：包含写入者、时间戳、提示词（Prompt）及数据来源。 |
| 记忆投毒 (Memory poisoning) | “幻觉扩散” | 单个智能体的错误进入共享状态后，被下游智能体误认为事实。 |
| 仅追加模式 (Append-only) | “禁止原地更新” | 修正操作以新条目形式追加并取代旧内容，从而保留完整的审计轨迹（Audit trail）。 |
| 不可写验证器 (Unwritable verifier) | “独立审计员” | 只读智能体，负责重新拉取原始数据并标记不一致之处。 |
| 状态投影 (Projection) | “限定范围视图” | 基于全局状态为各智能体计算出的专属视图。LangGraph 的归约器（Reducer）是该模式的典型实现。 |
| 知识源 (Knowledge Source) | “专家智能体” | Hayes-Roth 于 1985 年提出的术语，专指黑板架构中的参与方。 |

## 延伸阅读

- [Cemri 等人 — 为什么多智能体大语言模型系统会失败？](https://arxiv.org/abs/2503.13657) — MAST 分类体系（MAST taxonomy）；记忆投毒属于协调失败（Coordination failure）的子类
- [CA-MCP — 上下文感知多服务器 MCP](https://arxiv.org/abs/2601.11595) — 用于协调 MCP 服务器的共享上下文存储（Shared Context Store）
- [Matrix — 去中心化多智能体框架](https://arxiv.org/abs/2511.21686) — 基于消息队列的黑板架构，无需中央编排器（Orchestrator）
- [LangGraph 状态与归约器（Reducer）](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 生产环境中每个智能体的投影模式
- [Anthropic — 我们如何构建多智能体研究系统](https://www.anthropic.com/engineering/multi-agent-research-system) — 来自生产环境部署的数据溯源与验证实践笔记
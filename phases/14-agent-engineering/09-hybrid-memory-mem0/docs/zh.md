# 混合记忆（Hybrid Memory）：向量（Vector） + 图（Graph） + 键值（KV）（Mem0）

> Mem0（Chhikara 等人，2025）将记忆视为三个并行存储（Store）——向量用于语义相似度（Semantic Similarity）匹配，键值用于快速事实查找（Fact Lookup），图用于实体关系推理（Entity-Relationship Reasoning）。在检索（Retrieval）时，一个评分层（Scoring Layer）会将这三者融合。这已成为 2026 年外部记忆（External Memory）的生产环境标准。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** Phase 14 · 07（MemGPT），Phase 14 · 08（Letta Blocks）
**耗时：** 约 75 分钟

## 学习目标

- 解释为何单一存储（Store）不足以支撑智能体记忆（Agent Memory）。
- 列举 Mem0 的三个并行存储及其各自的优化目标。
- 描述 Mem0 的融合评分（Fusion Scoring）机制——相关性（Relevance）、重要性（Importance）、时效性（Recency）——并解释为何采用加权求和（Weighted Sum）而非层级结构（Hierarchy）。
- 使用标准库实现一个简易的三存储记忆系统，包含向三者同时写入的 `add()` 方法，以及融合检索结果的 `search()` 方法。

## 问题背景

单一存储无法同时满足以下三类查询需求：

- **语义相似度（Semantic Similarity）**——“上周我们关于智能体漂移（Agent Drift）讨论了些什么？”向量存储表现最佳；键值与图存储无法命中。
- **事实查找（Fact Lookup）**——“用户的电话号码是多少？”键值存储表现最佳；向量存储效率低下，图存储则过度设计。
- **关系推理（Relationship Reasoning）**——“哪些客户属于同一个计费实体（Billing Entity）？”图存储表现最佳；向量与键值存储无法作答。

生产环境中的智能体在一次会话中会同时发起这三类查询。单一存储记忆必然在其中两类查询上表现不佳。Mem0 的核心贡献在于，通过一个评分函数将三者融合，并统一封装在单一的 `add`/`search` 接口背后。

## 核心概念

### 三个并行存储

Mem0（arXiv:2504.19413，2025年4月）在执行 `add(text, user_id, metadata)` 时的流程：

1. 从文本中提取候选事实（由大语言模型（LLM）驱动的步骤）。
2. 将每个事实写入向量存储（Vector Store）（基于嵌入向量（Embedding）），用于语义搜索。
3. 将每个事实写入键值存储（Key-Value Store），以 `(user_id, fact_type, entity)` 为键，实现 O(1) 时间复杂度的查找。
4. 将每个事实作为带类型的边写入图存储（Graph Store）（Mem0g），用于关系查询。

在执行 `search(query, user_id)` 时：

1. 向量存储根据嵌入余弦相似度返回 Top-k 结果。
2. 键值存储根据查询派生的 `(user_id, type, entity)` 键返回直接匹配结果。
3. 图存储返回从查询实体可达的子图。
4. 评分层对三者结果进行融合。

### 融合评分（Fusion Scoring）

score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)

- **相关性（Relevance）** — 向量余弦相似度、键值存储精确匹配、图路径权重。
- **重要性（Importance）** — 在写入时打标或通过模型学习得出（某些事实更为关键：如姓名、ID、策略）。
- **时效性（Recency）** — 自上次写入或读取以来随时间呈指数衰减。

权重根据具体产品进行调优。聊天智能体（Chat Agent）侧重更高的 `w_recency`；合规智能体（Compliance Agent）侧重更高的 `w_importance`；检索智能体（Retrieval Agent）侧重更高的 `w_relevance`。

### Mem0g 与时序推理（Temporal Reasoning）

Mem0g 引入了冲突检测器。当新事实与现有边发生冲突时，现有边会被标记为无效而非直接删除。时序查询（例如“用户三月份所在的城市是什么？”）会遍历在特定时间点有效的子图。

这正是 Letta 的失效模式（Invalidation Pattern）所泛化的合规级行为。

### 基准测试数据（Benchmark Numbers）

Mem0 论文报告的数据（2025年）如下：

- **LoCoMo**（长对话记忆）：91.6
- **LongMemEval**（长周期情景记忆）：93.4
- **BEAM 1M**（100万 Token 记忆基准测试）：64.1

对比基线（全上下文 128k LLM、扁平向量存储、扁平键值存储）均落后 10 分以上。仅凭基准测试不足以证明技术选型——实际运行架构才是关键——但这些数据表明，融合设计带来的提升绝非微不足道的误差。

### 作用域分类（Scope Taxonomy）

Mem0 按作用域划分记忆：

- **用户记忆（User Memory）** — 跨会话持久化，以 `user_id` 为键。
- **会话记忆（Session Memory）** — 在单个对话线程内持久化。
- **智能体记忆（Agent Memory）** — 每个智能体实例的独立状态。

每次写入仅选择一个作用域。检索时可跨作用域查询，并应用各作用域专属的权重。若不假思索地混用作用域，就会导致“助手把 Bob 的项目透露给 Alice”这类数据泄露事件。

### 该模式的常见陷阱

- **嵌入漂移（Embedding Drift）。** 向量检索结果在前一百次查询中表现良好，但随着语料库增长会逐渐退化。需定期对使用频率最高的 Top-N 记录重新计算嵌入向量。
- **键值模式蔓延（KV Schema Creep）。** `(user_id, type, entity)` 看似简单，直到每个团队都自行添加自己的 `type`。建议每季度审计一次类型集合。
- **图爆炸（Graph Explosion）。** 一个噪声较大的提取器可能为每条消息添加 50 条边。应限制每次 `add` 调用的图写入数量，并丢弃低置信度的边。

## 动手构建

`code/main.py` 在标准库（stdlib）中实现了三存储模式（three-store pattern）：

- `VectorStore` —— 使用基础的词元重叠相似度作为嵌入（embedding）的替代方案。
- `KVStore` —— 以 `(user_id, fact_type, entity)` 为键的字典。
- `GraphStore` —— 带类型的边（subject, relation, object, valid）。
- `Mem0` —— 顶层门面（facade），提供 `add()`、`search()`、融合评分（fusion scoring）以及作用域感知（scope-aware）检索功能。
- 包含一个多用户、多会话对话的完整执行追踪（trace）。

运行方式：

python3 code/main.py

输出结果将展示三条独立的召回路径（recall paths）以及融合后的 Top-K 结果。修改 `main()` 顶部的评分权重，即可观察排序的变化。

## 使用方案

- **Mem0 (Apache 2.0)** —— 生产就绪。可基于 Postgres + Qdrant + Neo4j 自行托管，或使用其托管云服务。
- **Letta** —— 采用核心/召回/归档三层架构；支持接入自有的向量与图数据库后端。
- **Zep** —— 商业替代方案，内置时序知识图谱（temporal KG）与事实抽取功能。
- **自定义构建** —— 适用于需要精确控制抽取器（满足合规要求）或融合权重（例如在语音智能体中近期信息占主导的场景）的情况。

## 部署与交付

`outputs/skill-hybrid-memory.md` 会生成一个三存储记忆脚手架（memory scaffold），其中已内置融合评分器、作用域分类体系（scope taxonomy）以及时效失效机制（temporal invalidation）。

## 练习

1. 将简易的向量相似度替换为真实的嵌入模型（如 sentence-transformers、Ollama 或 OpenAI embeddings）。在合成的长对话数据集上评估 recall@10。在写入 1000 条数据后，排序是否会发生漂移？
2. 添加时序查询功能：`search(query, as_of=timestamp)`。仅返回在该时间点或之前有效的记录。哪个存储组件需要最多的改造工作？
3. 实现冲突检测器（conflict detector）：如果新传入的事实与现有图边冲突，则将旧边标记为失效，并记录两者。使用“用户居住在柏林” -> “用户居住在里斯本”进行测试。
4. 改造融合评分器，加入 `user_feedback`（用户反馈）维度（例如对检索到的记录点赞）。如何防止刷分/操纵行为（gaming）（例如智能体只返回它自己偏好的记录）？
5. 阅读 Mem0 文档（`docs.mem0.ai`）。将示例代码迁移至 `mem0` 客户端调用。在相同的 20 个测试查询上对比检索质量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 混合记忆（Hybrid memory） | “向量+图+键值” | 并行写入三个存储，检索时进行融合 |
| 事实抽取（Fact extraction） | “记忆摄入” | LLM 将文本拆解为（实体，关系，事实）元组的步骤 |
| 融合评分（Fusion scoring） | “相关性排序” | 相关性、重要性与时效性的加权和 |
| 作用域（Scope） | “记忆命名空间” | 用户/会话/智能体 —— 决定数据的可见范围 |
| Mem0g | “记忆图谱” | 带类型且具备时效有效性的边，用于关系查询 |
| 时效失效（Temporal invalidation） | “软删除” | 将冲突的边标记为无效；永不物理删除 |
| 嵌入漂移（Embedding drift） | “检索退化” | 随着语料库增长向量质量下降；需定期重新嵌入 |

## 延伸阅读

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — 原始论文
- [Mem0 文档](https://docs.mem0.ai/platform/overview) — 生产级应用程序接口（API）、软件开发工具包（SDK）与托管云服务
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 虚拟上下文（Virtual Context）架构的前身
- [Letta, Memory Blocks 博客](https://www.letta.com/blog/memory-blocks) — 三层同级（Sibling）设计
---
name: 混合记忆（hybrid-memory）
description: 生成一个具备融合评分器（fusion scorer）、作用域分类体系（scope taxonomy）与时间失效机制（temporal invalidation）的类 Mem0 三存储记忆系统（向量 + 键值 + 图）。
version: 1.0.0
phase: 14
lesson: 09
tags: [记忆, mem0, 向量, 图, 键值, 融合, 作用域]
---

给定目标运行时、向量后端（vector backend，如 Qdrant、pgvector、Chroma、sqlite-vec）、键值后端（KV backend，如 Postgres、Redis、dict）以及图后端（graph backend，如 Neo4j、内存边），请生成一个融合记忆系统。

生成内容如下：

1. 在 `add(text, user_id, session_id, scope, importance, tags)` 门面（facade）背后实现三个存储类。在写入时，提取器（extractor）会将 `text` 分解为记录、KV 三元组和图三元组。所有存储均为必需项，不可省略。
2. 一个融合评分器（fusion scorer）`score = w_rel * relevance + w_imp * importance + w_rec * recency`。将这三个权重全部暴露为配置项。按产品维度进行调优，而非按单次调用调优。
3. 作用域分类体系（scope taxonomy）：`user`、`session`、`agent`。检索（retrieval）必须严格遵守作用域限制。用户查询绝不允许泄露其他用户的记录。
4. 时间失效机制（temporal invalidation）。当出现数据矛盾时，将旧的边/记录标记为无效；切勿直接删除。暴露 `search(query, as_of=timestamp)` 接口以支持历史查询。
5. 一个提取器接口（extractor interface）。默认实现可由大语言模型（LLM）驱动；同时允许使用确定性正则表达式作为测试回退方案。限制每次 `add()` 调用生成的图边数量，以防止数据爆炸。

硬性拒绝条件：

- 将单存储记忆描述为“类 Mem0 架构”。仅支持向量、仅支持键值或仅支持图的产品本身没有问题，但它们不属于混合记忆。请勿错误命名。
- 在缺乏按作用域权重或显式 `scope=` 过滤器的情况下进行跨作用域检索。作用域泄露属于合规与隐私事故。
- 在出现矛盾时直接删除记录。应将其标记为无效并添加时间戳。删除操作会掩盖缺陷并破坏审计追踪。

拒绝规则：

- 如果用户要求“不进行重要性加权”，请拒绝。在百万级记录上采用扁平化的相关性排序，注定会导致检索失败。
- 如果图后端缺乏冲突检测器（conflict detector），请拒绝将生成的系统称为“类 Mem0 架构”。应降低其命名规格。
- 如果产品涉及个人身份信息（PII，如医疗、法律、人力资源领域），请拒绝交付未经产品负责人审计的提取器。

输出要求：每个存储对应一个独立文件，外加 `memory.py`（门面）、`config.py`（权重配置）以及 `README.md`（用于说明融合权重、作用域策略、提取器契约及失效语义）。结尾需包含“下一步阅读”指引：若智能体需要学习新技能，指向第 10 课；若记忆操作需要 OpenTelemetry（OTel）追踪跨度（spans），指向第 23 课；若检索环节需处理不可信输入，指向第 27 课。
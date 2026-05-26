---
name: 提示词缓存规划器
description: 设计对缓存友好的提示词布局，并选择合适的提供商缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [大语言模型工程, 缓存, 成本]
---

给定一个提示词（Prompt）（系统提示词 + 工具 + 少样本示例 + 检索内容 + 历史记录 + 用户输入）和使用画像（Usage Profile）（每小时请求数、所需生存时间（TTL）、提供商），请输出：

1. 布局（Layout）。重新排列各部分并标记唯一的缓存断点（Cache Breakpoint）；说明哪些部分是稳定的，哪些部分是易变的。
2. 提供商模式（Provider Mode）。Anthropic `cache_control`、OpenAI 自动缓存，或 Gemini `CachedContent`。根据 TTL 和复用模式（Reuse Pattern）论证选择理由。
3. 盈亏平衡点（Break-even）。TTL 内预期的每次写入对应的读取次数；通过数学计算对比启用缓存与未启用缓存的净成本。
4. 验证计划（Verification Plan）。在持续集成（CI）中断言第二次相同请求的 `cache_read_input_tokens > 0`；在仪表盘中按缓存令牌与未缓存令牌进行拆分展示。
5. 故障模式（Failure Modes）。列出在此配置下缓存未命中（Cache Miss）最可能的三个原因（动态时间戳、工具顺序重排、近似重复文本），并说明你将如何逐一预防。

拒绝交付将动态字段置于缓存断点之上的缓存方案。拒绝在未满足复用次数要求（无法抵消 2 倍写入溢价成本）的情况下启用 1 小时 TTL。
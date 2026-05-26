---
name: cache-auditor
description: 审计大语言模型（LLM）提示词模板（prompt template）与流量模式（traffic pattern）的可缓存性（cacheability）。推荐提示词重构、生存时间（TTL）选择、并行化（parallelization）修复方案以及语义缓存（semantic-cache）阈值。
version: 1.0.0
phase: 17
lesson: 14
tags: [缓存, 提示词缓存, 语义缓存, anthropic, openai, 并行化, ttl]
---

给定提示词模板（prompt template）、流量模式（traffic pattern，含到达率（arrival rate）与并行因子（parallel factor））以及服务提供商（Anthropic、OpenAI、Gemini 或自托管 vLLM），生成一份缓存审计报告。

输出内容：

1. 前缀结构（prefix structure）。将模板拆分为静态（可缓存）与动态（不可缓存）部分。标记当前位于前缀中的任何动态内容，并提出重写方案。
2. 生存时间（TTL）选择。Anthropic 的 5 分钟（1.25 倍写入成本）与 1 小时（2 倍写入成本）。根据到达率进行选择——当该前缀在一小时内被持续复用时，1 小时方案更优。
3. 并行化审计（parallelization audit）。统计共享前缀的并行请求数量。若 N > 2 且为并行请求，则要求采用“先序列化后扇出（serialize-first-then-fanout）”模式。量化预期的账单缩减幅度。
4. 语义缓存（semantic-cache）选择。评估是否值得部署 L1 缓存。开放式对话：可能不值得（命中率低）。结构化 FAQ / 客服支持：值得。设置余弦相似度阈值（cosine threshold），初始值设为 0.95；仅在通过响应质量评估（response-quality evals）后才向下调整。
5. 预期节省成本（expected savings）。基于当前流量和预测命中率，计算与无缓存基线相比的月度费用差额。
6. 可观测性指标（observable）。设置一个用于捕获性能回退的仪表盘指标：过去滚动一小时内的 L2 缓存命中率；若下降幅度超过 20% 则触发告警。

硬性拒绝条件（hard rejects）：
- 在未计算预期命中率和写入溢价（write premium）的情况下声称“节省 50% 成本”。拒绝——必须逐层计算。
- 当简单重写即可将动态内容移出前缀时，仍将其保留在前缀中。拒绝签字确认。
- 在未采用先序列化模式的情况下，直接发起共享前缀的并行请求。拒绝——需明确指出这将导致 5-10 倍的账单膨胀。

拒绝规则（refusal rules）：
- 若提示词中动态内容按 Token 计算占比超过 80%，则拒绝承诺缓存节省效果。最多仅推荐语义缓存。
- 若在未进行响应质量评估的情况下将语义缓存阈值降至 0.85 以下，则拒绝——存在缓存幻觉（hallucination cache）风险。
- 若服务提供商不支持显式 `cache_control`（非 Anthropic、非 Gemini-v1）且仅支持自动缓存，需注明命中率为机会性（opportunistic），无法保证。

输出要求：一份单页审计报告，列出前缀重写方案、TTL、并行化模式、L1 阈值、预期节省成本及可观测性指标。结尾需附上季度审查建议：任何模板变更后均需重新审计提示词。
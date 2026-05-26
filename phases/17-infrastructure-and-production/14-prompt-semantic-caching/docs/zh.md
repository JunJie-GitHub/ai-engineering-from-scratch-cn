# 提示词缓存 (Prompt Caching) 与语义缓存 (Semantic Caching) 的经济学

> **定价快照日期为 2026 年 4 月。** 下方的数值声明反映了本课程发布时捕获的供应商费率表；在向下游引用前，请对照链接文档进行核实。

> 缓存发生在两个层级。L2（供应商级）提示词/前缀缓存 (Prompt/Prefix Caching) 会针对重复的前缀复用注意力键值 (Attention KV)——Anthropic 的提示词缓存文档宣称，在长提示词场景下可降低成本高达 90%，延迟降低 85%；对于 Claude 3.5 Sonnet，缓存读取价格为 0.30 美元/百万 token，而标准输入（未缓存）为 3.00 美元/百万 token，5 分钟生存时间 (Time To Live, TTL) 选项下写入有 2 倍溢价（针对 1 小时 TTL 选项）（docs.anthropic.com, 2026-04）。OpenAI 的提示词缓存会自动应用于长度 ≥1024 token 的提示词，缓存输入的定价相比标准输入约有 90% 的折扣（platform.openai.com, 2026-04）；每个模型的具体缓存费率取决于实时费率表。L1（应用级）语义缓存 (Semantic Caching) 在嵌入相似度命中 (Embedding Similarity Hits) 时会完全跳过 LLM。供应商宣称的“95% 准确率”指的是匹配正确性，而非命中率 (Hit Rate)——实际生产环境报告的命中率范围从 10%（开放式聊天）到 70%（结构化 FAQ）不等；两家供应商均未发布官方基准数据，因此应将其视为社区遥测数据而非保证。生产环境中的常见陷阱：并行化 (Parallelization) 会破坏缓存（在首次缓存写入完成前发出 N 个并行请求会使成本膨胀数倍），以及前缀内的动态内容 (Dynamic Content) 会完全阻止缓存命中。ProjectDiscovery 报告称（2025-11），通过将动态文本移出可缓存前缀，其命中率从 7% 提升至 74%。

**Type:** 学习
**Languages:** Python（标准库，简易双层缓存模拟器）
**Prerequisites:** 第 17 阶段 · 04（vLLM 服务内部原理），第 17 阶段 · 06（SGLang RadixAttention）
**Time:** 约 60 分钟

## 学习目标

- 区分 L2 提示词/前缀缓存（供应商端的 KV 复用）与 L1 语义缓存（在相似提示词时绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记机制，以及两种 TTL 选项（5 分钟与 1 小时）及其对应的价格倍率。
- 根据命中率、提示词/响应比例以及 token 价格，计算预期的月度节省成本。
- 指出会使账单膨胀 5-10 倍的并行化反模式 (Anti-pattern)，以及会导致命中率骤降的动态内容反模式。

## 问题描述

你在检索增强生成 (Retrieval-Augmented Generation, RAG) 服务中启用了提示词缓存，但账单却毫无变化。你测量了命中率，发现仅为 7%。你的提示词看似静态，实则不然——系统提示词包含了精确到分钟的当前日期、请求 ID，以及为了增加多样性而随机重排的示例。每次请求都会写入一个新的缓存条目，而读取次数为零。

此外，你的智能体 (Agent) 针对每个用户问题会并行执行十次工具调用。这十个请求在首次缓存写入完成前就已全部到达供应商端。十次写入，零次读取。你的实际账单是“启用缓存后”预期成本的 5 到 10 倍。

缓存是一种协议，而非一个简单的标志位。两个层级，对应两种截然不同的失效模式。

## 核心概念

### L2 — 提供商提示词/前缀缓存（Provider Prompt/Prefix Caching）

提供商（Provider）会存储可缓存前缀的注意力键值（Attention KV），并在下一个匹配该前缀的请求中重复使用。你只需支付一次写入成本，后续读取成本几乎为零。

**Anthropic（Claude 3.5 / 3.7 / 4 系列）**：在请求中使用显式的 `cache_control` 标记。你可以指定哪些文本块可被缓存。生存时间（TTL）：5 分钟（写入成本为基准价的 1.25 倍）或 1 小时（写入成本为基准价的 2 倍）。缓存读取：在 Claude 3.5 Sonnet 上为 0.30 美元/百万 token，而未缓存输入（Fresh Input）为 3.00 美元/百万 token——便宜 10 倍（docs.anthropic.com，截至 2026-04）。不同模型的费率有所不同（Opus/Haiku 单独公布）；请务必核对实时定价页面。

**OpenAI**：对长度 ≥1024 token 的提示词自动进行缓存（platform.openai.com，2026-04）。无需显式标记。在当前的 gpt-4o/gpt-5 定价表中，缓存输入的成本约为未缓存输入的 1/10。官方文档和发布说明均未公布官方的缓存命中率（Hit Rate）基准；社区报告显示，在精心设计提示词的情况下，命中率通常集中在 30%–60% 之间。请监控 `usage.cached_tokens` 字段以测量你自己的实际命中率。

**Google（Gemini）**：通过显式 API 实现上下文缓存（Context Caching）；支持 100 万 token 的上下文窗口，这意味着缓存带来的收益更为显著。

**自托管（Self-hosted，如 vLLM、SGLang）**：Phase 17 · 06 节介绍了 RadixAttention 技术——你可以在自己的计算资源上实现相同的缓存模式。

### L1 — 应用级语义缓存（App-level Semantic Caching）

在调用大语言模型（LLM）之前，先对提示词进行哈希处理并生成嵌入向量（Embedding），然后查找是否存在相似的缓存请求（余弦相似度（Cosine Similarity）高于阈值，通常为 0.95 以上）。若命中（Hit），则直接返回缓存的响应；若未命中（Miss），则调用 LLM 并将结果缓存。

开源方案：Redis 向量相似度检索、GPTCache、Qdrant。商业方案：Portkey Cache、Helicone Cache。

供应商宣称的准确率指的是返回的缓存响应在语义上合适的频率，而非缓存命中的频率。生产环境中的实际命中率如下：

- 开放式对话：10%–15%。
- 结构化常见问题解答（FAQ）/ 客服支持：40%–70%。
- 代码相关问题：20%–30%（微小的变体就会导致未命中）。
- 语音智能体（Voice Agent）重复提示词：50%–80%（语音归一化后形成固定集合）。

### 并行化反模式（Parallelization Anti-pattern）

你的智能体（Agent）并行发起 10 次工具调用。这 10 次调用都包含相同的 4K token 系统提示词（System Prompt）。Anthropic 的缓存写入是按请求独立进行的；提供商在接收到提示词后，首次缓存写入大约需要 300 毫秒完成。请求 2 到 10 在同一毫秒级时间窗口内到达，因此每个请求都会遭遇缓存未命中。结果是你支付了 10 次写入溢价，却享受不到任何读取折扣。

修复方案：采用“首请求串行+后续批量”策略——先单独发送请求 1，待其缓存填充完毕后，再并发发送请求 2 到 10。这会给首次工具调用增加约 300 毫秒的延迟，但能节省 5 到 10 倍的账单费用。

### 动态内容反模式（Dynamic Content Anti-pattern）

你的系统提示词可能长这样：

You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...

每个请求都是独一无二的。每个请求都会触发写入。命中率为零。

修复方案：将所有真正静态的内容移至可缓存前缀中；将动态内容追加到缓存边界之后：

[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.

ProjectDiscovery 通过这种方式将缓存命中率从 7% 提升至 74%，并公开了详细的技术剖析。

### 为夜间任务叠加批处理与缓存（Stack Batch + Cache）

批处理 API（Batch API，见 Phase 17 · 15）在 24 小时交付周期内提供 50% 的折扣。在此基础上叠加缓存输入，可再获得约 10 倍的成本优势。通过这种叠加策略，夜间分类、数据标注和报告生成等任务的成本可降至同步未缓存模式的约 10%。

### 关键数据备忘

以下定价数据截取自 2026-04 的供应商官方文档链接，且每隔几个月就会发生变动——在依赖这些数据前请务必重新核实。

- Anthropic 缓存读取：Claude 3.5 Sonnet 为 0.30 美元/百万 token，比未缓存输入便宜约 10 倍（docs.anthropic.com）。
- Anthropic 缓存写入溢价：1.25 倍（5 分钟 TTL）或 2 倍（1 小时 TTL）。
- OpenAI 自动缓存：适用于长度 ≥1024 token 的提示词；在当前定价表中，缓存输入的价格约为未缓存输入的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：开放式对话约 10%；结构化 FAQ 最高可达约 70%。此非供应商官方基准数据。
- ProjectDiscovery：通过将动态内容移出前缀，命中率从 7% 提升至 74%（项目博客，2025-11）。
- 并行化反模式：典型报告显示，当 N 个并行请求错过首次缓存写入时，账单费用通常会膨胀 5 到 10 倍。

## 使用它

`code/main.py` 模拟了混合工作负载下的 L1 + L2 缓存（L1 + L2 Caching）。该脚本会报告命中率（Hit Rate）、费用账单，并展示并行化带来的额外开销（Parallelization Penalty）。

## 交付成果

本课时将生成 `outputs/skill-cache-auditor.md`。在给定提示词模板（Prompt Template）和请求流量的情况下，该工具会评估提示词的可缓存性（Cacheability）并提供结构优化建议。

## 练习

1. 运行 `code/main.py`。切换并行化开关（Parallelization Flag）。账单费用发生了多大变化？
2. 你的系统提示词（System Prompt）中包含日期。将其移出。展示调整前后的命中率计算过程。
3. 根据你的请求到达率（Request Arrival Rate），计算 1 小时 TTL（Time-To-Live，2 倍写入成本）与 5 分钟 TTL（1.25 倍写入成本）的盈亏平衡点（Break-even）。
4. 语义缓存（Semantic Cache）在 0.95 阈值下的命中率为 20%。降至 0.85 时命中率升至 50%，但会出现错误的缓存响应。请选择合适的阈值并阐述理由。
5. 你针对每个用户问题批量处理 10 个并行子查询。在不增加端到端延迟（End-to-End Latency）的前提下，重写逻辑以提升缓存友好性（Cache-friendliness）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| L2 提示词缓存 (L2 Prompt Cache) | “前缀缓存” (Prefix Cache) | 服务提供商为重复的前缀存储键值对 (KV Cache) |
| `cache_control` | “Anthropic 缓存标记” | 用于显式标记可缓存代码块的属性 |
| 缓存写入溢价 (Cache Write Premium) | “写入税” (Write Tax) | 首次缓存未命中并写入时产生的额外成本（1.25 倍或 2 倍） |
| L1 语义缓存 (L1 Semantic Cache) | “嵌入缓存” (Embedding Cache) | 在调用大语言模型 (LLM) 前，在应用层进行哈希与嵌入处理 |
| GPTCache | “LLM 缓存库” | 流行的开源 (OSS) L1 缓存库 |
| 缓存命中率 (Cache Hit Rate) | “命中数 / 总请求数” | 由缓存直接提供服务的请求占比 |
| 并行化反模式 (Parallelization Anti-pattern) | “N 次写入陷阱” (N-write Trap) | N 个并行请求会导致 N 次缓存未命中 |
| 动态内容陷阱 (Dynamic Content Trap) | “提示词内含时间陷阱” (Time-in-prompt Trap) | 前缀中的动态字节会严重降低命中率 |
| RadixAttention | “副本内缓存” (Intra-replica Cache) | SGLang 框架中的前缀缓存实现方案 |

## 延伸阅读

- [Anthropic 提示词缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` 语义与 TTL 说明。
- [OpenAI 提示词缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动缓存行为与适用条件。
- [TianPan — 面向 LLM 生产环境的语义缓存](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — 通过提示词缓存降低 59% 的 LLM 成本](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — 提示词缓存](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
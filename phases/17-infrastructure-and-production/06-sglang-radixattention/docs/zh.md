# 面向前缀密集型工作负载的 SGLang 与 RadixAttention

> SGLang 将 KV 缓存 (KV Cache) 视为存储在基数树 (Radix Tree) 中的一等公民 (first-class) 级可复用资源。与 vLLM 采用先来先服务 (First-Come, First-Served, FCFS) 策略调度请求不同，SGLang 的缓存感知调度器 (Cache-Aware Scheduler) 会优先处理共享前缀更长的请求——这本质上是一种深度优先基数树遍历 (Depth-First Radix Traversal)，从而确保热点分支常驻于高带宽内存 (High Bandwidth Memory, HBM) 中。在 Llama 3.1 8B 模型上处理类似 ShareGPT 的 1K 提示词时，SGLang 的吞吐量达到约 16,200 tok/s，而 vLLM 约为 12,500 tok/s，领先约 29%。在前缀密集型 RAG 工作负载中，这一优势可达 6.4 倍。在语音克隆类工作负载中，缓存命中率超过 86%。截至 2026 年，该技术已部署于 xAI、LinkedIn、Cursor、Oracle、GCP、Azure 和 AWS 的 40 万+ GPU 上。需要注意的是，如果前缀排序不一致，6.4 倍的性能优势将荡然无存——排序正是工程师手中的关键杠杆。

**Type:** 学习
**Languages:** Python（标准库，简易基数树缓存 + 缓存感知调度器）
**Prerequisites:** 第 17 阶段 · 04（vLLM 服务内部机制），第 14 阶段（智能体 RAG）
**Time:** 约 75 分钟

## 学习目标

- 绘制 RadixAttention 架构图：说明前缀如何存储在基数树中，以及 KV 块 (KV Blocks) 如何在共享同一分支根节点的序列间复用。
- 解释缓存感知调度机制，并说明为何先来先服务 (FCFS) 策略不适用于前缀密集型流量。
- 根据前缀缓存命中率 (Prefix-Cache Hit Rate) 和提示词长度分布，计算工作负载的预期加速比。
- 指出能够实现 6.4 倍性能提升而非错失该优势的提示词排序规范。

## 问题背景

传统的模型服务架构将每个请求的提示词视为不透明数据。即使 5,000 个 RAG 请求都以相同的 2,000 token 系统提示词和相同的检索前导文本开头，vLLM 仍会将该 2,000 token 的前缀预填充 (Prefill) 5,000 次。GPU 因此反复执行相同的计算。

核心观察：在智能体 (Agentic) 和 RAG 工作负载中，提示词几乎总是共享较长的前缀。系统提示词、工具模式 (Tool Schemas)、少样本示例 (Few-Shot Examples)、检索头信息、对话历史——这些内容在不同请求间高度重复。如果只需为该前缀计算并存储一次 KV 缓存，后续请求即可直接复用，从而避免重复预填充。

RadixAttention 正是为此而生。Token 被索引至一棵基数树中；树中的每个节点都负责维护从根节点到该节点路径上对应 Token 序列的 KV 块。当新请求到来时，系统会遍历该树：只要路径上的 Token 匹配，即可直接复用对应节点的 KV 块。此时，预填充的计算开销仅与“新增”的后缀长度成正比，而不再取决于完整提示词的长度。

真正的挑战在于调度。假设两个请求共享 2,000 token 的前缀，而第三个请求仅共享该前缀的 200 token，理想的做法是将前两个请求集中处理，以确保长前缀持续驻留在高带宽内存 (HBM) 中。FCFS 策略则恰恰相反——它仅按到达顺序处理请求，这可能导致在下一个长前缀请求到来前，热点分支已被提前逐出 (Evict) 内存。

## 核心概念

### 基数树（Radix Tree）作为 KV 索引

基数树（Radix Tree，一种紧凑前缀树 Compact Trie）用于存储 Token 序列。每个节点拥有一个 Token 范围以及为该范围计算出的 KV 块（KV Blocks）。子节点会将序列扩展一个或多个 Token。

root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)

当新请求包含系统提示词（System Prompt）+ "Context: <doc A>" + "Question: Carol" 时，调度器（Scheduler）会进行遍历匹配：系统前缀匹配成功（复用 124 个块），doc-A 分支匹配成功（复用 31 个块），随后仅为 "Question: Carol" 分配全新的块（4 个块）。预填充（Prefill）成本：仅需处理 4 个新 Token 块。若无此树结构：需处理 160 个块。预填充开销节省约 40 倍。

### 缓存感知调度（Cache-Aware Scheduling）

如果缓存频繁抖动（Cache Churn），基于基数树的复用将毫无意义。两项关键策略如下：

1. **深度优先分发（Depth-First Dispatch）**。从队列中挑选下一个请求时，优先选择与当前运行请求集位于同一分支根部的请求。这能确保热点分支（Hot Branch）保持驻留。
2. **分支级而非块级的最近最少使用（LRU）策略**。淘汰整个分支（从使用时长最短的叶子节点开始），而非单个块，从而使缓存形态与基数树形态保持一致。

先来先服务（FCFS）策略同时违背了这两点。一个共享 2,000 个 Token 的请求排在一个仅共享 50 个 Token 的请求之后，结果导致 2,000 Token 的分支被淘汰，以便接纳那个 50 Token 的请求。

### 值得牢记的基准测试数据

- Llama 3.1 8B，H100 显卡，ShareGPT 1K 提示词：SGLang 约 16,200 tok/s，对比 vLLM 约 12,500 tok/s（领先约 29%）。
- 前缀密集型检索增强生成（RAG）（相同系统提示词 + 相同文档，仅问题不同）：SGLang 性能最高可达 6.4 倍。
- 语音克隆工作负载：前缀缓存命中率（Prefix-Cache Hit Rate）达 86.4%。
- SGLang 客户在生产环境中的命中率：50% 至 99% 不等，具体取决于提示词规范的严格程度。
- 截至 2026 年，已部署于超过 400,000 块 GPU 上。

### 顺序陷阱（Ordering Gotcha）

6.4 倍的性能提升依赖于一致的提示词模板（Prompt Template）顺序。如果客户端在某些请求中按 `[system, tools, context, history, question]` 构建提示词，而在其他请求中按 `[system, context, tools, history, question]` 构建，基数树将无法找到共享前缀。在人类看来是共享前缀的内容，对基数树而言却是两个完全不同的序列。

工程师的发力点：你的提示词模板就是缓存键（Cache Key）。固定顺序。将所有不可变内容（系统提示词、工具定义、模式 Schema）放在最前面。接着放置检索上下文。最后放置用户问题。切勿将动态内容穿插到前缀中。

研究中的真实案例：仅通过一次变更，将动态内容移出可缓存前缀，就使某次部署的缓存命中率从 7% 跃升至 74%。

### RadixAttention 的优劣场景

优势场景：
- RAG（相同的检索引导语，不同的问题）。
- 智能体（Agent）（相同的工具模式，不同的查询）。
- 带有长系统提示词的对话。
- 带有重复引导语的语音/视觉工作负载。

劣势场景（吞吐量回落至 vLLM 水平）：
- 使用唯一提示词的单次生成（代码补全、无系统提示词的开放式对话）。
- 动态提示词，即每个请求都将独特内容穿插到前缀中。

### 为何这是调度问题，而非单纯的算子（Kernel）问题

你可以将 KV 复用实现为一种底层算子（Kernel）优化技巧。SGLang 的核心洞察在于：只有当调度器保持热点分支常驻内存时，复用才能带来收益。在混合负载下，朴素的“有空闲就复用”策略会导致缓存频繁抖动。正是基于基数树索引的调度器，将这一算子技巧转化为了 29% 的生产环境性能优势。

### 与 vLLM 的协同与竞争

这两个系统并非严格的竞争对手。2026 年，vLLM 增加了前缀缓存功能（`--enable-prefix-caching`）以及缓存感知路由器（Rust 编写的 vLLM Router）。两者差距有所缩小，但并未完全消失——SGLang 的整个技术栈以基数树为核心设计；而 vLLM 则是后期集成该功能。对于以前缀复用为主的工作负载，SGLang 仍是默认首选。对于缺乏强前缀模式的通用服务场景，vLLM 的表现依然持平或更优。

## 使用它

`code/main.py` 实现了一个简易的基数树（radix tree）KV 缓存，以及包含两种策略的调度器（scheduler）：先来先服务（FCFS）与缓存感知（cache-aware）。该脚本使用相同的工作负载（workload）分别运行这两种策略，并报告前缀缓存命中率（prefix-cache hit rate）与吞吐量差异（throughput delta）。随后，它会运行一个“顺序打乱”的工作负载，以展示性能 6.4 倍的骤降。

## 交付它

本课时将生成 `outputs/skill-radix-scheduler-advisor.md` 文件。根据提供的工作负载描述（提示词模板结构、检索模式、并发租户数量），该文件会输出提示词排序方案，并给出是否采用 SGLang 的决策建议（go/no-go）。

## 练习

1. 运行 `code/main.py`。在相同工作负载下对比 FCFS 与缓存感知策略。性能差异源自何处——是预填充（prefill）阶段的计算节省、解码（decode）阶段的计算节省，还是队列延迟（queue delay）？
2. 修改工作负载，使提示词随机打乱 `[system, tools, context]` 的顺序。重新运行。命中率会发生什么变化？为什么？
3. 计算在 Llama 3.1 8B 模型上，将一个 2,000 token 的系统提示词作为单个基数分支常驻内存的高带宽内存（HBM）成本。将其与不使用前缀复用的 16 序列批处理（batch）成本进行对比。
4. 阅读 SGLang RadixAttention 论文。用三句话解释：为何在重前缀负载下，树形最近最少使用（LRU）淘汰策略优于块状 LRU。
5. 客户报告缓存命中率仅为 8%。列出三个可能的原因，并说明针对每个原因你会运行的诊断方法。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| RadixAttention | “SGLang 的那个功能” | 将 KV 缓存索引为基数树，使共享前缀能够复用缓存块 |
| 基数树（Radix tree） | “紧凑的字典树” | 树中每个节点拥有一个 token 范围及其对应的 KV 块 |
| 缓存感知调度器（Cache-aware scheduler） | “优先处理热分支” | 优先调度共享常驻分支请求的调度器 |
| 前缀缓存命中率（Prefix-cache hit rate） | “你的提示词有多少是免费的” | 通过复用 KV 块提供的提示词 token 所占的比例 |
| 先来先服务（FCFS） | “先到先得” | 会破坏前缀局部性的默认调度策略 |
| 分支级 LRU（Branch-level LRU） | “淘汰叶子节点” | 与基数树形状相匹配的淘汰策略 |
| 提示词模板排序（Prompt template ordering） | “缓存键” | 提示词组件的顺序决定了树能够共享哪些内容 |
| 系统提示词固定（System prompt pinning） | “常驻前缀” | 将不可变的系统部分固定驻留，以避免频繁淘汰抖动 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源代码与文档。
- [SGLang 文档](https://sgl-project.github.io/) — RadixAttention 与调度细节。
- [SGLang 论文 — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — 设计参考。
- [LMSYS 博客 — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — 基准测试数据与调度器设计依据。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 自身的类基数树实现，供对比参考。
# 提示词缓存 (Prompt Caching) 与上下文缓存 (Context Caching)

> 你的系统提示词 (System Prompt) 包含 4,000 个词元 (Token)。你的检索增强生成 (Retrieval-Augmented Generation, RAG) 上下文包含 20,000 个词元。每次请求你都会发送这两部分内容，并且每次都要为此付费。提示词缓存允许服务商在服务器端保持该前缀处于“预热”状态，复用时仅按正常费率的 10% 计费。若使用得当，它可将推理成本 (Inference Cost) 降低 50%–90%，并将首字延迟 (First-Token Latency) 缩短 40%–85%。

**类型：** 构建
**语言：** Python
**前置条件：** 第 11 阶段 · 01（提示词工程），第 11 阶段 · 05（上下文工程），第 11 阶段 · 11（缓存与成本）
**耗时：** 约 60 分钟

## 问题所在

一个编程智能体 (Coding Agent) 在对话的每一轮都会向 Claude 发送相同的 15,000 词元系统提示词。按输入词元单价 3 美元/百万词元计算，仅 20 轮对话的输入成本就高达 0.90 美元——这甚至还没算上用户的实际消息。如果每天进行 10,000 次对话，账单将飙升至每天 9,000 美元，而花费全在那些从未改变过的文本上。

你无法在不损害质量的前提下压缩提示词。你也无法避免发送它——模型在每一轮都需要它。唯一的出路是停止为服务商已经见过的前缀支付全额费用。

这一方案就是提示词缓存。Anthropic 于 2024 年 8 月推出了该功能（2025 年又增加了 1 小时延长生存时间 (Extended-TTL) 的变体），OpenAI 在同一年晚些时候实现了自动化支持，Google 则随 Gemini 1.5 推出了显式上下文缓存。如今，这三家厂商均已将其作为前沿模型 (Frontier Models) 的一等公民功能提供。

## 核心概念

![提示词缓存：一次写入，廉价读取](../assets/prompt-caching.svg)

**工作原理。** 当请求的前缀与近期某个请求的前缀匹配时，服务提供商会直接提供上一次运行生成的键值缓存（KV-cache），而无需重新编码词元（tokens）。首次使用时需支付少量的写入溢价，此后每次读取均可享受大幅折扣。

**2026 年三大服务商的实现方案。**

| 服务商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存量 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在内容块上显式使用 `cache_control` 标记 | 输入费用减免 90% | 加收 25% | 5 分钟（可延长至 1 小时） | 1,024 个词元（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入费用减免 50% | 无 | 最长 1 小时（尽力而为） | 1,024 个词元 |
| Google (Gemini) | 显式调用 `CachedContent` API | 按存储计费；读取费用约为正常价格的 25% | 按词元·小时收取存储费 | 用户自定义（默认 1 小时） | 4,096 个词元（Flash），32,768（Pro） |

**核心原则。** 这三家服务商均仅缓存前缀。如果请求间的任意词元存在差异，则从第一个不同词元开始的所有内容均视为缓存未命中。请将*稳定*的部分置于顶部，*可变*的部分置于底部。

### 适配缓存的布局结构

[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)

一旦打乱此顺序——例如将用户消息置于系统提示词（system prompt）之上，或在少样本示例（few-shot examples）之间穿插动态检索内容——缓存将永远无法命中。

### 盈亏平衡计算

Anthropic 25% 的写入溢价意味着，一个缓存块至少需要被读取两次才能实现净成本节约。1 次写入 + 1 次读取的平均单次请求成本为原价的 0.675 倍（节省 32%）；1 次写入 + 10 次读取的平均成本为 0.205 倍（节省 80%）。经验法则：对于在生存时间（TTL）内预计至少会复用 3 次的内容，都应进行缓存。

## 动手构建

### 步骤 1：使用显式标记的 Anthropic 提示词缓存（Prompt Caching）

import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )

`cache_control` 标记会指示 Anthropic 将该数据块缓存 5 分钟。在此时间窗口内复用将触发缓存命中（Cache Hit）；窗口过期后再次复用则会失效并重新写入。

**响应用量字段：**

response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x

在持续集成（CI）环境中检查这两个字段——如果 `cache_read_input_tokens` 在多次请求中始终为零，说明你的缓存键（Cache Keys）正在发生漂移。

### 步骤 2：一小时延长生存时间（TTL）

对于长时间运行的批处理任务，默认的 5 分钟缓存会在任务间隔期间过期。可通过设置 `ttl` 来延长：

{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}

1 小时的 TTL 会产生 2 倍的写入溢价（超出基准价 50%，而非 25%），但只要批处理任务复用该前缀超过 5 次，成本就能迅速收回。

### 步骤 3：OpenAI 自动缓存

OpenAI 无需任何手动配置。只要前缀超过 1,024 个词元（Tokens）且与近期请求匹配，系统就会自动给予 50% 的折扣。

from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion

同样适用对缓存友好的布局规则。有两种情况会破坏 OpenAI 的缓存，但不会影响 Anthropic：更改 `user` 字段（该字段被用作缓存键的组成部分）以及调整工具（Tools）的顺序。

### 步骤 4：Gemini 显式上下文缓存（Context Caching）

Gemini 将缓存视为一等对象（First-class Object），你需要显式创建并为其命名：

from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)

Gemini 会按缓存存活期间的词元·小时（Token·Hour）收取存储费用，读取费用约为正常输入费率的 25%。当你需要在数天内的多个会话中重复使用同一个超长提示词时，这种计费模式最为合适。

### 步骤 5：在生产环境中测量命中率（Hit Rate）

请参阅 `code/main.py`，其中包含一个模拟的三提供商成本核算器，用于跟踪写入/读取/未命中（Miss）次数，并计算每 1,000 次请求的综合成本。建议以目标命中率为门槛设置部署门禁（Deployment Gate）——在预热完成后，大多数生产环境的 Anthropic 配置应能达到 80% 以上的缓存读取比例。

## 2026年仍会踩中的陷阱

- **顶部动态时间戳。** 将 `"Current time: 2026-04-22 15:30:02"` 放在系统提示词（System Prompt）顶部会导致每次请求都缓存未命中（Cache Miss）。请将时间戳移至缓存断点（Cache Breakpoint）之后。
- **工具顺序重排。** 以稳定的顺序序列化工具——部署之间的字典（Dict）重排会导致所有缓存命中（Cache Hit）失效。
- **自由文本近似重复。** "You are helpful." 与 "You are a helpful assistant." ——仅一个字节之差就会导致完全未命中。
- **缓存块过小。** Anthropic 强制要求最低 1,024 个 Token（Haiku 模型为 2,048）。小于该阈值的块将静默失效，无法触发缓存。
- **缺乏明细的成本看板。** 将“输入 Token（Input Tokens）”拆分为已缓存与未缓存两部分。否则，流量下降会被误认为是缓存带来的收益。

## 实际应用

2026 年缓存技术栈选型：

| 场景 | 推荐方案 |
|-----------|------|
| 拥有稳定 10k+ 系统提示词且多轮对话的智能体（Agent） | 使用 Anthropic 的 `cache_control`，设置 5 分钟 TTL（Time To Live） |
| 批量任务在 30 分钟以上复用相同前缀 | 使用 Anthropic 并设置 `ttl: "1h"` |
| 基于 GPT-5 的无服务器（Serverless）端点，无自定义基础设施 | OpenAI 自动缓存（只需确保前缀稳定且足够长） |
| 跨多日复用大型代码/文档语料库 | Gemini 显式使用 `CachedContent` |
| 跨提供商降级/容灾 | 在各提供商间保持可缓存前缀的布局一致，以确保任一缓存命中均可生效 |

在用户消息层结合语义缓存（Semantic Caching，第 11 阶段 · 第 11 课）使用：提示词缓存（Prompt Caching）处理 *Token 完全一致* 的复用，语义缓存处理 *语义一致* 的复用。

## 交付上线

保存至 `outputs/skill-prompt-caching-planner.md`：

---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.

## 练习

1. **简单。** 使用包含 5,000 个 token 的系统提示词（system prompt）与 Claude 进行 10 轮对话。分别在不启用和启用 `cache_control` 的情况下运行该对话，并报告各自的输入 token 费用。
2. **中等。** 编写一个测试框架（test harness），在给定提示词模板（prompt template）和请求日志（request log）的情况下，计算各服务提供商的预期命中率（hit rate）及节省的费用（Anthropic 5分钟、Anthropic 1小时、OpenAI 自动模式、Gemini 显式模式）。
3. **困难。** 构建一个布局优化器（layout optimizer）：给定一个提示词和标记为 `stable=True/False` 的字段列表，重写该提示词，在不丢失信息的前提下，将单个缓存断点（cache breakpoint）置于最有利于缓存的位置。在真实的 Anthropic 端点（endpoint）上进行验证。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 提示词缓存（Prompt caching） | “让长提示词变便宜” | 复用服务提供商侧的键值缓存（KV-cache）以匹配前缀；对重复的输入 token 提供 50-90% 的折扣。 |
| `cache_control` | “Anthropic 标记” | 内容块属性，用于声明“到此为止的内容均可缓存”；值为 `{"type": "ephemeral"}`。 |
| 缓存写入（Cache write） | “支付溢价” | 首次填充缓存的请求；在 Anthropic 上按约 1.25 倍的输入费率计费，在 OpenAI 上免费。 |
| 缓存读取（Cache read） | “享受折扣” | 后续匹配该前缀的请求；按 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）的费率计费。 |
| 生存时间（TTL） | “缓存存活多久” | 缓存保持活跃状态的秒数；Anthropic 默认为 5 分钟（可延长至 1 小时），OpenAI 尽力维持最长 1 小时，Gemini 由用户自定义。 |
| 延长生存时间（Extended TTL） | “1 小时 Anthropic 缓存” | 配置为 `{"type": "ephemeral", "ttl": "1h"}`；写入溢价翻倍，但非常适合批量重复使用。 |
| 前缀匹配（Prefix match） | “为什么我的缓存未命中” | 仅当从开头到断点的每一个 token 在字节级别完全一致时，缓存才会命中。 |
| 上下文缓存（Context caching，Gemini） | “显式缓存” | Google 提供的具名且按存储计费的缓存对象；最适合大型语料库的多日重复使用。 |

## 延伸阅读

- [Anthropic — 提示词缓存 (Prompt Caching)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1小时生存时间 (TTL)、成本平衡表。
- [OpenAI — 提示词缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — 上下文缓存 (Context Caching)](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与存储定价。
- [Anthropic 工程团队 — 面向长上下文工作负载的提示词缓存](https://www.anthropic.com/news/prompt-caching) — 首发博文，附带延迟数据。
- 第11阶段 · 05（上下文工程 (Context Engineering)）—— 提示词应在何处切分以便命中缓存。
- 第11阶段 · 11（缓存与成本 (Caching and Cost)）—— 将提示词缓存与用户消息的语义缓存 (Semantic Cache) 结合使用。
- [Pope 等人，《Efficiently Scaling Transformer Inference》（2022年）](https://arxiv.org/abs/2211.05102) —— 提示词缓存向用户暴露的 KV缓存 (KV-Cache) 内存模型；解释了为何读取已缓存的前缀比重新计算的成本低约10倍。
- [Agrawal 等人，《SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills》（2023年）](https://arxiv.org/abs/2308.16369) —— 预填充 (Prefill) 阶段正是提示词缓存所跳过的环节；本文解释了为何在缓存命中 (Cache Hit) 时首字延迟 (TTFT) 会显著下降，而词元生成时间 (TPOT) 不受影响。
- [Leviathan 等人，《Fast Inference from Transformers via Speculative Decoding》（2023年）](https://arxiv.org/abs/2211.17192) —— 提示词缓存与投机解码 (Speculative Decoding)、Flash Attention 以及 MQA/GQA 同为优化推理成本曲线的关键杠杆；如需了解其他三项技术，请阅读此文。
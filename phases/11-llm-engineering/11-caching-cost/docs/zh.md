# 缓存 (Caching)、速率限制 (Rate Limiting) 与成本优化 (Cost Optimization)

> 大多数 AI 初创公司并非死于模型不佳，而是死于糟糕的单位经济效益 (Unit Economics)。单次 GPT-4o 调用仅需几分之一美分。但如果一万个用户每天各调用十次，仅输入 Token (Input Tokens) 的成本就高达 250 美元——这甚至还没开始向你收取一分钱。能够存活下来的公司，会将每一次 API 调用 (API Call) 视为一笔财务交易，而非单纯的函数调用。

**类型：** 构建
**语言：** Python
**前置条件：** 第 11 阶段 第 09 课（函数调用 (Function Calling)）
**时长：** 约 45 分钟
**相关课程：** 第 11 阶段 · 第 15 课（提示词缓存 (Prompt Caching)）—— 本课涵盖应用层缓存 (Application-layer Caching)（语义缓存 (Semantic Cache)、精确哈希缓存 (Exact Hash Cache)、模型路由 (Model Routing)）。第 15 课涵盖提供商层提示词缓存（Anthropic cache_control、OpenAI 自动缓存、Gemini CachedContent）。将两者结合可实现 50-95% 的成本降低。

## 学习目标

- 实现语义缓存 (Semantic Caching)，直接从缓存中响应重复或相似的查询，而非发起新的 API 调用
- 计算跨提供商的单次请求成本，并实现基于 Token 感知的速率限制 (Token-aware Rate Limiting) 与预算告警 (Budget Alerts)
- 构建成本优化层，集成提示词压缩 (Prompt Compression)、模型路由（区分昂贵与廉价模型）以及响应缓存 (Response Caching)
- 针对不同查询类型，设计分层缓存策略 (Tiered Caching Strategy)，结合精确匹配 (Exact Match)、语义相似度 (Semantic Similarity) 与前缀缓存 (Prefix Caching)

## 问题所在

你构建了一个检索增强生成 (Retrieval-Augmented Generation, RAG) 聊天机器人。它运行完美，用户非常喜欢。

直到账单寄来。

GPT-5 的定价为每百万输入 Token 5 美元，每百万输出 Token 15 美元。Claude Opus 4.7 为输入 15 美元 / 输出 75 美元。Gemini 3 Pro 为输入 1.25 美元 / 输出 5 美元。GPT-5-mini 为 0.25 美元 / 2 美元。以下价格仅为示例；请务必查阅各提供商最新的定价页面。

以下是拖垮初创公司的数学账：

- 10,000 名日活跃用户
- 每位用户每天 10 次查询
- 每次查询 1,000 个输入 Token（系统提示词 + 上下文 + 用户消息）
- 每次响应 500 个输出 Token

**每日输入成本：** 10,000 x 10 x 1,000 / 1,000,000 x $2.50 = **$250/天**
**每日输出成本：** 10,000 x 10 x 500 / 1,000,000 x $10.00 = **$500/天**
**月度总计：** **$22,500/月**

这还仅仅是大语言模型 (Large Language Model, LLM) 的费用。再加上嵌入向量 (Embeddings)、向量数据库 (Vector Database) 托管和基础设施开销，一个聊天机器人的月成本将直奔 30,000 美元。

残酷的现实在于：40-60% 的查询几乎是重复的。用户只是用略微不同的措辞询问相同的问题。你的系统提示词在每次请求中完全一致，却每次都被计费。通过 RAG 检索的上下文文档，在询问相同主题的不同用户之间也会大量重复。

你正在为冗余计算支付全额费用。

## 核心概念

### LLM（大语言模型）调用的成本构成

每次 API 调用都包含五个成本组成部分。

graph LR
    A[User Query] --> B[System Prompt<br/>500-2000 tokens]
    A --> C[Retrieved Context<br/>500-4000 tokens]
    A --> D[User Message<br/>50-500 tokens]
    B --> E[Input Cost<br/>$2.50/1M tokens]
    C --> E
    D --> E
    E --> F[Model Processing]
    F --> G[Output Cost<br/>$10.00/1M tokens]

系统提示词（System Prompt）是隐形的成本杀手。每次请求都附带一个 1,500 Token 的系统提示词，仅此前缀部分在每百万次请求中就会产生 3.75 美元的成本。若每天处理 10 万次请求，这笔费用将达到每天 375 美元——每月 11,250 美元——而这段文本实际上从未改变过。

### 供应商缓存（Provider Caching）：内置折扣

到 2026 年，三大主流供应商均提供供应商侧的提示词缓存（Prompt Caching）功能，但具体机制各不相同。深入解析请参阅“第 11 阶段 · 15”。

| Provider | Mechanism | Discount | Minimum | Cache Duration |
|----------|-----------|----------|---------|----------------|
| Anthropic | 显式 `cache_control` 标记 | 缓存命中时 90%（写入时额外支付 25%） | 1,024 Token（Sonnet/Opus），2,048（Haiku） | 默认 5 分钟；延长至 1 小时（写入溢价翻倍） |
| OpenAI | 自动前缀匹配 | 缓存命中时 50% | 1,024 Token | 尽力而为，最长 1 小时 |
| Google Gemini | 显式 `CachedContent` API | 降低约 75%（另加存储费） | 4,096（Flash）/ 32,768（Pro） | 用户可配置的 TTL（生存时间） |

**Anthropic 的方案**采用显式标记。你需要使用 `cache_control: {"type": "ephemeral"}` 标记提示词中的特定部分。首次请求需支付 25% 的写入溢价。后续具有相同前缀的请求可享受 90% 的折扣。一个正常成本为 0.005 美元的 2,000 Token 系统提示词，在缓存命中时成本仅为 0.000625 美元。在 10 万次请求中，这每天可节省 437.50 美元。

**OpenAI 的方案**是自动化的。任何与历史请求匹配的提示词前缀均可获得 50% 的折扣，无需添加标记。其权衡在于：折扣力度较小、控制力较弱，但实现成本为零。

### 语义缓存（Semantic Caching）：自定义缓存层

供应商缓存仅对完全相同的前缀有效。语义缓存则处理更复杂的情况：字面不同但语义相同的查询。

“退货政策是什么？”和“我该如何退货？”是两个不同的字符串，但意图完全一致。语义缓存会对这两个查询进行嵌入（Embedding）处理，计算余弦相似度（Cosine Similarity），若相似度超过设定阈值（通常为 0.92-0.95），则直接返回缓存的响应。

flowchart TD
    A[User Query] --> B[Embed Query]
    B --> C{Similar query<br/>in cache?}
    C -->|sim > 0.95| D[Return Cached Response]
    C -->|sim < 0.95| E[Call LLM API]
    E --> F[Cache Response<br/>with Embedding]
    F --> G[Return Response]
    D --> G

嵌入处理的成本微乎其微。OpenAI 的 `text-embedding-3-small` 模型每百万 Token 仅需 0.02 美元。与完整的 LLM 调用相比，检查缓存的成本几乎可以忽略不计。

### 精确缓存（Exact Caching）：哈希匹配

对于确定性调用（`temperature=0`、相同模型、相同提示词），精确缓存更为简单高效。只需对完整提示词进行哈希（Hash）计算，检查缓存，若命中则直接返回。

该方案非常适用于以下场景：
- 系统提示词 + 固定上下文 + 完全相同的用户查询
- 使用相同工具定义（Tool Definitions）的函数调用（Function Calling）
- 同一文档被多次处理的批处理任务

### 速率限制（Rate Limiting）：保护你的预算

速率限制不仅关乎公平性，更关乎系统的生存。

**令牌桶算法（Token Bucket Algorithm）**：每个用户分配一个容量为 N 的令牌桶，以每秒 R 的速率补充令牌。每次请求会消耗桶中的令牌。若桶为空，则拒绝请求。该机制在强制执行平均速率的同时，允许突发流量（一次性消耗完整桶容量）。

**单用户配额**：根据用户层级设置每日/每月的 Token 限制。

| Tier | Daily Token Limit | Max Requests/min | Model Access |
|------|------------------|------------------|-------------|
| Free | 50,000 | 10 | 仅限 GPT-4o-mini |
| Pro | 500,000 | 60 | GPT-4o, Claude Sonnet |
| Enterprise | 5,000,000 | 300 | 所有模型 |

### 模型路由（Model Routing）：为任务匹配合适的模型

并非所有查询都需要调用 GPT-4o。

“商店几点关门？”这类问题根本不需要使用输出成本为 10 美元/百万 Token 的模型。输出成本仅 0.60 美元/百万 Token 的 GPT-4o-mini 即可完美处理。输出成本为 1.25 美元/百万 Token 的 Claude Haiku 同样胜任。通过一个简单的分类器，即可将低成本查询路由至廉价模型，将复杂查询路由至昂贵模型。

flowchart TD
    A[User Query] --> B[Complexity Classifier]
    B -->|Simple: lookup, FAQ| C[GPT-4o-mini<br/>$0.15/$0.60 per 1M]
    B -->|Medium: analysis, summary| D[Claude Sonnet<br/>$3.00/$15.00 per 1M]
    B -->|Complex: reasoning, code| E[GPT-4o / Claude Opus<br/>$2.50/$10.00+]

一个调优良好的路由器仅模型成本一项即可节省 40%-70%。

### 成本追踪（Cost Tracking）：掌握资金流向

无法衡量，就无法优化。请记录每次 API 调用的以下信息：

- 时间戳
- 模型名称
- 输入 Token 数
- 输出 Token 数
- 延迟（毫秒）
- 计算成本（美元）
- 用户 ID
- 缓存命中/未命中
- 请求类别

这些数据能揭示哪些功能成本高昂、哪些用户是重度消耗者，以及缓存在何处能发挥最大效用。

### 批处理（Batching）：批量折扣

OpenAI 的 `Batch API` 以异步方式处理请求，并提供 50% 的折扣。你只需提交最多 50,000 个请求的批次，结果将在 24 小时内返回。

批处理适用于：
- 夜间文档处理
- 批量分类
- 评估任务运行
- 数据增强流水线

不适用于：面向用户的实时查询（延迟敏感型场景）。

### 预算告警与熔断机制（Circuit Breakers）

熔断机制会在达到预设限额时自动停止支出。若缺乏此机制，一个漏洞或恶意滥用可能在几小时内耗尽你的月度预算。

设置三个阈值：
1. **警告**（预算的 70%）：发送告警通知
2. **限流**（预算的 85%）：仅切换至成本更低的模型
3. **停止**（预算的 95%）：拒绝新请求，仅返回缓存响应

### 优化技术栈（Optimization Stack）

请按顺序应用这些技术。每一层的效果都会在前一层的基础上叠加。

| Layer | Technique | Typical Savings | Implementation Effort |
|-------|-----------|----------------|----------------------|
| 1 | 供应商提示词缓存 | 30-50% | 低（添加缓存标记） |
| 2 | 精确缓存 | 10-20% | 低（哈希 + 字典） |
| 3 | 语义缓存 | 15-30% | 中（嵌入向量 + 相似度计算） |
| 4 | 模型路由 | 40-70% | 中（分类器） |
| 5 | 速率限制 | 预算保护 | 低（令牌桶） |
| 6 | 提示词压缩 | 10-30% | 中（重写提示词） |
| 7 | 批处理 | 符合条件者 50% | 低（Batch API） |

应用了第 1 至 5 层技术的 RAG（检索增强生成）应用，通常能将成本从每月 22,500 美元降至 4,000-6,000 美元。这决定了你是在烧光融资跑道，还是在稳健地构建业务。

### 实际节省：优化前后对比

以下是一个服务于 10,000 日活跃用户（DAU）的 RAG 聊天机器人的真实成本明细。

| Metric | Before Optimization | After Optimization | Savings |
|--------|--------------------|--------------------|---------|
| 月度 LLM 成本 | $22,500 | $5,200 | 77% |
| 单次查询平均成本 | $0.0075 | $0.0017 | 77% |
| 缓存命中率 | 0% | 52% | -- |
| 路由至 mini 模型的查询比例 | 0% | 65% | -- |
| P95 延迟 | 2,800ms | 900ms（缓存命中：50ms） | 68% |
| 月度嵌入成本 | $0 | $180 | （新增成本） |
| 月度总成本 | $22,500 | $5,380 | 76% |

语义缓存的嵌入成本（每月 180 美元）在缓存命中的第一个小时内即可收回成本。

## 构建项目

### 步骤 1：成本计算器 (Cost Calculator)

构建一个 Token 成本计算器，用于掌握主流模型的当前定价。

import hashlib
import time
import json
import math
from dataclasses import dataclass, field


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
    "claude-opus-4": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.3125},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cached_input": 0.0375},
}


def calculate_cost(model, input_tokens, output_tokens, cached_input_tokens=0):
    if model not in MODEL_PRICING:
        return {"error": f"Unknown model: {model}"}
    pricing = MODEL_PRICING[model]
    non_cached = input_tokens - cached_input_tokens
    input_cost = (non_cached / 1_000_000) * pricing["input"]
    cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + cached_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "input_cost": round(input_cost, 6),
        "cached_input_cost": round(cached_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }

### 步骤 2：精确缓存 (Exact Cache)

对完整提示词 (Prompt) 进行哈希 (Hash) 处理，并为完全相同的请求返回缓存的响应。

class ExactCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash(self, model, messages, temperature):
        key_data = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, model, messages, temperature=0.0):
        if temperature > 0:
            self.misses += 1
            return None
        key = self._hash(model, messages, temperature)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                entry["access_count"] += 1
                return entry["response"]
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, model, messages, temperature, response):
        if temperature > 0:
            return
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        key = self._hash(model, messages, temperature)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        }

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.cache),
        }

### 步骤 3：语义缓存 (Semantic Cache)

对查询进行文本嵌入 (Embedding) 处理，当语义相似度超过预设阈值时，返回缓存的响应。

def simple_embed(text):
    words = text.lower().split()
    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    norm = math.sqrt(sum(v * v for v in vocab.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vocab.items()}


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    all_keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    return dot


class SemanticCache:
    def __init__(self, similarity_threshold=0.85, max_size=500, ttl_seconds=3600):
        self.entries = []
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = simple_embed(query)
        now = time.time()
        best_match = None
        best_sim = 0.0
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry
        if best_match and best_sim >= self.threshold:
            self.hits += 1
            best_match["access_count"] += 1
            return {"response": best_match["response"], "similarity": round(best_sim, 4), "original_query": best_match["query"]}
        self.misses += 1
        return None

    def put(self, query, response):
        if len(self.entries) >= self.max_size:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries.pop(0)
        self.entries.append({
            "query": query,
            "embedding": simple_embed(query),
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        })

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.entries),
        }

### 步骤 4：速率限制器 (Rate Limiter)

基于令牌桶 (Token Bucket) 算法的速率限制器，支持按用户分配调用配额。

class TokenBucketRateLimiter:
    def __init__(self):
        self.buckets = {}
        self.tiers = {
            "free": {"capacity": 50_000, "refill_rate": 500, "max_requests_per_min": 10},
            "pro": {"capacity": 500_000, "refill_rate": 5_000, "max_requests_per_min": 60},
            "enterprise": {"capacity": 5_000_000, "refill_rate": 50_000, "max_requests_per_min": 300},
        }

    def _get_bucket(self, user_id, tier="free"):
        if user_id not in self.buckets:
            tier_config = self.tiers.get(tier, self.tiers["free"])
            self.buckets[user_id] = {
                "tokens": tier_config["capacity"],
                "capacity": tier_config["capacity"],
                "refill_rate": tier_config["refill_rate"],
                "last_refill": time.time(),
                "request_timestamps": [],
                "max_rpm": tier_config["max_requests_per_min"],
                "tier": tier,
                "total_tokens_used": 0,
            }
        return self.buckets[user_id]

    def _refill(self, bucket):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill = int(elapsed * bucket["refill_rate"])
        if refill > 0:
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + refill)
            bucket["last_refill"] = now

    def check(self, user_id, tokens_needed, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        self._refill(bucket)
        now = time.time()
        bucket["request_timestamps"] = [t for t in bucket["request_timestamps"] if now - t < 60]
        if len(bucket["request_timestamps"]) >= bucket["max_rpm"]:
            return {"allowed": False, "reason": "rate_limit", "retry_after_seconds": 60 - (now - bucket["request_timestamps"][0])}
        if bucket["tokens"] < tokens_needed:
            deficit = tokens_needed - bucket["tokens"]
            wait = deficit / bucket["refill_rate"]
            return {"allowed": False, "reason": "token_limit", "tokens_available": bucket["tokens"], "retry_after_seconds": round(wait, 1)}
        return {"allowed": True, "tokens_available": bucket["tokens"]}

    def consume(self, user_id, tokens_used, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        bucket["tokens"] -= tokens_used
        bucket["request_timestamps"].append(time.time())
        bucket["total_tokens_used"] += tokens_used

    def get_usage(self, user_id):
        if user_id not in self.buckets:
            return {"error": "User not found"}
        b = self.buckets[user_id]
        return {
            "user_id": user_id,
            "tier": b["tier"],
            "tokens_remaining": b["tokens"],
            "capacity": b["capacity"],
            "total_tokens_used": b["total_tokens_used"],
            "utilization": round(b["total_tokens_used"] / b["capacity"], 4) if b["capacity"] else 0,
        }

### 步骤 5：成本追踪器 (Cost Tracker)

记录每次 API 调用日志，并实时计算累计成本总额。

class CostTracker:
    def __init__(self, monthly_budget=1000.0):
        self.logs = []
        self.monthly_budget = monthly_budget
        self.alerts = []

    def log_call(self, model, input_tokens, output_tokens, cached_input_tokens=0, latency_ms=0, user_id="anonymous", cache_status="miss"):
        cost = calculate_cost(model, input_tokens, output_tokens, cached_input_tokens)
        entry = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "latency_ms": latency_ms,
            "cost": cost["total_cost"],
            "user_id": user_id,
            "cache_status": cache_status,
        }
        self.logs.append(entry)
        self._check_budget()
        return entry

    def _check_budget(self):
        total = self.total_cost()
        pct = total / self.monthly_budget if self.monthly_budget > 0 else 0
        if pct >= 0.95 and not any(a["level"] == "stop" for a in self.alerts):
            self.alerts.append({"level": "stop", "message": f"Budget 95% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.85 and not any(a["level"] == "throttle" for a in self.alerts):
            self.alerts.append({"level": "throttle", "message": f"Budget 85% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.70 and not any(a["level"] == "warning" for a in self.alerts):
            self.alerts.append({"level": "warning", "message": f"Budget 70% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})

    def total_cost(self):
        return round(sum(e["cost"] for e in self.logs), 6)

    def cost_by_model(self):
        by_model = {}
        for e in self.logs:
            m = e["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0}
            by_model[m]["calls"] += 1
            by_model[m]["cost"] = round(by_model[m]["cost"] + e["cost"], 6)
            by_model[m]["input_tokens"] += e["input_tokens"]
            by_model[m]["output_tokens"] += e["output_tokens"]
        return by_model

    def cache_savings(self):
        cache_hits = [e for e in self.logs if e["cache_status"] == "hit"]
        if not cache_hits:
            return {"saved": 0, "cache_hits": 0}
        saved = 0
        for e in cache_hits:
            full_cost = calculate_cost(e["model"], e["input_tokens"], e["output_tokens"])
            saved += full_cost["total_cost"]
        return {"saved": round(saved, 4), "cache_hits": len(cache_hits)}

    def summary(self):
        if not self.logs:
            return {"total_calls": 0, "total_cost": 0}
        total_latency = sum(e["latency_ms"] for e in self.logs)
        cache_hits = sum(1 for e in self.logs if e["cache_status"] == "hit")
        return {
            "total_calls": len(self.logs),
            "total_cost": self.total_cost(),
            "avg_cost_per_call": round(self.total_cost() / len(self.logs), 6),
            "avg_latency_ms": round(total_latency / len(self.logs), 1),
            "cache_hit_rate": round(cache_hits / len(self.logs), 4),
            "cost_by_model": self.cost_by_model(),
            "cache_savings": self.cache_savings(),
            "budget_remaining": round(self.monthly_budget - self.total_cost(), 2),
            "budget_utilization": round(self.total_cost() / self.monthly_budget, 4) if self.monthly_budget > 0 else 0,
            "alerts": self.alerts,
        }

### 步骤 6：模型路由 (Model Router)

根据查询复杂度，将请求智能路由至能够胜任且成本最低的模型。

SIMPLE_KEYWORDS = ["what time", "hours", "address", "phone", "price", "return policy", "hello", "hi", "thanks", "yes", "no"]
COMPLEX_KEYWORDS = ["analyze", "compare", "explain why", "write code", "debug", "architect", "design", "trade-off", "evaluate"]


def classify_complexity(query):
    q = query.lower()
    if len(q.split()) <= 5 or any(kw in q for kw in SIMPLE_KEYWORDS):
        return "simple"
    if any(kw in q for kw in COMPLEX_KEYWORDS):
        return "complex"
    return "medium"


def route_model(query, tier="pro"):
    complexity = classify_complexity(query)
    routing_table = {
        "simple": {"free": "gpt-4.1-nano", "pro": "gpt-4o-mini", "enterprise": "gpt-4o-mini"},
        "medium": {"free": "gpt-4o-mini", "pro": "claude-sonnet-4", "enterprise": "claude-sonnet-4"},
        "complex": {"free": "gpt-4o-mini", "pro": "gpt-4o", "enterprise": "claude-opus-4"},
    }
    model = routing_table[complexity].get(tier, "gpt-4o-mini")
    return {"query": query, "complexity": complexity, "model": model, "tier": tier}

### 步骤 7：运行演示 (Run the Demo)

def simulate_llm_call(model, query):
    input_tokens = len(query.split()) * 4 + 500
    output_tokens = 150 + (len(query.split()) * 2)
    latency = 200 + (output_tokens * 2)
    return {
        "model": model,
        "response": f"[Simulated {model} response to: {query[:50]}...]",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency,
    }


def run_demo():
    print("=" * 60)
    print("  Caching, Rate Limiting & Cost Optimization Demo")
    print("=" * 60)

    print("\n--- Model Pricing ---")
    for model, pricing in list(MODEL_PRICING.items())[:6]:
        cost_1k = calculate_cost(model, 1000, 500)
        print(f"  {model}: ${cost_1k['total_cost']:.6f} per 1K in + 500 out")

    print("\n--- Cost Comparison: 100K Requests ---")
    for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "claude-haiku-3.5"]:
        cost = calculate_cost(model, 1000 * 100_000, 500 * 100_000)
        print(f"  {model}: ${cost['total_cost']:.2f}")

    print("\n--- Anthropic Cache Savings ---")
    no_cache = calculate_cost("claude-sonnet-4", 2000, 500, 0)
    with_cache = calculate_cost("claude-sonnet-4", 2000, 500, 1500)
    saving = no_cache["total_cost"] - with_cache["total_cost"]
    print(f"  Without cache: ${no_cache['total_cost']:.6f}")
    print(f"  With 1500 cached tokens: ${with_cache['total_cost']:.6f}")
    print(f"  Savings per call: ${saving:.6f} ({saving/no_cache['total_cost']*100:.1f}%)")

    exact_cache = ExactCache(max_size=100, ttl_seconds=300)
    semantic_cache = SemanticCache(similarity_threshold=0.75, max_size=100)
    rate_limiter = TokenBucketRateLimiter()
    tracker = CostTracker(monthly_budget=100.0)

    print("\n--- Exact Cache ---")
    messages_1 = [{"role": "user", "content": "What is the return policy?"}]
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  First lookup: {'HIT' if result else 'MISS'}")
    exact_cache.put("gpt-4o-mini", messages_1, 0.0, "You can return items within 30 days.")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  Second lookup: {'HIT' if result else 'MISS'} -> {result}")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.7)
    print(f"  With temp=0.7: {'HIT' if result else 'MISS (non-deterministic, skip cache)'}")
    print(f"  Stats: {exact_cache.stats()}")

    print("\n--- Semantic Cache ---")
    test_queries = [
        ("What is the return policy?", "Items can be returned within 30 days with receipt."),
        ("How do I return an item?", None),
        ("What are your store hours?", "We are open 9am-9pm Monday through Saturday."),
        ("When does the store open?", None),
        ("Tell me about quantum computing", "Quantum computers use qubits..."),
        ("Explain quantum mechanics", None),
    ]
    for query, response in test_queries:
        cached = semantic_cache.get(query)
        if cached:
            print(f"  '{query[:40]}' -> CACHE HIT (sim={cached['similarity']}, original='{cached['original_query'][:40]}')")
        elif response:
            semantic_cache.put(query, response)
            print(f"  '{query[:40]}' -> MISS (stored)")
        else:
            print(f"  '{query[:40]}' -> MISS (no match)")
    print(f"  Stats: {semantic_cache.stats()}")

    print("\n--- Rate Limiting ---")
    for i in range(12):
        check = rate_limiter.check("user_1", 1000, "free")
        if check["allowed"]:
            rate_limiter.consume("user_1", 1000, "free")
        status = "OK" if check["allowed"] else f"BLOCKED ({check['reason']})"
        if i < 5 or not check["allowed"]:
            print(f"  Request {i+1}: {status}")
    print(f"  Usage: {rate_limiter.get_usage('user_1')}")

    print("\n--- Model Routing ---")
    routing_queries = [
        "What time do you close?",
        "Summarize this quarterly earnings report",
        "Analyze the trade-offs between microservices and monoliths",
        "Hello",
        "Write code for a binary search tree with deletion",
    ]
    for q in routing_queries:
        route = route_model(q, "pro")
        print(f"  '{q[:50]}' -> {route['model']} ({route['complexity']})")

    print("\n--- Full Pipeline: Before vs After Optimization ---")
    queries = [
        "What is the return policy?",
        "How do I return something?",
        "What are your hours?",
        "When do you open?",
        "Explain the difference between TCP and UDP",
        "Compare TCP vs UDP protocols",
        "Hello",
        "What is your phone number?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of serverless architecture",
    ]

    print("\n  [Before: no caching, single model (gpt-4o)]")
    tracker_before = CostTracker(monthly_budget=1000.0)
    for q in queries:
        result = simulate_llm_call("gpt-4o", q)
        tracker_before.log_call("gpt-4o", result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
    before = tracker_before.summary()
    print(f"  Total cost: ${before['total_cost']:.6f}")
    print(f"  Avg cost/call: ${before['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {before['avg_latency_ms']}ms")

    print("\n  [After: caching + routing + rate limiting]")
    exact_c = ExactCache()
    semantic_c = SemanticCache(similarity_threshold=0.75)
    tracker_after = CostTracker(monthly_budget=1000.0)

    for q in queries:
        messages = [{"role": "user", "content": q}]
        cached = exact_c.get("gpt-4o", messages, 0.0)
        if cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=5, cache_status="hit")
            continue
        sem_cached = semantic_c.get(q)
        if sem_cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=15, cache_status="hit")
            continue
        route = route_model(q)
        result = simulate_llm_call(route["model"], q)
        tracker_after.log_call(route["model"], result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
        exact_c.put(route["model"], messages, 0.0, result["response"])
        semantic_c.put(q, result["response"])

    after = tracker_after.summary()
    print(f"  Total cost: ${after['total_cost']:.6f}")
    print(f"  Avg cost/call: ${after['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {after['avg_latency_ms']}ms")
    print(f"  Cache hit rate: {after['cache_hit_rate']:.0%}")

    if before["total_cost"] > 0:
        savings_pct = (1 - after["total_cost"] / before["total_cost"]) * 100
        print(f"\n  SAVINGS: {savings_pct:.1f}% cost reduction")
        print(f"  Latency improvement: {(1 - after['avg_latency_ms'] / before['avg_latency_ms']) * 100:.1f}% faster")

    print("\n--- Budget Alerts Demo ---")
    alert_tracker = CostTracker(monthly_budget=0.01)
    for i in range(5):
        alert_tracker.log_call("gpt-4o", 5000, 2000, latency_ms=500)
    print(f"  Total spent: ${alert_tracker.total_cost():.6f} / ${alert_tracker.monthly_budget}")
    for alert in alert_tracker.alerts:
        print(f"  ALERT [{alert['level'].upper()}]: {alert['message']}")

    print("\n--- Cost Breakdown by Model ---")
    multi_tracker = CostTracker(monthly_budget=500.0)
    for _ in range(50):
        multi_tracker.log_call("gpt-4o-mini", 800, 200, latency_ms=150)
    for _ in range(30):
        multi_tracker.log_call("claude-sonnet-4", 1500, 500, latency_ms=400)
    for _ in range(10):
        multi_tracker.log_call("gpt-4o", 2000, 800, latency_ms=600)
    for _ in range(10):
        multi_tracker.log_call("claude-opus-4", 3000, 1000, latency_ms=1200)
    breakdown = multi_tracker.cost_by_model()
    for model, data in sorted(breakdown.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {model}: {data['calls']} calls, ${data['cost']:.6f}, {data['input_tokens']:,} in / {data['output_tokens']:,} out")
    print(f"  Total: ${multi_tracker.total_cost():.6f}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()


## Use It

### Anthropic 提示词缓存 (Prompt Caching)

# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     system=[
#         {
#             "type": "text",
#             "text": "You are a helpful customer support agent for Acme Corp...",
#             "cache_control": {"type": "ephemeral"},
#         }
#     ],
#     messages=[{"role": "user", "content": "What is the return policy?"}],
# )
#
# print(f"Input tokens: {response.usage.input_tokens}")
# print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
# print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")

首次调用会将内容写入缓存（Cache，需支付 25% 的溢价）。此后，所有使用相同系统提示词前缀（System Prompt Prefix）的调用都将从缓存中读取（享受 90% 的折扣）。缓存有效期为 5 分钟，每次命中（Hit）缓存时都会重置计时器。

### OpenAI 自动缓存 (Automatic Caching)

# from openai import OpenAI
#
# client = OpenAI()
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful customer support agent..."},
#         {"role": "user", "content": "What is the return policy?"},
#     ],
# )
#
# print(f"Prompt tokens: {response.usage.prompt_tokens}")
# print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
# print(f"Completion tokens: {response.usage.completion_tokens}")

OpenAI 会自动进行缓存。任何长度超过 1,024 个词元（Token）且与近期请求匹配的提示词前缀，均可享受 50% 的折扣。无需修改代码——只需检查响应中的 `prompt_tokens_details.cached_tokens` 字段即可验证缓存是否生效。

### OpenAI 批处理 API (Batch API)

# import json
# from openai import OpenAI
#
# client = OpenAI()
#
# requests = []
# for i, query in enumerate(queries):
#     requests.append({
#         "custom_id": f"request-{i}",
#         "method": "POST",
#         "url": "/v1/chat/completions",
#         "body": {
#             "model": "gpt-4o-mini",
#             "messages": [{"role": "user", "content": query}],
#         },
#     })
#
# with open("batch_input.jsonl", "w") as f:
#     for r in requests:
#         f.write(json.dumps(r) + "\n")
#
# batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
# batch = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
# print(f"Batch ID: {batch.id}, Status: {batch.status}")

批处理 API 对所有词元提供统一的 50% 折扣。结果将在 24 小时内返回。非常适合非实时工作负载（Workload）：如模型评估、数据标注和批量摘要生成。

### 基于 Redis 的生产环境语义缓存 (Semantic Cache)

# import redis
# import numpy as np
# from openai import OpenAI
#
# r = redis.Redis()
# client = OpenAI()
#
# def get_embedding(text):
#     response = client.embeddings.create(model="text-embedding-3-small", input=text)
#     return response.data[0].embedding
#
# def semantic_cache_lookup(query, threshold=0.95):
#     query_emb = np.array(get_embedding(query))
#     keys = r.keys("cache:emb:*")
#     best_sim, best_key = 0, None
#     for key in keys:
#         stored_emb = np.frombuffer(r.get(key), dtype=np.float32)
#         sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
#         if sim > best_sim:
#             best_sim, best_key = sim, key
#     if best_sim >= threshold and best_key:
#         response_key = best_key.decode().replace("cache:emb:", "cache:resp:")
#         return r.get(response_key).decode()
#     return None

在生产环境中，建议使用向量索引（Vector Index，如 Redis Vector Search、Pinecone 或 pgvector）替代线性扫描（Linear Scan）。线性扫描适用于条目数少于 1,000 的场景。若数据量更大，则应使用近似最近邻（Approximate Nearest Neighbor, ANN）算法，以实现 O(log n) 复杂度的查询。

## 交付上线

本课时将生成 `outputs/prompt-cost-optimizer.md` —— 一个可复用的提示词 (Prompt)，用于分析你的大语言模型 (Large Language Model, LLM) 应用，并推荐具体的成本优化方案及预期节省的费用。

同时还会生成 `outputs/skill-cost-patterns.md` —— 一个决策框架，帮助你根据具体用例选择合适的缓存策略、限流配置以及模型路由规则。

## 练习

1. **为语义缓存 (Semantic Cache) 实现 LRU 淘汰机制。** 将原有的最早优先淘汰策略替换为最近最少使用 (Least Recently Used, LRU) 策略。记录每个缓存条目的最后访问时间，当缓存满时，淘汰访问时间最旧的条目。在 100 次查询中对比这两种策略的命中率 (Hit Rate)。

2. **构建成本预测工具。** 基于 API 调用日志（即 CostTracker 日志），根据过去 7 天的平均值预测月度成本。需考虑工作日与周末的使用模式差异。若预测的月度成本超出预算 20% 以上，则触发告警。

3. **实现分层语义缓存。** 设置两个相似度阈值 (Similarity Threshold)：0.98 用于高置信度命中（直接返回结果），0.90 用于中置信度命中（返回时附带提示语：“基于类似的历史问题……”）。记录每次命中所属的层级，并评估用户满意度的差异。

4. **构建模型路由分类器。** 将基于关键词的分类器替换为基于嵌入 (Embedding) 的分类器。对 50 条已标注的查询（简单/中等/复杂）进行向量化嵌入，然后通过寻找最近的已标注样本来对新查询进行分类。使用包含 20 条查询的测试集评估分类准确率。

5. **实现具备多级降级 (Degradation Levels) 功能的熔断器 (Circuit Breaker)。** 当预算消耗达到 70% 时，记录警告日志；达到 85% 时，自动将所有路由切换至最便宜的模型（gpt-4o-mini）；达到 95% 时，仅返回缓存响应并拒绝新查询。通过模拟 1,000 次请求（预算设为 1.00 美元）进行测试，验证各阈值能否正确触发。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 提示词缓存 (Prompt Caching) | “缓存系统提示词” | 提供商级别的缓存机制，对重复的提示词前缀提供折扣（Anthropic 90%，OpenAI 50%）——OpenAI 无需修改代码，Anthropic 需使用显式标记。 |
| 语义缓存 (Semantic Caching) | “智能缓存” | 将查询转换为向量嵌入，计算其与历史查询的相似度，若超过阈值则返回缓存响应——能够捕获精确匹配会遗漏的同义改写。 |
| 精确缓存 (Exact Caching) | “哈希缓存” | 对完整提示词（模型 + 消息 + 温度参数）进行哈希计算，对完全相同的输入返回缓存响应——仅适用于 temperature=0 的确定性调用。 |
| 令牌桶 (Token Bucket) | “速率限制器” | 一种算法，为每个用户分配容量为 N 的令牌桶，并以每秒 R 的速率补充令牌——允许突发请求达到 N，同时强制执行平均速率 R。 |
| 模型路由 (Model Routing) | “低成本路由” | 使用分类器将简单查询路由至低成本模型（GPT-4o-mini, Haiku），将复杂查询路由至高成本模型（GPT-4o, Opus）——可节省 40-70% 的模型调用成本。 |
| 成本追踪 (Cost Tracking) | “用量计量” | 记录每次 API 调用的模型、Token 数量、延迟、成本及用户 ID，以便精确掌握资金流向及高成本功能。 |
| 熔断机制 (Circuit Breaker) | “紧急开关” | 当支出接近预算上限时，自动降级服务（切换至低成本模型或仅使用缓存）或完全停止请求。 |
| 批量 API (Batch API) | “批量折扣” | OpenAI 提供的异步处理服务，享受 50% 折扣——最多可提交 50,000 个请求，24 小时内返回结果。 |
| 提示词压缩 (Prompt Compression) | “Token 节食” | 重写系统提示词和上下文以减少 Token 消耗，同时保留原意——更短的提示词成本更低，且通常表现更好。 |
| 缓存命中率 (Cache Hit Rate) | “缓存效率” | 通过缓存而非调用大语言模型 (LLM) 直接返回的请求比例——生产环境聊天机器人通常为 40-60%，成本可按比例节省。 |

## 延伸阅读

- [Anthropic 提示词缓存 (Prompt Caching) 指南](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) -- Anthropic 官方文档，介绍显式 cache_control 标记、定价策略以及缓存生命周期行为
- [OpenAI 提示词缓存](https://platform.openai.com/docs/guides/prompt-caching) -- OpenAI 的自动缓存机制、如何通过 usage 字段验证缓存命中 (cache hits)，以及最小前缀长度要求
- [OpenAI 批量 API (Batch API)](https://platform.openai.com/docs/guides/batch) -- 异步处理享 50% 折扣、JSONL 格式要求、24 小时完成窗口以及 5 万次请求限制
- [GPTCache](https://github.com/zilliztech/GPTCache) -- 开源语义缓存 (Semantic Caching) 库，支持多种嵌入 (Embedding) 后端、向量存储 (Vector Stores) 及淘汰策略 (Eviction Policies)
- [Martian 模型路由 (Model Routing)](https://docs.withmartian.com) -- 生产级模型路由方案，可自动为每个查询选择能够处理该请求的最便宜模型
- [Not Diamond](https://www.notdiamond.ai) -- 基于机器学习 (ML-based) 的模型路由器，通过学习您的流量模式来优化跨供应商的成本与质量权衡 (Cost/Quality Tradeoffs)
- [Helicone](https://www.helicone.ai) -- 大语言模型可观测性 (LLM Observability) 平台，以代理层 (Proxy Layer) 形式提供成本追踪、缓存、速率限制 (Rate Limiting) 及预算告警功能
- [Dean & Barroso, "The Tail at Scale" (CACM 2013)](https://research.google/pubs/the-tail-at-scale/) -- 延迟 (Latency)、吞吐量 (Throughput)、TTFT/TPOT 百分位数及对冲请求 (Hedged Requests)；阐述了“在满足 P95 延迟要求的前提下选择最便宜模型”这一策略背后的成本模型
- [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)](https://arxiv.org/abs/2309.06180) -- vLLM 的奠基论文；解释了为何分页键值缓存 (Paged KV-Cache) 结合连续批处理 (Continuous Batching) 能在吞吐量上超越传统服务器 24 倍，属于“缓存与成本”主题下的基础设施层内容
- [Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024)](https://arxiv.org/abs/2307.08691) -- 与提示词缓存正交的底层内核级 (Kernel-Level) 成本优化方案；建议结合投机解码 (Speculative Decoding) 与分组查询注意力 (Grouped-Query Attention, GQA) 一同阅读，以全面了解成本曲线 (Cost Curve)
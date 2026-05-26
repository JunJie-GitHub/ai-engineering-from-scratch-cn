# 上下文工程（Context Engineering）：窗口、预算、记忆与检索

> 提示词工程（Prompt Engineering）只是其中的一部分，上下文工程才是全局。提示词是你输入的一段字符串，而上下文则是输入模型窗口的所有内容：系统指令、检索文档、工具定义、对话历史、少样本示例（Few-shot Examples）以及提示词本身。2026 年最顶尖的 AI 工程师都是上下文工程师。他们决定哪些内容该放入、哪些该剔除，以及以何种顺序排列。

**类型：** 构建
**语言：** Python
**前置要求：** 第 10 阶段（LLMs from Scratch），第 11 阶段 第 01-02 课
**时长：** 约 90 分钟
**相关章节：** 第 11 阶段 · 15（提示词缓存（Prompt Caching））—— 缓存友好的布局是上下文工程的延伸。第 5 阶段 · 28（长上下文评估（Long-Context Evaluation））介绍了如何使用 NIAH/RULER 评估“中间丢失”（Lost-in-the-Middle）现象。

## 学习目标

- 计算上下文窗口（Context Window）各组件的 Token 预算（系统提示词、工具、历史记录、检索文档、生成预留空间）
- 实现上下文窗口管理策略：针对对话历史的截断、摘要与滑动窗口机制
- 对上下文组件进行优先级排序与排列，以最大化模型对最相关信息的注意力
- 构建上下文组装器，根据查询类型和可用窗口空间动态分配 Token

## 问题描述

Claude Opus 4.7 拥有 200K 的上下文窗口 (Context Window)（测试版为 1M）。GPT-5 为 400K。Gemini 3 Pro 为 2M。Llama 4 宣称达到 10M。这些数字听起来非常庞大，直到你真正将它们填满。

以下是一个编程助手的实际消耗明细。系统提示词 (System Prompt)：500 个 token。50 个工具的定义 (Tool Definitions)：8,000 个 token。检索文档 (Retrieved Documentation)：4,000 个 token。对话历史 (Conversation History)（10 轮）：6,000 个 token。当前用户查询：200 个 token。生成预算 (Generation Budget)（最大输出）：4,000 个 token。总计：22,700 个 token。这仅占 128K 上下文窗口的 18%。

然而，注意力机制 (Attention Mechanism) 的计算开销并不会随上下文长度 (Context Length) 线性增长。拥有 128K 上下文 token 的模型需要承担二次方注意力计算开销 (Quadratic Attention Cost)（在原始 Transformer 架构 (Vanilla Transformers) 中为 O(n^2)，尽管大多数生产环境模型已采用高效注意力变体 (Efficient Attention Variants)）。更重要的是，检索准确率 (Retrieval Accuracy) 会随之下降。“大海捞针”测试 (Needle in a Haystack Test) 表明，模型在长上下文的中间位置查找信息时会遇到困难。Liu 等人（2023）的研究显示，大语言模型 (Large Language Models, LLMs) 在长上下文的开头和结尾检索信息的准确率接近完美，但对于位于中间位置（上下文的 40%-70% 处）的信息，准确率会下降 10%-20%。这种“中间迷失”效应 (Lost-in-the-Middle Effect) 因模型而异，但会影响当前所有的架构。

实际经验表明：拥有 200K 的可用 token 并不意味着塞满 200K 就能取得好效果。经过精心筛选的 10K token 上下文，其表现往往优于直接堆砌的 100K token 上下文。上下文工程 (Context Engineering) 正是致力于在上下文窗口内最大化信噪比 (Signal-to-Noise Ratio) 的一门学科。

窗口中的每一个 token 都会挤占原本可以承载更相关信息的 token 空间。每一个无关的工具定义、每一轮过时的对话、每一段无法回答问题的检索文本——都会让模型在执行任务时的表现略微下降。

## 核心概念

### 上下文窗口（Context Window）是一种稀缺资源

将上下文窗口视为内存（RAM），而非硬盘。它速度快且可直接访问，但容量有限。你无法装入所有内容，必须做出取舍。

graph TD
    subgraph Window["Context Window (128K tokens)"]
        direction TB
        S["System Prompt\n~500 tokens"] --> T["Tool Definitions\n~2K-8K tokens"]
        T --> R["Retrieved Context\n~2K-10K tokens"]
        R --> H["Conversation History\n~2K-20K tokens"]
        H --> F["Few-shot Examples\n~1K-3K tokens"]
        F --> Q["User Query\n~100-500 tokens"]
        Q --> G["Generation Budget\n~2K-8K tokens"]
    end

    style S fill:#1a1a2e,stroke:#e94560,color:#fff
    style T fill:#1a1a2e,stroke:#0f3460,color:#fff
    style R fill:#1a1a2e,stroke:#ffa500,color:#fff
    style H fill:#1a1a2e,stroke:#51cf66,color:#fff
    style F fill:#1a1a2e,stroke:#9b59b6,color:#fff
    style Q fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#0f3460,color:#fff

各个组件都在争夺空间。增加工具定义（Tool Definitions）就意味着对话历史（Conversation History）的空间减少。增加检索上下文（Retrieved Context）就意味着少样本示例（Few-shot Examples）的空间减少。上下文工程（Context Engineering）的艺术就在于如何分配这一预算，以最大化任务性能。

### 中间迷失现象（Lost-in-the-Middle）

这是上下文工程中最重要的一项经验发现。模型对位于上下文开头和结尾的信息关注度更高。处于中间位置的信息获得的注意力分数较低，更容易被忽略。

Liu 等人（2023）对此进行了系统测试。他们将一份相关文档与 20 份无关文档混合，放置在不同位置，并测量回答准确率。当相关文档位于首位或末位时，准确率为 85%-90%；当它位于中间（第 10 位，共 20 份）时，准确率降至 60%-70%。

这对工程实践有直接的指导意义：

- 将最重要的信息放在最前面（系统提示词 System Prompt、关键指令）
- 将当前查询（Query）和最相关的上下文放在最后（近期偏差 Recency Bias 会有所助益）
- 将上下文的中间区域视为最低优先级区域
- 如果必须在中间包含某些信息，请在末尾重复关键点

graph LR
    subgraph Attention["Attention Distribution Across Context"]
        direction LR
        P1["Position 0-20%\nHIGH attention\n(system prompt)"]
        P2["Position 20-40%\nMODERATE"]
        P3["Position 40-70%\nLOW attention\n(lost in middle)"]
        P4["Position 70-90%\nMODERATE"]
        P5["Position 90-100%\nHIGH attention\n(current query)"]
    end

    style P1 fill:#51cf66,color:#000
    style P2 fill:#ffa500,color:#000
    style P3 fill:#ff6b6b,color:#fff
    style P4 fill:#ffa500,color:#000
    style P5 fill:#51cf66,color:#000

### 上下文组件

**系统提示词（System Prompt）**：设定角色、约束条件和行为规则。它始终位于最前，且在多轮对话中保持不变。Claude Code 的系统提示词（包含工具定义和行为指令）大约占用 6,000 个 token。务必保持精简。系统提示词中的每个字都会在每次 API 调用时重复发送。

**工具定义（Tool Definitions）**：每个工具会增加 50-200 个 token（名称、描述、参数模式）。50 个工具按每个 150 个 token 计算，在对话开始前就会占用 7,500 个 token。动态工具选择（Dynamic Tool Selection）——仅包含与当前查询相关的工具——可将此开销降低 60%-80%。

**检索上下文（Retrieved Context）**：来自向量数据库的文档、搜索结果或文件内容。检索质量直接决定回答质量。糟糕的检索比不检索更糟——它会让窗口充满噪声，并主动误导模型。

**对话历史（Conversation History）**：包含所有先前的用户消息和助手回复。其长度随对话轮数线性增长。一场 50 轮、每轮 200 个 token 的对话会产生 10,000 个 token 的历史记录。其中大部分与当前查询无关。

**少样本示例（Few-shot Examples）**：用于展示期望行为的输入/输出对。两到三个精心挑选的示例，通常比数千个 token 的指令更能提升输出质量。但它们会占用空间。

**生成预算（Generation Budget）**：为模型回复预留的 token 数量。如果将窗口填满，模型将没有空间进行回答。请至少为生成预留 2,000-4,000 个 token。

### 上下文压缩策略

**历史摘要（History Summarization）**：与其逐字保留所有历史对话，不如定期对对话进行总结。用 100 个 token 概括“我们讨论了 X，决定了 Y，用户需要 Z”，即可替代原本占用 2,000 个 token 的 10 轮对话。当历史记录超过设定阈值（例如 5,000 个 token）时触发摘要生成。

**相关性过滤（Relevance Filtering）**：根据当前查询对每个检索到的文档进行打分，并丢弃低于阈值的文档。如果检索到 10 个文本块（Chunks），但只有 3 个相关，就丢弃其余 7 个。拥有 3 个高度相关的文本块，远胜于 10 个平庸的文本块。

**工具剪枝（Tool Pruning）**：对用户查询意图进行分类，仅包含与该意图相关的工具。代码问题不需要日历工具，日程安排问题不需要文件系统工具。这可将工具定义的 token 占用从 8,000 降至 1,000。

**递归摘要（Recursive Summarization）**：针对超长文档，采用分阶段摘要。先对每个章节进行摘要，再对摘要进行二次摘要。一份 50 页的文档最终可浓缩为一份 500 个 token 的要点摘要。

### 记忆系统

上下文工程涵盖三个时间维度。

**短期记忆（Short-term Memory）**：当前对话内容。直接存储在上下文窗口中，随对话轮数增长。通过摘要和截断进行管理。

**长期记忆（Long-term Memory）**：跨对话持久保存的事实与偏好。例如“用户偏好 TypeScript”或“项目使用 PostgreSQL”。存储在数据库中，在会话开始时检索。Claude Code 将其存储在 `CLAUDE.md` 文件中，ChatGPT 则通过其记忆功能进行存储。

**情景记忆（Episodic Memory）**：可能相关的特定历史交互记录。例如“上周二，我们在认证模块调试过类似问题”。以嵌入向量（Embeddings）形式存储，当当前对话与历史情景匹配时进行检索。

graph TD
    subgraph Memory["Memory Architecture"]
        direction TB
        STM["Short-term Memory\n(current conversation)\nDirect in context window"]
        LTM["Long-term Memory\n(facts, preferences)\nDB -> retrieved on session start"]
        EM["Episodic Memory\n(past interactions)\nEmbeddings -> retrieved on similarity"]
    end

    Q["Current Query"] --> STM
    Q --> LTM
    Q --> EM

    STM --> CW["Context Window"]
    LTM --> CW
    EM --> CW

    style STM fill:#1a1a2e,stroke:#51cf66,color:#fff
    style LTM fill:#1a1a2e,stroke:#0f3460,color:#fff
    style EM fill:#1a1a2e,stroke:#e94560,color:#fff
    style CW fill:#1a1a2e,stroke:#ffa500,color:#fff

### 动态上下文组装

核心洞察：不同的查询需要不同的上下文。静态的系统提示词 + 静态工具 + 静态历史是一种浪费。优秀的系统会根据每次查询动态组装上下文。

1. 对查询意图进行分类
2. 选择相关工具（而非全部工具）
3. 检索相关文档（而非固定集合）
4. 包含相关的历史对话轮次（而非全部历史）
5. 添加与任务类型匹配的少样本示例
6. 按重要性排序所有内容：关键信息置首，重要信息置尾，可选信息放中间

这正是区分优秀 AI 应用与卓越 AI 应用的关键。底层模型可能相同，但上下文管理才是决定差异的核心。

## 构建

### 步骤 1：Token 计数器（Token Counter）

无法衡量，就无法进行预算分配。构建一个简单的 Token 计数器（采用空白字符分割进行近似估算，因为精确计数取决于具体的分词器（Tokenizer））。

import json
import numpy as np
from collections import OrderedDict

def count_tokens(text):
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def count_tokens_json(obj):
    return count_tokens(json.dumps(obj))

### 步骤 2：上下文预算管理器（Context Budget Manager）

核心抽象层。预算管理器负责追踪各组件消耗的 Token 数量，并严格执行配额限制。

class ContextBudget:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        tokens = count_tokens(content)
        if max_tokens and tokens > max_tokens:
            words = content.split()
            target_words = int(max_tokens / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        used = sum(self.allocations.values())
        if used + tokens > self.available:
            allowed = self.available - used
            if allowed <= 0:
                return None, 0
            words = content.split()
            target_words = int(allowed / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        self.allocations[component] = tokens
        return content, tokens

    def remaining(self):
        used = sum(self.allocations.values())
        return self.available - used

    def utilization(self):
        used = sum(self.allocations.values())
        return used / self.max_tokens

    def report(self):
        total_used = sum(self.allocations.values())
        lines = []
        lines.append(f"Context Budget Report ({self.max_tokens:,} token window)")
        lines.append("-" * 50)
        for component, tokens in self.allocations.items():
            pct = tokens / self.max_tokens * 100
            bar = "#" * int(pct / 2)
            lines.append(f"  {component:<25} {tokens:>6} tokens ({pct:>5.1f}%) {bar}")
        lines.append("-" * 50)
        lines.append(f"  {'Used':<25} {total_used:>6} tokens ({total_used/self.max_tokens*100:.1f}%)")
        lines.append(f"  {'Generation reserve':<25} {self.generation_reserve:>6} tokens")
        lines.append(f"  {'Remaining':<25} {self.remaining():>6} tokens")
        return "\n".join(lines)

### 步骤 3：“中间丢失”重排序（Lost-in-the-Middle Reordering）

实现重排序策略：将最重要的内容置于首尾，最不重要的内容放置在中间。

def reorder_lost_in_middle(items, scores):
    paired = sorted(zip(scores, items), reverse=True)
    sorted_items = [item for _, item in paired]

    if len(sorted_items) <= 2:
        return sorted_items

    first_half = sorted_items[::2]
    second_half = sorted_items[1::2]
    second_half.reverse()

    return first_half + second_half

def score_relevance(query, documents):
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        if not query_words:
            scores.append(0.0)
            continue
        overlap = len(query_words & doc_words) / len(query_words)
        scores.append(round(overlap, 3))
    return scores

### 步骤 4：对话历史压缩器（Conversation History Compressor）

对早期的对话轮次（Conversation Turns）进行摘要，以回收 Token 预算。

class ConversationManager:
    def __init__(self, max_history_tokens=5000):
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return

        while total > self.max_history_tokens and len(self.turns) > 4:
            old_turns = self.turns[:2]
            summary = self._summarize_turns(old_turns)
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)

    def _summarize_turns(self, turns):
        parts = []
        for t in turns:
            content = t["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"{t['role']}: {content}")
        return "Previous: " + " | ".join(parts)

    def get_context(self):
        parts = []
        if self.summaries:
            parts.append("[Conversation Summary]")
            for s in self.summaries:
                parts.append(s)
        parts.append("[Recent Conversation]")
        for t in self.turns:
            parts.append(f"{t['role']}: {t['content']}")
        return "\n".join(parts)

    def token_count(self):
        return count_tokens(self.get_context())

### 步骤 5：动态工具选择器（Dynamic Tool Selector）

仅包含与当前查询相关的工具。先进行意图分类（Intent Classification），再进行过滤。

TOOL_REGISTRY = {
    "read_file": {
        "description": "Read contents of a file",
        "tokens": 120,
        "categories": ["code", "files"],
    },
    "write_file": {
        "description": "Write content to a file",
        "tokens": 150,
        "categories": ["code", "files"],
    },
    "search_code": {
        "description": "Search for patterns in codebase",
        "tokens": 130,
        "categories": ["code"],
    },
    "run_command": {
        "description": "Execute a shell command",
        "tokens": 140,
        "categories": ["code", "system"],
    },
    "create_calendar_event": {
        "description": "Create a new calendar event",
        "tokens": 180,
        "categories": ["calendar"],
    },
    "list_emails": {
        "description": "List recent emails",
        "tokens": 160,
        "categories": ["email"],
    },
    "send_email": {
        "description": "Send an email message",
        "tokens": 200,
        "categories": ["email"],
    },
    "web_search": {
        "description": "Search the web for information",
        "tokens": 140,
        "categories": ["research"],
    },
    "query_database": {
        "description": "Run a SQL query on the database",
        "tokens": 170,
        "categories": ["code", "data"],
    },
    "generate_chart": {
        "description": "Generate a chart from data",
        "tokens": 190,
        "categories": ["data", "visualization"],
    },
}

def classify_intent(query):
    query_lower = query.lower()

    intent_keywords = {
        "code": ["code", "function", "bug", "error", "file", "implement", "refactor", "debug", "test"],
        "calendar": ["meeting", "schedule", "calendar", "appointment", "event"],
        "email": ["email", "mail", "send", "inbox", "message"],
        "research": ["search", "find", "what is", "how does", "explain", "look up"],
        "data": ["data", "query", "database", "chart", "graph", "analytics", "sql"],
    }

    scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ["code"]

    max_score = max(scores.values())
    return [intent for intent, score in scores.items() if score >= max_score * 0.5]

def select_tools(query, token_budget=2000):
    intents = classify_intent(query)
    relevant = {}
    total_tokens = 0

    for name, tool in TOOL_REGISTRY.items():
        if any(cat in intents for cat in tool["categories"]):
            if total_tokens + tool["tokens"] <= token_budget:
                relevant[name] = tool
                total_tokens += tool["tokens"]

    return relevant, total_tokens

### 步骤 6：完整上下文组装流水线（Full Context Assembly Pipeline）

将所有模块串联集成。给定查询后，动态组装出最优的上下文（Context）。

class ContextEngine:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.budget = ContextBudget(max_tokens, generation_reserve)
        self.conversation = ConversationManager(max_history_tokens=5000)
        self.system_prompt = (
            "You are a helpful AI assistant. You have access to tools for "
            "code editing, file management, web search, and data analysis. "
            "Use the appropriate tools for each task. Be concise and accurate."
        )
        self.knowledge_base = [
            "Python 3.12 introduced type parameter syntax for generic classes using bracket notation.",
            "The project uses PostgreSQL 16 with pgvector for embedding storage.",
            "Authentication is handled by Supabase Auth with JWT tokens.",
            "The frontend is built with Next.js 15 using the App Router.",
            "API rate limits are set to 100 requests per minute per user.",
            "The deployment pipeline uses GitHub Actions with Docker multi-stage builds.",
            "Test coverage must be above 80% for all new modules.",
            "The codebase follows the repository pattern for data access.",
        ]

    def assemble(self, query):
        self.budget = ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)

        system_content, _ = self.budget.allocate("system_prompt", self.system_prompt, max_tokens=1000)

        tools, tool_tokens = select_tools(query, token_budget=2000)
        tool_text = json.dumps(list(tools.keys()))
        tool_content, _ = self.budget.allocate("tools", tool_text, max_tokens=2000)

        relevance = score_relevance(query, self.knowledge_base)
        threshold = 0.1
        relevant_docs = [
            doc for doc, score in zip(self.knowledge_base, relevance)
            if score >= threshold
        ]

        if relevant_docs:
            doc_scores = [s for s in relevance if s >= threshold]
            reordered = reorder_lost_in_middle(relevant_docs, doc_scores)
            doc_text = "\n".join(reordered)
            doc_content, _ = self.budget.allocate("retrieved_context", doc_text, max_tokens=3000)

        history_text = self.conversation.get_context()
        if history_text.strip():
            history_content, _ = self.budget.allocate("conversation_history", history_text, max_tokens=5000)

        query_content, _ = self.budget.allocate("user_query", query, max_tokens=500)

        return self.budget

    def chat(self, query):
        self.conversation.add_turn("user", query)
        budget = self.assemble(query)
        response = f"[Response to: {query[:50]}...]"
        self.conversation.add_turn("assistant", response)
        return budget


def run_demo():
    print("=" * 60)
    print("  Context Engineering Pipeline Demo")
    print("=" * 60)

    engine = ContextEngine(max_tokens=128000, generation_reserve=4000)

    print("\n--- Query 1: Code task ---")
    budget = engine.chat("Fix the bug in the authentication module where JWT tokens expire too early")
    print(budget.report())

    print("\n--- Query 2: Research task ---")
    budget = engine.chat("What is the best approach for implementing vector search in PostgreSQL?")
    print(budget.report())

    print("\n--- Query 3: After conversation history builds up ---")
    for i in range(8):
        engine.conversation.add_turn("user", f"Follow-up question number {i+1} about the implementation details of the system")
        engine.conversation.add_turn("assistant", f"Here is the response to follow-up {i+1} with technical details about the architecture")

    budget = engine.chat("Now implement the changes we discussed")
    print(budget.report())

    print("\n--- Tool Selection Examples ---")
    test_queries = [
        "Fix the bug in auth.py",
        "Schedule a meeting with the team for Tuesday",
        "Show me the database query performance stats",
        "Search for best practices on error handling",
    ]

    for q in test_queries:
        tools, tokens = select_tools(q)
        intents = classify_intent(q)
        print(f"\n  Query: {q}")
        print(f"  Intents: {intents}")
        print(f"  Tools: {list(tools.keys())} ({tokens} tokens)")

    print("\n--- Lost-in-the-Middle Reordering ---")
    docs = ["Doc A (most relevant)", "Doc B (somewhat relevant)", "Doc C (least relevant)",
            "Doc D (relevant)", "Doc E (moderately relevant)"]
    scores = [0.95, 0.60, 0.20, 0.80, 0.50]
    reordered = reorder_lost_in_middle(docs, scores)
    print(f"  Original order: {docs}")
    print(f"  Scores:         {scores}")
    print(f"  Reordered:      {reordered}")
    print(f"  (Most relevant at start and end, least relevant in middle)")


## 实践应用

### Claude Code 的上下文策略

Claude Code 采用分层架构来管理上下文（Context）。系统提示词（System Prompt）中包含了行为规则与工具定义（约 6K 个 Token）。当你打开文件时，文件内容会被注入上下文；执行搜索时，检索结果会被追加进来；历史对话轮次则会被自动摘要。此外，`CLAUDE.md` 文件提供了跨会话保持的长期记忆。

其核心工程决策在于：Claude Code 不会将整个代码库直接灌入上下文，而是按需检索相关文件。这正是上下文工程（Context Engineering）在实践中的体现。

### Cursor 的动态上下文加载

Cursor 会将你的整个代码库构建为嵌入向量（Embeddings）索引。当你输入查询时，系统会基于向量相似度检索出最相关的文件与代码块，并仅将这些片段送入上下文窗口（Context Window）。即便是 50 万行代码的庞大代码库，也会被精准压缩至 5 到 10 个最相关的代码块中。

其核心模式可概括为：全量嵌入、按需检索、仅保留关键信息。

### ChatGPT 的记忆功能

ChatGPT 会将用户偏好与关键事实存储为长期记忆。在每次对话启动时，系统会检索相关记忆并将其注入系统提示词中。例如，“用户偏好使用 Python”这一信息仅占用 5 个 Token，却能在后续多次对话中节省数百个 Token 的重复指令开销。

### 作为上下文工程的检索增强生成

检索增强生成（Retrieval-Augmented Generation, RAG）本质上是上下文工程的标准化实践。与其将知识硬编码进模型权重（通过训练）或写死在系统提示词（静态上下文）中，不如在查询发生时动态检索相关文档，并将其注入上下文窗口。整个 RAG 流水线（Pipeline）——包括文本分块（Chunking）、向量化嵌入、检索与重排序（Reranking）——其存在只为解决一个核心问题：确保将最准确的信息精准送入上下文窗口。

## 交付落地

本节内容将生成 `outputs/prompt-context-optimizer.md` 文件——这是一个可复用的提示词，专门用于审计上下文组装策略并提供优化建议。只需输入你的系统提示词、工具数量、平均历史对话长度以及检索策略，该提示词即可自动识别 Token 浪费点并给出改进方案。

同时还会生成 `outputs/skill-context-engineering.md` 文件——这是一套决策框架，旨在指导开发者根据任务类型、上下文窗口大小及延迟预算（Latency Budget）来设计上下文组装流水线。

## 练习

1. 在 ContextBudget 类中添加一个“Token 浪费检测器”（Token Waste Detector）。它应标记出占用预算超过 30% 的组件，并针对每种组件类型推荐特定的压缩策略（如历史对话摘要、工具剪枝、文档重排序）。

2. 为检索上下文（Retrieved Context）实现语义去重（Semantic Deduplication）。如果两份检索文档的相似度超过 80%（通过词重叠或嵌入向量（Embeddings）的余弦相似度（Cosine Similarity）计算），则仅保留得分较高的一份。评估该操作能回收多少 Token 预算。

3. 构建一个“上下文回放”（Context Replay）工具。给定一段对话记录，将其输入 ContextEngine 进行回放，并逐轮可视化预算分配的变化情况。绘制各组件随时间推移的 Token 使用量曲线。定位上下文开始被压缩的具体对话轮次。

4. 实现一个基于优先级的工具选择器（Priority-based Tool Selector）。摒弃传统的二元包含/排除机制，为每个工具分配一个针对当前查询的相关性得分。按相关性降序依次纳入工具，直至工具预算耗尽。对比在纳入 5、10、20 和 50 个工具时的任务性能表现。

5. 构建一个多策略上下文压缩器（Multi-strategy Context Compressor）。实现三种压缩策略（截断（Truncation）、摘要（Summarization）、关键句提取（Key Sentence Extraction）），并在包含 20 份文档的数据集上进行基准测试。评估压缩率（Compression Ratio）与信息保留率（Information Retention）之间的权衡关系（压缩后的版本是否仍包含查询所需的答案？）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 上下文窗口 (Context Window) | “模型能读取多少内容” | 模型在单次前向传播（forward pass）中处理的最大 Token 数量（输入 + 输出）—— GPT-5 为 400K，Claude Opus 4.7 为 200K（1M 测试版），Gemini 3 Pro 为 2M |
| 上下文工程 (Context Engineering) | “高级提示词工程” | 决定将哪些内容放入上下文窗口、以何种顺序排列以及赋予何种优先级的学科——涵盖检索、压缩、工具选择和记忆管理 |
| 中间迷失效应 (Lost-in-the-middle) | “模型会忘记中间的内容” | 经验性发现表明，大语言模型（LLM）对上下文开头和结尾的关注度更高，放置在中间的信息会导致准确率下降 10-20% |
| Token 预算 (Token Budget) | “你还剩多少 Token” | 将上下文窗口容量明确分配给各个组件（系统提示词、工具、历史记录、检索内容、生成内容），并为每个组件设定上限 |
| 动态上下文 (Dynamic Context) | “动态加载内容” | 根据意图分类、相关工具选择和检索结果，为每次查询动态组装不同的上下文窗口 |
| 历史摘要 (History Summarization) | “压缩对话内容” | 用简洁的摘要替换逐字记录的旧对话轮次，在保留关键信息的同时降低 Token 消耗 |
| 工具剪枝 (Tool Pruning) | “仅包含相关工具” | 对查询意图进行分类，仅包含匹配的工具定义，从而将工具的 Token 成本降低 60-80% |
| 长期记忆 (Long-term Memory) | “跨会话记忆” | 存储在数据库中并在会话开始时检索的事实与偏好——例如 CLAUDE.md、ChatGPT Memory 及类似系统 |
| 情景记忆 (Episodic Memory) | “记住特定的过往事件” | 将过往交互存储为向量嵌入（embeddings），当当前查询与历史对话相似时进行检索 |
| 生成预算 (Generation Budget) | “留给回答的空间” | 为模型输出预留的 Token 数量——如果上下文完全占满窗口，模型将没有空间进行回复 |

## 进一步阅读

- [Liu 等人，2023 -- "Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) -- 关于位置依赖注意力（position-dependent attention）的权威研究，表明模型在处理长上下文中间位置的信息时表现不佳。
- [Anthropic 的上下文检索（Contextual Retrieval）博客文章](https://www.anthropic.com/news/contextual-retrieval) -- 介绍了 Anthropic 如何实现上下文感知分块检索（context-aware chunk retrieval），将检索失败率降低了 49%。
- [Simon Willison 的《上下文工程（Context Engineering）》](https://simonwillison.net/2025/Jun/27/context-engineering/) -- 这篇博客文章正式命名了该学科，并将其与提示词工程（prompt engineering）区分开来。
- [LangChain 关于 RAG 的文档](https://python.langchain.com/docs/tutorials/rag/) -- 将检索增强生成（retrieval-augmented generation）作为上下文工程模式的实际实现指南。
- [Greg Kamradt 的“大海捞针（Needle in a Haystack）”测试](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) -- 该基准测试揭示了所有主流模型中普遍存在的位置依赖检索失败问题。
- [Pope 等人，《Efficiently Scaling Transformer Inference》（2022）](https://arxiv.org/abs/2211.05102) -- 解释了上下文长度为何会主导内存占用与延迟，以及键值缓存（KV cache）、多查询注意力（Multi-Query Attention, MQA）和分组查询注意力（Grouped-Query Attention, GQA）如何改变资源预算的计算方式。
- [Agrawal 等人，《SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills》（2023）](https://arxiv.org/abs/2308.16369) -- 阐述了推理的两个阶段如何导致长提示词在首词延迟（Time To First Token, TTFT）上开销高昂，而在每词生成时间（Time Per Output Token, TPOT）上成本较低；揭示了上下文打包（context-packing）权衡背后的核心原理。
- [Ainslie 等人，《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》（EMNLP 2023）](https://arxiv.org/abs/2305.13245) -- 这篇关于分组查询注意力（Grouped-Query Attention, GQA）的论文证明了在生产环境解码器中，该技术可在不损失质量的前提下将 KV 内存占用降低 8 倍。
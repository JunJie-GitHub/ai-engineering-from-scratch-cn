# RAG（检索增强生成）的分块策略（Chunking）

> 分块（Chunking）配置对检索质量的影响，与嵌入模型（Embedding Model）的选择同等重要（Vectara NAACL 2025）。如果分块策略出错，再多的重排序（Reranking）也无济于事。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 14（信息检索），第 5 阶段 · 22（嵌入模型）
**耗时：** 约 60 分钟

## 问题所在

你将一份 50 页的合同输入 RAG 系统。用户提问：“终止条款是什么？”检索器（Retriever）却返回了封面页。为什么？因为模型是基于 512 个词元（Token）的分块进行训练的，而终止条款位于第 20 页，恰好被分页符切断，且缺乏与查询直接关联的局部关键词。

解决方案并非“购买更好的嵌入模型”，而是优化分块策略。分块该设多大？重叠率多少？在何处切分？是否需要保留上下文？

2026 年 2 月的基准测试（Benchmark）揭示了令人惊讶的结果：

- Vectara 2026 年研究：递归分块（Recursive Chunking）的准确率击败了语义分块（Semantic Chunking），分别为 69% 对 54%。
- 在 Natural Questions 数据集上使用 SPLADE + Mistral-8B：重叠策略未带来任何可衡量的收益。
- 上下文悬崖（Context Cliff）：当上下文达到约 2,500 个词元时，回复质量会急剧下降。

那些看似“显而易见”的答案（语义分块、20% 重叠率、1000 词元）往往是错误的。本课程将帮助你建立对六种分块策略的直观理解，并指导你在不同场景下如何选择最合适的方案。

## 核心概念

![在一篇文本上可视化的六种分块策略](../assets/chunking.svg)

**固定分块（Fixed Chunking）。** 按每 N 个词元（Token）或字符进行切分。最基础的基线方法。会在句子中间断开。压缩率高，但连贯性差。

**递归分块（Recursive Chunking）。** 使用 LangChain 的 `RecursiveCharacterTextSplitter`。优先尝试按 `\n\n` 切分，其次是 `\n`，然后是 `.`，最后是空格。回退机制平滑。2026 年的默认方案。

**语义分块（Semantic Chunking）。** 对每个句子进行嵌入（Embedding）。计算相邻句子间的余弦相似度（Cosine Similarity）。在相似度低于阈值处进行切分。保留主题连贯性。速度较慢；有时会产生仅 40 个词元的微小片段，不利于检索。

**句子分块（Sentence Chunking）。** 按句子边界切分。每个分块包含一个句子或 N 个句子的滑动窗口。在约 5k 词元范围内，其效果可与语义分块媲美，但成本仅为后者的一小部分。

**父文档分块（Parent-Document Chunking）。** 存储较小的子分块用于检索，*同时*存储较大的父分块用于提供上下文。按子分块检索，返回父分块。具备优雅降级（Graceful Degradation）特性：即使子分块质量不佳，仍能返回合理的父分块。

**延迟分块（Late Chunking，2024）。** 首先在词元级别对整个文档进行嵌入，然后将词元嵌入池化（Pooling）为分块嵌入。保留跨分块上下文。适用于长上下文嵌入模型（如 BGE-M3、Jina v3）。计算开销较高。

**上下文检索（Contextual Retrieval，Anthropic，2024）。** 在每个分块前附加由大语言模型（Large Language Model, LLM）生成的该分块在文档中位置的摘要（例如“本分块属于终止条款第 3.2 节……”）。在 Anthropic 自身的基准测试中，检索效果提升 35-50%。索引成本较高。

### 优于所有默认设置的准则

将分块大小与查询类型相匹配：

| 查询类型 | 分块大小 |
|------------|-----------|
| 事实型（Factoid）（“CEO 的名字是什么？”） | 256-512 个词元 |
| 分析型 / 多跳（Multi-hop） | 512-1024 个词元 |
| 全章节理解 | 1024-2048 个词元 |

基于 NVIDIA 2026 年的基准测试。分块应足够大，以包含答案及局部上下文；同时应足够小，以确保检索器（Retriever）返回的 Top-K 结果聚焦于答案本身，而非上下文噪声。

## 开始构建

### 步骤 1：固定长度与递归分块 (Fixed and Recursive Chunking)

def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)

### 步骤 2：语义分块 (Semantic Chunking)

def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result

请根据你的具体业务领域调整 `threshold`。该参数设置过高会导致文本碎片化；设置过低则会产生一个巨大的分块。

### 步骤 3：父子文档分块 (Parent-Document Chunking)

def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents

核心要点：对父文档进行去重。多个子块可能映射至同一父文档，若全部返回将浪费宝贵的上下文窗口空间。

### 步骤 4：上下文检索 (Contextual Retrieval，Anthropic 模式)

def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]

将添加了上下文信息的分块建立索引。在查询阶段，检索过程将受益于这些额外的上下文信号。

### 步骤 5：评估 (Evaluation)

def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)

务必进行基准测试 (Benchmark)。针对你的语料库而言“最佳”的策略，未必与任何技术博客中介绍的方案一致。

## 常见陷阱

- **仅在事实型查询（factoid queries）上评估分块（chunking）。** 多跳查询（multi-hop queries）会揭示截然不同的最优方案。请使用按查询类型分层的评估集（query-type-stratified eval set）。
- **无最小尺寸限制的语义分块（semantic chunking）。** 会产生仅 40 个 token 的碎片，严重损害检索效果。务必强制设置 `min_tokens`。
- **盲目跟风设置重叠（overlap）。** 2026 年的研究表明，重叠往往毫无益处，反而会使索引成本翻倍。请通过实测验证，切勿主观臆断。
- **未强制执行最小/最大长度限制。** 5 个 token 或 5000 个 token 的分块都会破坏检索效果。务必进行边界限制（clamp）。
- **跨文档分块（cross-doc chunking）。** 绝不允许单个分块跨越两个文档。务必按文档独立分块，然后再进行合并。

## 使用指南

2026 年技术栈推荐：

| 场景 | 策略 |
|-----------|----------|
| 初次构建，语料库未知 | 递归分块（recursive chunking），512 token，无重叠 |
| 事实型问答（factoid QA） | 递归分块，256-512 token |
| 分析型 / 多跳查询（analytical / multi-hop） | 递归分块，512-1024 token + 父文档分块（parent-document chunking） |
| 重度交叉引用（合同、论文） | 延迟分块（late chunking）或上下文检索（contextual retrieval） |
| 对话 / 会话语料（conversational / dialog corpus） | 轮次级分块（turn-level chunks）+ 说话人元数据 |
| 短文本/短句（short utterances） | 单文档即单分块 |

从递归分块 512 token 开始。在包含 50 个查询的评估集上测量前5项召回率（Recall@5）。在此基础上进行调优。

## 交付使用

保存为 `outputs/skill-chunker.md`：

---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.

## 练习

1. **简单。** 使用固定分块 `fixed(512, 0)`、递归分块 `recursive(512, 0)` 和 `recursive(512, 100)` 对一份 20 页的文档进行分块。对比分块数量与边界质量。
2. **中等。** 基于 5 份文档构建包含 30 个查询的评估集。测量递归分块、语义分块和父文档分块的 Recall@5。哪种方案胜出？结果是否与博客文章中的结论一致？
3. **困难。** 实现上下文检索（contextual retrieval）。测量其相较于基线递归分块的 MRR（平均倒数排名）提升。报告索引成本（LLM 调用次数）与准确率增益的对比。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 文本块 (Chunk) | 文档片段 | 用于向量化 (Embedding)、建立索引和检索的子文档单元。 |
| 重叠 (Overlap) | 安全缓冲 | 相邻文本块间共享的 N 个 Token；在 2026 年的基准测试中往往已失去效用。 |
| 语义分块 (Semantic Chunking) | 智能分块 | 在相邻句子嵌入相似度显著下降处进行切分。 |
| 父文档 (Parent-Document) | 两级检索 | 检索细粒度子块，返回粗粒度父文档。 |
| 延迟分块 (Late Chunking) | 嵌入后分块 | 先对完整文档进行 Token 级嵌入，再池化聚合为文本块向量。 |
| 上下文检索 (Contextual Retrieval) | Anthropic 的优化技巧 | 在建立索引前，将大语言模型 (LLM) 生成的摘要作为前缀附加到每个文本块中。 |
| 上下文悬崖 (Context Cliff) | 2500 Token 瓶颈 | 在检索增强生成 (RAG) 系统中，当上下文长度达到约 2.5k Token 时出现的生成质量骤降现象（2026年1月观测）。 |

## 延伸阅读

- [Yepes 等人 / LangChain — 递归字符切分文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认方案。
- [Vectara (2024, NAACL 2025). 分块配置分析](https://arxiv.org/abs/2410.13070) — 分块策略的重要性不亚于嵌入模型的选择。
- [Jina AI — 长上下文嵌入模型中的延迟分块 (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 延迟分块技术的原始论文。
- [Anthropic — 上下文检索](https://www.anthropic.com/news/contextual-retrieval) — 使用 LLM 生成的上下文前缀可提升 35-50% 的检索效果。
- [NVIDIA 2026 分块大小基准测试 — Premai 总结](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型推荐的分块大小。
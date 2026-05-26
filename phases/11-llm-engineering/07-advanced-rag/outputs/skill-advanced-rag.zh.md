---
name: 高级RAG技能
description: 构建具备混合搜索、重排序和评估能力的生产级RAG系统
version: 1.0.0
phase: 11
lesson: 7
tags: [rag, 混合搜索, bm25, 重排序, hyde, 评估]
---

# 高级 RAG (Retrieval-Augmented Generation) 模式

基础 RAG：嵌入查询 -> 向量搜索 (Vector Search) -> 取 Top-K -> 生成。
高级 RAG：嵌入查询 + BM25 -> 融合排序 -> 重排序 (Reranking) -> 取 Top-K -> 生成。

query -> [vector search (top-50)] -+-> RRF fusion -> reranker (top-5) -> prompt -> LLM
                                   |
query -> [BM25 search (top-50)]  --+

## 何时从基础 RAG 升级

- 检索质量低于 70% 的 Recall@5（前5项召回率）
- 用户反馈答案错误或无关
- 语料库规模超过 10 万个文本块 (Chunks)
- 查询使用的词汇与文档存在差异
- 多跳问题 (Multi-hop Questions) 持续失败

## 实施清单

1. 在向量索引旁添加 BM25 索引
2. 并行执行两种搜索（各取 Top-50）
3. 使用倒数排名融合 (Reciprocal Rank Fusion, RRF，k=60) 进行合并
4. 使用交叉编码器 (Cross-Encoder) 对候选结果进行重排序
5. 选取 Top-5 用于最终提示词 (Prompt)
6. 在测试集上添加忠实度 (Faithfulness) 评估

## 技术选型指南

- **混合搜索 (Hybrid Search)**：生产环境中应始终使用。查询时不会产生额外成本。
- **重排序 (Reranking)**：当 Recall@50 表现良好但 Recall@5 较差时使用。会增加 50-200 毫秒的延迟。
- **HyDE (Hypothetical Document Embeddings)**：当查询模糊或与文档词汇不匹配时使用。会增加一次大语言模型 (LLM) 调用。
- **父子文本块 (Parent-Child Chunks)**：当小块缺乏上下文而大块稀释相关性时使用。
- **元数据过滤 (Metadata Filtering)**：当语料库具有明确分类（如日期、来源类型、部门）时使用。
- **查询分解 (Query Decomposition)**：用于需要从多个文档中获取信息的多跳问题。

## 常见误区

- 对 BM25 和向量搜索使用不同的文本块集合（两者必须检索同一语料库）
- 重排序的候选池过小（Top-10 太少；建议使用 Top-50）
- 为每个查询都添加 HyDE（仅在词汇不匹配成为瓶颈时才有效）
- 未对变更进行评估（在应用每项技术前后测量 Recall@k）
- 在未定位失败环节前过度设计流水线

## 评估工作流

1. 创建 50 个以上已知答案所在文本块的测试问题
2. 测量每种检索方法的 Recall@5 和 Recall@10
3. 针对检索成功的查询，评估生成答案的忠实度
4. 随着语料库增长，每周跟踪相关指标
5. 在引入更多技术前，先深入分析个别失败案例
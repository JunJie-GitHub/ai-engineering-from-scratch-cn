---
name: 检索栈选择器
description: 为给定的语料库和查询模式挑选检索栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [自然语言处理 (NLP), 检索 (Retrieval), 检索增强生成 (RAG), 搜索 (Search)]
---

根据给定需求（语料库规模、查询模式、延迟预算 (Latency Budget)、质量门槛、基础设施约束），输出以下内容：

1. 检索栈 (Retrieval Stack)。仅 BM25、仅稠密检索 (Dense Retrieval)、混合检索 (Hybrid Retrieval)（BM25 + 稠密检索 + 倒数排名融合 RRF）、混合检索 + 交叉编码器重排 (Cross-Encoder Reranking)，或三路检索（BM25 + 稠密检索 + 学习型稀疏检索 Learned-Sparse Retrieval）。
2. 稠密编码器 (Dense Encoder)。指定具体模型名称（`all-MiniLM-L6-v2`、`bge-large-en-v1.5`、`e5-large-v2`、`paraphrase-multilingual-MiniLM-L12-v2`）。需与目标语言、领域及上下文长度相匹配。
3. 重排器 (Reranker)。若使用交叉编码器模型，请指明具体名称（`cross-encoder/ms-marco-MiniLM-L-6-v2`、`BAAI/bge-reranker-large`）。需标注在 Top-30 结果上重排将增加约 30-100 毫秒的延迟。
4. 评估方案 (Evaluation Plan)。召回率@10 (Recall@10) 作为检索器的核心指标。多答案场景使用平均倒数排名 (MRR)。首先建立基线 (Baseline)，后续增量改进均以此为准进行衡量。

若语料库包含命名实体 (Named Entities)、错误代码或产品 SKU，除非用户能提供稠密模型可精准处理精确匹配 (Exact Matches) 的证据，否则拒绝推荐仅使用稠密检索的方案。对于高风险检索 (High-Stakes Retrieval) 场景（如法律、医疗领域），若最终 Top-5 结果将直接决定用户的答案，则拒绝跳过重排步骤。
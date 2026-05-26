---
name: codebase-rag
description: 构建一个跨仓库语义搜索系统，具备抽象语法树（AST）感知分块、混合检索（Hybrid Retrieval）、增量重新索引（Incremental Re-indexing）以及带引用的回答功能。
version: 1.0.0
phase: 19
lesson: 02
tags: [capstone, rag, code-search, tree-sitter, qdrant, bm25, hybrid-retrieval]
---

基于总计至少 200 万行代码的 10 个以上代码仓库，构建数据摄入流水线（Ingestion Pipeline）、混合索引（Hybrid Index）以及强制引用查询代理（Citation-Enforced Query Agent），使其能够使用可验证的 `文件:行号（file:line）` 锚点来回答跨仓库问题。

构建计划：

1. 使用 Tree-sitter 解析每个文件。在函数和类节点边界处进行分块。存储 `{repo, path, start_line, end_line, symbol, body}`。
2. 使用 Claude Haiku 4.5 或 Gemini 2.5 Flash，结合提示词缓存（Prompt Caching）的系统提示词对每个代码块进行摘要。将单句摘要与代码块一同存储。
3. 将数据索引至三种结构中：Qdrant（稠密向量 Dense Vector，使用 Voyage-code-3 或 nomic-embed-code）、Tantivy（带字段权重的 BM25）以及 Kuzu（用于导入、调用和继承关系的符号图边 Symbol Graph Edges）。
4. 构建一个包含三个节点的 LangGraph 查询代理：检索（稠密向量与 BM25 并行）、重排序（使用 Cohere rerank-3 或 bge-reranker-v2-gemma-2b）、合成（使用 Claude Sonnet 4.7，结合提示词缓存并强制要求 `文件:行号` 引用）。
5. 后过滤：拒绝任何缺乏可验证 `(repo/path:start-end)` 锚点的声明；重新提问或直接丢弃。
6. 接入 `git push` Webhook，计算符号级差异（Symbol-level Diff）并仅对变更的代码块重新进行向量化嵌入。目标：在 200 万行代码规模的集群上，实现 50 个文件的提交在 60 秒内可被检索。
7. 使用包含 100 个问题的预留测试集（Held-out Set）进行评估。报告 MRR@10、nDCG@10、引用忠实度（Citation Faithfulness）以及延迟百分位数（Latency Percentiles）。
8. 运行每周数据漂移检测任务（Data Drift Job），重新执行评估，并在 MRR@10 下降超过 5% 时触发告警。

评估标准：

| 权重 | 评估标准 | 测量指标 |
|:-:|---|---|
| 25 | 检索质量 | 在 100 题预留测试集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 答案声明中带有可验证 `文件:行号` 锚点的比例 |
| 20 | 延迟与规模 | 在索引语料库规模下，10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 50 个文件的提交从 `git push` 到可被检索所需的时间 |
| 15 | 用户体验与回答格式 | 引用可点击性、代码片段预览、后续交互引导 |

硬性拒绝条件：

- 使用固定大小 Token 分块而非 AST 感知分块。这将污染以生成代码为主的语料库。
- 仅使用余弦相似度检索，缺乏 BM25 或重排序。已知在精确符号名称查询上会失效。
- 回答未强制包含 `文件:行号` 引用。
- 每次 `git push` 都重新嵌入整个语料库；必须采用增量方式。

拒绝规则：

- 未阅读许可证前拒绝索引仓库。部分许可证禁止将代码嵌入第三方向量数据库。
- 拒绝回答声称引用了索引中从未出现过的文件的查询；返回结果前必须始终验证锚点。
- 拒绝提供 p95 延迟超过 4 秒的回答；改为返回部分结果并附带后续交互句柄。

输出要求：一个包含数据摄入流水线、LangGraph 查询代理、100 题标注评估集的代码仓库，一个 Langfuse 仪表板链接，以及一份说明文档。文档需明确指出你修复的三种检索失败模式（生成代码污染 Generated-Code Poisoning、长尾符号召回 Long-Tail Symbol Recall、跨仓库符号解析 Cross-Repo Symbol Resolution）以及修复每种模式的具体改动。
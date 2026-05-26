# 综合项目 02 — 基于代码库的检索增强生成（RAG）（跨仓库语义搜索）

> 到 2026 年，所有成熟的工程团队都会部署内部代码搜索系统，这类系统能够理解代码语义，而非仅仅进行字符串匹配。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业图谱、Aider 的 repomap、Pinterest 的内部 MCP —— 它们的底层架构如出一辙。摄入多个代码仓库，使用 tree-sitter 进行解析，对函数和类级别的文本块（chunks）进行向量化嵌入（embedding），执行混合搜索（hybrid search）与重排序（re-ranking），最终生成附带引用来源的答案。本项目要求你构建这样一个系统，使其能够处理跨越 10 个仓库的 200 万行代码，并在每次 `git push` 时稳定支持增量重新索引。

**类型：** 综合项目（Capstone）
**编程语言：** Python（数据摄入），TypeScript（API + UI）
**前置要求：** 第 5 阶段（自然语言处理基础），第 7 阶段（Transformer 架构），第 11 阶段（大语言模型工程），第 13 阶段（工具调用），第 17 阶段（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**预计耗时：** 30 小时

## 问题背景

到 2026 年，所有前沿的代码智能体（coding agent）都会内置代码库检索层，因为仅依赖上下文窗口（context window）无法解决跨仓库查询问题。Claude 的 100 万 token 上下文固然有用，但并不能消除对排序检索（ranked retrieval）的需求。对原始文本块进行简单的余弦相似度搜索（cosine search）会在处理生成代码、单体仓库（monorepo）重复代码以及极少被导入的长尾符号时严重干扰检索结果。工业界的标准解法是：基于感知抽象语法树（Abstract Syntax Tree, AST）的文本块，结合重排序器（re-ranker）执行混合搜索（稠密向量 + BM25），并由符号引用图谱提供底层支持。

你将通过对真实的代码仓库集群（而非单个教程仓库）建立索引来掌握这项技术，并重点评估 MRR@10（Mean Reciprocal Rank at 10）、引用忠实度（citation faithfulness）以及增量更新时效性（incremental freshness）。典型的故障场景往往出现在基础设施层面：例如包含 10 万个文件的单体仓库、一次触及半数文件的代码推送，或是需要横跨四个仓库才能给出正确答案的复杂查询。

## 核心概念

一套感知抽象语法树的数据摄入流水线（ingestion pipeline）会使用 tree-sitter 解析每个文件，提取函数与类节点，并严格在节点边界处进行分块，而非采用固定的 Token 窗口。每个文本块将生成三种表示形式：稠密向量嵌入（dense embedding，例如 Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及一段简短的自然语言摘要。摘要引入了第三种可检索模态（retrievable modality）——即使用户提问“X 是如何进行权限校验的”，而代码中仅包含 `check_permission`，摘要中也会明确提及“authz”。

检索采用混合架构。单次查询会同时触发稠密向量搜索与 BM25 搜索，合并 Top-K 结果后，将并集交由交叉编码器重排序器（cross-encoder re-ranker，例如 Cohere rerank-3 或 bge-reranker-v2-gemma-2b）处理。重排序后的结果列表将输入至长上下文合成器（long-context synthesizer，如启用提示词缓存（prompt caching）的 Claude Sonnet 4.7，或自托管的 Llama 3.3 70B），系统会严格指令模型对每一项结论按“文件路径及行号范围”进行引用。任何未附带引用的回答都将被后置过滤器（post-filter）直接拦截。

增量更新时效性是基础设施层面的核心挑战。每次 `git push` 都会触发差异分析（diff）：精准定位发生变更的文件与符号。系统仅对受影响的文本块执行重新向量化嵌入。同时，受影响的跨文件符号关联边（如导入语句、方法调用）也会被重新计算。借此，索引库能够始终保持一致，无需在每次代码提交时重新处理全部 200 万行代码。

## 架构设计

git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations

## 技术栈 (Tech Stack)

- 解析 (Parsing)：使用 tree-sitter 及 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密向量嵌入 (Dense Embeddings)：Voyage-code-3（托管服务）或 nomic-embed-code-v1.5（自托管），bge-code-v1 作为备选方案
- 稀疏索引 (Sparse Index)：基于 Rust 的 Tantivy 与 BM25F 算法，针对符号名称与代码主体进行字段加权
- 向量数据库 (Vector Database)：支持混合搜索的 Qdrant 1.12，或面向向量规模低于 5000 万的团队使用 pgvector + pgvectorscale
- 代码块摘要模型 (Chunk Summary Model)：Claude Haiku 4.5 或 Gemini 2.5 Flash，启用提示词缓存 (Prompt Caching)
- 重排序器 (Re-ranker)：Cohere rerank-3 或自托管的 bge-reranker-v2-gemma-2b
- 流程编排 (Orchestration)：使用 LlamaIndex Workflows 进行数据摄入，LangGraph 用于构建查询智能体
- 答案生成器 (Synthesizer)：Claude Sonnet 4.7（支持 100 万上下文窗口），配合提示词缓存
- 符号图 (Symbol Graph)：Neo4j（托管版）或 kuzu（嵌入式），用于构建导入与调用关系边
- 可观测性 (Observability)：通过 Langfuse 记录每次检索与生成步骤的追踪片段 (Spans)

## 动手构建

1. **数据摄取遍历器（Ingestion walker）。** 在每次推送钩子（push hook）触发时遍历 Git 历史记录，收集发生变更的文件。针对每个文件，使用 tree-sitter 进行解析，提取函数与类节点及其完整的源代码范围。最终输出代码块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **代码块摘要生成器（Chunk summarizer）。** 将代码块批量发送至 Haiku 4.5 模型进行调用，并在系统提示词前缀（system preamble）上启用提示词缓存（prompt caching）。提示词为：“用一句话总结该函数，说明其公开接口契约（public contract）与副作用（side effects）。”将生成的摘要与对应代码块一同存储。

3. **向量嵌入池（Embedding pool）。** 包含两条并行队列：稠密向量（dense）队列（使用 Voyage-code-3，批次大小为 128）与摘要向量（summary）队列（使用相同模型，但输入为摘要字符串）。将生成的向量写入 Qdrant，并附带载荷（payload） `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引（BM25 index）。** 采用基于字段加权（field-weighted）的 Tantivy 索引：符号名称权重设为 4，符号主体权重设为 1，摘要权重设为 2。该设计同时支持“查找名为 X 的函数”与“查找执行 X 操作的函数”两类查询。

5. **符号关系图（Symbol graph）。** 针对每个代码块记录边（edges）关系：导入（imports）关系（该文件使用了仓库 Z 中的符号 Y）、调用（calls）关系（该函数调用了类 C 的方法 M）以及继承（inheritance）关系。数据存储在 kuzu 中，在查询阶段用于跨仓库边界扩展检索范围。

6. **查询代理（Query agent）。** 基于 LangGraph 构建，包含三个节点。`retrieve` 节点并行触发稠密向量检索与 BM25 检索，并按 `(repo, path, symbol)` 进行去重。`rerank` 节点对前 50 个结果运行交叉编码器（cross-encoder），并保留前 10 个。`synth` 节点将重排序后的代码块作为上下文调用 Claude Sonnet 4.7，缓存系统提示词，并要求输出必须包含 `file:line` 格式的引用。

7. **引用强制校验（Citation enforcement）。** 解析模型输出内容；任何缺乏 `(repo/path:start-end)` 锚点的声明都将被标记为需重新提问或直接丢弃。最终仅向用户返回包含有效引用的答案。

8. **增量重建索引（Incremental re-index）。** 每次接收 Webhook 时，计算符号级别的差异（diff）。仅对文本发生变化的代码块重新进行向量嵌入（re-embed），并对导入关系发生变化的代码块重新计算符号边。性能指标：针对 200 万行代码（LOC）的代码库，一次包含 50 个文件的推送可在 60 秒内完成索引重建。

9. **评估（Eval）。** 标注 100 个跨仓库问题，并附带标准答案（gold answers）的 `file:line` 格式。测量指标包括：MRR@10、nDCG@10、引用忠实度（citation faithfulness，即带有可验证锚点的声明占比）以及 p50/p99 延迟（latency）。

## 使用指南

$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]

## 部署上线

交付技能 `outputs/skill-codebase-rag.md`。给定一组代码仓库语料库，该技能将部署数据摄入流水线（ingestion pipeline）、混合索引（hybrid index）和查询代理（query agent），并针对任何跨仓库问题返回带引用的答案。评估标准：

| 权重 | 评估指标 | 测量方法 |
|:-:|---|---|
| 25 | 检索质量（Retrieval quality） | 在包含 100 个问题的保留测试集（held-out set）上计算 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度（Citation faithfulness） | 答案中带有可验证的 `文件:行号` 锚点的声明所占比例 |
| 20 | 延迟与扩展性（Latency and scale） | 在索引语料库规模下，10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性（Incremental indexing correctness） | 从 `git push` 到包含 50 个文件的提交内容可被搜索所需的时间 |
| 15 | 用户体验与答案格式（UX and answer formatting） | 引用可点击性、代码片段预览、后续交互引导（follow-up affordance） |
| **100** | | |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。测量 MRR@10 的变化差值（delta）。报告在启用重排序（re-ranking）后差距是否缩小。
2. 向语料库中注入 20% 的生成代码（由大语言模型生成的样板代码）并重新评估。观察检索中毒（retrieval poisoning）现象。在数据负载（payload）中添加 "generated" 标志，并降低这些命中结果的权重。
3. 在您的语料库规模下，对 Qdrant 混合搜索与 pgvector + pgvectorscale 进行基准测试。报告批量大小（batch size）为 1 时的 p99 延迟。
4. 添加基于采样的漂移检查（drift check）：每周重新运行 100 题评估。当 MRR@10 下降超过 5% 时触发告警。
5. 扩展至跨语言符号解析（cross-language symbol resolution）：例如一个通过 gRPC 调用 Go 服务的 Python 函数。使用符号图（symbol graph）将它们关联起来。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| AST-aware chunking | “函数级切分” | 在 tree-sitter 节点边界处切分代码，而非使用固定的 token 窗口 |
| Hybrid search | “稠密 + 稀疏” | 并行运行 BM25 和向量搜索，合并 top-k 结果后进行重排序 |
| Cross-encoder rerank | “二阶段排序” | 对每个（查询，候选）对进行联合打分的模型，比余弦相似度更准确 |
| Prompt caching | “缓存系统提示词” | 2026 年 Claude / OpenAI 推出的功能，可对重复的前缀 token 提供高达 90% 的折扣 |
| Symbol graph | “代码图” | 表示跨文件和仓库的导入、调用、继承关系的边 |
| Citation faithfulness | “有依据的答案率” | 用户可通过点击锚点并阅读引用片段来验证的声明所占比例 |
| Incremental re-index | “推送至可搜索时间” | 从 `git push` 到变更符号可被查询的实际耗时（wall-clock） |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能（cross-repo code intelligence）
- [Sourcegraph Cody 检索增强生成（RAG）架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本核心综合项目的参考深度解析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — 基于 Tree-sitter 排序的仓库视图
- [Augment Code 企业级图谱](https://www.augmentcode.com) — 商业化符号图谱（symbol-graph）检索增强生成（RAG）
- [Qdrant 混合搜索（hybrid search）文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码嵌入（embeddings）](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 模型详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器（cross-encoder）参考
- [Pinterest MCP（模型上下文协议）内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考案例
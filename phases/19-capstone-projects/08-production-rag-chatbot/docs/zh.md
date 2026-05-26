# 综合实战项目 08 — 面向受监管垂直领域的生产级检索增强生成（RAG）聊天机器人

> 到 2026 年，Harvey、Glean、Mendable 和 LlamaCloud 均采用相同的生产架构。使用 docling 或 Unstructured 进行文档解析，并结合 ColPali 处理视觉内容。采用混合检索（Hybrid Search）。使用 bge-reranker-v2-gemma 进行重排序（Re-ranking）。借助提示词缓存（Prompt Caching）实现 60-80% 的命中率，并使用 Claude Sonnet 4.7 进行内容合成。通过 Llama Guard 4 和 NeMo Guardrails 实施安全防护。利用 Langfuse 和 Phoenix 进行系统监控。在包含 200 个问题的黄金测试集（Golden Set）上使用 RAGAS 进行评估打分。在受监管领域（法律、临床、保险）构建一个此类系统，本项目的通过标准即为：成功通过黄金测试集、红队测试（Red Teaming）以及数据漂移监控面板（Drift Dashboard）的考核。

**类型：** 综合实战项目
**编程语言：** Python（流水线 + API）、TypeScript（聊天界面）
**前置要求：** 第 5 阶段（自然语言处理 NLP）、第 7 阶段（Transformer 架构）、第 11 阶段（大语言模型工程 LLM Engineering）、第 12 阶段（多模态 Multimodal）、第 17 阶段（基础设施 Infrastructure）、第 18 阶段（安全 Safety）
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18
**预计耗时：** 30 小时

## 问题背景

面向受监管领域的检索增强生成（RAG，如法律合同、临床试验方案、保险保单）是 2026 年交付量最大的生产架构，因为其投资回报率（ROI）显而易见，且业务风险明确具体。Harvey（Allen & Overy 律所）将其应用于法律领域。Mendable 专注于开发者文档场景。Glean 则覆盖企业级搜索。其核心模式为：高保真文档解析、结合重排序的混合检索、强制引用与提示词缓存的内容合成、多层安全防护，以及持续的数据漂移（Data Drift）监控。

真正的难点并不在于模型本身。而是涉及司法管辖区合规性（如 HIPAA、GDPR、SOC2）、引用级别的审计追踪能力、成本控制（在高命中率下，提示词缓存可带来 60-90% 的成本折扣）、基于 RAGAS 忠实度（Faithfulness）指标的幻觉检测（Hallucination Detection），以及当源文档更新但索引未能及时同步时的漂移检测。本综合实战项目要求你在包含 200 个问题的黄金测试集上交付完整系统，并配套红队测试套件。

## 核心概念

该流水线（Pipeline）包含两个核心环节。**数据摄取（Ingestion）**：使用 docling 或 Unstructured 解析结构化文档；ColPali 处理视觉密集型文档；文本块（chunks）会附加摘要、标签以及基于角色的访问控制标签。向量数据将存入 pgvector + pgvectorscale（适用于 5000 万向量以下规模）或 Qdrant Cloud；同时并行运行稀疏检索 BM25。**对话交互（Conversation）**：由 LangGraph 负责记忆管理与多轮对话；每次查询均执行混合检索（hybrid retrieval），使用 bge-reranker-v2-gemma-2b 进行重排序（rerank），通过 Claude Sonnet 4.7（启用提示词缓存）进行内容合成，输出结果需经过 Llama Guard 4 与 NeMo Guardrails 的安全过滤，最终生成附带引用锚点的回复。

评估栈（Eval stack）分为四个层级。**黄金数据集（Golden set）**（包含 200 条带引用的标注问答对）用于验证答案正确性。**红队测试（Red team）**（涵盖越狱攻击、PII 提取尝试及跨领域提问）用于保障系统安全性。**RAGAS** 用于自动按轮次评估忠实度（faithfulness）、答案相关性（answer relevance）与上下文精确度（context precision）。**漂移监控面板（Drift dashboard）**（基于 Arize Phoenix）每周监控检索质量与幻觉分数（hallucination score）。

提示词缓存（Prompt caching）是控制成本的关键杠杆。Claude 4.5+ 与 GPT-5+ 均支持对系统提示词及检索到的上下文进行缓存。当缓存命中率（hit rate）达到 60%-80% 时，单次查询成本可降低 3-5 倍。流水线架构必须针对稳定前缀（stable prefixes，即优先固定系统提示词与重排序后的上下文）进行设计，以实现较高的缓存命中率。

## Architecture

documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)

## Stack

- 数据摄取（Ingestion）：使用 Unstructured.io 或 docling 处理结构化文档；ColPali 处理视觉密集型 PDF
- 向量数据库（Vector DB）：5000 万向量以下使用 pgvector + pgvectorscale；超出则使用 Qdrant Cloud
- 稀疏检索（Sparse）：Tantivy BM25，支持字段权重配置
- 编排（Orchestration）：LlamaIndex Workflows（负责数据摄取）+ LangGraph（负责对话交互）
- 重排序（Re-ranker）：自托管 bge-reranker-v2-gemma-2b 或使用托管版 Voyage rerank-2
- 大语言模型（LLM）：Claude Sonnet 4.7（启用提示词缓存）；备用方案为自托管 Llama 3.3 70B
- 评估（Eval）：在线运行 RAGAS 0.2；使用 DeepEval 进行幻觉与越狱测试套件评估
- 可观测性（Observability）：自托管 Langfuse 并配置人工标注队列；使用 Arize Phoenix 监控数据漂移
- 安全护栏（Guardrails）：Llama Guard 4 输入/输出分类器、NeMo Guardrails v0.12 策略引擎、Presidio PII 脱敏
- 合规性（Compliance）：文本块附加基于角色的访问控制标签；添加司法管辖区标签以满足 GDPR/HIPAA 要求

## 构建实施

1. **数据摄取（Ingestion）。** 使用 Unstructured 或 docling 解析你的语料库（生产级构建建议包含 1000-10000 份文档）。针对扫描件或视觉密集型页面，通过 ColPali 进行路由处理。生成带有摘要、角色标签及管辖权标签的文本块（Chunks）。

2. **索引（Index）。** 将稠密向量嵌入（Dense Embeddings）（使用 Voyage-3 或 Nomic-embed-v2）存入 pgvector + pgvectorscale。通过 Tantivy 构建 BM25 辅助索引（BM25 Side-index）。将角色与管辖权过滤器作为负载（Payload）附加。

3. **混合检索（Hybrid Retrieval）。** 首先按角色与管辖权进行过滤；随后并行执行稠密检索与 BM25 检索；使用倒数排名融合（Reciprocal Rank Fusion）合并结果；将 Top-20 送入重排序器（Reranker）；将 Top-5 送入生成模块（Synthesis）。

4. **基于提示词缓存的生成（Synthesize with Prompt Caching）。** 将系统提示词（System Prompt）与静态策略置于缓存头部；将重排序后的上下文作为缓存扩展；用户提问作为未缓存的后缀。在稳定运行状态下，目标缓存命中率（Cache Hit Rate）为 60-80%。

5. **安全护栏（Guardrails）。** 输入端接入 Llama Guard 4；使用 NeMo Guardrails 拦截非业务领域问题或策略禁止的主题；通过 Presidio 清理输出中意外泄露的个人身份信息（PII）；在后处理阶段强制执行引用校验。

6. **黄金数据集（Golden Set）。** 由领域专家标注的 200 组问答对，包含（答案，引用来源）。基于精确引用匹配、答案正确性及忠实度（Faithfulness，使用 RAGAS 框架）对智能体进行评分。

7. **红队测试（Red Teaming）。** 准备 50 个对抗性提示词（Adversarial Prompts）：越狱攻击（Jailbreaks，如 PAIR、TAP）、PII 窃取尝试、域外提问及跨管辖权数据泄露。以通过/失败及严重程度进行评分。

8. **漂移监控面板（Drift Dashboard）。** 使用 Arize Phoenix 每周跟踪检索质量（nDCG、引用忠实度）。当指标下降 5% 时触发告警。

9. **成本报告（Cost Report）。** 通过 Langfuse 监控：提示词缓存命中率、单次查询 Token 消耗、以及按处理阶段划分的单次查询成本（$/query）明细。

## 使用方式

$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]

## 交付上线

`outputs/skill-production-rag.md` 描述了交付物。一个部署了合规标签的受监管领域聊天机器人，已通过评估量规（Rubric），并配备实时漂移监控。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 黄金数据集（200 组问答）上的在线评分 |
| 20 | 引用正确性 | 包含可验证来源锚点的答案占比 |
| 20 | 护栏覆盖率 | Llama Guard 4 通过率 + 越狱测试套件结果 |
| 20 | 成本/延迟工程优化 | 提示词缓存命中率、P95 延迟（P95 Latency）、$/query |
| 15 | 漂移监控面板 | 包含每周检索质量趋势的 Phoenix 实时面板 |
| **100** | | |

## 练习

1. 在另一个合规辖区（jurisdiction）下构建第二个语料切片（corpus slice）（例如，将 HIPAA 与 GDPR 并列）。通过一个包含 20 个问题的跨辖区探测（cross-jurisdiction probe）测试，验证“角色+辖区”过滤（role+jurisdiction filtering）能否有效防止跨区数据泄露（cross-leak）。

2. 统计一周生产环境流量下的提示词缓存命中率（prompt-cache hit rate）。识别哪些查询会破坏缓存前缀（cache prefix），并据此进行重构。

3. 引入带有 10k token 摘要缓冲区（summary buffer）的多轮对话记忆（multi-turn memory）机制。评估随着对话轮次增加，回答的忠实度（faithfulness）是否会出现下降。

4. 将 Claude Sonnet 4.7 替换为本地部署（self-hosted）的 Llama 3.3 70B 模型。对比测量单次查询成本（$/query）与忠实度变化量（faithfulness delta）。

5. 增加“不确定”模式：若重排序（reranked）后的最高得分低于设定阈值，智能体将回复“我无法提供确切的引用依据”而非直接作答。评估该模式对降低虚假置信度（false-confidence）的效果。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Prompt caching (提示词缓存) | “缓存的系统提示词+上下文” | Claude/OpenAI 特性：命中缓存前缀 token 时可享受 60-90% 的费用折扣 |
| RAGAS | “RAG 评估器” | 对忠实度（faithfulness）、答案相关性（answer relevance）和上下文精确度（context precision）进行自动化评分 |
| Golden set (黄金数据集) | “标注评估集” | 包含 200+ 条专家标注的问答及引用数据；作为评估的基准真相（ground truth） |
| Jurisdiction tag (辖区标签) | “合规标签” | 附加在文本块（chunks）上的 GDPR/HIPAA/SOC2 合规范围标识；由检索过滤器强制执行 |
| Citation faithfulness (引用忠实度) | “有据回答率” | 由可检索到的原文片段（source spans）所支撑的声明占比 |
| Drift (数据漂移) | “检索质量衰减” | nDCG 或引用得分的周度变化；告警阈值设为 5% |
| Red team (红队测试) | “对抗性评估” | 发布前的越狱攻击（jailbreak）、PII（个人身份信息）提取及跨领域探测测试 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 法律领域生产环境技术栈参考
- [Glean enterprise search](https://www.glean.com) — 企业级规模 RAG 参考
- [Mendable documentation](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管式数据摄入（managed ingestion）方案
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本优化杠杆参考
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 标准 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移可观测性（drift observability）参考
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 版安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略护栏（policy rail）框架
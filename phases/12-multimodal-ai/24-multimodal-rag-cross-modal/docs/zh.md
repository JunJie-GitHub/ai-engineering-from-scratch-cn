# 多模态检索增强生成 (Multimodal RAG) 与跨模态检索 (Cross-Modal Retrieval)

> 视觉原生文档 RAG (Vision-Native Document RAG) 仅是其中一环。生产级多模态 RAG (Production Multimodal RAG) 的应用边界更为广阔——它能够跨越文本、图像、音频和视频进行检索，适用于旅行规划（“帮我找一家有自然光、安静的素食早午餐店”）、医疗分诊（“这张照片加上这些笔记对应什么伤情”）、电子商务（“找出和我这张自拍穿搭相似、且尺码合适的服装”）以及现场运维（“根据这段引擎声音和零件照片进行故障诊断”）等工作流。2025 年的三篇综述文献（Abootorabi 等人、Mei 等人、Zhao 等人）对这些子问题进行了系统化梳理：跨模态检索、检索融合 (Retrieval Fusion)、生成依据 (Generation Grounding) 以及多模态评估 (Multimodal Evaluation)。本课程将深入研读这些综述，并设计一套生产级流水线。

**类型：** 构建
**语言：** Python（标准库，带融合功能的跨模态检索器 + 带依据的生成器）
**前置条件：** Phase 12 · 23 (ColPali)，Phase 11（RAG 基础）
**时长：** 约 180 分钟

## 学习目标

- 设计跨模态检索：文本 → 图像、图像 → 文本、音频 → 视频等。
- 比较三种融合策略：分数融合 (Score Fusion)、基于注意力的融合 (Attention-Based Fusion)、混合专家融合 (MoE Fusion)。
- 解释生成依据：当知识来源混合了多种模态时，“引用你的来源”具体呈现为何种形式。
- 列举 2025 年三篇经典的多模态 RAG 综述文献及其子问题分类体系 (Taxonomy)。

## 问题背景

单模态 RAG (Single-Modality RAG) 已是一种成熟范式：对查询进行嵌入 (Embedding)，对文本块进行嵌入，执行检索，然后将结果输入大语言模型 (LLM)。而多模态 RAG 则需要：

1. 多个检索头 (Retrieval Heads)（每种模态都需要在兼容的向量空间中进行嵌入）。
2. 跨模态检索结果的融合。
3. 能够跨模态引用来源的生成依据机制。
4. 涵盖跨模态信号的评估指标。

2025 年的这几篇综述文献均得出了相同的分类体系。

## 核心概念

### 跨模态检索 (Cross-modal retrieval)

给定模态 A 的查询，检索模态 B 的文档。三种模式：

1. 共享嵌入空间 (Shared embedding space)。CLIP 和 CLAP 在共享空间中生成文本+图像/文本+音频的嵌入表示 (embeddings)。跨模态的余弦相似度 (Cosine similarity) 可直接计算。但仅限于 CLIP 训练过的配对数据。

2. 单模态编码器加转换模块 (Per-modality encoder + translation)。文本编码器 + 图像编码器 + 一个用于在空间之间映射的小型转换模块。例如 Gupta 等人提出的 Sen2Sen 及其他 2024 年的设计。灵活性高，但增加了系统复杂度。

3. 视觉语言模型 (Vision-Language Model, VLM) 作为编码器。使用 VLM 的隐藏状态 (hidden states) 作为检索表示。VLM 支持的任何模态均可适用。质量更高，但计算成本也更高。

选型建议：文本+图像使用 CLIP / SigLIP 2；文本+音频使用 CLAP；追求前沿质量的跨模态检索可使用 VLM 隐藏状态。

### 融合策略 (Fusion strategies)

假设你检索到了 10 个结果：5 张图像、3 段文本、2 个音频片段。如何进行合并？

分数融合 (Score fusion)（成本最低）。每个模态使用独立的检索器，各自返回分数。先在模态内部对分数进行归一化 (Normalize)，然后求和。方法简单，且通常有效。

基于注意力的融合 (Attention-based fusion)。将所有检索项拼接，通过一个小型注意力网络 (Attention network) 进行加权。需要额外训练。

混合专家融合 (Mixture of Experts, MoE fusion)。门控网络 (Gating network) 将请求路由至特定模态的专家模型。不同类型的查询路由方式不同——例如视觉类问题会赋予图像更高的权重。

生产环境默认方案：采用分数融合，并对查询的主导模态施加轻微偏置。若 A/B 测试在你的业务领域显示出明确优势，可升级至 MoE 方案。

### 生成溯源 (Generation grounding)

大语言模型 (Large Language Model, LLM) 应明确指出支撑每项主张的检索来源。针对多模态场景：

- 文本来源：标准引用格式 `[1]`。
- 图像来源：`[img 3]` 附带简短说明。
- 音频来源：`[audio 2 at 0:34]`。

使用具备溯源意识的数据训练生成器：训练目标中的每项主张均标注了来源索引。在推理阶段，模型会自然地输出引用标记。

### 2025 年综述文献 (2025 surveys)

Abootorabi 等人（arXiv:2502.08826，《Ask in Any Modality》）：多模态检索增强生成 (Multimodal RAG) 的分类体系。涵盖检索、融合与生成环节。覆盖范围最广。

Mei 等人（arXiv:2504.08748，《A Survey of Multimodal RAG》）：聚焦子任务基准测试 (Benchmarks) 与失败模式分析。对评估方案设计极具参考价值。

Zhao 等人（arXiv:2503.18016）：以视觉为核心的综述。对 ColPali 系列工作的分析尤为深入。

通读这三篇文献即可掌握截至 2025 年春季的技术前沿 (State of the art)。大多数子问题仍处于开放研究阶段。

### MuRAG —— 奠基性论文 (MuRAG — the foundational paper)

MuRAG（Chen 等人，2022 年）是首个多模态 RAG 系统。它从多模态知识库 (Knowledge Base, KB) 中检索图像与文本并生成答案。在 VLM 浪潮兴起前便验证了该路线的可行性。现代系统（如 REACT、VisRAG、M3DocRAG）均在此基础上构建。

### 生产环境旅行规划示例 (A production trip-planner example)

查询：“帮我找一家安静、有自然光线的纯素食早午餐店。”

处理流程：

1. 查询分解。“安静” → 音频/评论关键词；“纯素食早午餐” → 菜单项；“自然光线” → 图像特征。
2. 按模态检索：
   - 基于评论的文本检索：“纯素食早午餐，环境安静。”
   - 基于餐厅照片的图像检索：“自然光线，空间通透。”
   - 基于环境声音片段的音频检索：“低分贝，无背景音乐。”
3. 分数融合。为每家餐厅计算综合得分。
4. 选取 Top-k 餐厅 → 输入 VLM 生成器并附带所有证据 → 输出带引用的答案。

这已远超纯文本 RAG 的能力范畴。每种模态都补充了单一文本无法捕捉的信号。

### 智能体多模态 RAG (Agentic multimodal RAG)

多跳 (Multi-hop)：若首次检索未能返回高置信度答案，LLM 将重新构建查询并再次检索。第 14 阶段介绍的智能体 RAG (Agentic RAG) 模式在此同样适用。示例：

- 检索初始 Top-10 → LLM 判断“太嘈杂，过滤掉 >40 分贝的” → 重新检索。
- 检索图像 → LLM 发现其中一张包含菜单 → 检索该菜单文本 → 生成答案。

虽增加了系统复杂度，但能处理单次检索 (Single-shot retrieval) 无法解决的查询。

### 评估 (Evaluation)

跨模态评估体系尚不成熟。常用代理指标 (Proxies) 包括：

- 各模态的召回率@k (Recall@k)。
- 融合后的 Top-k 准确率 (Top-k accuracy)。
- 人工评估的端到端满意度 (End-to-end satisfaction)。
- 任务特定指标（如完成预订数、实际购买数）。

目前尚无覆盖所有模态的标准基准测试。大多数论文仅在特定领域任务上进行评估。

## 使用它

`code/main.py`：

- 三个模拟检索器（Retriever）（文本、图像、音频），在共享的餐厅语料库（Corpus）上运行。
- 分数融合（Score Fusion）机制，通过可配置的权重组合各模态（Modality）的得分。
- 一个生成器存根（Generator Stub），用于输出带有引用来源的最终答案。
- 一个简单的智能体循环（Agentic Loop），当置信度（Confidence）较低时会重新构建查询。

## 交付它

本课时将生成 `outputs/skill-multimodal-rag-designer.md`。给定包含多模态查询流程（Multimodal Query Flow）的产品规格说明，该文件将设计检索器、融合模块、生成器及评估方案。

## 练习

1. 设计一个用于医疗分诊的多模态检索增强生成（Multimodal RAG）系统：查询条件 = 伤情照片 + 文本症状。哪些模态应从哪些知识库（Knowledge Base, KB）中检索信息？

2. 分数融合（Score Fusion）本质上是简单的加权求和。它存在哪种失效模式（Failure Mode），而混合专家融合（Mixture of Experts Fusion, MoE Fusion）能够避免？

3. 阅读 Abootorabi 等人的分类体系（Taxonomy）（第 3 节）。其中包含哪三个经典子问题？它们如何映射到你选择的产品中？

4. 为旅行规划多模态 RAG 设计一份评估规范（Evaluation Spec）。哪些评估指标能够覆盖图像召回率（Recall）、音频召回率以及综合正确性（Composite Correctness）？

5. 智能体多跳检索增强生成（Agentic Multi-hop RAG）在每次往返交互（Round-trip）中都会产生延迟开销（Latency Tax）。当查询难度达到何种程度时，准确率的提升才足以抵消延迟带来的代价？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 跨模态检索（Cross-modal Retrieval） | “用一种模态查询，检索另一种模态” | 文本查询检索图像；图像查询检索文本；需要共享嵌入空间或模态转换器 |
| 分数融合（Score Fusion） | “合并得分” | 各模态检索得分的加权求和；最基础的融合方式 |
| 混合专家融合（MoE Fusion） | “按模态路由的专家” | 门控网络（Gating Network）根据每次查询决定信任哪个模态的得分 |
| 基于依据的生成（Grounded Generation） | “注明你的来源” | 答案中的每个主张都标记有对应的来源索引 |
| MuRAG | “首个多模态 RAG” | 2022 年发表的论文，确立了多模态 RAG 的基础范式 |
| 智能体多跳（Agentic Multi-hop） | “重构查询并重试” | 大语言模型（LLM）在首次检索置信度较低时，会重新向检索器发起查询 |

## 延伸阅读

- [Abootorabi 等人 — 任意模态提问 (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei 等人 — 多模态 RAG 综述 (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao 等人 — 视觉 RAG 综述 (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen 等人 — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu 等人 — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
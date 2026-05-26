# 综合项目 04 — 多模态文档问答（视觉优先的 PDF、表格与图表）

> 2026 年的文档问答（Document QA）前沿技术已从“先 OCR 后文本（OCR-then-text）”转向“视觉优先的晚期交互（vision-first late interaction）”。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，使用多向量晚期交互（multi-vector late interaction）进行嵌入（embedding），并让查询直接关注图像块（patches）。在处理金融 10-K 报表、学术论文和手写笔记时，该模式大幅优于传统的 OCR 优先方案。请在 1 万页文档上端到端地构建该流水线（pipeline），并发布与“先 OCR 后文本”方案的对比结果。

**类型：** 综合项目（Capstone）
**编程语言：** Python（流水线）、TypeScript（查看器 UI）
**前置要求：** 第 4 阶段（计算机视觉）、第 5 阶段（自然语言处理）、第 7 阶段（Transformer）、第 11 阶段（大语言模型工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**预计耗时：** 30 小时

## 问题背景

企业积累了大量 PDF 文档，但传统的 OCR 流水线往往处理不佳：包含旋转表格的扫描版 10-K 报表、密布公式的学术论文、仅作为图像才有意义的图表，以及手写批注。若将这些内容优先视为纯文本处理，将丢失近一半的有效信号。2026 年的解决方案是对原始页面图像进行基于晚期交互的多向量检索（late-interaction multi-vector retrieval）。ColPali（由 Illuin Tech 提出）开创了这一方法；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步提升了准确率。在 ViDoRe v3 基准测试中，视觉优先检索的得分显著高于“先 OCR 后文本”方案——且在处理图表、表格和手写内容时，优势更为明显。

该方案的代价在于存储空间与延迟。ColQwen 的嵌入表示每页包含约 2048 个图像块向量，而非单一的 1024 维向量，导致原始存储需求急剧膨胀。DocPruner（2026）技术可在不造成可测量精度损失的前提下实现 50% 的向量剪枝（pruning）。你将完成 1 万页文档的索引构建，测量 ViDoRe v3 的 nDCG@5 指标，确保答案响应时间低于 2 秒，并与“先 OCR 后文本”基线进行直接对比。

## 核心概念

晚期交互（late interaction）是指查询中的每个词元（token）都会与每个图像块词元进行打分，并将每个查询词元的最高分进行求和。这种方式无需依赖单一的池化向量即可实现细粒度匹配。多向量索引（如 Vespa、Qdrant 多向量模式或 AstraDB）负责存储每个图像块的嵌入向量，并在检索时执行 MaxSim 计算。

回答模块采用视觉语言模型（vision-language model），它将查询语句与检索到的 Top-K 页面图像作为输入，生成包含证据区域（边界框 bounding boxes 或页码引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿模型选择。针对公式和科学记号，系统会接入 OCR 备用方案（如 Nougat、dots.ocr）作为可选的文本通道。

评估体系采用二维矩阵。一个维度为内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、公式）；另一个维度为检索方法（视觉优先晚期交互 vs 先 OCR 后文本 vs 混合检索 hybrid）。矩阵中的每个单元格将记录 nDCG@5 得分与答案准确率。最终的评估报告即为交付物。

## 架构设计

PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page

## 技术栈 (Stack)

- 页面渲染：使用 PyMuPDF (fitz) 以 180 DPI 渲染，并进行纵向标准化 (portrait-normalized)
- 晚期交互模型 (Late-interaction model)：ColQwen2.5-v0.2 或 ColQwen3-omni（由 Hugging Face 上的 vidore 团队开发）
- 索引：带有**多向量字段 (multi-vector field)** 的 Vespa，或 Qdrant 多向量，或支持最大相似度匹配 (MaxSim) 的 AstraDB
- 剪枝：DocPruner 2026 策略（保留高方差图像块，在精度损失 < 0.5% 的情况下实现 50% 压缩）
- OCR 回退方案（针对公式/密集表格）：dots.ocr 或 Nougat
- VLM 回答器：自托管的 Qwen3-VL-30B 或云端托管的 Gemini 2.5 Pro；InternVL3 作为备用方案
- 评估：ViDoRe v3 基准测试，以及用于多页推理的 M3DocVQA
- 查看器 UI：Next.js 15，使用 Canvas 叠加层显示证据区域

## 构建流程 (Build It)

1. **数据摄入 (Ingest)。** 遍历包含 10-K 年报 (10-K filings)、学术论文和扫描文档在内的 1 万页 PDF 语料库。将每页渲染为 1536x2048 的 PNG 图像。持久化存储 `{doc_id, page_num, image_path}`。

2. **嵌入 (Embed)。** 对每页图像运行 ColQwen2.5-v0.2。输出形状约为 2048 个维度为 128 的图像块嵌入 (patch embeddings)。应用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或 Qdrant 多向量索引。

3. **查询 (Query)。** 对每个传入查询，使用查询塔 (query tower) 进行嵌入（词元级嵌入 (token-level embeddings)）。针对索引运行最大相似度匹配 (MaxSim)：对每个查询词元，计算其与页面图像块嵌入的最大点积，然后求和。返回 top-k 页面。

4. **合成 (Synthesize)。** 将查询与 top-5 页面图像输入 Qwen3-VL-30B。提示词：“仅使用提供的页面进行回答。每个主张需按 (doc_id, page) 引用，并注明区域类型（图表、表格、段落）。”

5. **证据区域 (Evidence regions)。** 对答案进行后处理以提取引用区域。如果 VLM 输出了边界框 (bounding boxes)（Qwen3-VL 支持此功能），则在查看器中将其渲染为叠加层。

6. **OCR 回退 (OCR fallback)。** 对于被识别为公式密集的页面（基于图像方差的启发式规则 (heuristic on image variance)），运行 Nougat 或 dots.ocr，并将 OCR 文本作为额外通道与图像一同传入。

7. **评估 (Eval)。** 运行 ViDoRe v3（检索归一化折损累计增益@5 (nDCG@5)）和 M3DocVQA（多页问答准确率）。同时在相同语料库上使用相同的合成器运行“先 OCR 后文本”流水线。生成内容类型 × 方法的对比矩阵。

8. **UI。** 首先使用 Streamlit 构建原型；生产环境查看器采用 Next.js 15，支持逐页显示证据区域叠加层。

## 使用它 (Use It)

$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4

## 交付上线

`outputs/skill-doc-qa.md` 描述了交付物：一个以视觉为先的多模态文档问答系统（Vision-first Multimodal Document QA System），该系统针对特定语料库（Corpus）进行了优化，并在 ViDoRe v3 上与“先OCR后文本”基线（OCR-then-text Baseline）进行了对比评估。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 基准测试得分对比“先OCR后文本”基线及已发布的排行榜 |
| 20 | 证据区域定位（Evidence-region Grounding） | 实际包含答案片段（Answer Span）的引用区域所占比例 |
| 20 | 存储与延迟工程（Storage and Latency Engineering） | DocPruner 压缩率、索引 p95 延迟、答案生成 p95 延迟 |
| 20 | 多页推理（Multi-page Reasoning） | 在人工标注的 100 题多页数据集上的准确率 |
| 15 | 源文件检查用户体验（Source-inspection UX） | 查看器清晰度、覆盖层保真度、并排对比工具 |
| **100** | | |

## 练习

1. 在同一语料库（Corpus）上对比测试 ColQwen2.5-v0.2 与 ColQwen3-omni。哪些页面一个模型能正确检索而另一个会遗漏？在索引中添加“内容类别”（Content Class）标签，以便按类型进行路由。
2. 对嵌入向量（Embeddings）进行激进剪枝（75%、90%）。寻找“压缩悬崖”（Compression Cliff）：即 ViDoRe nDCG@5 指标跌破 OCR 基线时的临界点。
3. 构建混合检索架构：并行运行“先OCR后文本”（OCR-then-text）与 ColQwen 流程，使用 RRF（Reciprocal Rank Fusion）进行融合，并使用交叉编码器（Cross-encoder）进行重排序。混合架构是否优于任一单一方案？在哪些场景下提升最显著？
4. 将 Qwen3-VL-30B 替换为参数量更小的视觉语言模型（VLM，如 Qwen2.5-VL-7B）。评估“单位成本准确率”（Accuracy-per-dollar）曲线。
5. 增加手写笔记支持。渲染手写语料库，使用 ColQwen 进行嵌入表示，并测量检索效果。与手写 OCR 流水线进行对比。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| Late interaction（晚期交互） | “ColPali 式检索” | 查询词元（Query Tokens）独立地与页面图像块（Page Patches）计算得分；通过 MaxSim 进行聚合 |
| Multi-vector（多向量） | “逐块嵌入” | 每个文档包含多个向量，而非单一池化向量（Pooled Vector） |
| MaxSim（最大相似度） | “晚期交互打分” | 针对每个查询词元，取其与文档向量间的最大相似度，然后求和 |
| DocPruner | “图像块压缩” | 2026 年提出的剪枝技术，保留 50% 的图像块且准确率损失可忽略不计 |
| ViDoRe v3 | “文档检索基准” | 2026 年用于衡量视觉文档检索性能的标准基准 |
| Evidence region（证据区域） | “引用边界框” | 源页面上用于定位答案片段的边界框（Bounding Box） |
| OCR fallback（OCR 回退） | “公式通道” | 与视觉模型并行使用的文本处理流水线，专门针对公式或表格密集的页面 |

## 延伸阅读

- [ColPali（Illuin Tech）代码库](https://github.com/illuin-tech/colpali) — 晚期交互（late-interaction）文档检索参考实现
- [ColPali 论文 (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 核心方法论文
- [Hugging Face 上的 ColQwen 系列模型](https://huggingface.co/vidore) — 生产就绪的模型检查点（checkpoints）
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态检索增强生成（RAG）基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈（serving stack）
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备选索引方案
- [AstraDB 多向量支持](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备选托管索引服务
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式识别的备用光学字符识别（OCR）方案
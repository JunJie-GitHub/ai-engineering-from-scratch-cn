# ColPali 与视觉原生文档检索增强生成（Retrieval-Augmented Generation）

> 传统的检索增强生成（Retrieval-Augmented Generation）将 PDF 解析为文本，切分为文本块（chunk），对块进行嵌入（embedding），并存储向量。每一步都会丢失信号：光学字符识别（OCR）会丢弃图表数据，分块会打断表格行，文本嵌入会忽略图像。ColPali（Faysse 等人，2024 年 7 月）提出了一个更简单的问题：为什么非要提取文本呢？直接通过 PaliGemma 对页面图像进行嵌入，在检索时使用类似 ColBERT 的延迟交互（late interaction）机制，从而保留文档携带的所有布局、图像、字体和格式信号。已发布的基准测试表明：在视觉密集型文档（visually-rich documents）上，其端到端准确率比基于文本的 RAG 高出 20-40%。ColQwen2、ColSmol 和 VisRAG 进一步扩展了这一模式。本课程将解读视觉原生 RAG 的核心论文，并构建一个轻量级的类 ColPali 索引器（indexer）。

**Type:** 构建实践
**Languages:** Python（标准库，多向量索引器 + MaxSim 评分器）
**Prerequisites:** 第 11 阶段（大语言模型工程 — RAG 基础），第 12 阶段 · 05（LLaVA）
**Time:** 约 180 分钟

## 学习目标

- 解释双编码器检索（bi-encoder retrieval，每文档一个向量）与延迟交互检索（late-interaction retrieval，每文档多个向量）之间的区别。
- 描述 ColBERT 的最大相似度（MaxSim）操作，以及 ColPali 如何将其从文本词元（text tokens）推广到图像块（image patches）。
- 构建一个轻量级的类 ColPali 索引器：页面 → 图像块嵌入 → 针对查询词元嵌入的 MaxSim 计算 → 返回前 k 个（top-k）页面。
- 在发票/财务报告用例中，对比 ColPali + Qwen2.5-VL 生成器与基于文本的 RAG + GPT-4 的效果。

## 问题背景

在 PDF 上使用基于文本的 RAG 会丢弃文档的大部分信息。财务报告的第三季度营收增长通常体现在图表中；医疗报告的诊断结果往往位于带标注的图像里；法律合同的签名栏属于布局事实，而非纯文本事实。

基于文本的 RAG 流程如下：

1. PDF → 通过 OCR / pdftotext 转换为文本。
2. 文本 → 切分为 300-500 个词元（token）的块。
3. 文本块 → 双编码器嵌入（bi-encoder embedding，生成单个向量）。
4. 用户查询 → 嵌入 → 余弦相似度（cosine similarity）计算 → 获取 top-k 文本块。
5. 文本块 + 查询 → 输入大语言模型（Large Language Model）。

这五个步骤均存在信息损耗。图表无法被捕获。表格在分块时被割裂。多栏布局被展平。图像标注随之消失。

ColPali 的解决方案：跳过 OCR，直接对页面图像进行嵌入。在检索时采用类似 ColBERT 的延迟交互机制，使模型能够在查询阶段关注细粒度的图像块。

## 核心概念

### ColBERT (2020)

ColBERT（Khattab & Zaharia, arXiv:2004.12832）是一种文本检索方法。与为每个文档生成单一向量不同，它为每个词元（token）生成一个向量。在查询阶段：

- 查询词元会获得各自的嵌入（embeddings）（N_q 个向量）。
- 文档词元也会获得嵌入（N_d 个向量，通常会被缓存）。
- 得分 = 对每个查询词元，计算其与所有文档词元余弦相似度（cosine similarity）的最大值，再将这些最大值求和：Σ_i max_j cos(q_i, d_j)。

这就是最大相似度匹配（MaxSim）操作。每个查询词元会“挑选”出与其最匹配的文档词元，最终得分为这些匹配得分的总和。

优点：召回率（recall）高，能处理词元级语义。缺点：每个文档需存储 N_d 个向量，存储成本较高。

### ColPali

ColPali（Faysse 等, arXiv:2407.01449）将 ColBERT 的模式应用到了图像领域。

- 每个页面通过 PaliGemma（ViT + 语言模型）编码为图像块嵌入（patch embeddings）：每页生成 N_p 个向量。
- 每个用户查询（文本）被编码为查询词元嵌入：N_q 个向量。
- 得分 = Σ_i max_j cos(q_i, p_j)，即在查询文本词元与页面图像块之间执行 MaxSim 操作。
- 根据总得分检索 top-k 页面。

在文档摄入（document-ingestion）阶段：使用 PaliGemma 对每个页面进行嵌入编码，并存储所有图像块向量。在查询阶段：对查询词元进行嵌入编码，与所有已存储的页面嵌入计算 MaxSim，返回得分最高的 k 个页面。

优点：在视觉信息丰富的文档上，端到端性能比文本检索增强生成（text-RAG）高出 20-40%。每个图像块向量都能捕捉局部的版面布局与内容。

缺点：每页 N_p 个图像块 × 4 字节浮点数 × D 维向量 = 存储量增长迅速。可通过乘积量化（PQ）/ 优化乘积量化（OPQ）进行缓解。

### ColQwen2 and ColSmol

ColQwen2（illuin-tech, 2024-2025）将 PaliGemma 替换为 Qwen2-VL。基础编码器更优，检索效果更好。

ColSmol 是面向本地/边缘设备使用的小规模变体。参数量约 10 亿（~1B）的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等, arXiv:2410.10594）是另一种变体：它不在图像块上执行 MaxSim，而是使用视觉语言模型（VLM）将每个页面池化（pool）为单一向量，随后通过双编码器（bi-encoder）进行检索。索引速度更快、存储更小，但召回率较弱。

质量与成本的权衡：追求质量选 ColPali，追求规模选 VisRAG。

### M3DocRAG

M3DocRAG（Cho 等, arXiv:2411.04952）将多模态检索扩展至跨页面、跨文档的推理任务。它能够跨文档检索相关页面，并为 VLM 构建多页上下文。

### ViDoRe — the benchmark

ColPali 的配套基准测试。全称为视觉文档检索评估（Visual Document Retrieval Evaluation）。任务涵盖财务报告、科学论文、行政公文、医疗记录、操作手册等。评估指标：nDCG@5（归一化折损累计增益@5）。

ColPali-v1 在 ViDoRe 上的 nDCG@5 得分约为 80%；而在相同文档上，文本 RAG 的得分仅为 50-60%。

### The end-to-end RAG pipeline

针对原生视觉检索增强生成（vision-native RAG）：

1. 摄入（Ingest）：PDF → 页面图像 → PaliGemma 编码 → 存储所有图像块嵌入。
2. 查询（Query）：用户文本 → 查询词元嵌入 → 与所有已索引页面计算 MaxSim → 返回 top-k 页面。
3. 生成（Generate）：top-k 页面图像 + 查询文本 → 输入 VLM（如 Qwen2.5-VL 或 Claude）→ 生成答案。

全程无需光学字符识别（OCR）。图表、字体、版面布局等信息均直接融入最终答案。

### Storage math

一份 50 页的财务报告，每页包含 729 个图像块，嵌入维度为 128：

- ColPali：50 * 729 * 128 * 4 字节 ≈ 18 MB（原始），经 PQ 量化后 ≈ 4 MB。
- 文本 RAG：50 个文本块 * 768 维 * 4 字节 ≈ 150 kB。

ColPali 单文档存储量约为文本 RAG 的 30 倍。在大规模应用中，OPQ / PQ 可将其降至 5-10 倍，通常处于可接受范围。

### When text-RAG still wins

- 纯文本且无版面结构信息的文档（如维基百科文章、聊天记录）。文本 RAG 更简单且存储成本更低。
- 数百万页规模的档案库，其中存储成本占据主导。
- 严格的合规性要求，规定检索时必须附带可提取的 OCR 文本。

对于 2026 年的其他所有场景——财务报告、科学论文、法律合同、医疗记录、用户体验（UX）文档——原生视觉 RAG 将胜出。

## 使用它

`code/main.py`：

- 示例 Patch 编码器（Toy patch encoder）：将“页面”（特征向量的小型网格）映射为 Patch 嵌入（patch embeddings）数组。
- MaxSim 评分器（MaxSim scorer）：计算查询词元（token）嵌入集与页面 Patch 集之间的 ColBERT 风格得分。
- 索引 5 个示例页面，执行 3 次查询，返回带得分的 Top-k 结果。

## 交付成果

本课时将生成 `outputs/skill-vision-rag-designer.md`。针对文档检索增强生成（Document-RAG）项目，该工具可协助选择 ColPali / ColQwen2 / VisRAG / 文本 RAG 方案，并估算所需的存储容量。

## 练习题

1. 一份 200 页的年度报告，每页包含 729 个图像块（patch），嵌入向量（embedding）维度为 128，采用 4 字节浮点数存储。请计算原始存储空间与经过乘积量化（Product Quantization, PQ）压缩（8 倍）后的存储空间。
2. MaxSim 的计算公式为 Σ_i max_j cos(q_i, p_j)。与简单的平均相似度相比，该求和运算捕捉到了哪些关键信息？
3. ColPali 将页面作为图像块集合进行索引。如果改为在词元级别进行索引（如 ColBERT 的做法），架构会发生什么变化？各自的权衡（trade-offs）是什么？
4. 为一个包含 100 万页的语料库设计端到端流水线（end-to-end pipeline），要求单次查询的延迟预算（latency budget）为 500ms。请在 ColQwen2 / VisRAG 中做出选择并论证理由。
5. 阅读 M3DocRAG（arXiv:2411.04952）。描述其多页注意力模式（multi-page attention pattern），并说明它与单页 ColPali 检索机制的差异。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 晚期交互（Late interaction） | “ColBERT 风格” | 使用每个词元或图像块的嵌入向量结合 MaxSim 进行检索，而非使用单一文档向量 |
| MaxSim | “Patch 取最大值” | 针对每个查询词元，选取相似度最高的文档词元；最后对所有查询词元的结果求和 |
| 双编码器（Bi-encoder） | “单向量” | 每个文档仅对应一个向量；速度更快，但会损失细粒度信息 |
| 多向量（Multi-vector） | “单文档多向量” | 每个文档/页面存储 N_p 个向量；存储成本增加，但召回率提升 |
| Patch 嵌入（Patch embedding） | “页面特征” | 视觉语言模型（VLM）编码器为每个图像块生成的向量，按页面缓存 |
| ViDoRe | “视觉文档基准” | ColPali 用于视觉文档检索的基准测试套件 |
| PQ 量化（PQ quantization） | “乘积量化” | 在保持向量相似度的同时，将存储空间压缩约 8 倍的技术 |

## 延伸阅读

- [Faysse 等人 — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu 等人 — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho 等人 — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub 仓库](https://github.com/illuin-tech/colpali)
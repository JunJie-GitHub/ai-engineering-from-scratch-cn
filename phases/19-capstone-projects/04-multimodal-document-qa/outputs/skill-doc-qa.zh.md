---
name: doc-qa
description: 基于 1 万页文档构建以视觉为先的多模态文档问答（QA）系统，采用延迟交互检索（late-interaction retrieval）并附带证据区域引用（evidence-region citations）。
version: 1.0.0
phase: 19
lesson: 04
tags: [综合项目, 多模态, 检索增强生成, colpali, colqwen, 延迟交互, pdf]
---

给定一个 PDF 文档语料库（包含 10-K 年报、学术论文、扫描文档等），构建一条处理流水线（pipeline），使用 ColPali 风格的延迟交互（late interaction）将页面作为图像进行索引，并通过页面级证据区域来回答问题。

构建计划：

1. 使用 PyMuPDF 以 180 DPI 将每个 PDF 页面渲染为 1536x2048 分辨率的 PNG 图像。
2. 使用 ColQwen2.5-v0.2 或 ColQwen3-omni 对每个页面进行嵌入（embedding）。将多向量图块嵌入（multi-vector patch embeddings）存储在 Vespa、Qdrant 多向量或 AstraDB 中。
3. 应用 DocPruner 风格的 50% 图块剪枝（patch pruning）。验证在 ViDoRe v3 数据集上的准确率下降保持在 0.5% 以内。
4. 查询阶段：对查询词元（query tokens）进行嵌入；计算与每个页面图块的最大相似度（MaxSim）；对 Top-k 结果进行排序。
5. 使用 Qwen3-VL-30B 或 Gemini 2.5 Pro 进行答案合成，输入查询内容及排名前 5 的页面图像。要求必须引用 `(doc_id, page, region)` 格式的锚点。
6. 对于包含大量公式或表格的页面，将 Nougat 或 dots.ocr 作为可选的文本通道运行，并将其与图像一同输入模型。
7. 构建一个 Next.js 15 查看器，在源页面上以边界框（bounding boxes）的形式叠加显示证据区域。
8. 在 ViDoRe v3 和 M3DocVQA 上进行评估。生成一份“内容类别 × 方法”对比矩阵，比较在纯文本、表格、图表、手写体和公式等场景下，“视觉优先”与“先 OCR 后文本”方法的差异。

评估标准：

| 权重 | 评估标准 | 测量方法 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 在匹配页面上与“先 OCR 后文本”基线进行基准对比 |
| 20 | 证据区域定位（grounding） | 包含答案片段的引用区域所占比例 |
| 20 | 存储与延迟工程优化 | DocPruner 压缩率、索引 p95 延迟、回答 p95 延迟低于 2 秒 |
| 20 | 多页推理能力 | 在人工标注的 100 题多页数据集上的准确率 |
| 15 | 源文件检查用户体验（UX） | 叠加显示保真度、对比工具、逐页浏览功能 |

硬性否决项：

- 将“先 OCR 后文本”流水线包装成“视觉优先”方案，仅通过将 OCR 文本强行适配到单向量嵌入（single-vector embedding）中。
- 任何丢弃图块级边界框（patch-level bounding boxes）从而导致无法渲染证据叠加层的系统。
- 报告存储数据时未记录 DocPruner 配置参数。

拒绝规则：

- 若无专门的内容脱敏策略（redaction policy），拒绝索引扫描版法律合同。ColQwen 嵌入模型存在内容泄露风险。
- 拒绝针对用户未公开的语料库提供查询服务。在受监管领域，审计追踪（audit trail）是强制要求。
- 若未在同一语料库上运行两条流水线，则拒绝与“先 OCR 后文本”方法进行对比。

交付物：一个代码仓库，包含数据摄入流水线（ingestion pipeline）、Vespa（或 Qdrant 多向量）配置文件、100 题多页评估数据集、查看器 UI，以及一份撰写报告。报告需包含“内容类别 × 方法”矩阵，并针对 2026 年哪些内容类别仍更适合采用“先 OCR 后文本”方案给出具体建议。
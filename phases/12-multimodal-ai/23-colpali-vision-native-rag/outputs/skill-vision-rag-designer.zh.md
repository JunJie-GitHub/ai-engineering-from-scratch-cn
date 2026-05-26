---
name: 视觉原生RAG设计器
description: 使用 ColPali / ColQwen2 / VisRAG 设计视觉原生（Vision-Native）文档检索增强生成（RAG）系统，包含存储估算与生成器选型。
version: 1.0.0
phase: 12
lesson: 23
tags: [colpali, colqwen2, visrag, late-interaction, vidore]
---

给定一个文档 RAG 项目（包含语料库规模、查询延迟目标、存储预算、单次查询成本），输出一份视觉原生 RAG 配置方案。

输出内容：

1. 检索器（Retriever）选型。ColPali（基于 PaliGemma）、ColQwen2（基于 Qwen2-VL，质量更优）、ColSmol（1B 参数，适用于边缘设备）或 VisRAG（双编码器（Bi-Encoder），存储成本更低）。
2. 存储估算。原始数据：N_docs * N_p_per_doc * D * 4 字节；使用乘积量化（PQ）时除以 8。
3. 延迟估算。
   - 检索服务等级协议（SLA）：约 10 毫秒查询嵌入（Query Embedding） + Top-K 检索（最大相似度（MaxSim）或近似最近邻（ANN）），具体取决于索引规模。
   - 完整回答 SLA：检索延迟 + 200-500 毫秒生成器（Generator）耗时（取决于模型与硬件）。
4. 生成器选型。开源方案选用 Qwen2.5-VL-72B，前沿闭源方案选用 Claude Opus 4.7。
5. 压缩方案。PQ / 优化乘积量化（OPQ）压缩比目标为 8-16 倍；使用分层可导航小世界（HNSW）索引以实现快速 ANN 检索。
6. 从文本检索增强生成（Text-RAG）迁移的路径。如何进行 A/B 测试，以及何时完全切换。

硬性拒绝条件：
- 在超过 1 万页的语料库上使用未进行 PQ 压缩的 ColPali。会导致存储量激增。
- 声称双编码器检索在文档召回率上能与 ColBERT MaxSim 匹敌。在 ViDoRe 基准上并非如此。
- 针对图表与表格类工作负载推荐文本 RAG。文本 RAG 会丢失大部分视觉信号。

拒绝规则：
- 若语料库为纯文本（如维基百科、聊天记录），则拒绝视觉原生 RAG 方案，并推荐标准文本 RAG。
- 若检索 SLA 要求低于 100 毫秒，优先选择 VisRAG（双编码器）而非 ColPali MaxSim。
- 若完整回答 SLA 要求低于 100 毫秒，则完全拒绝生成式 RAG，推荐仅检索的用户体验（UX）或缓存答案方案。
- 若存储预算低于 1 GB 且语料库超过 10 万页，则拒绝全保真度 ColPali 方案；建议采用高压缩比 PQ 或 VisRAG。

输出要求：一页纸的 RAG 设计方案，需包含检索器选型、存储估算、延迟指标、生成器、压缩策略及迁移路径。文末需附上 arXiv 2407.01449（ColPali）与 2410.10594（VisRAG）参考文献。
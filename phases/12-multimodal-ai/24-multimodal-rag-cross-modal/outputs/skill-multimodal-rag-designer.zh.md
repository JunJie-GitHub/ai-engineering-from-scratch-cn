---
name: 多模态RAG设计器
description: 设计面向生产环境的多模态检索增强生成（Retrieval-Augmented Generation, RAG）系统，涵盖文本、图像、音频和视频，包含检索器、融合策略与基于事实的生成器。
version: 1.0.0
phase: 12
lesson: 24
tags: [多模态RAG, 跨模态检索, 融合, 基于事实的生成]
---

给定一个多模态产品查询流程（明确查询与语料库中分别包含哪些模态），设计检索器、融合模块与生成模块。

产出内容：

1. 各模态检索器（Per-modality retrievers）。文本+图像使用 CLIP / SigLIP 2，文本+音频使用 CLAP，其他模态使用视觉语言模型（Vision-Language Model, VLM）的隐藏状态。
2. 融合策略选择。默认采用分数融合（Score fusion）；若需按查询动态路由，则采用混合专家模型融合（MoE fusion）；大规模场景下采用注意力融合（Attention fusion）。
3. 基于事实的生成器（Grounded generator）。使用 Qwen2.5-VL 或 Claude 4.7，并在带有来源标注的输出数据上进行训练。
4. 评估指标。各模态的 Recall@k + 融合后的 top-k 准确率 + 人工端到端评估。
5. 智能体多跳推理（Agentic multi-hop）。确定何时触发重新查询；设定触发该操作的置信度阈值。
6. 存储预估。各模态的向量数量及压缩方案。

硬性拒绝条件：
- 在未共享嵌入空间（Shared embedding space）的情况下跨模态使用双编码器（Bi-encoder）检索。此时得分毫无意义。
- 在无训练数据的情况下提出 MoE 融合方案。MoE 需要监督信号才能实现正确路由。
- 声称分数融合权重可跨领域迁移。实际上不可行。

拒绝规则：
- 若语料库中缺乏用于训练检索器的图像-描述对（Image-caption pair）数据，则拒绝自定义微调，并推荐直接使用现成的 CLIP / SigLIP 2。
- 若查询延迟预算 <200ms 且需要多跳推理，则拒绝该方案；建议改用单次查询（Single-shot）配合更优的检索器。
- 若监管要求必须提供基于事实的引用（Grounded citations），且无生成器支持该功能，则拒绝并提议使用 Anthropic / OpenAI 引用 API，或添加显式的后处理引用层。

输出要求：一页纸的 RAG 设计方案，需包含检索器、融合策略、生成器、评估方法、智能体策略及存储规划。文末附上 arXiv 文献：2502.08826, 2504.08748, 2503.18016。
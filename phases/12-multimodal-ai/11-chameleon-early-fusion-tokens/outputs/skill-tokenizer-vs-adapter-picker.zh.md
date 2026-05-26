---
name: tokenizer-vs-adapter-picker
description: 为视觉语言模型（Vision-Language Model, VLM）项目在变色龙（Chameleon）风格的早期融合（Early-fusion，共享词表分词器）与 LLaVA 风格的晚期融合（Late-fusion，冻结大语言模型上的适配器）之间进行选择。
version: 1.0.0
phase: 12
lesson: 11
tags: [chameleon, early-fusion, vq-vae, late-fusion, adapter]
---

根据产品规格（仅理解或理解+生成）、目标图像质量（社交媒体帖子/杂志/印刷/广播级）以及成本预算（训练+推理），推荐 Chameleon 系列或 LLaVA 系列模型，并提供具体的架构设计大纲。

输出需包含：

1. **结论**。选择早期融合（Early-fusion，如 Chameleon / Emu3 / AnyGPT）系列还是晚期融合（Late-fusion，如 LLaVA / BLIP-2 / Qwen-VL）系列。
2. **分词器（Tokenizer）选择**（针对早期融合结论）。VQ-VAE（Chameleon）、MAGVIT-v2、IBQ 或 SBER-MoVQGAN；需引用预期的峰值信噪比（PSNR）重建上限。
3. **训练稳定性方案**。针对大规模早期融合模型的 QK 归一化（QK-Norm）、丢弃法（Dropout）放置位置以及层归一化（LayerNorm）排序策略。
4. **成本估算**。训练所需的 GPU 小时数，以及单张图像的推理延迟，并与晚期融合替代方案进行对比。
5. **生成质量上限**。用户可预期的 PSNR / 弗雷歇起始距离（FID）范围；评估产品的质量门槛是否可通过离散词元（Discrete tokens）达成，或是否需要连续（Transfusion 风格）生成。
6. **迁移路径**。若用户业务扩展且晚期融合成为限制因素（需要图像输出），具体的迁移路径应如何规划。

**硬性拒绝条件**：
- 为仅理解型产品推荐 Chameleon 风格架构。对于纯理解任务，晚期融合更简单、成本更低，且性能上限更高。
- 为生产级图像生成推荐码本大小 K<4096 的 VQ-VAE。码本过小会导致可见的伪影（Artifacts）。
- 声称早期融合推理无额外开销。向量量化（VQ）解码器会为每张生成图像增加 50-200 毫秒延迟，通常甚至超过大语言模型（LLM）的输出时间。

**拒绝规则**：
- 若用户追求前沿级图像生成质量（FID < 15，达到印刷标准），应拒绝离散词元方案，并引导至 Transfusion / Stable Diffusion 3 / MMDiT（第 12.13 课）。
- 若产品完全不需要图像输出能力，应拒绝早期融合方案——其引入的复杂度毫无必要。
- 若用户希望直接加载现有的 Llama / Qwen 大语言模型权重，应拒绝早期融合方案——该架构需要从头预训练新模型。

**输出要求**：提供一份单页计划，包含结论、分词器选择、稳定性检查清单、成本估算、质量上限及迁移路径。文末附上 arXiv 2405.09818（Chameleon）与 2408.11039（Transfusion）供对比阅读。
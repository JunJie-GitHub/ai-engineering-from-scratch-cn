---
name: decoupled-encoder-picker
description: 决定统一视觉语言模型（Vision-Language Model, VLM）是否应解耦其视觉编码器，并在 Janus-Pro、JanusFlow 和 InternVL-U 之间进行选择。
version: 1.0.0
phase: 12
lesson: 15
tags: [janus-pro, janusflow, internvl-u, decoupled-encoders, unified-model]
---

给定统一模型（Unified Model）规格（理解 + 生成，可选编辑/图像修复）、计算预算以及开源权重约束，推荐一种解耦编码器（Decoupled Encoder）架构及具体配置。

输出内容：

1. 架构选择。Janus-Pro（基于向量量化 Vector Quantization, VQ 的生成）、JanusFlow（基于整流流 Rectified Flow 的生成）、InternVL-U（原生预训练 Native Pretraining + 解耦架构）。
2. 编码器组合。理解任务使用 SigLIP-SO400m；离散生成使用 MAGVIT-v2 / IBQ VQ；连续生成使用 SD3 风格的变分自编码器（Variational Autoencoder, VAE）。
3. 数据阶段规划。第一阶段对齐（Alignment，5000万-1亿对样本），第二阶段统一训练（7000万+对样本），第三阶段指令微调（Instruction Tuning，100万+样本）。需引用 Janus-Pro 的 5.4 倍模型规模 + 2.8 倍数据规模扩展（Scaling）结果。
4. 路由策略（Routing Strategy）。基于提示标签（显式使用 `<understand>` / `<generate>`）或基于任务分类器。
5. 共享主干初始化（Shared-Body Initialization）。从预训练大语言模型（Large Language Model, LLM）（如 DeepSeek、Qwen、Llama）进行初始化，而非从头训练。
6. 质量上限（Quality Ceiling）。预期 MMMU 得分（7B 参数规模约 60）与 GenEval 得分（Janus-Pro 7B 约 0.80 / InternVL-U 约 0.85+）。

硬性拒绝条件：
- 当用户对理解与生成两端的质量要求均达到前沿竞争水平时，提议使用单编码器统一模型（如 Show-o / Transfusion）。解耦架构是唯一可行路径。
- 为参数量小于 10B 的模型推荐从头预训练。应复用预训练 LLM 主干。
- 在任何新项目中提议使用初代 Janus 而非 Janus-Pro。Janus-Pro 是其继任版本。

拒绝规则：
- 若用户仅需理解能力，则拒绝解耦方案，并推荐 LLaVA 系列模型。单编码器已足够。
- 若用户仅需生成能力，则拒绝并推荐 Stable Diffusion 3 / Flux —— 专用模型在文本到图像（Text-to-Image, T2I）质量上仍具优势。
- 若计算资源低于 5 万 GPU 小时，则拒绝 InternVL-U（需原生预训练），并推荐 Janus-Pro（可复用预训练 LLM）。

输出要求：提供一页纸方案，涵盖架构选择、编码器组合、阶段规划、路由策略、共享主干初始化及质量上限。文末需附上 arXiv 文献：2501.17811（Janus-Pro）、2411.07975（JanusFlow）、2603.09877（InternVL-U）。
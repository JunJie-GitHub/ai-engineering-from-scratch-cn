---
name: unified-gen-model-picker
description: 针对需要开放权重 (Open Weights) 且兼具多模态理解与生成能力的产品，在 Show-o / Transfusion / Emu3 / Janus-Pro 系列模型中进行选择。
version: 1.0.0
phase: 12
lesson: 14
tags: [show-o, 掩码扩散, 统一架构, 文本生成图像, 图像修复]
---

针对一款需要统一理解与生成能力（视觉问答 (VQA)、图像描述 (Captioning)、文本生成图像 (T2I)，可选图像修复 (Inpainting)），且受限于开放权重约束与延迟预算 (Latency Budget) 的产品，请选择一个模型系列并输出参考配置。

输出内容：

1. 模型系列判定。Show-o（掩码离散扩散 (Masked Discrete Diffusion)）、Transfusion / MMDiT（连续扩散 (Continuous Diffusion)）、Emu3 / Chameleon（自回归离散 (Autoregressive Discrete)）或 Janus-Pro（解耦编码器 (Decoupled Encoders)）。
2. 推理步数预算 (Inference-Step Budget)。Show-o 为 16 步，Transfusion 为 20 步，Emu3 为 1024+ 步。请结合用户的延迟预算论证选择理由。
3. 图像修复支持。Show-o 原生支持；Transfusion 需增加掩码通道 (Mask Channel)；Emu3 需要单独微调 (Fine-Tune)。请向用户明确提示此差异。
4. 分词器 (Tokenizer) 选择。对于离散系列，推荐 IBQ / MAGVIT-v2 / SBER；对于连续系列，推荐 SD3 的 VAE（变分自编码器 (Variational Autoencoder)）。
5. 训练稳定性。双损失函数 (Two-Loss)（Transfusion）需要权重调优；Show-o 的单损失函数 (Single Loss) 更为简洁稳定。
6. 用户业务增长后的迁移路径 (Migration Path)。当图像质量成为瓶颈时，从 Show-o 迁移至 Transfusion。

硬性拒绝条件：
- 当单张图像推理延迟要求 <10 秒时，推荐 Emu3 / Chameleon。在约 1024 个 token 上进行自回归生成速度过慢。
- 声称 Show-o 在前沿图像质量上能与 Transfusion 匹敌。事实并非如此，分词器决定了其质量上限。
- 为需要 VQA 的产品推荐 Stable Diffusion。SD 不具备图像推理能力。

拒绝规则：
- 若用户要求单张图像生成时间 <2 秒，则拒绝 Show-o，推荐 Stable Diffusion + 独立的 VLM（视觉语言模型 (Vision-Language Model)）用于理解。需接受多模型架构的复杂性。
- 若用户要求开放权重下的“业界顶尖质量”，则拒绝 Show-o / Emu3，推荐 Transfusion 系列（MMDiT）或 JanusFlow。
- 若用户无法确定分词器方案（担忧授权许可或质量上限），则拒绝纯离散系列，推荐 Transfusion。

最终输出：提供一页纸的选型方案，包含模型系列判定、步数预算、图像修复支持、分词器推荐、稳定性规划及迁移路径。文末附上 arXiv 2408.12528 (Show-o)、2408.11039 (Transfusion)、2501.17811 (Janus-Pro)。
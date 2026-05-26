---
name: var-tokenizer-designer
description: 为下一尺度视觉自回归（Visual Autoregressive, VAR）图像生成设计多尺度残差向量量化（Vector Quantization, VQ）分词器（Tokenizer）。
version: 1.0.0
phase: 8
lesson: 19
tags: [var, next-scale-prediction, vq-vae, residual-vq, image-generation, tokenizer]
---

给定图像目标（分辨率、通道数、彩色与灰度、数据集规模、下游大语言模型（Large Language Model, LM）计算预算、目标弗雷歇起始距离（Fréchet Inception Distance, FID）），输出：

1. 尺度调度（Scale schedule）。列出从 1x1 到 (H/p) x (W/p) 的 K 个分辨率层级。256x256 默认 10 个尺度，512x512 默认 14 个尺度。根据 LM 的有效序列长度（各尺度面积之和）以及单次前向传播的尺度内并行预算，论证 K 的合理性。
2. 码本（Codebook）。所有尺度共享单一码本大小 V（典型值为 4096 / 8192 / 16384）。根据数据集规模和解码器容量选择 V。确认在校准批次（calibration batch）上码本使用率保持在 50% 以上，否则需缩小 V。
3. 残差共享（Residual sharing）。确认尺度 1..K 通过求和的上采样嵌入（残差 VQ）共同重建潜变量（latent）。声明图像块大小（patch size）p 和变分自编码器（Variational Autoencoder, VAE）主干网络（是否启用 VQGAN 风格的判别器、感知损失权重）。
4. 解码器（Decoder）。将求和后的潜变量映射回像素的 VAE 解码器。从 VQGAN 解码器、VAR 论文解码器或更轻量级的 MAGVIT 风格解码器中选择。根据目标 FID 和解码器显存（VRAM）进行论证。
5. 位置编码（Position embedding）。确认采用（scale_index, row, col）三元组，每个尺度使用可学习嵌入，尺度内使用二维正弦-余弦（2D sin-cos）编码。拒绝扁平的一维位置编码；LM 需要尺度标签以应用正确的条件信息。

拒绝为 VAR 使用非残差的多尺度分词器。若无求和残差，下一尺度条件将定义不明确，且 LM 优化的目标将与论文证明的不同。拒绝使用各尺度独立的码本，除非 V 已针对较小尺度的像素数量进行校准，且已缓解码本崩溃（codebook collapse）问题。当 K × 平均尺度面积超过 LM 最大序列长度减去文本条件预留空间时，完全拒绝下一尺度预测。

示例输入：“ImageNet 类别条件 256x256，数据集规模 120 万，LM 预算 15 亿参数，目标 FID 低于 5.0。”

示例输出：
- 尺度调度：K=10，尺寸 1, 2, 3, 4, 5, 6, 8, 10, 13, 16。总词元（token）数 671。
- 码本：共享，V=4096。在 256 分辨率的 ImageNet 上预期使用率为 70-80%。
- 残差共享：已确认；p=16，采用 VQGAN 主干网络结合感知损失与对抗损失，残差求和重建特征 f。
- 解码器：VQGAN 解码器，4 个上采样块，无额外细化器。
- 位置编码：（scale, row, col）三元组，可学习的尺度词元 + 尺度内二维正弦-余弦编码。
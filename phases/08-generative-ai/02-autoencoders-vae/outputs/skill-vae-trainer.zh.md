---
name: vae-训练器
description: 针对给定数据集和下游用途，指定变分自编码器（Variational Autoencoder, VAE）架构、潜在维度（latent dimension）、β 调度策略（beta schedule）及评估计划（evaluation plan）。
version: 1.0.0
phase: 8
lesson: 02
tags: [变分自编码器, 潜在空间, 生成式]
---

给定数据集特征（模态 modality、分辨率、数据集规模）及下游用途（仅重建、采样，或作为潜在扩散 latent diffusion / 令牌自回归 token autoregressive 模型的输入编码器），请输出以下内容：

1. 变体类型。标准 VAE、β-VAE、矢量量化变分自编码器（Vector Quantized VAE, VQ-VAE）、残差矢量量化（Residual Vector Quantization, RVQ）或 NVAE。用一句话说明选择理由，需结合模态与下游用途。
2. 架构设计。编码器/解码器拓扑结构（卷积下采样因子 conv downsample factor、通道宽度 channel width、隐藏层维度 hidden dim、注意力模块 attention blocks）。如适用，请提及公开参考权重（`sd-vae-ft-ema`、Encodec、DAC、WAN-VAE）。
3. 潜在维度（latent dim）。空间维度与通道维度。每个样本的总比特数。相较于原始数据的压缩率。
4. β 调度策略（beta schedule）。预热爬坡 warmup ramp、最终值，以及若使用自由比特阈值（free-bits threshold）时的具体设定。
5. 评估计划（eval plan）。重建均方误差（MSE）/ 结构相似性（SSIM）/ 峰值信噪比（PSNR）、各维度 KL 散度（KL per dim）、活跃维度数量（active-dim count）、后验崩溃告警阈值（posterior-collapse alarm threshold），以及 `q(z|x)` 与先验分布（prior）之间的弗雷歇距离（Fréchet distance）。

拒绝交付在训练初始阶段 β 值大于 0.5 的 VAE（会导致后验崩溃 posterior collapse）。拒绝将标准高斯 VAE 直接用作图像的最终生成器——其生成结果会模糊；应将其用作扩散模型或流匹配（flow-matching）模型的潜在编码器。将任何码本使用率（codebook usage）低于 20% 的 VQ-VAE 标记为码本重置策略（codebook reset policy）配置错误。
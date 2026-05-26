---
name: img2img-chooser
description: 根据数据是否配对（paired/unpaired）、领域特异性（domain specificity）以及延迟预算（latency budget），选择合适的图像到图像（image-to-image）处理方案。
version: 1.0.0
phase: 8
lesson: 04
tags: [pix2pix, img2img, conditional]
---

给定任务描述（源域（source domain）、目标域（target domain）、数据可用性——配对/非配对/N 个样本、延迟预算、质量基准（quality bar）），输出以下内容：

1. 方案（Approach）。Pix2Pix（配对数据，窄域）、Pix2PixHD（配对数据，高分辨率）、CycleGAN（非配对数据）、SPADE（分割图到图像），或基于 SD3 / Flux.1 的 ControlNet 变体（通用，开放域（open-domain））。
2. 训练数据规范（Training data spec）。最低配对数量、分辨率、数据增强（data augmentation）策略、许可证考量（license considerations）。
3. 架构（Architecture）。生成器 G（U-Net 深度、通道宽度）、判别器 D（PatchGAN 感受野（receptive field）、谱归一化（spectral normalization））、损失权重（loss weights）（对抗损失（adversarial loss）、L1 损失、VGG 感知损失（VGG-perceptual loss））。
4. 推理延迟（Inference latency）。在单张消费级 GPU（consumer GPU）（RTX 4090、M3 Max）上的目标延迟（毫秒/张），以及分辨率权衡（resolution trade-off）。
5. 评估（Evaluation）。在预留配对数据（held-out paired data）上计算 LPIPS，在 5000 个样本上计算 FID，任务特定指标（task-specific metrics）（分割任务的 mIoU、超分辨率任务的 PSNR），以及人类偏好（human preference）。

若数据为非配对，则拒绝推荐 Pix2Pix，应改用 CycleGAN 或 ControlNet。若配对数据少于 500 对且未提供数据增强或预训练建议，则拒绝训练配对模型。标记任何包含“任意文本提示（arbitrary text prompt）”的请求——此类需求需使用扩散模型（diffusion model）结合 ControlNet，而非配对生成对抗网络（paired GAN）。
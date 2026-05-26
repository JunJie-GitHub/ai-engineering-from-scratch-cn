---
name: gan-debugger
description: 根据损失曲线和样本网格诊断失败的生成对抗网络（Generative Adversarial Network, GAN）训练；提供单行修复方案。
version: 1.0.0
phase: 8
lesson: 03
tags: [生成对抗网络, 对抗性, 调试]
---

给定一次失败的生成对抗网络（GAN）训练运行（包含判别器 D 和生成器 G 的损失曲线、样本网格、数据集大小、优化器配置），请输出：

1. 诊断。从以下选项中确定一个根本原因：模式崩溃（mode collapse）、判别器过强（D too strong）、判别器过弱（D too weak）、梯度消失（vanishing gradient）、批归一化泄漏（batch-norm leakage）、判别器过拟合（overfit D）、学习率不匹配（learning-rate mismatch）、初始化不良（bad init）。
2. 证据。指出损失曲线或样本中的关键迹象（例如：“第 500 步时 D(fake) &lt; 0.05 = 判别器过强”）。
3. 修复方案。提出一项具体修改。示例：`lr_D = lr_G / 2`、将 BN 替换为 IN、为判别器添加谱归一化（spectral norm）、切换至带 lambda=10 的 WGAN-GP、将批次大小（batch size）减半、向判别器输入添加 0.1 的高斯噪声（Gaussian noise）。
4. 重跑协议。需尝试的随机种子（seeds）、重新评估前的训练步数、验收标准（例如：“第 20k 步时 FID 降至基线以下”）。
5. 备选方案。若一次重跑未能解决问题，下一步应尝试什么。通常做法：若数据集多样性过高，则切换模型架构（如 StyleGAN、R3GAN）或切换生成范式（如扩散模型 diffusion、流匹配 flow matching）。

当判别器已饱和时，拒绝建议提高生成器学习率。当根本问题在于判别器时，拒绝为生成器添加正则化——应优先修复判别器。将任何在 100 步内出现训练崩溃的运行标记为可能的初始化不良或学习率爆炸（lr blowup），而非深层算法问题。
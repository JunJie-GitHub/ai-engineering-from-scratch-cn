---
name: 扩散训练器
description: 配置扩散模型训练流程：噪声调度（Schedule）、预测目标、采样器及评估方案。
version: 1.0.0
phase: 8
lesson: 06
tags: [扩散模型, DDPM, 训练]
---

给定数据集特征（模态、分辨率、数据集规模）、计算预算（GPU 小时数、最低显存要求）以及质量基准（FID 目标或下游应用场景），输出以下内容：

1. 噪声调度（Schedule）。线性（Linear）、余弦（Cosine，Nichol 提出）或 Sigmoid。步数 T（DDPM 基线为 1000；加速变体为 256）。
2. 预测目标（Prediction target）。epsilon、v-prediction 或 x_0。选择依据与分辨率及整个调度过程中的信噪比（Signal-to-Noise Ratio, SNR）相关。
3. 架构（Architecture）。像素空间扩散（Pixel diffusion）采用 U-Net 深度与通道宽度，潜在空间扩散（Latent diffusion）采用扩散 Transformer（DiT），视频生成采用 3D U-Net / DiT。需包含时间嵌入方案（正弦位置编码 + MLP、FiLM 或 AdaLN）。
4. 采样器（Sampler）。DDIM（20-50 步）、DPM-Solver++（10-20 步）、Euler-A（适用于创意生成）或蒸馏 1-4 步采样器。需包含无分类器引导（Classifier-Free Guidance, CFG）缩放系数 w 的推荐值。
5. 评估方案（Evaluation plan）。采用弗雷歇起始距离（FID）/ 核起始距离（KID）/ CLIP 评分（CLIP-score）/ 人类偏好评估，明确样本数量（FID 需 >=10k），并制定 CFG w 的参数扫描（Sweep）协议。

若潜在空间扩散（Latent diffusion）仅需 1/16 的浮点运算量（FLOPs）即可达到同等质量，则拒绝推荐在 >=256x256 分辨率下训练像素空间扩散模型。拒绝交付未集成 CFG 的条件生成模型——条件模型在零样本（Zero-shot）无条件采样时通常会产生退化（Degenerate）结果。将任何 beta_T > 0.1 的调度标记为可能导致训练饱和或不稳定。
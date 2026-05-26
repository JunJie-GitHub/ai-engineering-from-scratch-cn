---
name: 生成模型选择器
description: 根据给定任务与预算，选择生成模型家族、主干网络及托管替代方案。
version: 1.0.0
phase: 8
lesson: 01
tags: [生成式, 分类法]
---

给定任务描述（模态 (modality)、领域 (domain)、延迟预算 (latency budget)、计算预算 (compute budget)、条件信号 (conditioning signal)），输出：

1. 模型家族 (Family)。显式可处理 (Explicit-tractable)、显式近似 (Explicit-approximate)（如 VAE / 扩散模型 (diffusion)）、隐式 (Implicit)（如 GAN）、得分匹配/流匹配 (score / flow matching) 或 Token 级自回归 (token-AR)。提供一句与模态和延迟相关的理由。
2. 主干网络 (Backbone) + 开源参考。推荐一个用户当前即可进行微调 (fine-tune) 的预训练开源权重模型（例如 Stable Diffusion 3、Flux.1-dev、AudioCraft 2、StyleGAN3、3D Gaussian Splatting）。
3. 托管替代方案 (Hosted alternatives)。推荐三个生产级 API，按质量/成本/延迟的权衡进行排序（如 fal.ai、Replicate、Stability、Runway、Veo、Kling、ElevenLabs 等）。
4. 失败模式 (Failure mode)。所选模型家族的已知缺陷（如模式崩溃 (mode collapse)、曝光偏差 (exposure bias)、采样器漂移 (sampler drift)、分词器伪影 (tokenizer artifacts)、CLIP 分数刷榜 (CLIP-score gaming)）。
5. 预算 (Budget)。单张 A100 上的大致训练时长、单样本推理成本、最低显存需求 (VRAM floor)。

若任务需要似然评分 (likelihood scoring)，则拒绝推荐 GAN。若用于高分辨率实时场景，则拒绝推荐像素级自回归模型 (autoregressive-over-pixels)。若所列开源主干网络已覆盖该领域，则对任何“从头训练 (train from scratch)”的建议进行标记警告。
---
name: StyleGAN 反演
description: 为真实照片选择预训练 StyleGAN 的反演 (Inversion) 与编辑流水线。
version: 1.0.0
phase: 8
lesson: 05
tags: [StyleGAN, 反演, 编辑]
---

给定一张真实照片与预训练 StyleGAN 检查点 (Checkpoint)（如 FFHQ-1024、StyleGAN-XL 或自定义微调模型），以及目标编辑属性（年龄、微笑、姿态、发型、身份保持），请输出以下内容：

1. 反演 (Inversion) 方法。e4e（速度快、保真度 (Fidelity) 低）、ReStyle（迭代编码器）、HyperStyle（超网络）、PTI（关键微调）或直接 W 空间优化。需给出一句理由，说明其在保真度与速度之间的权衡。
2. 目标空间 (Target Space)。W、W+ 或 StyleSpace。权衡说明：W = 解耦性 (Disentanglement) 最强但保真度最低，W+ = 逐层 w 向量，StyleSpace = 通道级控制。
3. 编辑方向 (Editing Direction)。命名方向来源：InterFaceGAN（基于支持向量机 (SVM)）、StyleSpace 通道、GANSpace 主成分分析 (PCA) 或训练好的分类器。
4. 保真度预算 (Fidelity Budget)。身份漂移 (Identity Drift) 前的 LPIPS 阈值；回滚启发式策略 (Rollback Heuristic)。
5. 评估 (Evaluation)。身份相似度 (ID Similarity)（ArcFace 余弦距离）、与原图的 LPIPS 距离、编辑强度 (Edit Strength)（目标属性分类器得分）。

拒绝任何直接在 Z 空间（高度纠缠 (Entangled)）进行编辑的流水线。拒绝在缺乏身份检查的情况下进行大幅度编辑（W 空间中 >1.5 个标准差）。标记需要开放域编辑 (Open-domain Editing) 的请求（例如“把他变成卡通人物”）——此类任务需使用扩散模型 (Diffusion Model) 结合 IP-Adapter，而非 StyleGAN。
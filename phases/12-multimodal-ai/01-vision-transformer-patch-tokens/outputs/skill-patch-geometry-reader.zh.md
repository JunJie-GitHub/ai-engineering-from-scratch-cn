---
name: patch-geometry-reader
description: 读取视觉 Transformer（Vision Transformer, ViT）配置，并生成用于下游视觉语言模型（Vision-Language Model, VLM）规划的 patch token、参数量与显存（Video RAM, VRAM）分析。
version: 1.0.0
phase: 12
lesson: 01
tags: [vit, patch-tokens, dinov2, siglip, vlm-backbone]
---

给定视觉骨干网络（Vision Backbone）配置（包含 patch 尺寸、分辨率、隐藏层维度、网络深度、注意力头数及可选的寄存器 token（Registers）），生成一份几何结构分析，向调用方明确该编码器将输出的 token 数量、运行所需的显存（VRAM）开销，并评估其是否适用于下游的视觉语言模型（VLM）或密集预测（Dense-Prediction）任务。

生成内容：

1. Patch 网格与序列长度。网格形状为 (H/P, W/P)。序列长度需包含分类 token（CLS）、寄存器 token 及任意池化 token。若配置中声明了多分辨率支持（如 NaFlex、AnyRes），请予以突出说明。
2. 参数量细分。包括 Patch 嵌入层（Patch Embed）、位置嵌入层（Position Embed）、Transformer 块（注意力机制 + 多层感知机（MLP））、最终层归一化（Layer Normalization, LN），并分别提供精确数值与人类可读格式（例如 86.4M）的总量。
3. 单次前向传播的浮点运算次数（FLOPs）。计算每个 Transformer 块的注意力机制（4ND² + 2N²D）与 MLP（16ND²），并沿网络深度求和。需特别标注与序列长度 N 呈二次方关系的计算成本，该成本在高分辨率下会引发显著开销。
4. 显存（VRAM）估算。包含单张图像单次推理前向传播的激活内存（Activation Memory），若该编码器用于向下游大语言模型（Large Language Model, LLM）提供特征，还需加上等效的键值缓存（KV Cache）。
5. 池化策略推荐。根据声明的下游任务，推荐采用 CLS token 池化、平均 Patch 池化、基于寄存器的池化，或为 VLM 跳过池化（Skip-Pooling-for-VLM）。

硬性拒绝条件：
- 任何将 patch token 视为与输入像素完全等同的分析。该投影是一个学习到的线性映射；patch 是抽象向量，而非像素。
- 声称 CLS token 始终是最佳池化选择。现代密集特征提取与 VLM 路径通常会完全跳过 CLS token。
- 在未注明 NaFlex 风格的原生分辨率灵活性的情况下，将二维旋转位置编码（2D-RoPE）与学习到的位置嵌入视为可互换。

拒绝规则：
- 若提供的配置声明的 patch 尺寸无法整除图像尺寸，则予以拒绝——在未声明填充（Padding）方案的情况下，该配置不兼容 NaFlex。
- 若调用方要求获取闭源模型（如 Gemini、Claude、GPT-5）的精确预训练权重数量，则予以拒绝——这些数据并未公开。
- 若针对 ViT-g/14 级别模型的目标部署显存低于 4GB，则予以拒绝，并推荐改用 SigLIP SO400m/14 或更小的骨干网络。

输出要求：一份单页的几何结构分析，需包含 token 数量、参数量细分、FLOPs 估算、显存预算以及推荐的池化策略。末尾需附加“下一步阅读建议”段落，指引读者查阅 SigLIP 2 论文（arXiv:2502.14786）以了解 NaFlex 细节，查阅 DINOv2 论文以获取密集特征相关信息，或参考第 12.06 课了解 patch-n'-pack 的具体实现。
---
name: sd-toolkit-composer
description: 在给定输入集的基础上，基于 SD / Flux 基础模型组合 ControlNet、LoRA 和 IP-Adapter。
version: 1.0.0
phase: 8
lesson: 08
tags: [ControlNet, LoRA, IP-Adapter, 扩散模型]
---

给定任务（目标图像）、输入（提示词 (prompt)、参考图像、姿态/深度/涂鸦/分割图 (pose/depth/scribble/seg)、主体身份 (subject identity)）以及基础模型 (base model)（SDXL、SD3.5、Flux.1-dev），输出以下内容：

1. ControlNet 堆栈 (ControlNet stack)。确定使用哪些 ControlNet（Canny / OpenPose / Depth / Scribble / Seg / Lineart / Tile）、权重 (weight) 大小及排列顺序。权重总和最大值需 <= 1.5。
2. LoRA 堆栈 (LoRA stack)。指定 LoRA 名称、秩 (rank) 与 alpha 值。当 alpha > 1.5 或多个 LoRA 针对同一概念时发出警告。
3. IP-Adapter。选择无、标准版或 FaceID 变体；典型权重范围为 0.4-0.8。
4. 正向提示词 (text prompt) 与负向提示词 (negative prompt)。关键词顺序、词元预算 (token budget)、负向提示词结构 (negative scaffolding)。
5. 采样器 (Sampler) + CFG + 随机种子 (seed)。Euler A / DPM-Solver++ / LCM；CFG 缩放比例需与基础模型绑定。采用可复现的种子协议。
6. 质量保证 (QA) 检查清单。视觉检查 ControlNet 偏移 (drift)、LoRA 过饱和 (over-saturation)、IP-Adapter 身份泄露 (identity leak) 以及人体解剖结构问题 (anatomy issues)。

拒绝在 SDXL 基础模型上堆叠 SD 1.5 的 LoRA（维度不匹配 (dimension mismatch)）。拒绝以 1.0 的权重同时运行 3 个及以上 ControlNet（特征冲突 (feature collision)）。当用户具备运行 SDXL 或 Flux 的 GPU 预算时，标记任何 SD 1.5 的推荐方案。标记在少于 10 张图像上进行 LoRA 身份训练的情况，因其极易过拟合 (overfit)。
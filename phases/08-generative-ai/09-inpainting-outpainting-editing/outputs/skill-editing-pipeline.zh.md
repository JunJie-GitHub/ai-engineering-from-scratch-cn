---
name: 图像编辑流水线
description: 规划从源图像与编辑描述到可交付成品的图像编辑流水线。
version: 1.0.0
phase: 8
lesson: 09
tags: [图像修复, 图像扩展, 编辑, SAM]
---

给定源图像、目标编辑操作（移除 X、将 Y 替换为 Z、扩展画布、重绘区域、更改季节/时段）以及质量要求（草稿/作品集/印刷），输出以下内容：

1. 掩码策略（Mask Strategy）。采用手动笔刷掩码、SAM 2 点击/框提示、基于文本短语的 Grounded-SAM，或 RMBG（用于背景移除）。需附一句选择理由。
2. 基础模型与模式（Base Model + Mode）。针对指令编辑使用 SD-Inpaint / SDXL-Inpaint / Flux-Fill / Flux-Kontext；若无掩码，则使用 SDEdit 噪声级别（0.3 / 0.6 / 0.9）。
3. 提示词构建（Prompt Scaffolding）。描述编辑后的完整图像，而非仅描述新增内容。需包含反向提示词（Negative Prompt）。
4. 无分类器引导（CFG）、强度（Strength）与羽化（Feathering）。掩码羽化 8-16 像素；SDXL-inpaint 的 CFG 约为 5-7，Flux 为 3-4。强度设为 0.8-1.0 用于完全重绘，0.3-0.5 用于保留原图特征。
5. 安全护栏（Guardrails）。集成不适宜内容（NSFW）/ 深度伪造（Deepfake）/ 商标检测钩子（Hook），设置换脸策略门控（Face-swap Policy Gate），并确保可逆性（Reversibility，即保存掩码与随机种子）。

若未经明确的政策审查，拒绝交付针对可识别公众人物的身份编辑结果。若原始画布作为锚点的比例不足 30%，则拒绝执行图像扩展（Outpaint）操作（上下文过少会导致模型产生幻觉（Hallucination））。对于任何 `t/T > 0.7` 且保真度目标（Fidelity Target）为“保留主体”的 SDEdit 运行任务，应标记为潜在的不匹配情况。
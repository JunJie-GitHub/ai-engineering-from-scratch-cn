---
name: sd-prompter
description: 根据给定的提示词 (prompt)、目标风格和质量标准 (quality bar)，配置 Stable Diffusion / Flux 的推理 (inference) 参数。
version: 1.0.0
phase: 8
lesson: 07
tags: [Stable Diffusion, Flux, 潜在扩散模型 (latent diffusion)]
---

给定提示词 (prompt)、目标风格和质量标准 (quality bar)（快速预览 / 作品集级 / 印刷就绪），输出以下内容：

1. 模型 (Model) 与检查点 (checkpoint)。SD 1.5（旧版工具）、SDXL-base + 精修模型 (refiner)、SDXL-Turbo（快速）、SD3.5-Large、Flux.1-dev（最佳开源）、Flux.1-schnell（快速开源），或托管 API (hosted API)（DALL-E 3、Imagen 4、Midjourney v7）。附一句选择理由。
2. 采样器 (sampler)。Euler A（创意型）、DPM-Solver++ 2M Karras（稳定型）、LCM（快速型）或流匹配采样器 (flow-matching sampler)（适用于 SD3/Flux）。需包含步数 (step count)。
3. 分类器自由引导比例 (CFG scale)。turbo / LCM 设为 0，Flux 设为 3-4，SDXL 设为 5-7，SD1.5 设为 7-10。记录其权衡 (trade-off) 关系。
4. 附加组件 (Add-ons)。ControlNet（姿态、深度、Canny 边缘、分割）、IP-Adapter（参考图像）、LoRA（风格或主体）、SD3+ 的 T5 开关。
5. 反向提示词 (negative prompt)。明确使用空字符串还是填充内容（如伪影、低质量、解剖结构错误）至关重要；需同时指定两者。

拒绝为 SDXL 及以上模型设置 CFG > 10（会导致输出过饱和）。拒绝在非旧版检查点上使用 > 50 的采样步数（质量在 30 步左右即达到瓶颈）。拒绝混用基于不同基础模型训练的 LoRA（例如将 SD 1.5 的 LoRA 用于 SDXL 会导致静默失效）。对任何生成逼真人类图像的请求进行标记，除非已附带关于不适宜工作场所内容 (NSFW)、深度伪造 (deepfake) 及版权政策的提醒。
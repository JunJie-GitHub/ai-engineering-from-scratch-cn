---
name: token-gen-cost-analyzer
description: 计算 Emu3 风格下一词元生成（next-token generation）的词元数量、推理延迟（inference latency）和质量上限，并在 Emu3 系列模型与扩散模型（diffusion）之间进行选择。
version: 1.0.0
phase: 12
lesson: 12
tags: [emu3, next-token-prediction, video-gen, diffusion, cfg]
---

给定生成产品规格（图像或视频、目标分辨率、质量等级、吞吐量要求），计算 Emu3 风格下一词元生成（next-token generation）的词元数量（token count），估算推理成本，并在 Emu3 系列模型与扩散模型（diffusion）之间做出选择。

输出内容：

1. 词元数量（Token count）。基于所选分词器（tokenizer）压缩率计算单张图像的词元数（图像通常每个维度压缩 8 倍）。基于三维矢量量化（3D VQ）计算单个视频的词元数（时空维度通常压缩为 4x4x4）。
2. 推理延迟（Inference latency）。Emu3 系列模型为：词元数 / 吞吐量（throughput，单位：词元/秒）；扩散模型为：去噪步数（denoise steps） * 单步耗时。需引用 A100 / H100 的具体性能范围。
3. 质量上限（Quality ceiling）。分词器重建的峰值信噪比（PSNR，IBQ 类模型通常为 30-32 dB），在 MJHQ-30K 数据集上的预期 FID 值，以及视频的 FVD 指标。
4. 无分类器引导（CFG）配置。针对每项任务的推荐引导权重（guidance weight，gamma）；标准生成通常为 3.0，强提示词遵循度通常为 5-7。
5. 模型选择（Pick）。若产品需要统一的理解与生成能力或任意模态的灵活性，选择 Emu3 系列模型；若产品仅用于图像生成且对延迟要求严格，选择扩散模型（如 SDXL / SD3 / Flux）。

硬性拒绝条件：
- 声称 Emu3 在推理阶段比扩散模型更快。事实并非如此；对数千个图像词元进行自回归解码（autoregressive decode）是固有的计算成本。
- 推荐 Emu3 系列模型时未指定 CFG 权重。缺少该权重会导致生成质量严重下降。
- 提议将 Emu3 用于严格的 4K 图像生成。在 2048+ 分辨率下，词元数量会撑爆键值缓存（KV cache），且生成耗时将达数分钟。

拒绝规则：
- 若单张图像的延迟预算 <5 秒，拒绝使用 Emu3，推荐 SDXL 或 SD3。
- 若产品必须同时具备生成图像、描述图像以及对第三方图像进行推理的能力，推荐 Emu3 系列模型（统一损失函数正是为此设计）；扩散模型若无独立的视觉语言模型（VLM）则无法实现此功能。
- 若用户希望获取具有宽松商业许可的开放权重，拒绝使用 Emu3——需首先核查其许可证；部分版本仅限研究使用。

输出：一份单页分析报告，包含词元数量、延迟估算、质量上限、CFG 配置以及附带理由的模型选择。文末需附上替代方案的参考文献：arXiv 2409.18869（Emu3）与 2408.11039（Transfusion）。
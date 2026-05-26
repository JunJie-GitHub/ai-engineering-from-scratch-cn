# Emu3：用于图像和视频生成的下一词元预测（Next-Token Prediction）

> 北京智源人工智能研究院（BAAI）的 Emu3（Wang 等人，2024年9月）是2024年的一项成果，本应终结扩散模型（Diffusion Model）与自回归模型（Autoregressive Model）之争。该模型采用单一的类 Llama 纯解码器 Transformer（Decoder-only Transformer），仅在下一词元预测（Next-Token Prediction）目标上进行训练，并使用涵盖文本、VQ 图像词元（VQ Image Tokens）与 3D VQ 视频词元（3D VQ Video Tokens）的统一词表。其在图像生成方面超越了 SDXL，在视觉感知（Perception）方面超越了 LLaVA-1.6。无需 CLIP 损失（CLIP Loss），也无需扩散调度（Diffusion Schedule）。推理阶段虽使用无分类器引导（Classifier-Free Guidance）以提升生成质量，但核心训练目标仍是结合教师强制（Teacher Forcing）的下一词元预测。该成果发表于《自然》（Nature）期刊。本课程将深入解读 Emu3 的核心论点——为何更优的分词器（Tokenizer）配合模型规模（Scale）便已足够——并将其与扩散模型方法进行对比。

**类型：** 学习
**编程语言：** Python（标准库，3D 视频分词器数学原理 + 自回归采样器骨架）
**前置知识：** 第12阶段 · 第11课（Chameleon）
**预计时长：** 约120分钟

## 学习目标

- 解释为何 Emu3 的单一损失下一词元目标能够奏效，尽管长期以来业界一直认为图像质量必须依赖扩散模型。
- 描述 3D 视频分词器（3D Video Tokenizer）：时空 VQ 码本（Spatiotemporal VQ Codebook）的结构是怎样的，以及为何图像块（Patches）需要跨越时间维度。
- 从（训练算力、推理成本、质量上限）三个维度对比 Emu3 与 Stable Diffusion XL。
- 指出同一 Emu3 模型所扮演的三种角色：Emu3-Gen（图像生成）、Emu3-Chat（视觉感知）、Emu3-Stage2（视频生成）。

## 问题背景

截至2024年的普遍共识是：图像生成必须依赖扩散模型。其核心论据在于：离散的图像词元会丢失过多信息，难以重建细节；且自回归采样（Autoregressive Sampling）在生成数千个词元时会累积误差。Stable Diffusion、DALL-E 3、Imagen 和 Midjourney 均采用某种形式的扩散模型。Chameleon（第12.11课）虽在小规模实验中部分推翻了这一观点，但在生成质量上仍未能匹敌 SDXL。

Emu3 直接正面挑战了这一论断。其核心主张为：更优的视觉分词器（Visual Tokenizer） + 充足的模型规模（Scale） + 下一词元损失（Next-Token Loss） = 在同一模型中实现超越扩散模型的图像生成能力，且该模型同时具备视觉感知功能。

该成果发表之初曾引发广泛争议。两年后的今天，开源统一生成模型家族（Emu3、Show-o、Janus-Pro、Transfusion）已成为学术界的研究默认路径；而工业界的前沿模型似乎也已采用其某种变体架构。

## 核心概念

### Emu3 分词器 (Tokenizer)

其核心组件是视觉分词器 (Visual Tokenizer)。Emu3 训练了一款自定义的 IBQ 类分词器（逆瓶颈量化器 Inverse Bottleneck Quantizer，隶属于 SBER-MoVQGAN 家族），每个词元 (token) 对应 8x8 的分辨率压缩。在码本 (codebook) 大小为 32768 的情况下，一张 512x512 的图像会被转换为 64x64 = 4096 个词元。

这一数量高于 Chameleon 模型在 K=8192 时每张 512x512 图像生成的 1024 个词元，但单个词元的计算成本更低（码本查找规模更小，编解码器更简单）。关键指标：重建峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR) 达到 30.5 dB，与 Stable Diffusion 连续潜在空间 (continuous latent space) 的 32 dB 表现相当。

针对视频：3D VQ 分词器 (3D Vector Quantization Tokenizer) 将一个时空块 (spatiotemporal patch，4x4x4 像素) 编码为单个整数。一段 8 FPS 的 4 秒视频包含 32 帧；在 256x256 分辨率下，经过 4 倍空间降维和 4 倍时间降维后，词元数量为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 个。

分词器的质量决定了性能上限。Emu3 的贡献之一在于“我们训练出了一个非常优秀的分词器”。

### 单一损失函数训练 (Single-Loss Training)

Emu3 采用单一优化目标：在文本词元、2D 图像词元和 3D 视频词元共享的词表 (vocabulary) 上进行下一词元预测 (next-token prediction)。训练期间，损失权重会乘以特定模态 (modality) 的系数以平衡各模态的贡献，但损失函数本身保持一致。

训练数据混合了以下格式：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：格式类似。
- 纯文本：标准的下一词元预测 (Next-Token Prediction, NTP)。

模型通过数据分布自主学习何时输出图像词元，何时输出文本词元。生成能力源于模型在 `<image>` 标签后预测图像词元的过程。

### 无分类器引导与温度参数 (Classifier-Free Guidance and Temperature)

在推理阶段，引入无分类器引导 (Classifier-Free Guidance, CFG) 能显著提升自回归 (autoregressive) 图像生成的质量。Emu3 采用了该技术：进行两次生成，一次使用完整提示词 (caption)，一次使用空提示词，随后按引导权重（通常为 3.0-7.0）混合两者的对数几率 (logits)。这与扩散模型 (diffusion models) 使用的 CFG 技巧相同，现被借鉴至自回归架构中。

温度参数 (temperature) 至关重要：过高会导致伪影 (artifacts)，过低则引发模式崩溃 (mode collapse)。Emu3 推荐的温度参数为：感知任务 1.0，图像生成任务 0.8。

### 三种角色，同一模型

Emu3 对外提供三个功能各异的 API，但底层共享同一套权重：

- Emu3-Gen：图像生成。输入文本，输出图像词元。
- Emu3-Chat：视觉问答 (Visual Question Answering, VQA) 与图像描述。输入图像（词元），输出文本。
- Emu3-Stage2：视频生成与视频 VQA。输入文本或视频，输出文本或视频。

无需任务特定的输出头 (task-specific heads)。仅通过不同的提示词模板 (prompt templates) 切换。共享同一检查点 (checkpoint)。

### 基准测试 (Benchmarks)

根据 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K 数据集的 FID 分数上优于 SDXL（5.4 vs 5.6），GenEval 综合得分相当（0.54 vs 0.55——统计学上无显著差异），Deep-Eval 综合指标持平。
- 图像感知：在 VQAv2 数据集上优于 LLaVA-1.6（75.1 vs 72.4），在 MMMU 数据集上表现大致相当。
- 视频生成：4 秒视频片段的生成质量在 FVD 指标上与 Sora 时代公开评测的模型具有竞争力。

各项指标并非全面领先——Emu3 在不同任务间存在性能权衡——但“下一词元预测即所需的一切”这一主张在多模态场景下依然站得住脚。

### 计算成本 (Compute Cost)

Emu3 使用 70 亿参数 (7B) 的模型，在约 3000 亿多模态词元上进行了训练。其 GPU 算力消耗大致与 Llama-2-7B 预训练相当（在 A100 级别硬件上约需 2000-4000 GPU 年）。类似 Stable Diffusion 3 的扩散模型虽在相近的算力预算下训练，但需要独立的文本编码器 (text encoders) 和更复杂的流水线 (pipelines)。

在推理阶段，Emu3 单张图像的生成速度慢于 SDXL：以 30 tok/s 的速度生成 4096 个图像词元，每张 512x512 图像约需 2 分钟，而 SDXL 仅需 2-5 秒。投机解码 (speculative decoding) 和 KV 缓存 (KV-cache) 优化虽能缩小差距，但无法完全抹平。自回归图像生成计算密集度高，这是当前架构固有的权衡。

### 重要意义

Emu3 更深层次的贡献在于理念层面。如果下一词元预测的扩展能力能在图像生成上媲美扩散模型，那么统一模型路径（单一损失函数、单一主干网络、支持任意模态）将是切实可行的。未来的模型将不再需要独立的文本编码器、独立的扩散调度器 (diffusion schedulers) 或独立的变分自编码器 (Variational Autoencoders, VAEs)。仅需一个 Transformer 架构，为每种模态配备一个分词器，即可实现扩展。

Show-o、Janus-Pro 和 InternVL-U 等模型均在此基础上构建或对该理念发起挑战。截至 2025 年，中国实验室（如北京智源人工智能研究院 BAAI、深度求索 DeepSeek）在该方向上的论文发表力度已超过美国实验室。

## 使用方法

`code/main.py` 构建了两个演示模块：

- 2D 与 3D VQ 分词器（VQ Tokenizer）数量计算器：给定（resolution, patch, clip_length, FPS），计算图像与视频的词元（Token）数量。
- 一个结合温度参数与无分类器引导（Classifier-Free Guidance, CFG）的自回归（Autoregressive）图像词元采样器。

该 CFG 的实现遵循 Emu3 的方案——通过引导权重混合条件与非条件逻辑值（Logits）。

## 交付物

本课时将生成 `outputs/skill-token-gen-cost-analyzer.md`。给定生成任务的产品规格（图像或视频、目标分辨率、质量等级、延迟预算），该脚本会计算词元数量与推理成本，并在 Emu3 系列模型与扩散模型（Diffusion Model）之间进行选型对比。

## 练习

1. Emu3 在 8x8 降采样率下，每张 512x512 图像生成 4096 个词元。请计算 1024x1024 和 2048x2048 分辨率下的等效词元数量。推理延迟将如何变化？
2. 阅读 Emu3 论文第 3.3 节关于视频分词器的内容。描述 3D VQ 的图像块（Patch）形状，并解释为何采用 4x4x4 而非 8x8x1。
3. 无分类器引导权重设为 5.0 与 3.0 相比：会产生何种视觉效果？请在 `code/main.py` 中梳理其数学推导过程。
4. 计算 Emu3-7B 在 300B 词元训练规模下的浮点运算次数（FLOPs），并与 Stable Diffusion 3 进行对比。哪个模型的训练成本更高？
5. Emu3 在弗雷歇起始距离（FID）指标上优于 SDXL，但在 VQAv2 基准上不及专用的视觉语言模型（Vision-Language Models, VLMs）。请解释为何统一损失（Unified-Loss）方法在不同基准测试中，相较于专用模型会展现出不同的优势。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 下一词元预测（Next-token prediction） | "NTP" | 标准的自回归损失函数：根据 token[0..i] 预测 token[i+1]；只要数据经过分词处理，即可适用于所有模态 |
| IBQ 分词器（IBQ tokenizer） | "逆瓶颈量化器（Inverse bottleneck quantizer）" | 一类向量量化变分自编码器（VQ-VAE），具有更大的码本（32768+），且重建质量优于 Chameleon |
| 3D VQ（三维向量量化） | "时空量化器（Spatiotemporal quantizer）" | 码本由（time, row, col）索引；单个词元覆盖一个 4x4x4 的像素立方体 |
| 无分类器引导（Classifier-free guidance） | "CFG" | 使用权重 gamma 混合条件与非条件逻辑值；在推理阶段提升图像质量 |
| 统一词表（Unified vocabulary） | "共享词元（Shared tokens）" | 文本、图像和视频均从同一整数空间中采样；模型负责预测下一个出现的任意模态 |
| MJHQ-30K | "图像生成基准（Image gen benchmark）" | 包含 3 万条提示词的 Midjourney 质量级图像生成基准；Emu3 在此报告 FID 分数 |

## 延伸阅读

- [Wang 等人 — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun 等人 — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu 等人 — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu 等人 — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian 等人 — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
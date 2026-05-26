# 图像修复（Inpainting）、图像扩展（Outpainting）与图像编辑（Image Editing）

> 文生图（Text-to-Image）用于创造新事物，而图像修复则用于修改现有内容。在实际生产环境中，70% 的可计费图像工作属于编辑类——例如替换背景、移除 Logo、扩展画布或重新生成手部。图像修复正是扩散模型（Diffusion）真正发挥商业价值的地方。

**类型：** 构建
**语言：** Python
**前置要求：** 第 8 阶段 · 07（潜在扩散模型（Latent Diffusion）），第 8 阶段 · 08（ControlNet 与 LoRA）
**耗时：** 约 75 分钟

## 问题描述

客户发来一张完美的产品照片，但背景中有一个干扰视线的标牌。你需要擦除该标牌，同时保持图像其余部分的像素完全一致。你不能从头开始运行文生图流程，因为生成的结果在颜色、光照和产品角度上都会发生变化。你希望*仅*重新生成掩码（Mask）区域内的内容，并且要求生成结果与周围上下文保持一致。

这就是图像修复。其主要变体包括：

- **图像修复。** 在掩码内部重新生成内容，保留外部像素。
- **图像扩展。** 在掩码外部（或画布边界之外）重新生成内容，保留内部区域。
- **图像编辑。** 重新生成整张图像，但保持与原图在语义或结构上的一致性（如 SDEdit、InstructPix2Pix）。

到 2026 年，所有主流的扩散模型管线（Pipeline）均内置了图像修复模式。例如 Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint 和 DALL-E 3 Edit。它们均基于相同的底层原理运行。

## 核心概念

![图像修复（Inpainting）：结合上下文保留重注入的掩码感知去噪](../assets/inpainting.svg)

### 朴素方法（及其缺陷）

结合掩码运行标准的文生图（text-to-image）流程。在每一步采样中，将噪声潜变量（noisy latent）中未掩码区域替换为经过前向扩散（forward diffusion）的干净图像。该方法虽然可行……但效果极差。由于模型对掩码区域内的内容毫无先验信息，边界伪影（boundary artifacts）会严重渗透。

### 专用的图像修复模型

训练一个改进的 U-Net，使其接收 9 个输入通道而非 4 个：

input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)

额外的通道由 VAE 编码后的源图像副本与单通道掩码组成。训练时，随机遮盖图像的某些区域，并训练模型仅对遮盖区域进行去噪，未遮盖区域则作为清晰的条件信号（conditioning signal）输入。推理时，模型能够“感知”掩码区域周围的环境，从而生成连贯的补全内容。

SD-Inpaint、SDXL-Inpaint 和 Flux-Fill 均采用这种 9 通道（或类似结构）的输入方式。Diffusers 库中的 `StableDiffusionInpaintPipeline` 和 `FluxFillPipeline` 也基于此实现。

### SDEdit（Meng 等，2022）—— 自由编辑

对源图像添加噪声至某一中间时间步 `t`，随后结合新的提示词（prompt）从 `t` 开始反向运行扩散链直至 0。全程无需重新训练。起始时间步 `t` 的选择决定了保真度与创作自由度之间的权衡：

- `t/T = 0.3` → 与源图像几乎一致，仅产生细微的风格变化
- `t/T = 0.6` → 中等程度编辑，保留粗略结构
- `t/T = 0.9` → 从近似纯噪声状态生成，几乎不保留源图像特征

### InstructPix2Pix（Brooks 等，2023）

在 `(input_image, instruction, output_image)` 三元组数据集上对扩散模型进行微调（fine-tune）。推理时，模型同时以输入图像和文本指令（例如“改为日落场景”、“添加一条龙”）作为条件输入。该模型采用双 CFG 尺度（CFG scales）：图像尺度与文本尺度。

### RePaint（Lugmayr 等，2022）

沿用标准的无条件扩散模型（unconditional diffusion model）。在每一步反向去噪过程中引入重采样（resample）机制——偶尔回退至噪声更高的状态并重新生成。该方法能有效避免边界伪影。适用于缺乏专用图像修复模型时的替代方案。

## 动手实现

`code/main.py` 实现了一个基于5维数据的简易一维图像修复（inpainting）方案。我们在5维混合数据上训练了一个去噪扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM），其中每个样本由来自两个聚类之一的5个浮点数组成。在推理（inference）阶段，我们对5个维度中的2个进行“掩码（mask）”处理，在每一步注入未掩码的3个维度的前向加噪版本，并仅重新生成被掩码的维度。

### 步骤 1：5维 DDPM 数据

def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster

### 步骤 2：在所有5个维度上训练去噪器（denoiser）

采用标准的 DDPM。网络针对5维加噪输入输出5维的噪声预测值。

### 步骤 3：推理阶段的掩码感知反向过程（mask-aware reverse）

def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t

这是一种朴素（naive）的方法，在简易一维数据上有效。实际的图像修复通常使用9通道输入，因为纹理连贯性（texture coherence）更为重要。

### 步骤 4：图像外扩（outpainting）

图像外扩本质上是掩码反转的图像修复：对新增的（原先不存在的）画布区域进行掩码处理，其余部分填充原始图像。训练目标完全相同。

## 常见陷阱（Pitfalls）

- **接缝问题。** 朴素方法会留下可见边界，因为梯度信息无法跨越掩码流动。修复方法：将掩码膨胀（dilate）8-16个像素，或使用专用的修复模型。
- **掩码泄漏。** 如果条件图像（conditioning image）的未掩码区域质量较低或含有噪声，会污染掩码内部的生成结果。可先进行轻微去噪或模糊处理。
- **CFG 与掩码尺寸的交互。** 在小掩码上使用高无分类器引导（Classifier-Free Guidance, CFG）会导致色块过饱和。进行局部微调时应降低 CFG。
- **SDEdit 的保真度断崖。** 将 `t/T` 从 `0.5` 调整到 `0.6` 可能会导致主体特征丢失。建议进行参数扫描（sweep）并保存检查点（checkpoint）。
- **提示词不匹配。** 提示词（prompt）应描述*整张*图像，而不仅仅是新增内容。例如使用“一只坐在椅子上的猫”，而不是仅用“一只猫”。

## 应用场景（Use It）

| 任务 | 工作流（pipeline） |
|------|----------|
| 移除物体（小掩码） | SD-Inpaint 或 Flux-Fill，使用标准提示词 |
| 替换天空 | SD-Inpaint + “日落时的蓝天” |
| 扩展画布 | SDXL 外扩模式（8像素羽化）或使用外扩掩码的 Flux-Fill |
| 重绘手部/面部 | SD-Inpaint（配合重新描述主体的提示词）+ ControlNet-Openpose |
| 更改局部区域风格 | 在掩码区域使用 `t/T=0.5` 的 SDEdit |
| “改为日落效果” | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 掩码 → SD-Inpaint |
| 超高保真度 | 针对高难度场景使用 Flux-Fill 或 GPT-Image（云端托管版） |

SAM（Meta 的 Segment Anything 模型，2023）结合扩散模型修复已成为 2026 年主流的背景移除工作流。SAM 2（2024）已支持视频处理。

## 部署上线（Ship It）

保存 `outputs/skill-editing-pipeline.md`。该技能（Skill）接收原始图像、编辑描述以及可选的掩码（或 SAM 提示词），并输出以下内容：掩码生成方案、基础模型（Base Model）、CFG 引导系数（CFG Scales，含图像与文本）、SDEdit-t 或图像修复模式（Inpainting Mode），以及质量检查清单（QA Checklist）。

## 练习

1. **简单。** 在 `code/main.py` 中，将掩码覆盖的维度比例从 0.2 逐步调整至 0.8。请问在何种比例下，图像修复质量（Inpaint Quality，即掩码区域内的残差）会与无条件生成（Unconditional Generation）的效果相当？
2. **中等。** 实现 RePaint 算法：在每第 10 个反向步骤（Reverse Step）时，回退 5 步（重新添加噪声）并再次去噪。评估该方法是否能有效降低掩码边缘的边界残差（Boundary Residual）。
3. **困难。** 使用 Hugging Face `diffusers` 库进行对比测试：在 20 个人脸重建任务中，对比 SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill 的表现。请分别对姿态遵循度（Pose Adherence）和身份保持度（Identity Preservation）进行独立评分。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 图像修复（Inpainting） | “填补空洞” | 在掩码区域内重新生成内容；保留掩码外的像素。 |
| 图像扩展（Outpainting） | “扩展画布” | 在画布外部重新生成内容；保留内部区域。 |
| 9通道 U-Net | “标准的图像修复模型” | 以 `noisy | encoded-source | mask` 作为输入的 U-Net 架构。 |
| SDEdit | “带噪声级别的图生图” | 将图像加噪至时间步 `t`，再使用新提示词进行去噪。 |
| InstructPix2Pix | “纯文本编辑” | 基于（图像，指令，输出）三元组微调的扩散模型。 |
| RePaint | “无需重新训练” | 在反向过程中周期性重新加噪，以减少拼接痕迹。 |
| SAM（Segment Anything Model） | “分割一切” | 通过点击或边界框生成掩码的工具；常与图像修复配合使用。 |
| Flux-Kontext | “基于上下文的编辑” | Flux 的变体模型，可接收参考图像与编辑指令进行内容修改。 |

## 生产环境提示：编辑流水线对延迟高度敏感

用户在进行图像编辑时，通常期望端到端往返延迟（Round Trips）低于 5 秒。在 L4 显卡上，以 1024² 分辨率运行 30 步的 SDXL-Inpaint 耗时约 3-4 秒，再加上 SAM 掩码生成（约 200 毫秒）与 VAE 编解码（VAE Encode/Decode，合计约 500 毫秒）。在生产环境架构中，此类任务受限于首字生成时间（Time To First Token, TTFT）而非整体吞吐量（Throughput）——因此需采用批处理大小为 1（Batch Size 1）、低并发的策略，并尽可能压缩每个处理阶段的耗时：

- **SAM-H 是主要耗时环节。** 在 1024² 分辨率下，SAM-H 耗时约 200 毫秒；而 SAM-ViT-B 仅需约 40 毫秒，且画质损失微乎其微。SAM 2（视频模型）会引入额外的时间维度开销，请勿将其用于单图编辑任务。
- **尽可能跳过编码步骤。** `pipe.image_processor.preprocess(img)` 会将图像编码为潜变量（Latents）。若你已持有上一轮生成的潜变量（这在迭代式编辑界面中很常见），请直接通过 `latents=...` 参数传入，从而跳过一次 VAE 编码过程。
- **掩码膨胀（Mask Dilation）同样影响吞吐量。** 掩码区域过小会导致 U-Net 的大部分前向传播（Forward Pass）计算被浪费（因为未掩码区域的像素最终仍会被强制约束）。`diffusers` 库中的 `StableDiffusionInpaintPipeline` 无论如何都会执行完整的 U-Net 前向计算；只有采用 9 通道输入的标准图像修复变体（Proper-Inpaint Variants）才能真正利用掩码区域进行计算优化。
- **Flux-Kontext 是 2025 年的终极方案。** 仅需对 `(source_image, instruction)` 执行单次前向传播——无需独立生成掩码，也无需 SDEdit 的噪声扫描（Noise Sweep）步骤。在 H100 显卡上，它可在约 1.5 秒内交付编辑结果。其带来的架构启示是：尽可能合并处理阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 免训练图像修复（inpainting）。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑（text-instruction editing）。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，掩码（mask）来源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频版 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力层级编辑（attention-level editing）。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 年工具集。
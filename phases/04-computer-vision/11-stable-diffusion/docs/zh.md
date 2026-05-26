# 稳定扩散（Stable Diffusion）—— 架构与微调

> 稳定扩散（Stable Diffusion）是一种去噪扩散概率模型（DDPM），它在预训练变分自编码器（VAE）的潜在空间（latent space）中运行，通过交叉注意力（cross-attention）机制以文本为条件，使用快速确定性常微分方程（ODE）求解器进行采样，并由无分类器引导（classifier-free guidance）进行控制。

**类型：** 学习 + 应用
**编程语言：** Python
**前置知识：** 第4阶段第10课（扩散模型），第7阶段第02课（自注意力机制）
**预计时长：** 约75分钟

## 学习目标

- 梳理稳定扩散（Stable Diffusion）流水线的五个核心组件：变分自编码器（VAE）、文本编码器（text encoder）、U-Net、调度器（scheduler）和安全检查器（safety checker），并理解它们各自的具体作用
- 解释潜在扩散（latent diffusion）的原理，以及为何在 4x64x64 的潜在空间（而非 3x512x512 的图像空间）中进行训练能在不损失质量的情况下将计算量降低 48 倍
- 使用 `diffusers` 库生成图像，并执行图生图（image-to-image）、图像修复（inpainting）以及 ControlNet 引导的生成任务
- 在小型自定义数据集上使用低秩自适应（LoRA）微调稳定扩散模型，并在推理阶段加载 LoRA 适配器

## 问题背景

直接在 512x512 的 RGB 图像上训练 DDPM 成本极高。每个训练步骤都需要通过 U-Net 进行反向传播，该网络需处理 3x512x512 = 786,432 个输入值；而采样过程则需要在同一个 U-Net 上进行 50 次以上的前向传播。若要达到 2022 年发布的 Stable Diffusion 1.5 的质量水平，在像素空间进行扩散训练大约需要 256 个 GPU 月的算力，且在消费级 GPU 上生成单张图像需耗时 10 到 30 秒。

让开源文生图模型得以落地的关键技巧是**潜在扩散（latent diffusion）**（Rombach 等人，CVPR 2022）。训练一个 VAE，将 3x512x512 的图像映射为 4x64x64 的潜在张量（latent tensor），然后再映射回图像空间，随后在该潜在空间中执行扩散过程。计算量因此下降了 `(3*512*512)/(4*64*64) = 48x`。在同一块 GPU 上，采样时间也从数十秒缩短至两秒以内。

如今几乎所有的主流图像生成模型（如 SDXL、SD3、FLUX、HunyuanDiT、Wan-Video）本质上都是潜在扩散模型，仅在自编码器（autoencoder）、去噪器（denoiser，如 U-Net 或 DiT）以及文本条件机制（text conditioning）上有所变体。掌握了 Stable Diffusion，你就掌握了这类模型的通用范式。

## 核心概念

### 处理流程 (Pipeline)

flowchart LR
    TXT["Text prompt"] --> TE["Text encoder<br/>(CLIP-L or T5)"]
    TE --> CT["Text<br/>embedding"]

    NOISE["Noise<br/>4x64x64"] --> UNET["UNet<br/>(denoiser with<br/>cross-attention<br/>to text)"]
    CT --> UNET

    UNET --> SCHED["Scheduler<br/>(DPM-Solver++,<br/>Euler)"]
    SCHED --> LATENT["Clean latent<br/>4x64x64"]
    LATENT --> VAE["VAE decoder"]
    VAE --> IMG["512x512<br/>RGB image"]

    style TE fill:#dbeafe,stroke:#2563eb
    style UNET fill:#fef3c7,stroke:#d97706
    style SCHED fill:#fecaca,stroke:#dc2626
    style IMG fill:#dcfce7,stroke:#16a34a

- **变分自编码器 (VAE)** — 冻结的自编码器。编码器将图像转换为潜在表示 (latents，用于图生图 img2img 和训练)。解码器将潜在表示还原回图像。
- **文本编码器 (Text encoder)** — CLIP 文本编码器（用于 SD 1.x/2.x）、CLIP-L + CLIP-G（用于 SDXL）或 T5-XXL（用于 SD3/FLUX）。用于生成词元嵌入 (token embeddings) 序列。
- **U-Net** — 去噪器。包含交叉注意力层 (cross-attention layers)，在每个分辨率层级上使潜在表示关注文本嵌入。
- **调度器 (Scheduler)** — 采样算法（如 DDIM、Euler、DPM-Solver++）。负责选择噪声水平 (sigmas)，并将预测的噪声混合回潜在表示中。
- **安全检查器 (Safety checker)** — 可选的 NSFW（不适宜工作场所内容）/非法内容过滤器，用于过滤输出图像。

### 无分类器引导 (Classifier-free guidance, CFG)

普通的文本条件化会为每个提示词 `c` 学习 `epsilon_theta(x_t, t, c)`。CFG 在训练时，有 10% 的概率丢弃条件 `c`（替换为空嵌入），从而让同一个模型能够同时预测条件噪声和无条件噪声。在推理阶段：

eps = eps_uncond + w * (eps_cond - eps_uncond)

`w` 为引导尺度 (guidance scale)。`w=0` 表示无条件生成，`w=1` 表示普通条件生成，`w>1` 会促使输出“更严格地遵循提示词”，但会牺牲多样性。Stable Diffusion 的默认值为 `w=7.5`。

CFG 是文生图 (text-to-image) 能够达到生产级质量的关键。没有它，提示词对输出的影响较弱；有了它，提示词将主导生成结果。

### 潜在空间几何 (Latent space geometry)

VAE 的 4 通道潜在表示 (latent) 不仅仅是一张压缩图像。它是一个流形 (manifold)，其上的算术运算大致对应于语义编辑（提示词工程与插值均在此空间进行），扩散 U-Net 的全部建模预算也集中于此。解码一个随机的 4x64x64 潜在表示并不会生成一张看似随机的图像——而是会产生无意义的噪声，因为只有特定的潜在子流形才能解码为有效图像。

由此带来两个重要特性：

1. **图生图 (Img2img)** = 将图像编码为潜在表示，添加部分噪声，运行去噪器，最后解码。由于编码过程近似可逆，图像结构得以保留；内容则根据提示词发生变化。
2. **图像修复 (Inpainting)** = 原理与图生图相同，但去噪器仅更新掩码 (masked) 区域；未掩码区域则保持原始编码的潜在表示不变。

### U-Net 架构

Stable Diffusion 的 U-Net 是第 10 课中 TinyUNet 的放大版本，主要增加了以下三个部分：

- 每个空间分辨率层级均包含 **Transformer 块 (Transformer blocks)**，内部包含自注意力 (self-attention) 以及针对文本嵌入的交叉注意力。
- 通过正弦编码 (sinusoidal encoding) 上的多层感知机 (MLP) 实现 **时间嵌入 (Time embedding)**。
- 在编码器与解码器之间相同分辨率层级处添加 **跳跃连接 (Skip connections)**。

SD 1.5 的总参数量约为 8.6 亿 (860M)。SDXL 约为 26 亿 (2.6B)。FLUX 约为 120 亿 (12B)。参数量的激增主要集中在注意力层。

### LoRA 微调 (LoRA fine-tuning)

对 Stable Diffusion 进行全量微调需要 20 GB 以上的显存 (VRAM)，并更新 8.6 亿个参数。低秩自适应 (LoRA, Low-Rank Adaptation) 保持基础模型冻结，仅向注意力层注入小型的秩分解矩阵。适用于 SD 的 LoRA 适配器通常仅 10-50 MB，在单张消费级 GPU 上训练 10-60 分钟即可完成，并在推理时作为即插即用的模块加载。

Original: W_q : (d_in, d_out)   frozen
LoRA:     W_q + alpha * (A @ B)   where A : (d_in, r), B : (r, d_out)

r is typically 4-32.

目前社区中几乎所有的微调模型都通过 LoRA 形式分发。CivitAI 和 Hugging Face 上托管着数百万个此类模型。

### 常见调度器 (Schedulers you will see)

- **DDIM** — 确定性算法，约需 50 步，实现简单。
- **Euler ancestral** — 随机性算法，需 30-50 步，生成的样本更具创造性。
- **DPM-Solver++ 2M Karras** — 确定性算法，需 20-30 步，生产环境默认选项。
- **LCM / TCD / Turbo** — 一致性模型 (consistency models) 及其蒸馏变体；仅需 1-4 步，但会牺牲部分图像质量。

在 `diffusers` 库中切换调度器通常只需修改一行代码，有时甚至无需重新训练即可解决采样问题。

## 构建

本课程将端到端地使用 `diffusers` 库，而非从零开始重建稳定扩散（Stable Diffusion）。重建所需的各个组件（变分自编码器（VAE）、文本编码器（text encoder）、U-Net、调度器（scheduler））均有各自的专题课程；本课程的目标是让你熟练掌握生产级 API 的使用。

### 步骤 1：文生图（Text-to-image）

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    prompt="a dog riding a skateboard in tokyo, studio ghibli style",
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("dog.png")

使用 `float16` 可将显存（VRAM）占用减半，且不会造成肉眼可见的质量损失。在使用默认的 DPM-Solver++ 时，`num_inference_steps=25` 的效果与使用 DDIM 时的 `num_inference_steps=50` 相当。

### 步骤 2：更换调度器（Scheduler）

from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

调度器（scheduler）的状态与 U-Net 的权重是解耦的。你可以使用 DDPM 进行训练，并使用任意调度器进行采样。

### 步骤 3：图生图（Image-to-image）

from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

init_image = Image.open("dog.png").convert("RGB").resize((512, 512))
out = img2img(
    prompt="a dog riding a skateboard, oil painting",
    image=init_image,
    strength=0.6,
    guidance_scale=7.5,
).images[0]

`strength` 参数控制在去噪前添加的噪声量（0.0 = 保持原图不变，1.0 = 完全重新生成）。0.5-0.7 是风格迁移的标准取值范围。

### 步骤 4：图像修复（Inpainting）

from diffusers import StableDiffusionInpaintPipeline

inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

image = Image.open("dog.png").convert("RGB").resize((512, 512))
mask = Image.open("dog_mask.png").convert("L").resize((512, 512))

out = inpaint(
    prompt="a cat",
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
).images[0]

掩码（mask）中的白色像素表示需要重新生成的区域，黑色像素表示保留区域。

### 步骤 5：加载 LoRA

pipe.load_lora_weights("sayakpaul/sd-lora-ghibli")
pipe.fuse_lora(lora_scale=0.8)

image = pipe(prompt="a village square in ghibli style").images[0]

`lora_scale` 用于控制强度；0.0 表示无效果，1.0 表示完全生效。`fuse_lora` 会将适配器（adapter）原地融合到权重中以提升推理速度，但这会导致无法再切换适配器。在加载其他适配器之前，请先调用 `pipe.unfuse_lora()`。

### 步骤 6：LoRA 训练（代码框架）

实际的 LoRA 训练代码通常位于 `peft` 或 `diffusers.training` 模块中。以下是核心流程框架：

# Pseudocode
for step, batch in enumerate(dataloader):
    images, prompts = batch
    latents = vae.encode(images).latent_dist.sample() * 0.18215

    t = torch.randint(0, num_train_timesteps, (batch_size,))
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    text_emb = text_encoder(tokenizer(prompts))

    pred_noise = unet(noisy_latents, t, text_emb)  # LoRA weights injected here

    loss = F.mse_loss(pred_noise, noise)
    loss.backward()
    optimizer.step()

训练过程中仅 LoRA 矩阵会接收梯度更新，基础的 U-Net、VAE 和文本编码器均保持冻结状态。在批次大小（batch size）为 1 且启用梯度检查点（gradient checkpointing）的情况下，该流程可在 8 GB 显存中运行。

## 实战应用

在实际生产环境中，你需要做出的具体决策包括：

- **模型系列（Model family）**：SD 1.5 适用于开源社区的微调模型，SDXL 适用于更高保真度场景，SD3 / FLUX 适用于追求最先进效果且有严格许可证要求的场景。
- **调度器（Scheduler）**：20-30 步推理时使用 DPM-Solver++ 2M Karras，延迟要求低于 1 秒时使用 LCM-LoRA。
- **精度（Precision）**：在 4080/4090 显卡上使用 `float16`，在 A100 及更新架构上使用 `bfloat16`，显存（VRAM）紧张时使用 `int8`（通过 `bitsandbytes` 或 `compel` 实现）。
- **条件控制（Conditioning）**：纯文本即可生效；若需更强控制力，可在基础流水线（pipeline）之上叠加 ControlNet（如边缘检测 canny、深度 depth、姿态 pose）。

针对批量生成任务，社区常用工具为 `AUTO1111` / `ComfyUI`；针对生产环境 API，推荐使用 `diffusers` + `accelerate`，或结合 TensorRT 编译的 `optimum-nvidia`。

## 交付成果

本课时将产出以下文件：

- `outputs/prompt-sd-pipeline-planner.md` —— 一个提示词（prompt），可根据延迟预算、保真度目标和许可证限制，自动选择 SD 1.5 / SDXL / SD3 / FLUX 模型，并搭配相应的调度器与精度设置。
- `outputs/skill-lora-training-setup.md` —— 一项技能（skill），可为自定义数据集生成完整的 LoRA 训练配置，包含图像描述（captions）、秩（rank）、批次大小（batch size）和学习率（learning rate）。

## 练习

1. **(简单)** 使用 `guidance_scale` 参数在 `[1, 3, 5, 7.5, 10, 15]` 范围内生成同一提示词对应的图像。描述图像的变化情况。在哪个引导值（guidance value）时开始出现伪影（artefacts）？
2. **(中等)** 选取任意真实照片，通过 `StableDiffusionImg2ImgPipeline` 以 `strength` 参数在 `[0.2, 0.4, 0.6, 0.8, 1.0]` 范围内进行处理。哪个强度值能在改变风格的同时保留原始构图？为什么设置为 1.0 时会完全忽略输入图像？
3. **(困难)** 使用 10-20 张单一主体（如宠物、Logo 或角色）的图像训练一个 LoRA 模型，并生成包含该主体的全新场景。报告在不过度拟合（overfitting）输入图像的前提下，能最佳保留主体特征（identity preservation）的 LoRA 秩（rank）和训练步数（training steps）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 潜在扩散 (Latent Diffusion) | “在潜在空间中扩散” | 在变分自编码器 (VAE) 的潜在空间（4x64x64）而非像素空间（3x512x512）中运行完整的去噪扩散概率模型 (DDPM)；计算量节省 48 倍 |
| VAE 缩放因子 (VAE Scale Factor) | “0.18215” | 用于将 VAE 原始潜在变量重新缩放至近似单位方差的常数；硬编码于每个 Stable Diffusion (SD) 流程中 |
| 无分类器引导 (Classifier-Free Guidance) | “CFG” | 混合条件与非条件噪声预测；对推理效果影响最大的单一调节参数 |
| 调度器 (Scheduler) | “采样器” | 将噪声与模型预测转化为去噪潜在轨迹的算法 |
| 低秩自适应 (LoRA) | “低秩适配器” | 小型秩分解矩阵，可在不改动基础权重的情况下微调注意力层 |
| 交叉注意力 (Cross-Attention) | “文本-图像注意力” | 从潜在词元 (token) 指向文本词元的注意力机制；在 U-Net 的每一层注入提示词信息 |
| ControlNet | “结构条件控制” | 独立训练的适配器，通过额外输入（如 Canny 边缘、深度图、姿态、分割图）引导 SD 生成 |
| DPM-Solver++ | “默认调度器” | 二阶确定性常微分方程 (ODE) 求解器；在 2026 年，于低步数（20-30 步）下仍能提供最佳生成质量 |

## 延伸阅读

- [High-Resolution Image Synthesis with Latent Diffusion (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752) — Stable Diffusion 的原始论文；包含所有用于论证该架构设计的消融实验
- [Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598) — 无分类器引导 (CFG) 的原始论文
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — LoRA 最初应用于自然语言处理 (NLP) 领域；几乎无需修改即可迁移至 SD
- [diffusers documentation](https://huggingface.co/docs/diffusers) — 所有 SD / SDXL / SD3 / FLUX 流程的官方参考文档
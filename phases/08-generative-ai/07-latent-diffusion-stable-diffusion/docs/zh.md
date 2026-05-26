# 潜在扩散模型（Latent Diffusion）与稳定扩散模型（Stable Diffusion）

> 在 512×512 的像素空间（pixel-space）中进行扩散计算堪称算力层面的“战争罪”。Rombach 等人（2022）指出，生成图像并不需要全部 78.6 万个维度——只需保留足以捕捉语义结构的维度，其余细节交由独立的解码器还原即可。在变分自编码器（VAE）的潜在空间（latent space）内运行扩散过程。正是这一核心思想，奠定了稳定扩散模型（Stable Diffusion）的基础。

**类型：** 构建实践
**编程语言：** Python
**前置知识：** 第 8 阶段 · 02（VAE）、第 8 阶段 · 06（DDPM）、第 7 阶段 · 09（ViT）
**预计耗时：** 约 75 分钟

## 问题所在

在 512² 分辨率下进行像素空间扩散，意味着 U-Net 需要处理形状为 `[B, 3, 512, 512]` 的张量。对于一个拥有 5 亿参数的 U-Net 而言，单次采样步骤的计算量约为 100 GFLOPS。五十步采样则意味着每张图像需要 5 TFLOPS 的算力。若在十亿张图像上进行训练，其计算成本将高得离谱。

绝大部分的浮点运算次数（FLOPs）都消耗在了让网络处理那些感知上不重要的细节上——即那些有损 VAE 本可压缩掉的高频纹理。Rombach 的思路是：仅训练一次 VAE（作为*第一阶段*），将其冻结，然后完全在 4 通道、64×64 的潜在空间内运行扩散过程（作为*第二阶段*）。使用相同的 U-Net，但像素数量仅为原来的 1/16。在保持相近生成质量的前提下，计算量大幅降低约 64 倍。

这便是稳定扩散模型（Stable Diffusion）的核心配方。SD 1.x / 2.x 在 `64×64×4` 的潜在向量上使用了 8.6 亿参数的 U-Net；SDXL 在 `128×128×4` 的潜在向量上使用了 26 亿参数的 U-Net；SD3 则结合流匹配（flow matching）技术，将 U-Net 替换为扩散 Transformer（DiT）。Flux.1-dev（Black Forest Labs, 2024）则搭载了一个 120 亿参数的 DiT-MMDiT 架构。它们均运行在同一套两阶段基础架构之上。

## 核心概念

![潜在扩散：VAE 压缩 + 潜在空间内的扩散过程](../assets/latent-diffusion.svg)

**两阶段独立训练。**

1. **第一阶段 —— VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩率：在每个空间轴上进行 8 倍下采样，并调整通道数，使潜在向量总大小约为原始像素数量的 1/16。损失函数 = 重建损失（L1 + LPIPS 感知损失）+ KL 散度（赋予较小权重，避免 `z` 被过度强制为高斯分布，因为我们并不需要从 `z` 中进行精确采样）。通常还会结合对抗损失（adversarial loss）进行训练，以确保解码后的图像边缘清晰锐利。

2. **第二阶段 —— 在 `z` 上进行扩散。** 将 `z = E(x_real)` 视为训练数据。训练一个 U-Net（或 DiT）来对 `z_t` 进行去噪。在推理阶段：通过扩散过程采样得到 `z_0`，随后通过 `x = D(z_0)` 还原图像。

**文本条件控制（Text conditioning）。** 引入两个额外组件。一个冻结的文本编码器（SD 1.x 使用 CLIP-L，SD 2/XL 使用 CLIP-L + OpenCLIP-G，SD3 和 Flux 使用 T5-XXL）。一个交叉注意力注入（cross-attention injection）机制：U-Net 的每个模块接收 `[Q = 图像特征, K = V = 文本词元]` 并将其融合。文本词元（tokens）是文本信息影响图像生成的唯一途径。

**损失函数与第 06 课完全一致。** 同样针对噪声计算 DDPM / 流匹配（flow matching）的均方误差（MSE）。你只需替换数据所在的域即可。

## 架构变体

| 模型 | 年份 | 骨干网络 | 潜在空间形状 | 文本编码器 | 参数量 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 个 token） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + 精炼器 | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏版 | 128×128×4 | 同上 | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT 蒸馏版 | 128×128×16 | T5-XXL + CLIP-L | 12B，1-4 步 |

发展趋势：使用扩散 Transformer（Diffusion Transformer, DiT）替代 U-Net，扩大文本编码器（Text Encoder）规模（T5 在提示词对齐度 Prompt Adherence 上优于 CLIP），增加潜在通道（Latent Channels）数量（从 4 提升至 16 为细节表现提供了更大余量）。

## 动手实现

`code/main.py` 在第 06 课的去噪扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM）之上堆叠了一个简易的一维变分自编码器（Variational Autoencoder, VAE；此处仅使用恒等编码器与解码器进行演示，真实的 VAE 应为卷积网络），并引入了结合无分类器引导（Classifier-Free Guidance）的类别条件控制。该示例表明，无论直接在原始一维数据上还是在编码后的数据上运行，扩散损失函数均同样有效——这正是核心所在。

### 步骤 1：编码器/解码器

def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0

真实的 VAE 拥有经过训练的权重。出于教学演示目的，此处的线性映射已足以说明：扩散过程直接作用于 `z`，而无需感知原始数据空间。

### 步骤 2：在 z 空间中进行扩散

沿用第 06 课的 DDPM 架构。网络实际处理的数据为 `z = E(x)`。在采样得到 `z_0` 后，通过 `D(z_0)` 进行解码。

### 步骤 3：无分类器引导（Classifier-Free Guidance）

在训练阶段，以 10% 的概率随机丢弃类别标签（替换为空标记）。在推理阶段，需同时计算条件噪声预测 `ε_cond` 与无条件噪声预测 `ε_uncond`，随后执行：

eps_cfg = (1 + w) * eps_cond - w * eps_uncond

`w = 0` 表示无引导（保留最大多样性），`w = 3` 为默认设置，`w = 7+` 则会导致图像饱和或过度锐化。

### 步骤 4：文本条件控制（概念说明，非代码实现）

使用冻结的文本编码器输出替换原有的类别标签。通过交叉注意力（Cross-Attention）机制将文本嵌入向量输入 U-Net：

h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)

这便是类别条件扩散模型与 Stable Diffusion 之间唯一的实质性差异。

## 常见陷阱

- **VAE 缩放不匹配（VAE-scale mismatch）。** SD 1.x 的 VAE（变分自编码器，Variational Autoencoder）在编码后会应用一个缩放常数（`scaling_factor ≈ 0.18215`）。忽略此步骤会导致 U-Net（U-Net 架构）在方差严重错误的潜在表示（latents）上进行训练。每个模型检查点（checkpoint）都会内置该参数。
- **文本编码器静默出错（Text encoder silently wrong）。** SD3 需要 T5-XXL 且输入长度需 >=128 个词元（tokens），若回退至仅使用 CLIP（对比语言-图像预训练模型）会导致信息有损。务必确保 `use_t5=True`，否则提示词保真度（prompt fidelity）将断崖式下降。
- **混合潜在空间（Mixing latent spaces）。** SDXL、SD3 和 Flux 均使用不同的 VAE。基于 SDXL 潜在表示训练的 LoRA（低秩自适应，Low-Rank Adaptation）无法在 SD3 上生效。Hugging Face diffusers（扩散模型库）0.30+ 版本会直接拒绝加载不匹配的模型检查点。
- **CFG 过高（CFG too high）。** 当 `w > 10` 时，生成的图像会出现色彩过饱和和“油腻感”，并以牺牲多样性为代价过度拟合提示词。最佳取值区间为 `w = 3-7`。
- **负向提示词泄漏（Negative prompts leaking）。** 空的负向提示词会转换为空词元（null token）；而填充了内容的负向提示词则会转换为 `ε_uncond`。两者并不等价；部分推理流水线（pipelines）会静默地默认回退至空词元。

## 使用指南

2026 年生产环境技术栈推荐：

| 目标场景 | 推荐主干模型（backbone） |
|--------|----------------------|
| 垂直领域、成对数据、从零训练模型 | SDXL 微调（fine-tune）（LoRA / 全量）—— 部署最快 |
| 开放域文生图、开放权重 | Flux.1-dev（12B，Apache / 非商用）或 SD3.5-Large |
| 最快推理速度、开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示词遵循度、云端托管 | GPT-Image / DALL-E 3（仍可用）、Midjourney v7、Imagen 4 |
| 图像编辑工作流 | Flux.1-Kontext（2024年12月）—— 原生支持图像+文本输入 |
| 学术研究、基线对比 | SD 1.5 —— 虽已陈旧但研究充分 |

## 部署上线

保存 `outputs/skill-sd-prompter.md`。该技能模块接收文本提示词与目标风格，并输出以下内容：模型及检查点、CFG（分类器自由引导，Classifier-Free Guidance）缩放系数、采样器（sampler）、负向提示词、分辨率、可选的 ControlNet/IP-Adapter（控制网络/图像适配器）组合，以及分步质量检查（QA）清单。

## 练习

1. **简单。** 使用引导系数 `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。按类别记录生成样本的均值。当 `w` 为何值时，类别均值会开始显著偏离真实数据的均值？
2. **中等。** 将简易的线性编码器替换为带有重建损失（reconstruction loss）的 tanh-MLP 编码器/解码器对。在新的潜在表示上重新训练扩散模型（diffusion model）。生成样本的质量是否会发生变化？
3. **困难。** 使用 diffusers 搭建真实的 Stable Diffusion 推理流程：加载 `sdxl-base`，设置 CFG=7 并运行 30 步 Euler 采样（Euler steps），记录推理耗时。随后切换至 `sdxl-turbo`，设置 4 步采样且 CFG=0。在相同主题下对比生成质量——请描述具体变化及其背后的原因。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 第一阶段 (First stage) | “VAE” | 经过训练的编码器/解码器对；将 512² 压缩至 64²。 |
| 第二阶段 (Second stage) | “U-Net” | 在潜空间 (latent space) 上运行的扩散模型 (diffusion model)。 |
| CFG | “引导尺度 (Guidance scale)” | `(1+w)·ε_cond - w·ε_uncond`；用于调节条件引导强度。 |
| 空标记 (Null token) | “空提示词嵌入 (Empty prompt embed)” | 用于 `ε_uncond` 的无条件嵌入 (unconditional embed)。 |
| 交叉注意力 (Cross-attention) | “文本如何注入” | U-Net 的每个模块将文本标记作为键 (K) 和值 (V) 进行注意力计算。 |
| DiT | “扩散 Transformer (Diffusion Transformer)” | 用处理潜块 (latent patches) 的 Transformer 替换 U-Net；扩展性更好。 |
| MMDiT | “多模态 DiT (Multi-modal DiT)” | SD3 的架构：文本与图像流通过联合注意力 (joint attention) 交互。 |
| VAE 缩放因子 (VAE scaling factor) | “魔法数字” | 将潜变量除以约 5.4，使扩散过程在单位方差空间中进行。 |

## 生产环境备注：在 8GB 消费级 GPU 上运行 Flux-12B

官方提供的 Flux 集成方案是解决“我只有消费级 GPU，能部署上线吗？”这一问题的标准范式。其核心技巧与生产环境推理文献中针对扩散 DiT (Diffusion Transformer) 列出的“三项核心参数调节”配方一致：

1. **分阶段加载 (Staggered loading)。** Flux 包含多个网络，它们无需同时驻留在显存 (VRAM) 中：T5-XXL 文本编码器（fp32 精度下约 10 GB）、CLIP-L（体积较小）、12B 参数的 MMDiT 以及 VAE。首先对提示词进行编码，*删除*编码器，加载 DiT，执行去噪，*删除* DiT，加载 VAE，最后解码。消费级 8GB GPU 每次仅能容纳一个阶段的模型。
2. **通过 bitsandbytes 进行 4 比特量化 (4-bit quantization)。** 对 T5 编码器和 DiT 均应用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。此举可将内存占用降低至原来的 1/8，且根据 Aritra 的基准测试（笔记本中已附链接），在文生图任务中画质下降几乎无法察觉。
3. **CPU 卸载 (CPU offload)。** `pipe.enable_model_cpu_offload()` 会在每次前向传播 (forward pass) 推进时，自动在 CPU 与 GPU 之间交换模块。这会增加 10-20% 的延迟，但能确保整个流水线得以运行。

显存占用核算如下：量化后的 T5 为 `10 GB T5 / 8 = 1.25 GB`，量化后的 DiT 为 `12 B params × 0.5 bytes = ~6 GB`，再加上激活值 (activations) 开销。用 stas00 的术语来说，这属于 TP=1 推理的极限配置——无模型并行 (model parallelism)，采用最大程度的量化。在生产环境中，你通常会在 H100 上运行 TP=2 或 TP=4；但对于单台开发笔记本而言，这就是标准方案。

## 延伸阅读

- [Rombach 等人 (2022)。基于潜在扩散模型 (Latent Diffusion Models) 的高分辨率图像合成](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell 等人 (2023)。SDXL：改进潜在扩散模型以实现高分辨率图像合成](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles 与 Xie (2023)。基于 Transformer 的可扩展扩散模型 (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser 等人 (2024)。扩展整流流 (Rectified Flow) Transformer 以实现高分辨率图像合成](https://arxiv.org/abs/2403.03206) — SD3、MMDiT。
- [Ho 与 Salimans (2022)。无分类器扩散引导 (Classifier-Free Diffusion Guidance)](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024)。Flux.1 — Black Forest Labs 公告](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 系列。
- [Hugging Face Diffusers 文档](https://huggingface.co/docs/diffusers/index) — 上述所有模型检查点 (checkpoint) 的参考实现。
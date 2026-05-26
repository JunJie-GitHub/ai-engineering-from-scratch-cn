# 扩散Transformer（Diffusion Transformer）与整流流（Rectified Flow）

> U-Net并非扩散模型的秘诀。将其替换为Transformer，将噪声调度（noise schedule）替换为直线流，你便瞬间拥有了SD3、FLUX以及所有2026年的文生图模型。

**Type:** 学习与实践
**Languages:** Python
**Prerequisites:** 第4阶段第10课（扩散DDPM）、第4阶段第14课（ViT）、第7阶段第02课（自注意力机制）
**Time:** 约75分钟

## 学习目标

- 梳理从U-Net DDPM（第10课）到扩散Transformer（DiT）、多模态扩散Transformer（MMDiT，用于SD3）以及单/双流DiT（single/double-stream DiT，用于FLUX）的演进路径
- 解释整流流（Rectified Flow）：为何噪声与数据之间的直线轨迹能让模型在20步而非1000步内完成采样
- 实现一个微型DiT模块与整流流训练循环，两者代码均不超过100行
- 根据架构、参数量与许可证区分不同模型变体（SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image、Qwen-Image）

## 问题背景

第10课构建了一个基于U-Net去噪器的去噪扩散概率模型（DDPM）。该方案在2020至2023年间占据主导地位：U-Net + beta调度（beta schedule） + 噪声预测损失（noise-prediction loss）。它催生了Stable Diffusion 1.5、2.1以及DALL-E 2。

截至2026年，所有最先进的文生图模型均已超越该架构。Stable Diffusion 3、FLUX、SD4、Z-Image、Qwen-Image、Hunyuan-Image——无一使用U-Net。它们均采用扩散Transformer（DiT）。SD3与FLUX还将DDPM的噪声调度替换为整流流（Rectified Flow），该机制拉直了从噪声到数据的生成路径，并结合一致性模型（consistency models）或蒸馏变体（distilled variants），实现了仅需1至4步的推理。

这一转变至关重要，因为它正是基于扩散的图像生成实现可控性、提示词精准度（SD3/SD4解决了文字渲染问题）以及生产级速度的根本原因。理解DiT与整流流，就是理解2026年生成式图像技术栈的核心。

## 核心概念

### 从 U-Net 到 Transformer

flowchart LR
    subgraph UNET["DDPM U-Net (2020)"]
        U1["Conv encoder"] --> U2["Conv bottleneck"] --> U3["Conv decoder"]
    end
    subgraph DIT["DiT (2023)"]
        D1["Patch embed"] --> D2["Transformer blocks"] --> D3["Unpatchify"]
    end
    subgraph MMDIT["MMDiT (SD3, 2024)"]
        M1["Text stream"] --> M3["Joint attention<br/>(separate weights per modality)"]
        M2["Image stream"] --> M3
    end
    subgraph FLUX["FLUX (2024)"]
        F1["Double-stream blocks<br/>(text + image separate)"] --> F2["Single-stream blocks<br/>(concat + shared weights)"]
    end

    style UNET fill:#e5e7eb,stroke:#6b7280
    style DIT fill:#dbeafe,stroke:#2563eb
    style MMDIT fill:#fef3c7,stroke:#d97706
    style FLUX fill:#dcfce7,stroke:#16a34a

- **DiT**（Peebles & Xie, 2023）—— 在潜在块（latent patches）上使用类似 ViT 的 Transformer 替换 U-Net。通过自适应层归一化（Adaptive Layer Normalization, AdaLN）进行条件注入。
- **MMDiT**（SD3, Esser 等, 2024）—— 采用双流架构，文本和图像 token 使用独立权重，但共享联合注意力（Joint Attention）机制。
- **FLUX**（Black Forest Labs, 2024）—— 前 N 个模块采用类似 SD3 的双流架构，后续模块为提升深层网络的计算效率，将特征拼接并共享权重（单流架构）。
- **Z-Image**（2025）—— 一款参数量为 60 亿的高效单流 DiT，挑战了“不惜一切代价扩大规模”的传统理念。

### 一段话理解整流（Rectified Flow）

去噪扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM）将前向过程定义为一个含噪随机微分方程（Stochastic Differential Equation, SDE），其中 `x_t` 的噪声逐渐增加。学习到的反向过程是另一个 SDE，通常通过 1000 个微小步长进行求解。

整流（Rectified Flow）在干净数据与纯噪声之间定义了一条**直线**插值路径：

x_t = (1 - t) * x_0 + t * epsilon,     t in [0, 1]

训练网络以预测速度 `v_theta(x_t, t) = epsilon - x_0`，即沿干净数据到噪声的直线路径的前进方向（`dx_t/dt`）。在采样阶段，通过反向积分该速度，逐步从噪声逼近数据。由此生成的常微分方程（Ordinary Differential Equation, ODE）轨迹更接近直线，因此采样所需的积分步数大幅减少。

SD3 将此方法称为**整流流匹配（Rectified Flow Matching）**。FLUX、Z-Image 以及大多数 2026 年的模型均采用相同的目标函数。典型推理过程：20-30 步欧拉法（确定性）对比旧版 DDPM 范式下的 50+ 步 DDIM。经过蒸馏（distilled）/ turbo / schnell / LCM 优化的变体可将步数进一步压缩至 1-4 步。

### AdaLN 条件注入

DiT 通过**自适应层归一化（Adaptive Layer Normalization, AdaLN）**对时间步和类别/文本进行条件注入：从条件向量中预测 `scale`（缩放）和 `shift`（平移）参数，并在 LayerNorm 之后应用。相比 U-Net 中 FiLM 风格的调制方式，该设计更为简洁，且已成为所有现代 DiT 的默认配置。

cond -> MLP -> (scale, shift, gate)
norm(x) * (1 + scale) + shift, then residual add * gate

### SD3 与 FLUX 中的文本编码器

- **SD3** 使用三个文本编码器：两个 CLIP 模型 + T5-XXL。将词嵌入（embeddings）拼接后，作为文本条件输入图像流。
- **FLUX** 使用一个 CLIP-L + T5-XXL。
- **Qwen-Image / Z-Image** 变体使用其自研的文本编码器，并与各自的基础大语言模型（Large Language Model, LLM）对齐。

文本编码器是 SD3/FLUX 在提示词（Prompt）理解与推理能力上远超 SD1.5 的关键原因之一。仅 T5-XXL 的参数量就高达 47 亿。

### 无分类器引导（Classifier-Free Guidance）依然适用

整流（Rectified Flow）改变的是采样器，而非条件注入机制。无分类器引导（Classifier-Free Guidance，训练时以 10% 的概率丢弃文本，推理时混合条件与无条件预测）在整流模型中同样有效。大多数 2026 年的模型将引导尺度（Guidance Scale）设为 3.5-5，低于 SD1.5 的 7.5，这是因为整流模型默认对提示词的遵循度更高。

### Consistency、Turbo、Schnell 与 LCM

这四个名称指向同一核心理念：将缓慢的多步模型蒸馏为快速的少步模型。

- **LCM（潜在一致性模型，Latent Consistency Model）**—— 训练一个学生模型，使其能够仅用一步从任意中间状态 `x_t` 预测出最终结果 `x_0`。
- **SDXL Turbo / FLUX schnell**—— 采用对抗扩散蒸馏（Adversarial Diffusion Distillation）训练的 1-4 步模型。
- **SD Turbo**—— 借鉴 OpenAI 风格的一致性模型（Consistency Models），并适配至潜在扩散架构。

任何新模型的生产部署都会同时提供“全质量”检查点（Checkpoint）和“turbo / schnell”变体。Schnell（德语意为“快速”，为 Black Forest Labs 的命名惯例）仅需 1-4 步即可运行，非常适合实时处理流水线。

### 2026 年模型生态概览

| 模型 | 参数量 | 架构 | 许可证 |
|-------|------|--------------|---------|
| Stable Diffusion 3 Medium | 20 亿 | MMDiT | SAI 社区协议 |
| Stable Diffusion 3.5 Large | 80 亿 | MMDiT | SAI 社区协议 |
| FLUX.1-dev | 120 亿 | 双流 + 单流 DiT | 非商业 |
| FLUX.1-schnell | 120 亿 | 同上，经蒸馏优化 | Apache 2.0 |
| FLUX.2 | — | FLUX.1 迭代版 | 混合协议 |
| Z-Image | 60 亿 | S3-DiT（可扩展单流） | 宽松协议 |
| Qwen-Image | 约 200 亿 | DiT + Qwen 文本塔 | Apache 2.0 |
| Hunyuan-Image-3.0 | 约 800 亿 | DiT | 研究用途 |
| SD4 Turbo | 30 亿 | DiT + 蒸馏 | SAI 商业协议 |

FLUX.1-schnell 是 2026 年开源领域的默认选择。Z-Image 在效率方面处于领先地位。FLUX.2 和 SD4 则代表了当前的质量巅峰。

### 为何这一范式转变至关重要

DDPM + U-Net 曾行之有效。而 DiT + 整流（Rectified Flow）则**效果更好、速度更快、且扩展性更优**。这一转变与自然语言处理领域从循环神经网络（Recurrent Neural Network, RNN）到 Transformer 的演进如出一辙：两种架构虽能解决相同问题，但 Transformer 具备更强的扩展能力并已成为主流。2026 年所有关于图像、视频或 3D 生成的论文均采用 DiT 结构的去噪器（Denoiser），且大多使用整流目标函数。U-Net DDPM 如今主要仅用于教学演示（如第 10 课）。

## 构建

### 步骤 1：带有自适应层归一化（AdaLN）的扩散变换器（DiT）模块

import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm with a gate. Predicts (scale, shift, gate) from the conditioning.
    Init such that the whole block starts as identity ("zero init").
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.mlp(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h, gate.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4, cond_dim=192):
        super().__init__()
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln2 = AdaLNZero(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        h, gate1 = self.adaln1(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * a
        h, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.mlp(h)
        return x

`AdaLNZero` 初始时表现为恒等映射（identity mapping），因为其多层感知机（MLP）的权重被初始化为零。训练过程会逐渐使该模块偏离恒等映射；这一机制能显著提升深层变换器扩散模型（transformer diffusion models）的稳定性。

### 步骤 2：微型 DiT

def timestep_embedding(t, dim):
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TinyDiT(nn.Module):
    def __init__(self, image_size=16, patch_size=2, in_channels=3, dim=96, depth=4, heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, cond_dim=dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        n = x.size(0)
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) + self.pos
        t_emb = self.time_mlp(timestep_embedding(t, self.pos.size(-1)))
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.norm_out(x)
        x = self.head(x)
        return self._unpatchify(x, n)

    def _unpatchify(self, x, n):
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        x = x.view(n, h, w, p, p, -1).permute(0, 5, 1, 3, 2, 4).reshape(n, -1, h * p, w * p)
        return x

### 步骤 3：整流流（Rectified Flow）训练

import torch.nn.functional as F

def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon

    target_velocity = epsilon - x0
    pred_velocity = model(x_t, t)

    loss = F.mse_loss(pred_velocity, target_velocity)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

与去噪扩散概率模型（DDPM）的噪声预测损失（noise-prediction loss，第 10 课）相比：两者结构相同，但优化目标不同。我们不再预测噪声 `epsilon`，而是预测**速度**（velocity）`epsilon - x_0`，该向量沿着直线插值（straight-line interpolation）路径从数据指向噪声。

### 步骤 4：欧拉采样器（Euler sampler）

整流流（Rectified Flow）本质上是一个常微分方程（ODE）。欧拉法（Euler's method）是最简单的求解器，对于训练良好的整流流模型而言，在 20 步以上的采样中，其精度几乎与高阶求解器（higher-order solvers）相当。

@torch.no_grad()
def rectified_flow_sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x

仅需 20 步。在训练好的模型上，其生成的样本质量可与 1000 步的 DDPM 相媲美。

### 步骤 5：端到端冒烟测试（smoke test）

import numpy as np

def synthetic_blobs(num=200, size=16, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((num, 3, size, size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for i in range(num):
        cx, cy = rng.uniform(4, size - 4, size=2)
        r = rng.uniform(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        colour = rng.uniform(-1, 1, size=3)
        for c in range(3):
            out[i, c][mask] = colour[c]
    return torch.from_numpy(out)

使用整流流在此数据集上训练 `TinyDiT`。经过 500 步训练后，采样输出的结果应呈现为淡淡的彩色色块。

## 使用它

对于使用 FLUX / SD3 / Z-Image 进行真实图像生成，`diffusers` 库为它们均提供了统一的 API：

from diffusers import FluxPipeline, StableDiffusion3Pipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

out = pipe(
    prompt="a golden retriever surfing a tsunami, hyperrealistic, studio lighting",
    guidance_scale=0.0,           # schnell was trained without CFG
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("surf.png")

仅需三行代码。`FLUX.1-schnell` 模型只需四步即可生成。若需更高质量的输出，可将模型 ID 替换为 `black-forest-labs/FLUX.1-dev`，并在启用分类器自由引导（Classifier-Free Guidance, CFG）的情况下运行 20-30 步。

对于 SD3：

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
).to("cuda")
out = pipe(prompt, guidance_scale=3.5, num_inference_steps=28).images[0]

## 交付内容

本章节将生成以下文件：

- `outputs/prompt-dit-model-picker.md` — 根据质量、延迟和许可证限制，在 SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image 和 SD4 Turbo 之间进行选择。
- `outputs/skill-rectified-flow-trainer.md` — 编写一个完整的整流流（Rectified Flow）训练循环，结合自适应层归一化扩散变换器（AdaLN DiT）与欧拉采样（Euler Sampling）。

## 练习

1. **(简单)** 使用上述的 TinyDiT 在合成斑点数据集（synthetic blob dataset）上训练 500 步。对比使用 10、20 和 50 步欧拉步数（Euler steps）生成的样本。
2. **(中等)** 通过将学习到的类别嵌入（class embedding）与时间嵌入（time embedding）拼接来添加文本条件控制（text conditioning）（10 个按颜色区分的斑点“类别”）。分别使用类别 0、5 和 9 进行采样，并验证生成的颜色是否匹配。
3. **(困难)** 计算在相同数据上以相同步数训练的同等规模网络中，整流流（Rectified Flow）版本与去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）版本生成样本之间的弗雷歇距离（Fréchet Distance，作为 FID 代理指标）。报告哪种方法收敛更快。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 扩散 Transformer（DiT） | "Diffusion transformer" | 使用 Transformer 替代 U-Net 作为扩散去噪器；在分块处理的潜在表示（patchified latents）上进行运算 |
| 自适应层归一化（AdaLN） | "Adaptive layer norm" | 通过在层归一化（LayerNorm）后应用学习到的缩放、平移和门控参数来实现时间步/文本条件注入；已成为所有现代 DiT 的标准配置 |
| 多模态 DiT（MMDiT） | "Multi-modal DiT (SD3)" | 文本与图像词元（Token）拥有独立的权重流，但共享联合自注意力（joint self-attention）机制 |
| 单流 / 双流架构（Single-stream / double-stream） | "FLUX trick" | 前 N 个模块采用双流架构（各模态独立权重），后续模块为提升效率转为单流架构（拼接后共享权重） |
| 整流流（Rectified Flow） | "Straight-line noise-to-data" | 数据与噪声之间的线性插值；网络预测速度场（velocity）；推理时所需的常微分方程（ODE）求解步数更少 |
| 速度目标（Velocity Target） | "epsilon - x_0" | 整流流中的回归目标；表示从干净数据指向噪声的方向 |
| 无分类器引导（CFG Guidance） | "classifier-free guidance" | 混合条件预测与无条件预测的结果；在整流流模型中依然广泛使用 |
| Schnell / turbo / LCM | "1-4 step distillation" | 由完整质量模型蒸馏而来的少步数变体；适用于生产级实时生成 |

## 延伸阅读

- [基于 Transformer 的可扩展扩散模型 (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) — DiT 的奠基论文
- [扩展整流流 Transformer (Esser 等, SD3 论文)](https://arxiv.org/abs/2403.03206) — 大规模应用 MMDiT 与整流流技术
- [FLUX.1 模型卡片与技术报告 (Black Forest Labs)](https://huggingface.co/black-forest-labs/FLUX.1-dev) — 双流与单流架构的详细说明
- [Z-Image：高效图像生成基础模型 (2025)](https://arxiv.org/html/2511.22699v1) — 60 亿参数规模的单流 DiT
- [阐明扩散模型的设计空间 (Karras 等, 2022)](https://arxiv.org/abs/2206.00364) — 探讨扩散模型各项设计权衡的权威参考
- [潜在一致性模型 (Luo 等, 2023)](https://arxiv.org/abs/2310.04378) — 详解 LCM-LoRA 如何实现 4 步推理
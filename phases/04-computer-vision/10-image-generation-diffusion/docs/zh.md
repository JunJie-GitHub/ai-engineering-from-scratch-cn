# 图像生成 — 扩散模型 (Diffusion Models)

> 扩散模型 (Diffusion Model) 学习的是去噪过程。训练它从带噪图像中去除少量噪声，将这一过程反向重复上千次，你就得到了一个图像生成器。

**类型：** 构建
**语言：** Python
**前置知识：** 第4阶段第07课（U-Net）、第1阶段第06课（概率论）、第3阶段第06课（优化器）
**时长：** 约75分钟

## 学习目标

- 推导前向加噪过程 `x_0 -> x_1 -> ... -> x_T`，并解释为何闭式解 `q(x_t | x_0)` 对任意 t 均成立
- 实现去噪扩散概率模型 (DDPM) 风格的训练目标，通过回归每一步添加的噪声进行优化，并实现一个从纯噪声逐步反向生成图像的采样器 (Sampler)
- 构建一个时间条件化 U-Net（规模足够小，可在 CPU 上训练），用于预测任意时间步 (Timestep) 的噪声
- 解释 DDPM 与去噪扩散隐式模型 (DDIM) 采样之间的区别及其适用场景（第23课将深入讲解流匹配 (Flow Matching) 与整流流 (Rectified Flow)）

## 问题背景

生成对抗网络 (GAN) 采用单次生成 (One-shot) 模式：输入噪声，经过一次前向传播 (Forward Pass) 即可输出图像。它们生成速度快，但训练难度极高。扩散模型则采用迭代生成模式：从纯噪声出发，通过多个小步长逐步去噪，最终呈现出图像。它们生成速度较慢，但训练相对容易。在过去五年中，后者的优势占据了主导地位：任何小型团队都能训练出一个扩散模型并获得质量合理的样本；而 GAN 的训练则更像是一门手艺，需要历经数年失败实验的打磨才能掌握。

除了训练稳定性之外，扩散模型的迭代结构正是解锁现代图像生成所有能力的关键：文本条件控制 (Text Conditioning)、图像修复 (Inpainting)、图像编辑、超分辨率重建 (Super-resolution) 以及可控风格生成。采样循环 (Sampling Loop) 中的每一步都可以作为注入新约束的切入点。正是这种“钩子”机制，使得 Stable Diffusion、Imagen、DALL-E 3、Midjourney 以及你未来将使用的所有可控图像模型，无一例外都基于扩散架构。

本课将构建一个最简化的 DDPM 模型，涵盖前向加噪、反向去噪以及训练循环。下一课（Stable Diffusion）将在此基础上，结合变分自编码器 (VAE)、文本编码器 (Text Encoder) 以及无分类器引导 (Classifier-Free Guidance)，将其接入生产级系统。

## 核心概念

### 前向过程 (Forward Process)

给定一张图像 `x_0`。向其添加微量的高斯噪声 (Gaussian noise) 得到 `x_1`。再添加微量噪声得到 `x_2`。重复此过程 T 步，直到 `x_T` 与纯高斯噪声几乎无法区分。

q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)

`beta_t` 是一个较小的方差调度 (variance schedule)，通常在 T=1000 步内从 0.0001 线性增加到 0.02。每一步都会略微衰减信号并注入新的噪声。

### 闭式跳跃 (Closed-form Jump)

逐步添加噪声构成一个马尔可夫链 (Markov chain)，但数学上可以将其折叠：你可以直接从 `x_0` 一步采样得到 `x_t`。

Define alpha_t = 1 - beta_t
Define alpha_bar_t = prod_{s=1..t} alpha_s

Then:
  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I)

Equivalently:
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  where epsilon ~ N(0, I)

正是这一个公式让扩散模型 (Diffusion Model) 具备了实际可行性。在训练过程中，你只需随机选择一个时间步 `t`，直接从 `x_0` 采样得到 `x_t`，并一步完成训练——无需模拟完整的马尔可夫链。

### 反向过程 (Reverse Process)

前向过程是固定的。神经网络学习的是反向过程 `p(x_{t-1} | x_t)`。扩散模型并不直接预测 `x_{t-1}`；而是预测在第 t 步添加的噪声 `epsilon`，然后通过数学公式从中推导出 `x_{t-1}`。

flowchart LR
    X0["x_0<br/>(clean image)"] --> Q1["q(x_t|x_0)<br/>add noise"]
    Q1 --> XT["x_t<br/>(noisy)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["predicted epsilon"]
    EPS --> LOSS["MSE against<br/>true epsilon"]

    XT -.->|sampling| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|repeat 1000x| X0S["x_0 (sampled)"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb

### 训练损失 (Training Loss)

对于每一个训练步骤：

1. 采样一张真实图像 `x_0`。
2. 从 [1, T] 区间内均匀采样一个时间步 `t`。
3. 采样噪声 `epsilon ~ N(0, I)`。
4. 计算 `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`。
5. 使用网络预测 `epsilon_theta(x_t, t)`。
6. 最小化 `|| epsilon - epsilon_theta(x_t, t) ||^2`。

就是这么简单。神经网络学习预测任意时间步的噪声。损失函数为均方误差 (Mean Squared Error, MSE)。这里没有对抗博弈 (Adversarial Game)，没有模式崩溃 (Mode Collapse)，也没有训练振荡。

### 采样器 (Sampler) (DDPM)

生成过程：从 `x_T ~ N(0, I)` 开始，逐步反向迭代。

for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    where z ~ N(0, I) if t > 1, else 0
return x_0

关键在于，尽管一般情况下反向条件概率没有闭式解，但针对这种特定的高斯前向过程，它是存在的。那些看起来复杂的系数正是贝叶斯法则 (Bayes' Rule) 推导出的结果。

### 为什么是 1000 步

前向噪声调度 (Noise Schedule) 的设计目标是让每一步添加的噪声量恰好使反向步骤近似服从高斯分布。步数太少会导致反向步骤严重偏离高斯分布，网络难以有效建模；步数太多则会使采样成本急剧上升，而收益递减。采用线性调度且 T=1000 是 DDPM 的默认设置。

### DDIM：20 倍加速采样

训练过程保持不变，仅改变采样方式。DDIM (Song et al., 2020) 定义了一种确定性的反向过程，无需重新训练即可跳过部分时间步。使用 DDIM 仅需 50 步采样，即可达到接近 DDPM 1000 步的生成质量。目前所有生产环境均采用 DDIM 或更快的变体（如 DPM-Solver、Euler ancestral）。

### 时间条件注入 (Time Conditioning)

网络 `epsilon_theta(x_t, t)` 需要知道当前正在对哪个时间步进行去噪。现代扩散模型通过正弦时间嵌入 (Sinusoidal Time Embeddings) 注入 `t`（其原理与 Transformer 中的位置编码 (Positional Encoding) 相同），并在 U-Net 的每一层将其加到特征图 (Feature Maps) 中。

t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)

若不进行时间条件注入，网络只能从图像本身推测噪声水平，虽然可行，但样本效率 (Sample Efficiency) 会大幅降低。

## 构建

### 步骤 1：噪声调度（Noise Schedule）

import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))

只需预先计算一次，在训练和采样阶段通过索引获取即可。

### 步骤 2：前向扩散（Forward Diffusion）（q_sample）

def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise

采用单行闭式解（Closed-form）。`t` 表示一个时间步（Timestep）批次，批次中的每张图像对应一个时间步。

### 步骤 3：轻量级时间条件 U-Net（Time-conditioned U-Net）

import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)

这是一个两级 U-Net 架构，时间条件（Time Conditioning）注入在瓶颈层（Bottleneck）。处理真实图像时，需相应增加网络的深度和宽度。

### 步骤 4：训练循环（Training Loop）

def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

这就是完整的训练循环。无需生成对抗网络（GAN）的博弈，无需设计特殊损失函数，仅需调用一次均方误差（MSE）损失。

### 步骤 5：采样器（Sampler）（DDPM）

@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x

生成一个批次的样本需要进行 1000 次前向传播（Forward Pass）。在实际代码中，通常会将其替换为 50 步的 DDIM 采样器。

### 步骤 6：DDIM 采样器（确定性，速度提升约 20 倍）

@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x

当 `eta=0` 时为完全确定性（Deterministic）过程（相同的噪声输入始终产生相同的输出）。当 `eta=1` 时则退化为标准的 DDPM。

## 使用方法

在生产环境中，请使用 `diffusers`：

from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)

该库内置了现成的调度器（Scheduler，如 DDPM、DDIM、DPM-Solver、Euler、Heun）、可配置的 U-Net 模型、文本生成图像（Text-to-Image）与图像生成图像（Image-to-Image）的流水线（Pipeline），以及 LoRA 微调（LoRA Fine-tuning）辅助工具。

对于学术研究，`k-diffusion`（由 Katherine Crowson 开发）提供了最权威的参考实现以及最优质的采样变体。

## 交付成果

本课时将产出以下内容：

- `outputs/prompt-diffusion-sampler-picker.md` — 一个提示词（Prompt），可根据质量目标、延迟预算和条件类型（Conditioning Type）自动选择 DDPM / DDIM / DPM-Solver / Euler 采样器。
- `outputs/skill-noise-schedule-designer.md` — 一项技能（Skill），可根据总步数 T 和目标破坏程度生成线性、余弦或 Sigmoid 的 Beta 调度（Beta Schedule），并附带信噪比（Signal-to-Noise Ratio）随时间变化的诊断图表。

## 练习

1. **(简单)** 可视化前向过程（Forward Process）：选取一张图像，绘制 `t in [0, 100, 250, 500, 750, 1000]` 时的 `x_t`。验证 `x_1000` 是否呈现为纯高斯噪声（Gaussian Noise）。
2. **(中等)** 在 synthetic-circles 数据集上训练 TinyUNet 模型 20 个 Epoch，并采样生成 16 个圆形。对比 DDPM（1000 步）与 DDIM（50 步）的采样结果——在相同的噪声种子（Noise Seed）下，它们生成的图像是否相似？
3. **(困难)** 实现余弦噪声调度（Cosine Noise Schedule，Nichol & Dhariwal, 2021）：`alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`。分别使用线性和余弦调度训练同一模型，并证明在步数较少时，余弦调度能生成质量更好的样本。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 前向过程（Forward Process） | “随时间添加噪声” | 固定的马尔可夫链（Markov Chain），在 T 步内将图像逐步破坏为高斯噪声 |
| 反向过程（Reverse Process） | “逐步去噪” | 学习到的概率分布，实现从噪声到图像的逆向生成 |
| 噪声预测（Epsilon Prediction） | “预测噪声” | 训练目标：`epsilon_theta(x_t, t)` 用于预测第 t 步所添加的噪声 |
| Beta 调度（Beta Schedule） | “噪声量” | 包含 T 个微小方差的序列，用于定义每一步引入的噪声量 |
| alpha_bar_t | “累积保留因子” | 截至时间 t 的 `(1 - beta_s)` 连乘积；t 越大，剩余信号越少 |
| DDPM 采样器（DDPM Sampler） | “祖先采样、随机过程” | 从条件高斯分布中采样每个 `x_{t-1}`；需 1000 步 |
| DDIM 采样器（DDIM Sampler） | “确定性、快速” | 将采样过程重写为确定性常微分方程（ODE）；20-100 步即可达到相近质量 |
| 时间条件注入（Time Conditioning） | “告诉模型当前是第几步” | 将 t 的正弦嵌入（Sinusoidal Embedding）注入 U-Net，使其感知当前噪声水平 |

## 延伸阅读

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) — 使扩散模型（Diffusion Models）走向实用化，并在 FID（Fréchet Inception Distance）指标上超越生成对抗网络（GANs）的论文
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) — 引入余弦调度（cosine schedule）与 v 参数化（v-parameterisation）
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502) — 采用确定性采样器（deterministic sampler），使实时推理（real-time inference）成为可能
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) — 对扩散模型各项设计选择提供统一视角的综述；当前最佳参考资料
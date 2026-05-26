# 自编码器 (Autoencoder) 与变分自编码器 (Variational Autoencoder, VAE)

> 普通自编码器仅执行压缩与重建。它擅长记忆，而非生成。只需加入一个技巧——强制潜在编码服从高斯分布——你就能得到一个采样器。正是这唯一的技巧，即重参数化 (Reparameterization) 公式 `z = μ + σ·ε`，使得你在 2026 年使用的每一个潜在扩散模型 (Latent Diffusion Model) 和流匹配 (Flow-Matching) 图像模型都在输入端配备了 VAE。

**类型:** 构建实践
**语言:** Python
**前置知识:** 第 3 阶段 · 02 (反向传播 Backprop), 第 3 阶段 · 07 (卷积神经网络 CNNs), 第 8 阶段 · 01 (模型分类 Taxonomy)
**预计耗时:** 约 75 分钟

## 问题背景

将一张 784 像素的 MNIST 手写数字图像压缩为 16 维编码，再进行重建。普通自编码器能在重建均方误差 (Mean Squared Error, MSE) 上表现优异，但其编码空间 (Code Space) 却杂乱无章。若在编码空间中随机选取一点并进行解码，得到的只会是噪声。它不具备采样能力，本质上只是一个披着外衣的压缩模型。

你真正需要的是：(a) 编码空间是一个干净、平滑且可采样的分布——例如各向同性高斯分布 (Isotropic Gaussian) `N(0, I)`；(b) 解码任意采样点都能生成合理的数字图像；(c) 编码器与解码器仍保持良好的压缩性能。三个目标，一套架构，一个损失函数。

Kingma 于 2013 年提出的 VAE 通过训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)` 解决了该问题。它利用 KL 散度惩罚 (KL Penalty) 将该分布拉向先验分布 `N(0, I)`，随后在解码前从 `q(z|x)` 中采样 `z`。在推理阶段，直接舍弃编码器，从 `z ~ N(0, I)` 采样并解码即可。正是 KL 惩罚项强制编码空间具备了结构化特性。

到了 2026 年，VAE 已极少单独部署——在原始图像生成质量上它已被扩散模型 (Diffusion Model) 超越——但它仍是所有潜在扩散模型（如 SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。掌握 VAE，就等于掌握了你所使用的每一个图像生成管线中不可见的第一层。

## 核心概念

![自编码器与 VAE 对比：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`, `x̂ = decoder(z)`, 损失 = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们共同定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧 (Reparameterization Trick)。** 直接从 `q(z|x)` 采样是不可微的。将采样过程重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。此时 `z` 变为 `(μ, σ)` 的确定性函数加上一个非参数噪声——梯度得以顺利流经 `μ` 和 `σ`。

**损失函数。** 证据下界 (Evidence Lower Bound, ELBO)，包含两项：

loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2

重建项推动 `x̂` 逼近 `x`。KL 项推动 `q(z|x)` 逼近先验分布。两者相互权衡。较小的 β (<1) 意味着采样更清晰，但编码空间偏离高斯分布；较大的 β (>1) 意味着编码空间更规整，但采样结果更模糊。β-VAE (Higgins, 2017) 让这一调节旋钮声名大噪，并由此开启了特征解耦 (Disentanglement) 研究。

**采样。** 推理阶段：从 `z ~ N(0, I)` 中采样，输入解码器进行前向传播。仅需一次前向传播——无需像扩散模型那样进行迭代采样。

## 动手实践

`code/main.py` 实现了一个不依赖 numpy 或 torch 的微型 VAE（变分自编码器，Variational Autoencoder）。输入数据为 8 维合成数据，采样自 8 维空间中的双分量高斯混合模型（Gaussian Mixture Model）。编码器（Encoder）与解码器（Decoder）均为单隐藏层的多层感知机（MLP）。代码中实现了 tanh 激活函数、前向传播（forward pass）、损失计算以及手写的反向传播（backward pass）。本代码仅供教学演示，非生产级实现。

### 步骤 1：编码器前向传播

def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2

使用 `log σ²` 而非直接输出 `σ`，可使网络输出不受约束（对 σ 使用 softplus 激活函数是一个常见陷阱——当 σ ≈ 0 时会导致梯度消失）。

### 步骤 2：重参数化与解码

def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])

### 步骤 3：ELBO（证据下界，Evidence Lower Bound）

def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl

由于先验与后验分布均为高斯分布，此处可直接使用精确的闭式 KL 散度（Kullback-Leibler Divergence）计算。切勿采用数值积分。即便到了 2026 年，仍有人发布使用蒙特卡洛（Monte Carlo）方法估计 KL 散度的代码——这毫无必要地使计算速度降低了 3 倍。

### 步骤 4：生成

def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)

这就是完整的生成模型。仅需五行代码。

## 常见陷阱

- **后验坍缩（Posterior Collapse）。** KL 项会过于激进地迫使 `q(z|x) → N(0, I)`，导致隐变量 `z` 完全不包含关于输入 `x` 的信息。解决方法：采用 β 退火（β-annealing，初始 β=0，逐渐增加至 1）、自由比特（free bits）技巧，或在非活跃维度上跳过 KL 计算。
- **生成样本模糊。** 高斯解码器的似然函数隐含了均方误差（Mean Squared Error, MSE）重建，这在 L2 损失下是贝叶斯最优的（即求均值）——一组合理数字的均值必然是一个模糊的数字。解决方法：使用离散化解码器（如 VQ-VAE、NVAE），或仅将 VAE 用作编码器，并在隐空间上堆叠扩散模型（Diffusion Model）（Stable Diffusion 正是采用此架构）。
- **β 值过大或过早引入。** 参见后验坍缩。建议从 β≈0.01 开始，并逐步增加。
- **隐空间维度（Latent Dimension）过小。** 16 维适用于 MNIST，256 维适用于 256² 分辨率的 ImageNet，2048 维适用于 1024² 分辨率的 ImageNet。Stable Diffusion 的 VAE 将 512×512×3 压缩至 64×64×4（空间面积下采样 32 倍，通道维度压缩 32 倍）。

## 如何使用

2026 年 VAE（变分自编码器，Variational Autoencoder）技术栈：

| 应用场景 | 推荐选择 |
|-----------|------|
| 扩散模型的图像潜在空间编码器 | Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE |
| 音频潜在空间编码器 | Encodec (Meta)、SoundStream 或 DAC (Descript) |
| 视频潜在变量 | Sora 的时空块（spatiotemporal patches）、Latte VAE、WAN VAE |
| 解耦表示学习（Disentangled representation learning） | β-VAE、FactorVAE、TCVAE |
| 离散潜在变量（用于 Transformer 建模） | VQ-VAE、RVQ（残差向量量化，ResidualVQ） |
| 用于生成的连续潜在变量 | 基础 VAE（Plain VAE），随后在该潜在空间中条件化流模型（flow model）或扩散模型（diffusion model） |

潜在扩散模型（Latent Diffusion Model）本质上是在 VAE 的编码器（encoder）与解码器（decoder）之间嵌入了一个扩散模型。VAE 负责粗粒度压缩，而扩散模型承担核心的生成任务。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen Transformer）也采用相同的架构范式。

## 交付上线

保存至 `outputs/skill-vae-trainer.md`。

Skill 接收以下输入：数据集概况 + 潜在维度目标 + 下游用途（重建、采样或作为潜在扩散模型的输入），并输出以下配置：架构选择（基础/β/VQ/RVQ）、β 调度策略、潜在维度、解码器似然函数（高斯分布 vs 分类分布）以及评估方案（重建 MSE、逐维 KL 散度、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 将 `code/main.py` 中的 `β` 修改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终的重建 MSE 和 KL 散度。对于你的合成数据，哪个 β 值能达到帕累托最优（Pareto-best）？
2. **中等。** 将解码器的高斯似然函数替换为伯努利似然函数（交叉熵损失）。在相同合成数据的二值化版本上对比生成样本的质量。
3. **困难。** 将 `code/main.py` 扩展为一个迷你 VQ-VAE：将连续的 `z` 替换为在包含 K=32 个条目的码本（codebook）中进行最近邻查找。对比重建 MSE，并报告实际被使用的码本条目数量（码本崩溃/codebook collapse 是真实存在的现象）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 自编码器（Autoencoder） | 编解码网络 | `x → z → x̂`，学习 MSE。不具备生成能力。 |
| 变分自编码器（VAE） | 带采样器的 AE | 编码器输出分布，KL 惩罚项塑造潜在空间。 |
| 证据下界（ELBO） | 证据下界 | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`；当 `q = p(z|x)` 时边界最紧。 |
| 重参数化技巧（Reparameterization Trick） | `z = μ + σ·ε` | 将随机节点重写为确定性部分加纯噪声。使梯度可通过采样过程反向传播。 |
| 先验分布（Prior） | `p(z)` | 潜在变量的目标分布，通常为 `N(0, I)`。 |
| 后验崩溃（Posterior Collapse） | “KL 项占主导” | 编码器忽略输入 `x`，直接输出先验分布；解码器只能盲目生成（hallucinate）。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。β 越高，解耦程度越好，但生成结果越模糊。 |
| VQ-VAE（向量量化变分自编码器） | 离散潜在变量 | 用码本中最近的向量替换连续的 `z`；便于使用 Transformer 进行建模。 |

## 生产环境提示：在扩散模型服务器中，VAE 是最关键的性能热点路径（hottest path）

在 Stable Diffusion / Flux / SD3 流水线（pipeline）中，变分自编码器（Variational Autoencoder, VAE）每次请求会被调用两次——一次用于编码（encode，若执行图生图 img2img / 局部重绘 inpainting），另一次用于解码（decode）。在 1024² 分辨率下，解码器（decoder）的前向传播通常是整个流水线中激活值内存峰值（activation-memory peak）最高的环节，因为它需要将 `128×128×16` 的潜变量（latents）上采样回 `1024×1024×3`。这带来两个实际影响：

- **对解码过程进行切片或分块（Slice or tile the decode）。** `diffusers` 库提供了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()` 接口。分块（Tiling）策略会以微小的拼接伪影（seam artifact）为代价，将内存占用从 `O(H·W)` 降至 `O(tile²)`。这对于在消费级 GPU（consumer GPUs）上运行 1024² 及以上分辨率至关重要。
- **解码器使用 bf16，最终缩放步骤使用 fp32 数值精度。** SD 1.x 的 VAE 最初以 fp32 发布，若在 1024²+ 分辨率下强制转换为 fp16，会*静默产生 NaN（非数字）值*。SDXL 官方提供了 `madebyollin/sdxl-vae-fp16-fix` 模型——请始终优先使用 fp16-fix 版本或直接采用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 的奠基论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解耦 β-VAE（disentangled β-VAE）。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — 矢量量化变分自编码器（VQ-VAE）。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 业界领先（state-of-the-art）的图像 VAE 模型。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；将 VAE 用作编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 的标准模型。
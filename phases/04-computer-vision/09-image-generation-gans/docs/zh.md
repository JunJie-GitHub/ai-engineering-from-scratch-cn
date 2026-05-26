# 图像生成 — 生成对抗网络（GAN）

> GAN 由两个神经网络组成，它们处于一场固定规则的博弈中。一个负责生成，一个负责判别。两者在协同训练中不断优化，直到生成的样本足以骗过判别器。

**类型：** 构建
**语言：** Python
**前置知识：** 第4阶段第03课（卷积神经网络 CNNs）、第3阶段第06课（优化器 Optimizers）、第3阶段第07课（正则化 Regularization）
**时长：** 约75分钟

## 学习目标

- 解释生成器（generator）与判别器（discriminator）之间的极小极大博弈（minimax game），并说明为何其均衡状态对应于 p_model = p_data
- 使用 PyTorch 实现深度卷积生成对抗网络（DCGAN），并在 60 行代码内使其生成结构连贯的 32x32 合成图像
- 运用三种标准技巧稳定 GAN 训练：非饱和损失（non-saturating loss）、谱归一化（spectral norm）以及双时间尺度更新规则（TTUR）
- 解读训练曲线，以区分健康收敛与模式崩溃（mode collapse）、训练振荡（oscillation）以及判别器完全胜出（discriminator-wins-completely）等异常情况

## 问题背景

分类任务教会网络将图像映射到标签。生成任务则将该问题反转：采样出看似来自同一分布的新图像。这里没有可供对比的“标准”输出，只有你需要模仿的目标分布。

标准的损失函数（如均方误差 MSE、交叉熵 cross-entropy）无法衡量“该样本是否来自真实分布”。最小化逐像素误差只会产生模糊的平均图像，而非逼真的样本。突破性的思路在于“学习损失函数”：训练第二个网络来负责区分真假样本，并利用其判别结果来驱动生成器优化。

生成对抗网络（GANs，Goodfellow 等人，2014）确立了这一框架。到 2018 年，StyleGAN 已能生成与真实照片难以区分的 1024x1024 人脸图像。此后，扩散模型（Diffusion models）在图像质量与可控性方面占据了主导地位，但让扩散模型得以实用的每一项技巧——归一化（normalisation）策略的选择、潜在空间（latent spaces）的构建、特征损失（feature losses）的设计——最初都是在 GAN 的研究中被深入理解的。

## 核心概念

### 两个网络

flowchart LR
    Z["z ~ N(0, I)<br/>noise"] --> G["Generator<br/>transposed convs"]
    G --> FAKE["Fake image"]
    REAL["Real image"] --> D["Discriminator<br/>conv classifier"]
    FAKE --> D
    D --> OUT["P(real)"]

    style G fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a

**生成器（Generator）** G 接收一个噪声向量 `z` 并输出一张图像。**判别器（Discriminator）** D 接收一张图像并输出一个标量：该图像为真实图像的概率。

### 对抗博弈

G 的目标是让 D 判断错误，而 D 的目标是做出正确判断。形式化表达如下：

min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]

从右向左解读：D 旨在最大化对真实图像（`log D(real)`）和伪造图像（`log (1 - D(fake))`）的分类准确率。G 则旨在最小化 D 对伪造图像的准确率——它希望 `D(G(z))` 的值尽可能高。

Goodfellow 证明了该极小极大（minimax）博弈存在一个全局均衡点，此时 `p_G = p_data`，D 在所有输入上的输出均为 0.5，且生成分布与真实分布之间的 Jensen-Shannon 散度（Jensen-Shannon divergence）为零。真正的难点在于如何达到这一状态。

### 非饱和损失（Non-saturating loss）

上述公式在数值上不稳定。在训练初期，对于所有伪造样本，`D(G(z))` 的值都接近于零，导致 `log(1 - D(G(z)))` 相对于 G 的梯度消失（vanishing gradients）。解决方法是：翻转 G 的损失函数。

L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
L_G = -E_z[log D(G(z))]                          # non-saturating

此时，当 `D(G(z))` 接近零时，G 的损失值会很大，且其梯度能提供有效的更新方向。所有现代生成对抗网络（GAN）均采用此变体进行训练。

### DCGAN 架构规则

Radford、Metz 和 Chintala（2015）从多年的失败实验中提炼出五条规则，使 GAN 的训练趋于稳定：

1. 使用步幅卷积（strided convolutions）替代池化层（适用于两个网络）。
2. 在生成器和判别器中均使用批归一化（batch normalization），但排除 G 的输出层和 D 的输入层。
3. 在更深的网络架构中移除全连接层。
4. G 的所有层均使用 ReLU 激活函数，输出层除外（输出层使用 tanh 将值限制在 [-1, 1] 区间）。
5. D 的所有层均使用 LeakyReLU 激活函数（`negative_slope=0.2`）。

如今所有基于卷积的 GAN（如 StyleGAN、BigGAN、GigaGAN）依然以这些规则为基础，并在此基础上逐一替换组件。

### 失败模式及其特征

flowchart LR
    M1["Mode collapse<br/>G produces a narrow<br/>set of outputs"] --> S1["D loss low,<br/>G loss oscillating,<br/>sample variety drops"]
    M2["Vanishing gradients<br/>D wins completely"] --> S2["D accuracy ~100%,<br/>G loss huge and static"]
    M3["Oscillation<br/>G and D keep trading<br/>wins forever"] --> S3["Both losses swing<br/>wildly with no downward trend"]

    style M1 fill:#fecaca,stroke:#dc2626
    style M2 fill:#fecaca,stroke:#dc2626
    style M3 fill:#fecaca,stroke:#dc2626

- **模式崩溃（Mode collapse）**：G 找到了一张能欺骗 D 的图像，并只生成该图像。解决方法：加入小批量判别（minibatch discrimination）、谱归一化（spectral norm）或标签条件化（label-conditioning）。
- **判别器过强（Discriminator wins）**：D 变得过强过快，导致 G 的梯度消失。解决方法：缩小 D 的规模、降低 D 的学习率，或对真实标签应用标签平滑（label smoothing）。
- **震荡（Oscillation）**：两个网络交替占优，始终无法逼近均衡点。解决方法：采用双时间尺度更新规则（TTUR，使 D 的学习速度比 G 快 2-4 倍），或切换至 Wasserstein 损失（Wasserstein loss）。

### 评估方法

GAN 没有基准真值（ground truth），那么如何判断其是否有效呢？

- **样本检查（Sample inspection）**：在每个训练轮次（epoch）结束时直接观察 64 个生成样本。必须严格执行。
- **FID（Fréchet Inception Distance，弗雷歇起始距离）**：计算真实数据集与生成数据集在 Inception-v3 特征空间中的分布距离。数值越低越好，是业界标准。
- **Inception Score（起始分数）**：较旧且更不稳定；建议优先使用 FID。
- **精确率/召回率（Precision/Recall）**：分别衡量生成质量（精确率）和分布覆盖度（召回率）。比单独使用 FID 能提供更多信息。

对于小规模的合成数据实验，仅进行样本检查通常就已足够。

## 开始构建

### 步骤 1：生成器（Generator）

一个小型的 DCGAN（深度卷积生成对抗网络，Deep Convolutional Generative Adversarial Network）生成器，接收 64 维噪声并生成 32x32 的图像。

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))

包含四个转置卷积层（Transposed Convolution），每层均配置为 `kernel_size=4, stride=2, padding=1`，从而将空间尺寸精确翻倍。最终通过 `tanh` 激活函数将输出值限制在 [-1, 1] 区间内。

### 步骤 2：判别器（Discriminator）

生成器的镜像结构。使用 LeakyReLU 激活函数与步长卷积（Strided Convolution），最终输出一个标量对数几率（Logit）。

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 2, feat * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x).view(-1)

最后一个卷积层将 `4x4` 的特征图（Feature Map）缩减为 `1x1`。每张图像输出一个标量；仅在计算损失（Loss）时应用 `sigmoid` 函数。

### 步骤 3：训练步骤

交替更新：每个批次（Batch）中先更新一次判别器 D，再更新一次生成器 G。

import torch.nn.functional as F

def train_step(G, D, real, z, opt_g, opt_d, device):
    real = real.to(device)
    bs = real.size(0)

    # D step
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
    loss_d.backward()
    opt_d.step()

    # G step
    opt_g.zero_grad()
    d_fake = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()

在判别器更新步骤中使用 `G(z).detach()` 至关重要：我们希望在更新判别器时，梯度不会回传到生成器 G。忽略这一点是初学者最常犯的经典错误。

### 步骤 4：在合成形状数据上运行完整训练循环

from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def synthetic_images(num=2000, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((num, 3, size, size), dtype=np.float32) - 1.0
    for i in range(num):
        r = rng.uniform(6, 12)
        cx, cy = rng.uniform(r, size - r, size=2)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.5, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = synthetic_images()
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

G = Generator(z_dim=64, img_channels=3, feat=32).to(device)
D = Discriminator(img_channels=3, feat=32).to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for (batch,) in loader:
        z = torch.randn(batch.size(0), 64, device=device)
        ld, lg = train_step(G, D, batch, z, opt_g, opt_d, device)
    print(f"epoch {epoch}  D {ld:.3f}  G {lg:.3f}")

`Adam(lr=2e-4, betas=(0.5, 0.999))` 是 DCGAN 的默认配置——较低的 `beta1` 值可防止动量项（Momentum）过度稳定对抗博弈过程。

### 步骤 5：采样

@torch.no_grad()
def sample(G, n=16, z_dim=64, device="cpu"):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1)

采样前务必切换至评估模式（Eval Mode）。对于 DCGAN 而言这很重要，因为此时将使用批归一化（Batch Normalization）的累积统计量，而非当前批次的统计量。

### 步骤 6：谱归一化（Spectral Normalization）

可直接替换判别器中批归一化（BN）的模块，能够保证网络满足 1-Lipschitz 连续性条件。可修复大多数“判别器过强（D wins too hard）”导致的训练失败问题。

from torch.nn.utils import spectral_norm

def build_sn_discriminator(img_channels=3, feat=64):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(img_channels, feat, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat, feat * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 4, 1, 4, 1, 0)),
    )

将 `Discriminator` 替换为 `build_sn_discriminator()` 后，通常就不再需要 TTUR（双时间尺度更新规则，Two-Time-Scale Update Rule）技巧。谱归一化（Spectral Norm）是提升模型鲁棒性最简单有效的单一改进方案。

## 实际应用

对于生产级生成任务，建议使用预训练权重或转向扩散模型（Diffusion Models）。两个标准库：

- `torch_fidelity` 可在无需编写自定义评估代码的情况下，直接计算生成器的 FID（Fréchet Inception Distance）/ IS（Inception Score）。
- `pytorch-gan-zoo`（旧版）和 `StudioGAN` 提供了经过验证的 DCGAN、WGAN-GP、SN-GAN、StyleGAN 和 BigGAN 实现。

截至 2026 年，生成对抗网络（GANs）在以下场景仍是最佳选择：实时图像生成（延迟 <10 ms）、风格迁移（Style Transfer）、以及需要精确控制的图像到图像转换（如 Pix2Pix、CycleGAN）。而扩散模型（Diffusion Models）则在照片级真实感和文本条件控制（Text Conditioning）方面占据优势。

## 交付成果

本课程的产出物包括：

- `outputs/prompt-gan-training-triage.md` —— 一个提示词（Prompt），用于读取训练曲线描述，识别故障模式（如模式崩溃/Mode Collapse、判别器占优/D-wins、训练振荡/Oscillation），并给出唯一的推荐修复方案。
- `outputs/skill-dcgan-scaffold.md` —— 一个技能脚本，可根据 `z_dim`、目标 `image_size` 和 `num_channels` 自动生成 DCGAN 基础代码框架，包含训练循环和样本保存功能。

## 练习

1. **（简单）** 在合成圆形数据集上训练上述 DCGAN，并在每个 Epoch（训练轮次）结束时保存一个包含 16 个样本的网格图。生成的圆形在第几个 Epoch 开始呈现出清晰的圆形轮廓？
2. **（中等）** 将判别器中的批归一化（Batch Norm）替换为谱归一化（Spectral Norm）。并行训练两个版本。哪一个收敛更快？在三个不同随机种子下，哪一个的方差更低？
3. **（困难）** 实现条件 DCGAN（Conditional DCGAN）：将类别标签同时输入生成器（G）和判别器（D）（在 G 中将独热编码（One-hot）与噪声拼接，在 D 中拼接类别嵌入通道）。使用第 7 课中的合成“圆形 vs 方形”数据集进行训练，并通过指定标签采样来验证类别条件控制的有效性。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 生成器（Generator, G） | “负责画图的神经网络” | 将噪声映射为图像；训练目标是欺骗判别器 |
| 判别器（Discriminator, D） | “评论家/裁判” | 二分类器；训练目标是区分真实图像与生成图像 |
| 极小极大博弈（Minimax） | “这场博弈” | 针对对抗损失在 G 上求最小、在 D 上求最大；均衡状态为生成数据分布 p_G 等于真实数据分布 p_data |
| 非饱和损失（Non-saturating Loss） | “数值上更稳定的版本” | G 的损失函数采用 -log(D(G(z))) 而非 log(1 - D(G(z)))，以避免训练初期出现梯度消失（Vanishing Gradients） |
| 模式崩溃（Mode Collapse） | “生成器只会做一样东西” | G 仅生成数据分布中的一小部分模式；可通过谱归一化（SN）、小批量判别（Minibatch Discrimination）或增大批量大小（Batch Size）来修复 |
| 双时标更新规则（TTUR） | “两个学习率” | D 的学习速度比 G 快，通常快 2-4 倍；用于稳定训练过程 |
| 谱归一化（Spectral Norm） | “1-Lipschitz 层” | 一种权重归一化方法，用于约束每一层的 Lipschitz 常数；防止判别器的梯度变得过于陡峭 |
| FID（Fréchet Inception Distance） | “弗雷歇 inception 距离” | 真实图像集与生成图像集在 Inception-v3 特征空间中的分布距离；业界标准评估指标 |

## 延伸阅读

- [生成对抗网络 (Generative Adversarial Networks) (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661) — 开启该领域的奠基之作
- [深度卷积生成对抗网络 (DCGAN) (Radford, Metz, Chintala, 2015)](https://arxiv.org/abs/1511.06434) — 奠定 GAN 可训练性的架构规范
- [GAN 的谱归一化 (Spectral Normalization) (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957) — 最为实用的训练稳定技巧
- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423) — 当前最先进的 (State-of-the-Art) GAN；堪称过去十年各项技巧的集大成之作
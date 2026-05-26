# 条件生成对抗网络 (Conditional GANs) 与 Pix2Pix

> 2014 至 2017 年间的首个重大突破在于控制生成对抗网络 (GAN) 的输出内容。你可以附加标签、图像或句子作为条件。Pix2Pix 实现了图像版本的条件控制，即便在如今，它在特定的图像到图像 (image-to-image) 任务上依然优于所有通用的文本到图像 (text-to-image) 模型。

**Type:** 构建实战
**Languages:** Python
**Prerequisites:** 第 8 阶段 · 03 (生成对抗网络 GANs)、第 4 阶段 · 06 (U-Net)、第 3 阶段 · 07 (卷积神经网络 CNNs)
**Time:** 约 75 分钟

## 问题所在

无条件生成对抗网络 (Unconditional GAN) 只能随机采样出任意人脸。这在演示中或许有用，但在实际生产中毫无价值。你的真实需求是：*将草图转换为照片*、*将地图转换为航拍图*、*将白天场景转换为夜晚*、*为灰度图像上色*。在所有这些任务中，你都会获得一张输入图像 `x`，并需要输出与之具有某种语义对应关系的图像 `y`。对于每个 `x`，都存在许多合理的 `y`。如果使用均方误差 (Mean-Squared Error) 损失，它会将这些可能性平均化，导致输出模糊成一团。而对抗损失 (Adversarial Loss) 则不会，因为“看起来真实”这一标准是明确且锐利的。

条件生成对抗网络 (Conditional GAN, Mirza & Osindero, 2014) 将条件 `c` 同时作为生成器 `G` 和判别器 `D` 的输入。Pix2Pix (Isola et al., 2017) 对此进行了专门化设计：条件是一张完整的输入图像，生成器采用 U-Net 架构，判别器采用基于图像块 (patch-based) 的分类器 (PatchGAN)，损失函数为对抗损失加 L1 损失。即便到了 2026 年，该方案在特定的图像到图像转换领域依然优于从零开始训练的文本到图像模型，原因在于它是在*配对数据 (paired data)* 上进行训练的——你恰好拥有模型所需的确切信号。

## 核心概念

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**条件生成器 (Conditional G)。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是生成器 `G` 内部的 Dropout 机制（不引入外部输入噪声——Isola 等人发现显式添加的噪声会被模型忽略）。

**条件判别器 (Conditional D)。** `D(x, y) → [0, 1]`。输入为*配对*数据（条件图像，输出图像）。这是关键区别所在：判别器 `D` 必须判断 `y` 是否与 `x` 保持一致，而不仅仅是判断 `y` 看起来是否真实。

**U-Net 生成器。** 带有跨越瓶颈层 (bottleneck) 的跳跃连接 (skip connections) 的编码器-解码器 (encoder-decoder) 架构。对于输入和输出共享底层结构（如边缘、轮廓）的任务至关重要。如果没有这些跳跃连接，高频细节将会丢失。

**PatchGAN 判别器。** 判别器 `D` 不再输出单一的“真实/伪造”评分，而是输出一个 `N×N` 的网格，其中每个单元格负责评估约 70×70 像素的感受野 (receptive field)。最终结果取平均值。这基于马尔可夫随机场 (Markov Random Field) 假设：真实性是局部的。这种设计训练速度更快、参数量更少，且输出更清晰锐利。

**损失函数 (Loss)。**

loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))

L1 项能够稳定训练过程，并推动生成器 `G` 逼近已知目标。相较于 L2 损失，L1 损失能产生更锐利的边缘（因为它优化的是中位数而非均值）。Pix2Pix 的默认超参数设置为 `λ = 100`。

## CycleGAN —— 当缺乏配对数据时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu 等人，2017）通过引入额外的损失函数——*循环一致性损失（cycle consistency loss）*，放弃了这一要求。它包含两个生成器 `G: X → Y` 和 `F: Y → X`。训练目标是使 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这使得无需配对样本即可实现马到斑马、夏季到冬季的图像转换。

到了 2026 年，非配对图像到图像（unpaired image-to-image）转换大多通过扩散模型（diffusion models，如 ControlNet、IP-Adapter）实现，而非 CycleGAN，但循环一致性（cycle-consistency）的思想几乎在每一篇非配对域适应（unpaired domain adaptation）论文中得以延续。

## 动手实现

`code/main.py` 在一维数据上实现了一个小型的条件生成对抗网络（conditional GAN）。条件 `c` 是一个类别标签（0 或 1）。任务目标：为给定类别生成符合其条件分布的样本。

### 步骤 1：将条件拼接到生成器 G 和判别器 D 的输入中

def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)

独热编码（one-hot encoding）是最简单的方法。规模更大的模型通常使用可学习嵌入（learned embeddings）、FiLM 调制（FiLM modulation）或交叉注意力机制（cross-attention）。

### 步骤 2：训练条件模型

for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)

生成器必须匹配*给定条件下*的真实分布，而非边缘分布（marginal distribution）。

### 步骤 3：验证每个类别的输出

for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)

## 常见陷阱

- **条件被忽略。** 生成器 G 倾向于学习边缘分布，而判别器 D 因条件信号较弱而从未施加惩罚。解决方法：更强烈地将条件注入判别器 D（例如在浅层而非仅在深层），或使用投影判别器（projection discriminator，Miyato & Koyama 2018）。
- **L1 权重过低。** 生成器 G 会偏向生成任意看似真实的输出，而非忠实于原图的输出。对于 Pix2Pix 类任务，建议初始 λ≈100。
- **L1 权重过高。** 由于 L1 本质上仍是 L_p 范数，生成器 G 会产生模糊的输出。待训练稳定后，应逐渐降低该权重。
- **判别器 D 的真实值（ground-truth）输入。** 应将 `(x, y)` 拼接作为 D 的输入，而非仅输入 `y`。否则 D 无法检查一致性。
- **按类别发生模式崩溃（mode collapse）。** 每个类别可能独立发生崩溃。需运行类别条件多样性检查。

## 使用指南

2026年图像到图像（image-to-image）任务现状：

| 任务 | 最佳方案 |
|------|---------------|
| 草图 → 照片，同领域，成对数据（paired data） | Pix2Pix / Pix2PixHD（依然快速，依然清晰） |
| 草图 → 照片，非成对数据（unpaired data） | 结合涂鸦条件模型（Scribble conditioning model）的 ControlNet |
| 语义分割图（semantic segmentation map） → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移（style transfer） | 结合 IP-Adapter 或 LoRA 的扩散模型（diffusion model）；GAN 方法已成为过时技术 |
| 深度图（depth map） → 照片 | 基于 Stable Diffusion 的 ControlNet-Depth |
| 超分辨率（super-resolution） | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散模型） |
| 图像着色（colorization） | ColTran、基于扩散模型的着色器或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气转换 | CycleGAN 或基于 ControlNet 的方案 |

当满足以下条件时，Pix2Pix 依然是合适的工具：(a) 拥有数千组成对样本，(b) 任务范围狭窄且可重复，(c) 需要快速推理（inference）。在通用的开放域（open-domain）任务中，扩散模型则占据优势。

## 交付上线

保存 `outputs/skill-img2img-chooser.md`。该技能模块（Skill）接收任务描述、数据可用性（成对与非成对、样本数量 N）以及延迟/质量预算，随后输出：所选方案（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据需求、推理成本以及评估协议（LPIPS、FID、特定任务指标）。

## 练习

1. **简单。** 修改 `code/main.py` 以添加第三个类别。确认生成器（G）是否仍能将每个类别的噪声映射到正确的模式。
2. **中等。** 在一维（1-D）设置中，将 L1 损失替换为感知风格损失（perceptual-style loss）（例如，使用一个小型冻结的判别器 D 作为特征提取器）。这是否会改变条件分布（conditional distribution）的清晰度？
3. **困难。** 在一维设置中设计一个 CycleGAN：包含两个分布、两个生成器（generators）和循环一致性损失（cycle loss）。证明它能够在没有成对数据的情况下学习两者之间的映射。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 条件生成对抗网络（Conditional GAN） | “带标签的 GAN” | G(z, c), D(x, c)。两个网络均接收条件输入。 |
| Pix2Pix | “图像到图像 GAN” | 基于成对数据的条件 GAN（cGAN），采用 U-Net 生成器（G）和 PatchGAN 判别器（D），并结合 L1 损失。 |
| U-Net | “带跳跃连接的编码器-解码器” | 对称卷积网络；跳跃连接（skips）用于保留高频信息（high-frequency information）。 |
| PatchGAN | “局部真实性分类器” | 判别器（D）输出每个图像块（patch）的得分，而非全局得分。 |
| CycleGAN | “非成对图像翻译” | 包含两个生成器（G）和循环一致性损失（cycle-consistency loss）；无需成对数据。 |
| SPADE | “GauGAN” | 使用语义图对中间激活值进行归一化；用于分割图到图像的生成。 |
| FiLM | “特征级线性调制” | 根据条件对每个特征进行仿射变换（affine transform）；一种低成本的条件注入（conditioning）方式。 |

## 生产环境提示：将 Pix2Pix 作为受延迟限制的基线模型

当你拥有配对数据（paired data）且任务范围较窄（narrow task）（例如：草图 → 渲染图、语义分割图 → 照片、白天 → 夜晚）时，Pix2Pix 的单次推理（one-shot inference）在延迟（latency）方面比扩散模型（diffusion model）快一个数量级。生产环境中的典型对比通常如下：

| 模型路径 | 推理步数 | 单张 L4 显卡上 512² 分辨率的典型延迟（latency） |
|------|-------|----------------------------------------|
| Pix2Pix（U-Net 前向传播） | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

在静态批次（static batches）处理中（每个请求的浮点运算次数 FLOPs 相同），Pix2Pix 在吞吐量（throughput）上占据优势。而扩散模型则在生成质量与泛化能力（generalization）上更胜一筹。当前的常见做法是：针对窄域任务部署一个 Pix2Pix 风格的蒸馏模型（distilled model），并为长尾输入（tail inputs）准备扩散模型作为兜底方案（fallback）。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — 条件生成对抗网络（cGAN）的奠基论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix 原始论文。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN 原始论文。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD 原始论文。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN 原始论文。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影判别器（projection discriminator）相关论文。
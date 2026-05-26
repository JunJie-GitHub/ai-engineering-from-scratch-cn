# 生成对抗网络（GAN）—— 生成器与判别器

> Goodfellow 在 2014 年的巧妙之处在于完全跳过了密度估计（density estimation）。两个网络：一个负责造假，一个负责识破。它们相互博弈，直到生成的假样本与真实样本无法区分。这听起来本不该奏效，事实上也常常失败。但一旦成功，它在特定领域生成的样本依然是文献中最清晰的。

**Type:** 实战构建
**Languages:** Python
**Prerequisites:** 第 3 阶段 · 02（反向传播），第 3 阶段 · 08（优化器），第 8 阶段 · 02（变分自编码器）
**Time:** 约 75 分钟

## 问题背景

变分自编码器（VAE）生成的样本往往模糊，因为其解码器的均方误差（MSE）损失函数在贝叶斯最优（Bayes-optimal）意义下对*平均*图像是最优的——而许多合理数字的平均值就是一个模糊的数字。你需要一种能够奖励*合理性*的损失函数，而不是追求与某个单一目标在像素级上的接近。合理性没有闭式解（closed-form），你必须通过模型去学习它。

Goodfellow 的思路是：训练一个分类器 `D(x)` 来区分真实图像与伪造图像；同时训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号取决于 `D` 当前认为“什么样本看起来像真的”。随着 `G` 的改进，该信号会不断更新，如同追逐一个移动靶。如果两个网络都能收敛，`G` 就学会了数据分布，而全程无需显式写出 `log p(x)`。

这就是对抗训练（adversarial training）。其数学本质是一个极小极大博弈（minimax game）：

min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]

到了 2026 年，GAN 已不再是最先进（SOTA）的生成模型（扩散模型和流匹配（flow matching）技术已摘得桂冠）。但 StyleGAN 2/3 依然是迄今发布的最清晰的人脸模型；GAN 的判别器被用作扩散模型训练中的*感知损失*（perceptual loss）；而对抗训练则支撑了快速的单步蒸馏技术（如 SDXL-Turbo、SD3-Turbo、LCM），让你能够部署实时扩散模型。

## 核心概念

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**生成器 `G(z)`：** 将噪声向量 `z ~ N(0, I)` 映射为样本 `x̂`。采用解码器结构的网络（全连接层或转置卷积）。

**判别器 `D(x)`：** 将样本映射为一个标量概率（或分数）。真实样本 → 1，伪造样本 → 0。

**损失函数。** 采用交替更新策略：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。基于真实标签为 1、伪造标签为 0 的二元交叉熵（binary cross-entropy）。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 采用的*非饱和*（non-saturating）形式（原始公式 `log(1 - D(G(z)))` 在 `D` 置信度较高时会发生饱和，导致梯度消失）。

**训练循环。** 执行一步 `D` 更新，再执行一步 `G` 更新。循环往复。

**为何有效。** 如果 `G` 完美拟合了真实数据分布 `p_data`，那么 `D` 的判别能力将退化为随机猜测，处处输出 0.5；此时 `G` 不再获得梯度。系统达到均衡状态。

**为何失效。** 模式崩溃（mode collapse，`G` 找到一个 `D` 无法区分的单一模式并无限重复生成）、梯度消失（vanishing gradient，`D` 学习过快导致 `log D` 饱和）、训练不稳定（学习率、批次大小等超参数敏感）。

## 让 GAN 真正可用的变体

| 年份 | 创新 | 修复/改进 |
|------|------------|-----|
| 2015 | DCGAN | 卷积/反卷积 (Conv/deconv)、批归一化 (batch norm)、LeakyReLU —— 首个稳定的架构。 |
| 2017 | WGAN, WGAN-GP | 使用 Wasserstein 距离 (Wasserstein distance) + 梯度惩罚 (gradient penalty) 替代二元交叉熵 (BCE)。修复了梯度消失 (vanishing gradient) 问题。 |
| 2017 | 谱归一化 (Spectral normalization) | 对判别器 (discriminator) 施加 Lipschitz 约束 (Lipschitz-bound)。在 2026 年的判别器中仍在使用。 |
| 2018 | 渐进式 GAN (Progressive GAN) | 先训练低分辨率，再逐层添加。首次实现百万像素级生成结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 (Mapping network) + 自适应实例归一化 (adaptive instance norm)。在固定域照片级真实感生成方面处于业界最先进水平 (State of the art)。 |
| 2021 | StyleGAN3 | 无混叠 (Alias-free)、平移等变 (translation-equivariant) —— 截至 2026 年仍是人脸生成的黄金标准。 |
| 2022 | StyleGAN-XL | 条件生成 (Conditional)、类别感知 (class-aware)、更大规模。 |
| 2024 | R3GAN | 通过更强的正则化 (regularization) 重构；无需特殊技巧即可在 1024² 分辨率上运行。 |

## 构建实践 (Build It)

`code/main.py` 在一维数据（两个高斯分布 (Gaussians) 的混合）上训练一个微型生成对抗网络 (GAN)。生成器 (Generator) 与判别器 (Discriminator) 均为单隐藏层的多层感知机 (MLP)。我们手动实现前向传播 (forward)、反向传播 (backward) 以及极小极大循环 (minimax loop)。目标是直观地观察两种关键失败模式（模式崩溃 (mode collapse) 与梯度消失 (vanishing gradient)）的发生过程。

### 步骤 1：非饱和损失 (non-saturating loss)

原始的 Goodfellow 损失函数 `log(1 - D(G(z)))` 在判别器 D 以高置信度将生成器 G 的伪造样本判定为假时，会趋近于 0。此时 G 的梯度基本为零，导致 G 无法继续优化。非饱和形式 `-log D(G(z))` 则具有相反的渐近特性：当 D 置信度很高时，该值会急剧增大，从而为 G 提供强烈的优化信号。

def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)

### 步骤 2：生成器每步对应一次判别器更新

for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)

为 G 提供新生成的伪造样本，否则梯度会过时失效。

### 步骤 3：监控模式崩溃 (mode collapse)

if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")

典型症状：两个真实数据模式 (mode) 中的一个停止被生成。由于判别器从未将其视为伪造样本，因此不再对其进行纠正。

## 常见陷阱 (Pitfalls)

- **判别器（Discriminator）过强。** 将 D 的学习率（Learning Rate）降低 2-5 倍，或添加实例/层噪声（Instance/Layer Noise）。若 D 的准确率超过 95%，G 将彻底失效。
- **生成器（Generator）陷入模式记忆（Mode Memorization）。** 向 D 的输入添加噪声，使用小批量判别器（Minibatch Discriminator）层，或切换至 WGAN-GP。
- **批归一化（Batch Normalization）统计信息泄露。** 真实批次与伪造批次流经同一个 BN 层会导致统计信息混合。请改用实例归一化（Instance Normalization）或谱归一化（Spectral Normalization）。
- **初始分数（Inception Score）指标博弈。** 在样本量较少时，FID 和 IS 指标波动较大。评估时请使用 ≥10k 个样本。
- **条件任务中的一次性采样（One-shot Sampling）纯属误导。** 你仍然需要调整无分类器引导尺度（Classifier-Free Guidance Scale）、截断技巧（Truncation Trick）以及重采样，才能获得可用的输出。

## 使用场景

2026 年 GAN 技术栈：

| 场景 | 推荐方案 |
|-----------|------|
| 逼真人脸，固定姿态 | StyleGAN3（最清晰、模型最小） |
| 动漫/风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译（Image-to-Image Translation） | Pix2Pix / CycleGAN（Phase 8 · 04）或 ControlNet（Phase 8 · 08） |
| 快速单步文本生成图像 | 扩散模型（Diffusion Model）的对抗蒸馏（Adversarial Distillation）（SDXL-Turbo, SD3-Turbo） |
| 扩散训练器中的感知损失（Perceptual Loss） | 在图像裁剪块上使用小型 GAN 判别器 |
| 任何多模态、开放域任务 | 不要使用 —— 请改用扩散模型或流匹配（Flow Matching） |

GAN 生成的图像锐利但适用范围较窄。一旦你的任务域变得开放——如通用照片、任意文本提示词或视频——请切换至扩散模型。对抗技巧（Adversarial Trick）如今已作为组件（如感知损失、蒸馏）延续其价值，而非作为独立的生成器使用。

## 交付上线

保存 `outputs/skill-gan-debugger.md`。该调试技能（Skill）会接收一次失败的 GAN 运行数据（损失曲线、样本网格、数据集大小），并输出按可能性排序的故障原因列表、一行修复代码以及重新运行协议。

## 练习

1. **简单。** 使用默认配置运行 `code/main.py`。然后将 `D_LR = 5 * G_LR` 并重新运行。观察 G 的损失（Loss）多快会坍缩为一个常数？
2. **中等。** 将 Goodfellow 的二元交叉熵损失（Binary Cross-Entropy Loss）替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 D 的权重裁剪至 `[-0.01, 0.01]`。训练是否更稳定？对比实际运行时间（Wall-clock Time）的收敛速度。
3. **困难。** 将一维示例扩展至二维数据（环形分布的 8 个高斯混合模型）。记录生成器在第 1k、5k、10k 步时捕获了 8 个模式（Modes）中的几个。实现小批量判别（Minibatch Discrimination）并重新测量。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 生成器 (Generator) | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器 (Discriminator) | "D" | 分类器 `D: x → [0, 1]`，用于区分真实与伪造样本。 |
| 极小极大 (Minimax) | “博弈” | 联合目标函数的 `min_G max_D` 优化过程。 |
| 非饱和损失 (Non-saturating loss) | “修正方案” | 生成器 G 的损失函数使用 `-log D(G(z))` 替代 `log(1 - D(G(z)))`。 |
| 模式崩溃 (Mode collapse) | “G 只记住了一种东西” | 尽管训练数据多样，生成器仍只能产出少量不同的样本。 |
| WGAN (Wasserstein GAN) | “Wasserstein” | 用推土机距离 (Earth-Mover distance) 加梯度惩罚替代二元交叉熵 (BCE)；梯度更平滑。 |
| 谱范数 (Spectral norm) | “Lipschitz 技巧” | 约束判别器 D 的权重范数以限制其斜率；稳定训练过程。 |
| StyleGAN | “真正能用的那个” | 映射网络 + AdaIN；人脸生成领域的顶尖模型，截至 2026 年依然如此。 |

## 生产环境备注：单步推理是 GAN 的持久优势

在开放域生成任务中，生成对抗网络 (GAN) 在样本质量上已不再占据优势，但在推理成本方面依然胜出。在生产推理领域的术语体系中，GAN 具备以下特点：

- **无预填充 (prefill)，无解码 (decode) 阶段。** 仅需一次 `G(z)` 前向传播。首字延迟 (TTFT) ≈ 总延迟。
- **无键值缓存 (KV-cache) 压力。** 唯一的状态就是模型权重。批量大小 (batch size) 受限于激活内存，而非缓存。
- **易于实现连续批处理 (continuous batching)。** 由于每个请求所需的浮点运算次数 (FLOPs) 固定且相同，在服务器目标占用率下使用静态批处理通常是最优解。无需处理进行中请求的调度器。

这正是 GAN 蒸馏 (GAN distillation)（如 SDXL-Turbo、SD3-Turbo、ADD、LCM）成为 2026 年快速文生图主流技术的原因：它将原本需要 20-50 步的扩散模型 (diffusion model) 流程压缩为 1-4 次类 GAN 的前向传播，同时保留了基础扩散模型的分布特性。对抗损失 (adversarial loss) 作为一种训练阶段的调节参数得以保留，用于将缓慢的生成器转化为快速生成器。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— GAN 的开山之作。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) —— 首个稳定的架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) —— WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) —— 谱归一化 (SN)。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) —— StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) —— StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) —— SDXL-Turbo。
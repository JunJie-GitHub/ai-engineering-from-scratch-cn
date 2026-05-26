# StyleGAN

> 大多数生成器会同时将 `z` 注入到网络的每一层。StyleGAN 将其拆分：首先将 `z` 映射为中间表示 `w`，然后通过自适应实例归一化（AdaIN）在每一个分辨率层级上*注入* `w`。这一项改动解耦了潜在空间（latent space），使得生成照片级逼真人脸在长达七年的时间里成为一个已解决的问题。

**类型：** 构建
**语言：** Python
**前置知识：** 第 8 阶段 · 03（生成对抗网络（GANs））、第 4 阶段 · 08（归一化（Normalization））、第 3 阶段 · 07（卷积神经网络（CNNs））
**预计耗时：** 约 45 分钟

## 核心问题

深度卷积生成对抗网络（DCGAN）通过堆叠的转置卷积（transposed convolutions）将 `z` 映射为图像。其问题在于：`z` 同时控制着姿态、光照、身份和背景等所有属性，这些属性高度纠缠（entangled）。沿着 `z` 的某一维度移动，上述四个属性会同时发生变化。你无法向模型提出“保持同一人物，仅改变姿态”的要求，因为其潜在表示并未以这种因子化的方式进行组织。

Karras 等人（2019，NVIDIA）提出：停止将 `z` 直接输入卷积层。改为将一个固定的 `4×4×512` 张量（tensor）作为网络输入。训练一个 8 层的多层感知机（MLP），将 `z ∈ Z` 映射为 `w ∈ W`。通过自适应实例归一化（AdaIN）在每一个分辨率层级注入 `w`：先对每个卷积特征图进行归一化，再利用 `w` 的仿射投影（affine projections）进行缩放和平移。此外，在每一层添加噪声以生成随机细节（如皮肤毛孔、发丝）。

其结果是：`W` 空间中大致形成了正交（orthogonal）的坐标轴，分别对应“高层风格”（姿态、身份）与“细粒度风格”（光照、色彩）。你可以通过在低分辨率层级使用图像 A 的 `w`、在高分辨率层级使用图像 B 的 `w`，来实现两张图像之间的风格交换。这一突破解锁了图像编辑、跨域风格化，以及整个“StyleGAN 反演（StyleGAN-inversion）”研究方向。

## 核心概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**映射网络（Mapping network）。** `f: Z → W`，一个 8 层的多层感知机（MLP）。`Z = N(0, I)^512`。`W` 空间不强制服从高斯分布——它会学习适应数据本身的分布形态。

**合成网络（Synthesis network）。** 从一个学习得到的固定 `4×4×512` 张量开始。每个分辨率块的处理流程为：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率逐级翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias

其中 `y_scale` 和 `y_bias` 来源于 `w` 的仿射投影。先对每个特征图进行归一化，再进行风格重塑。此处的“风格”指的是特征图的一阶和二阶统计量（均值与方差）。

**逐层噪声（Per-layer noise）。** 向每个特征图添加单通道高斯噪声，并通过学习得到的逐通道缩放因子进行调节。该机制用于控制随机细节，且不会影响全局结构。

**截断技巧（Truncation trick）。** 在推理阶段，采样 `z`，计算 `w = mapping(z)`，然后执行 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是大量样本 `w` 的均值。当 `ψ < 1` 时，模型会以牺牲多样性为代价换取更高的生成质量。几乎所有的 StyleGAN 演示都采用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新点 |
|---------|------|------------|
| StyleGAN | 2019 | 映射网络（Mapping Network）+ 自适应实例归一化（AdaIN）+ 噪声注入 + 渐进式增长（Progressive Growing）。 |
| StyleGAN2 | 2020 | 权重解调（Weight Demodulation）取代 AdaIN（修复水滴伪影）；跳跃/残差架构；路径长度正则化（Path-Length Regularization）。 |
| StyleGAN3 | 2021 | 无混叠卷积（Alias-Free Convolution）+ 等变核（Equivariant Kernels）；消除纹理附着于像素网格的问题。 |
| StyleGAN-XL | 2022 | 类别条件生成（Class-Conditional），1024² 分辨率，基于 ImageNet 训练。 |
| R3GAN | 2024 | 采用更强的正则化重新设计；在 FFHQ-1024 数据集上以少 20 倍的参数量缩小了与扩散模型（Diffusion Models）的差距。 |

截至 2026 年，StyleGAN3 仍是以下场景的默认选择：(a) 高帧率下的窄域照片级真实感生成；(b) 少样本领域自适应（Few-Shot Domain Adaptation）（使用 100 张新图像训练并冻结映射网络）；(c) 基于反演的编辑（Inversion-Based Editing）（寻找能重建真实照片的潜在向量 `w`，随后对该 `w` 进行编辑）。对于开放域的文生图（Text-to-Image）任务，它并非合适工具——扩散模型才是。

## 构建示例

`code/main.py` 实现了一个一维的“轻量版 Style-GAN”玩具示例：包含一个映射多层感知机（Mapping MLP）、一个合成函数（该函数接收一个可学习的常量向量，并使用源自 `w` 的缩放/偏置对其进行调制），以及逐层噪声注入。该示例表明，通过仿射调制（Affine Modulation）注入 `w` 的效果匹配甚至优于将 `z` 直接拼接到生成器输入中。

### 步骤 1：映射网络

def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h

### 步骤 2：自适应实例归一化

def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]

每个特征图（Feature Map）的缩放和偏置参数通过线性投影从 `w` 中获取。

### 步骤 3：逐层噪声

def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]

每个通道的 Sigma（标准差）是可学习的。

## 常见陷阱

- **水滴伪影（Droplet Artifacts）。** StyleGAN 1 会在特征图中产生团状的水滴伪影，因为 AdaIN 将均值归零。StyleGAN 2 通过缩放卷积权重而非特征图本身，利用权重解调修复了该问题。
- **纹理附着（Texture Sticking）。** StyleGAN 1 和 2 的纹理会跟随像素坐标而非物体坐标移动（在插值时尤为明显）。StyleGAN 3 通过引入加窗 sinc 滤波器（Windowed Sinc Filters）的无混叠卷积解决了此问题。
- **模式覆盖（Mode Coverage）。** 截断（Truncation）阈值 `ψ < 0.7` 生成的图像看起来更干净，但采样范围局限于一个狭窄的锥体；若需要多样性，请使用 `ψ = 1.0`。
- **反演存在信息损失（Inversion is Lossy）。** 将真实照片反演到 `W` 空间通常通过优化或编码器（如 e4e、ReStyle、HyperStyle）完成。经过多次迭代后，结果会发生漂移。

## 使用指南

| 应用场景 | 方案 |
|----------|----------|
| 照片级真实人脸（动漫、产品、垂直领域） | StyleGAN3 FFHQ / 自定义微调 (custom fine-tune) |
| 基于照片的人脸编辑 | e4e 图像反演 (inversion) + StyleSpace / InterFaceGAN 方向 |
| 人脸替换 / 表情驱动 (reenactment) | StyleGAN + 编码器 (encoder) + 融合 (blending) |
| 虚拟化身 (Avatar) 流水线 | 结合自适应判别器增强 (ADA) 的 StyleGAN3 用于小数据微调 |
| 基于少量图像的域适应 (domain adaptation) | 冻结映射网络 (mapping network)，微调合成网络 (synthesis) |
| 多模态或文本条件生成 | 不建议 — 请使用扩散模型 (diffusion) |

对于产品级演示，当需求是“生成一张人脸照片”时，在相同的画质标准下，StyleGAN 在推理成本 (inference cost)（单次前向传播 (forward pass)，在 RTX 4090 上耗时 <10ms）和图像清晰度方面均优于扩散模型。

## 交付上线 (Ship It)
保存 `outputs/skill-stylegan-inversion.md`。该技能模块接收一张真实照片，并输出以下信息：图像反演方法（e4e / ReStyle / HyperStyle）、预期的潜在空间损失 (latent loss)、编辑预算 (editing budget)（在 `W` 空间中移动多远会出现伪影），以及一组已知有效的编辑方向（年龄、表情、姿态）。

## 练习
1. **简单。** 分别使用 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。对比固定潜在向量 (latent) 与扰动潜在向量下输出结果的分布差异。
2. **中等。** 实现混合正则化 (mixing regularization)：在一个训练批次中，计算 `w_a` 和 `w_b`，在合成网络的前半部分应用 `w_a`，后半部分应用 `w_b`。观察解码器 (decoder) 是否学习到了解耦 (disentangled) 的风格特征？
3. **困难。** 加载预训练的 StyleGAN3 FFHQ 模型 (`ffhq-1024.pkl`)。通过在标注样本上训练支持向量机 (SVM)，找到控制“微笑”的 `w` 方向；报告在身份特征发生偏移 (identity drift) 之前，该方向的最大可调节范围。

## 关键术语
| 术语 | 通俗叫法 | 实际含义 |
|------|-----------------|-----------------------|
| 映射网络 (Mapping network) | “多层感知机 (MLP)” | `f: Z → W`，共 8 层，将潜在空间的几何结构与数据统计特性解耦。 |
| W 空间 (W space) | “风格空间” | 映射网络的输出；大致实现了解耦。 |
| 自适应实例归一化 (AdaIN) | “自适应实例归一化” | 对特征图进行归一化，随后通过 `w` 投影进行缩放与平移。 |
| 截断技巧 (Truncation trick) | “Psi (ψ)” | `w = mean + ψ·(w - mean)`，当 ψ<1 时，以牺牲多样性为代价换取生成质量。 |
| 路径长度正则化 (Path-length regularization) | “PL 正则化” | 惩罚 `w` 单位变化引起的图像剧烈变化；使 `W` 空间更平滑。 |
| 权重解调 (Weight demodulation) | “StyleGAN2 的修复方案” | 对卷积权重而非激活值进行归一化；消除水滴状伪影。 |
| 无混叠 (Alias-free) | “StyleGAN3 的秘诀” | 采用加窗 sinc 滤波器；消除纹理附着在像素网格上的问题。 |
| 图像反演 (Inversion) | “为真实图像寻找 w” | 优化或编码 `x → w`，使得 `G(w) ≈ x`。 |

## 生产环境备注：为何 StyleGAN 在 2026 年依然被广泛部署

在 RTX 4090 上运行 StyleGAN3 生成一张 1024² 分辨率的 FFHQ 人脸图像耗时不到 10 毫秒——`num_steps = 1`，无需 VAE 解码（VAE decode），也无交叉注意力（cross-attention）计算。从生产环境的角度来看，这代表了任何图像生成器的延迟下限（floor latency）。在相同分辨率下，采用 50 步推理的 SDXL + VAE 解码流水线耗时约为 3 秒。这形成了 **300 倍的差距**，对于垂直领域产品（如虚拟化身服务、证件照处理流水线、图库人脸生成等），它在总拥有成本（TCO）上具有显著优势。

两个运维层面的影响：

- **无需调度器（scheduler），无需批处理器（batcher）。** 在目标占用率下采用静态批处理（static batching）是最优解。连续批处理（continuous batching，对大语言模型和扩散模型至关重要）在此毫无收益，因为每个请求所需的浮点运算次数（FLOPs）完全相同。
- **截断参数 `ψ`（truncation `ψ`）是安全调节旋钮。** 当 `ψ < 0.7` 时，采样点将集中在映射网络（mapping network）取值范围的一个狭窄锥体内。这是服务层控制样本方差（sample variance）的唯一手段。在负载高峰期降低 `ψ` 值，为高级用户提高该值。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演（e4e inversion）。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 方案。
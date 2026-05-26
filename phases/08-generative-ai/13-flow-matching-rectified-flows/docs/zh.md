# 流匹配 (Flow Matching) 与 整流流 (Rectified Flows)

> 扩散模型 (Diffusion Models) 需要 20-50 个采样步数，因为它们从噪声到数据的生成路径是弯曲的。流匹配 (Flow Matching)（Lipman 等人，2023）和整流流 (Rectified Flow)（Liu 等人，2022）通过训练得到了直线路径。路径越直，所需步数越少，推理速度也就越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 均在 2024 年转向了流匹配。

**类型:** 构建
**语言:** Python
**前置知识:** 第 8 阶段 · 06 (DDPM), 第 1 阶段 · 微积分
**预计时间:** ~45 分钟

## 问题所在

DDPM 的逆向过程是一个从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 将其压缩为 20-50 步的确定性步骤。你希望步数更少——理想情况下只需一步。阻碍在于，求解逆向过程的常微分方程 (ODE) 具有刚性 (stiff)，且路径是弯曲的。

如果能训练模型，使从噪声到数据的路径变为一条*直线*，那么只需从 `t=1` 到 `t=0` 执行一次欧拉步 (Euler step) 即可。流匹配直接构建了这一机制：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练一个向量场 (vector field) `v_θ(x, t)` 以匹配其时间导数，并在推理阶段进行积分。

整流流 (Rectified Flow)（Liu, 2022）更进一步：通过重流 (reflow) 过程迭代地拉直路径，从而生成逐渐逼近线性的 ODE。经过两次重流迭代后，仅需 2 步的采样器即可达到 50 步 DDPM 的生成质量。

## 核心概念

![流匹配 (Flow Matching)：噪声与数据之间的直线插值](../assets/flow-matching.svg)

### 直线流 (Straight-line flow)

定义：

x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]

其中 `x_0 ~ data` 且 `x_1 ~ N(0, I)`。沿该直线的时间导数为常数：

dx_t / dt = x_1 - x_0

定义一个神经向量场 `v_θ(x_t, t)` 并训练它以匹配该导数：

L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²

这就是**条件流匹配 (Conditional Flow Matching)** 损失函数 (Lipman 2023)。其训练过程是免仿真的 (Simulation-free)：你无需在训练时展开常微分方程 (ODE)。只需对 `(x_0, x_1, t)` 进行采样并执行回归即可。

### 采样 (Sampling)

在推理阶段，沿时间*反向*积分学习到的向量场：

x_{t-Δt} = x_t - Δt · v_θ(x_t, t)

从 `x_1 ~ N(0, I)` 开始，使用欧拉步进 (Euler-step) 逐步降至 `t=0`。

### 整流流 (Rectified Flow) (Liu 2022)

直线流虽然有效，但学习到的路径*实际上并非直线*——它们会发生弯曲，因为多个 `x_0` 可能映射到同一个 `x_1`。整流流的“重流 (Reflow)”步骤如下：

1. 使用随机配对训练流模型 v_1。
2. 通过从 `x_1` 积分 v_1 至其落点 `x_0`，采样 N 对 `(x_1, x_0)`。
3. 在这些配对样本上训练 v_2。由于这些配对现在是“ODE匹配 (ODE-matched)”的，它们之间的直线插值会真正变得更加平坦（更接近直线）。
4. 重复上述过程。

在实践中，仅需 2 次重流迭代即可使路径接近线性，从而实现 2-4 步推理。SDXL-Turbo、SD3-Turbo 和 LCM 均是基于流匹配模型蒸馏而来的。

### 为何它在 2024 年成为图像生成的首选

原因有三：

1. **免仿真训练 (Simulation-free training)** —— 训练期间无需展开 ODE，实现起来非常简单。
2. **更优的损失几何 (Loss geometry)** —— 直线路径具有一致的信噪比 (Signal-to-Noise Ratio)，而 DDPM 的 ε-损失在噪声调度 (Noise schedule) 边缘处的信噪比表现较差。
3. **更快的推理 (Faster inference)** —— 在 SDXL-Turbo 质量下仅需 4-8 步；结合一致性蒸馏 (Consistency distillation) 甚至可实现 1 步推理。

## 流匹配 vs DDPM —— 精确的数学联系

采用高斯条件路径 (Gaussian-conditional path) 的流匹配，本质上就是*带有特定噪声调度*的扩散模型。若选择 `x_t = α(t) x_0 + σ(t) x_1` 调度，流匹配即可还原为 Stratonovich 重构扩散 (Stratonovich-reformulated diffusion) 形式，其中 `v = α'·x_0 - σ'·x_1`。对于高斯路径，两者在代数上是等价的。

流匹配带来的额外价值在于：目标更加*清晰*（即直接预测速度向量）、损失函数更简洁，以及允许自由尝试非高斯插值路径。

## 动手实现 (Build It)

`code/main.py` 在双峰高斯混合分布（Two-mode Gaussian Mixture）上实现了一维流匹配（1-D Flow Matching）。向量场（Vector Field）`v_θ(x, t)` 是一个使用直线路径目标（Straight-line Target）训练的小型多层感知机（MLP）。在推理（Inference）阶段，分别执行 1、2、4 和 20 步欧拉积分（Euler Steps），并比较生成样本的质量。

### 步骤 1：训练损失（Training Loss）

def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update

### 步骤 2：多步推理（Multi-step Inference）

def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x

### 步骤 3：比较步数

预期 4 步采样器（Sampler）的质量已能与 20 步采样器相媲美——这对降低延迟（Latency）至关重要。

## 常见陷阱

- **时间参数化（Time Parameterization）。** 流匹配使用 `t ∈ [0, 1]`，其中 `t=0` 对应数据分布，`t=1` 对应噪声分布。去噪扩散概率模型（DDPM）使用 `t ∈ [0, T]`，其中 `t=0` 对应数据，`t=T` 对应噪声。方向相同，但尺度不同。论文中经常在此处出错。
- **调度策略选择（Schedule Choice）。** 整流流（Rectified Flow）的直线路径是“标准”的流匹配调度，但你可以使用余弦（cosine）或 logit-normal 时间采样（Logit-normal T-sampling）（SD3 采用了此方法）以获得更好的尺度覆盖。
- **重流成本（Reflow Cost）。** 为重流（Reflow）生成配对数据集需要对每个样本执行一次完整的推理过程。仅在你确实需要 1-2 步推理时才进行重流。
- **无分类器引导（Classifier-Free Guidance）依然适用。** 只需在线性组合中将 ε 替换为 v 即可：`v_cfg = (1+w) v_cond - w v_uncond`。

## 应用场景

| 应用场景 | 2026 技术栈 |
|----------|-----------|
| 文生图，最佳质量 | 流匹配：SD3, Flux.1-dev |
| 文生图，1-4 步 | 蒸馏流匹配（Distilled Flow Matching）：Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 实时推理 | 基于流匹配基座的一致性蒸馏（Consistency Distillation）（LCM, PCM） |
| 音频生成 | 流匹配：Stable Audio 2.5, AudioCraft 2 |
| 视频生成 | 流匹配与扩散模型混合（Sora, Veo, Stable Video） |
| 科学/物理（粒子轨迹、分子） | 流匹配 + 等变向量场（Equivariant Vector Field） |

在 2025-2026 年的论文中，只要提到“比扩散模型更快”，几乎无一例外都是流匹配结合蒸馏技术。

## 部署上线

保存 `outputs/skill-fm-tuner.md`。Skill 工具会接收类扩散模型的配置规范，并将其转换为流匹配训练配置：包括调度策略选择、时间采样分布（均匀分布 / logit-normal 分布）、优化器、重流计划、目标步数以及评估协议。

## 练习

1. **简单。** 运行 `code/main.py`，比较 1 步与 20 步的均方误差（MSE）相对于真实数据分布的差异。
2. **中等。** 将 `t` 的均匀采样切换为 logit-normal 采样（采样集中在中间时刻 t）。模型质量是否会提升？
3. **困难。** 实现一次重流迭代：通过积分第一个模型生成配对数据 `(x_0, x_1)`，在这些配对数据上训练第二个模型，并比较 1 步采样的质量。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 流匹配 (Flow Matching) | “直线扩散” | 训练 `v_θ(x, t)` 使其沿插值路径匹配 `x_1 - x_0`。 |
| 整流流 (Rectified Flow) | “重流” | 一种迭代过程，用于将学习到的流路径拉直。 |
| 速度场 (Velocity Field) | “v_θ” | 模型的输出——即移动 `x_t` 的方向。 |
| 直线插值 (Straight-line Interpolant) | “路径” | `x_t = (1-t)·x_0 + t·x_1`；目标导数形式极为简单。 |
| 欧拉采样器 (Euler Sampler) | “一阶常微分方程求解器” | 最基础的积分器；在路径为直线时效果良好。 |
| Logit正态分布 t (Logit-normal t) | “SD3 采样” | 将 `t` 的采样集中在梯度最强的中间值区域。 |
| 一致性蒸馏 (Consistency Distillation) | “单步采样器” | 训练学生模型，将任意 `x_t` 直接映射到 `x_0`。 |
| 基于速度的分类器自由引导 (CFG with Velocity) | “v-CFG” | `v_cfg = (1+w) v_cond - w v_uncond`；同样的技巧，只是换了变量。 |

## 生产实践说明：Flux.1-schnell 是流匹配的极速实现

流匹配在生产环境中的成功典范是 Flux.1-schnell——它将经过流匹配训练的扩散变换器 (DiT) 蒸馏至仅需 1-4 步推理，同时保持了 Flux-dev 级别的质量。Niels 的《在 8GB 显存机器上运行 Flux》Notebook 提供了标准的部署方案：T5 + CLIP 编码、量化多模态扩散变换器 (MMDiT) 去噪（schnell 版本 4 步，dev 版本 50 步）、变分自编码器 (VAE) 解码。算力开销对比如下：

| 变体 | 步数 | L4 显卡 1024² 分辨率延迟 | 总浮点运算次数 (FLOPs)（相对值） |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (原始版) | 50 | ~15 秒 | 1.0× |
| Flux.1-schnell | 4 | ~1.2 秒 | 0.08×（快 12 倍） |
| SDXL-base | 30 | ~4 秒 | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 秒 | 0.03× |

生产部署的黄金法则：**流匹配基座模型 + 蒸馏 = 2026 年快速文生图 (Text-to-Image) 的默认标准。** 各大主流厂商均已推出该组合：SD3-Turbo（SD3 + 流匹配 + 蒸馏）、Flux-schnell（Flux-dev + 整流流路径拉直）、CogView-4-Flash。纯粹的扩散 (Diffusion) 基座模型如今仅存于旧版权重中。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 整流流。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 流匹配。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模整流流应用。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 涵盖流匹配与扩散模型的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 扩散/流模型的单步蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — Turbo 变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 流匹配的生产级应用。
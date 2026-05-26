# 扩散模型（Diffusion Models）—— 从零实现 DDPM

> Ho、Jain 和 Abbeel（2020）为该领域提供了一套令人欲罢不能的配方。通过上千个微小步骤逐步用噪声破坏数据。训练一个神经网络来预测这些噪声。在推理阶段逆转该过程。如今，所有主流图像、视频、3D 和音乐模型都基于这一循环运行，可能在此基础上叠加了流匹配（Flow Matching）或一致性（Consistency）等技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 第 3 阶段 · 02（反向传播 Backprop），第 8 阶段 · 02（变分自编码器 VAE）
**耗时：** 约 75 分钟

## 问题背景

你需要一个针对 `p_data(x)` 的采样器。生成对抗网络（Generative Adversarial Networks, GAN）通常采用极小极大博弈（Minimax Game），且容易发散。变分自编码器（Variational Autoencoders, VAE）通过高斯解码器生成的样本往往较为模糊。你真正需要的训练目标应满足：(a) 单一且稳定的损失函数（无鞍点 Saddle Point，无极小极大博弈）；(b) 提供 `log p(x)` 的下界（从而具备似然 Likelihood 估计）；(c) 生成样本的质量达到当前最优（State-of-the-Art, SOTA）水平。

Sohl-Dickstein 等人（2015）给出了理论解答：定义一个逐步添加高斯噪声（Gaussian Noise）的马尔可夫链（Markov Chain）`q(x_t | x_{t-1})`，并训练一个逆向链 `p_θ(x_{t-1} | x_t)` 来进行去噪。Ho、Jain 和 Abbeel（2020）证明该损失函数可简化为一行代码——预测噪声，并理顺了相关数学推导。2020 年，它还只是一个新奇的概念；2021 年，它生成了达到 SOTA 水平的样本；2022 年，它演变为 Stable Diffusion；到了 2026 年，它已成为底层基石。

## 核心概念

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个微小步骤中逐步添加高斯噪声。其闭式解（Closed Form）——也是数学上可处理的原因——在于累积步骤同样服从高斯分布：

q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)` 对应 `β_t` 的调度策略（Schedule）。在 T=1000 步中将 `β_t` 从 1e-4 线性增加到 0.02，此时 `x_T` 近似服从 `N(0, I)`。

**逆向过程 `p_θ`。** 训练一个神经网络 `ε_θ(x_t, t)` 来预测所添加的噪声。给定 `x_t`，通过以下公式进行去噪：

x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z

其中 `σ_t` 可取 `sqrt(β_t)` 或学习得到的方差。该表达式看似繁琐，但本质只是代数运算——基于后验分布（Posterior Distribution）`q(x_{t-1} | x_t, x_0)` 求解 `x_{t-1}`，并将 `x_0` 替换为其噪声预测估计值。

**训练损失。**

L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]

从数据中采样 `x_0`，随机选取时间步 `t`，采样 `ε ~ N(0, I)`，通过闭式解一步计算出带噪声的 `x_t`，并对噪声进行回归。单一损失函数，无需极小极大博弈，无需 KL 散度（Kullback-Leibler Divergence, KL），也无需重参数化技巧（Reparameterization Trick）。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代执行逆向步骤。完成。

## 为什么有效

三个核心直觉：

1. **去噪容易，生成困难。** 在 `t=T` 时，数据已退化为纯噪声——网络只需解决一个极其简单的问题。在 `t=0` 时，网络仅需修复少量像素。在中间时刻 `t`，问题虽然复杂，但来自各个噪声级别的梯度会大量流经相同的权重参数。

2. **伪装的分数匹配（Score Matching）。** Vincent (2011) 证明了预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*分数（score）*。反向随机微分方程（Stochastic Differential Equation, SDE）利用该分数沿概率密度梯度上升——这是一种朝向高概率区域的引导式随机游走。

3. **证据下界（Evidence Lower Bound, ELBO）简化为简单的均方误差（Mean Squared Error, MSE）。** 完整的变分下界在每个时间步都包含一个 KL 散度（Kullback-Leibler Divergence）项。在去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）的参数化设定下，这些 KL 项可简化为带有特定系数的噪声预测均方误差；Ho 等人直接去掉了这些系数（称之为“简单”损失函数），结果生成质量反而*提升*了。

## 动手实现

`code/main.py` 实现了一个一维 DDPM。数据为双峰混合分布。该“网络”是一个小型多层感知机（Multilayer Perceptron, MLP），接收 `(x_t, t)` 并输出预测的噪声。训练过程仅由一行损失函数代码构成。采样过程则通过迭代反向扩散链完成。

### 步骤 1：前向扩散调度（闭式解）

betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)

### 步骤 2：一步采样 `x_t`

def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps

### 步骤 3：单步训练

def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)

### 步骤 4：反向采样

def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x

对于包含 40 个时间步和 24 个单元 MLP 的一维问题，该模型仅需约 200 个训练周期（epoch）即可拟合双峰混合分布。

## 时间步条件注入（Time Conditioning）

网络需要知晓当前正在对哪个时间步进行去噪。两种标准方案如下：

- **正弦位置嵌入（Sinusoidal Embedding）。** 类似于 Transformer 的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过一个 MLP 处理后，广播注入到网络各层。
- **FiLM / 组归一化条件注入（FiLM / Group-Norm Conditioning）。** 在每个网络模块中，将嵌入向量投影为逐通道的缩放系数与偏置（特征级线性调制，Feature-wise Linear Modulation, FiLM）。

我们的示例代码采用正弦嵌入 → 拼接（concat）的方式。实际部署的 U-Net 则普遍采用 FiLM。

## 常见陷阱

- **调度策略（Schedule）至关重要。** 线性 `β` 是去噪扩散概率模型（DDPM）的默认设置，但在相同计算量下，余弦调度（cosine schedule）（Nichol & Dhariwal, 2021）能带来更优的 Fréchet Inception Distance（FID）分数。如果生成质量进入停滞期，请尝试切换调度策略。
- **时间步嵌入（Timestep embedding）非常脆弱。** 将原始时间步 `t` 作为浮点数直接传入仅适用于简单的一维示例模型，在图像任务中会失效；务必使用规范的时间步嵌入方法。
- **V-prediction 与 ε-prediction 的对比。** 在特定区间（极小或极大的 `t` 值）内，`ε` 预测的信噪比（signal-to-noise ratio）较差。V-prediction（`v = α·ε - σ·x`）更为稳定；SDXL、SD3 和 Flux 均采用了该方案。
- **无分类器引导（Classifier-free guidance）。** 在推理阶段，需同时计算条件 `ε` 与无条件 `ε`，然后通过公式 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` 进行融合，其中引导权重 `w ≈ 3-7`。详见第 08 课。
- **1000 步的采样过程过于冗长。** 实际生产环境中通常采用 DDIM（20-50 步）、DPM-Solver（10-20 步）或模型蒸馏（distillation，1-4 步）。详见第 12 课。

## 实际应用

| 应用场景 | 2026 年典型技术栈 |
|------|-----------------------|
| 图像像素空间扩散（小型/示例模型） | DDPM + U-Net |
| 图像潜在空间扩散（Latent diffusion） | 变分自编码器（VAE）编码器 + U-Net 或扩散 Transformer（DiT）（第 07 课） |
| 视频潜在空间扩散 | 时空 DiT（Sora, Veo, WAN） |
| 音频潜在空间扩散 | Encodec + 扩散 Transformer（diffusion transformer） |
| 科学计算（分子、蛋白质、物理） | 等变扩散模型（Equivariant diffusion，EDM, RFdiffusion, AlphaFold3） |

扩散模型（Diffusion）已成为通用的生成式基础架构。流匹配（Flow matching，第 13 课）是 2024-2026 年间的主要竞争技术，在同等生成质量下，其推理速度通常更具优势。

## 部署与交付

保存至 `outputs/skill-diffusion-trainer.md`。该模块接收数据集与算力预算作为输入，并输出以下配置：调度策略（线性/余弦/Sigmoid）、预测目标（ε/v/x）、采样步数、引导系数（guidance scale）、采样器家族（sampler family）以及评估协议（eval protocol）。

## 练习

1. **简单。** 在 `code/main.py` 中将 `T` 从 40 改为 10。观察样本质量（输出结果的可视化直方图）如何下降？在哪个 `T` 值时双峰结构会发生坍塌？
2. **中等。** 将预测目标从 ε-prediction 切换为 v-prediction。重新推导反向扩散步骤，并对比最终的样本生成质量。
3. **困难。** 添加无分类器引导（classifier-free guidance）。以类别标签 `c ∈ {0, 1}` 作为条件，在训练过程中以 10% 的概率随机丢弃该条件；在采样阶段使用公式 `ε = (1+w)·ε_cond - w·ε_uncond`。测量在 `w = 0, 1, 3, 7` 时的条件模式命中率（conditional-mode-hit rate）。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 前向过程 (Forward process) | “添加噪声” | 固定的马尔可夫链 (Markov chain) `q(x_t | x_{t-1})`，用于破坏原始数据。 |
| 反向过程 (Reverse process) | “去噪” | 学习得到的链 `p_θ(x_{t-1} | x_t)`，用于重建数据。 |
| β 调度 (β schedule) | “噪声阶梯” | 每一步的方差；可采用线性、余弦或 Sigmoid 形式。 |
| α̅ (Alpha bar) | “Alpha 累积值” | 累积乘积 `∏(1 - β)`；提供从 `x_0` 直接计算 `x_t` 的闭式解。 |
| 简单损失 (Simple loss) | “噪声的均方误差” | `||ε - ε_θ(x_t, t)||²`；所有变分推导最终都简化为此形式。 |
| ε 预测 (ε-prediction) | “预测噪声” | 模型输出为添加的噪声；标准 DDPM 的做法。 |
| V 预测 (V-prediction) | “预测速度” | 模型输出为 `α·ε - σ·x`；在不同时间步 `t` 下具有更好的数值条件。 |
| DDPM | “那篇论文” | Ho 等人 2020 年提出；使用线性 β、1000 步和 U-Net。 |
| DDIM | “确定性采样器” | 非马尔可夫采样器，20-50 步，训练目标相同。 |
| 无分类器引导 (Classifier-free guidance) | “CFG” | 混合条件与非条件噪声预测，以增强条件控制效果。 |

## 生产环境备注：扩散模型推理的核心是步数问题

DDPM 论文中采用了 T=1000 个反向步骤。在实际生产部署中，没有人会直接使用这种配置。每个实际的推理栈都会从以下三种策略中选择一种——且每种策略都清晰地对应了生产环境中分析“延迟来源”的框架：

1. **更快的采样器，相同的模型。** DDIM（20-50 步）、DPM-Solver++（10-20 步）、UniPC（8-16 步）。直接替换反向循环，训练好的 `ε_θ` 权重保持不变。可将延迟降低 20~50 倍。
2. **蒸馏 (Distillation)。** 训练学生模型以更少的步数拟合教师模型：渐进式蒸馏 (Progressive Distillation)（2 → 1 步）、一致性模型 (Consistency Models)（任意步数 → 1~4 步）、LCM、SDXL-Turbo、SD3-Turbo。可进一步将延迟降低 5~10 倍，但需要重新训练。
3. **缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散模型后端、`xformers`/SDPA 注意力机制、bf16 权重。可将单步延迟降低约 2 倍。该策略可与 (1) 和 (2) 叠加使用。

对于生产环境的扩散模型服务器，其性能预算讨论与大型语言模型 (Large Language Model, LLM) 的生产文献描述一致：延迟为 `num_steps × step_cost + VAE_decode`，吞吐量为 `batch_size × (num_steps × step_cost)^-1`。首字延迟 (Time To First Token, TTFT) 很小（仅需一步）；而等效的每词生成时间 (Time Per Output Token, TPOT) 则等同于完整的响应时间，因为从用户视角来看，图像生成是“一次性”完成的。

## 延伸阅读

- [Sohl-Dickstein 等人 (2015)。基于非平衡态热力学的深度无监督学习](https://arxiv.org/abs/1503.03585) —— 扩散模型（Diffusion Model）的奠基之作，理念领先于时代。
- [Ho, Jain, Abbeel (2020)。去噪扩散概率模型](https://arxiv.org/abs/2006.11239) —— DDPM（去噪扩散概率模型）。
- [Song, Meng, Ermon (2021)。去噪扩散隐式模型](https://arxiv.org/abs/2010.02502) —— DDIM（去噪扩散隐式模型），采样步数更少。
- [Nichol & Dhariwal (2021)。改进版 DDPM](https://arxiv.org/abs/2102.09672) —— 余弦调度（Cosine Schedule）与可学习方差（Learned Variance）。
- [Dhariwal & Nichol (2021)。扩散模型在图像合成上击败 GAN](https://arxiv.org/abs/2105.05233) —— 分类器引导（Classifier Guidance）。
- [Ho & Salimans (2022)。无分类器扩散引导](https://arxiv.org/abs/2207.12598) —— CFG（无分类器引导，Classifier-Free Guidance）。
- [Karras 等人 (2022)。阐明基于扩散的生成模型的设计空间 (EDM)](https://arxiv.org/abs/2206.00364) —— 统一符号表示，提供最清晰的实现范式。
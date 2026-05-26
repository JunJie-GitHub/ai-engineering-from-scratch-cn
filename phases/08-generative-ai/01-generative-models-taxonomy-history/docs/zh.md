# 生成模型（Generative Models）—— 分类与历史

> 无论是图像模型、文本模型、视频模型还是 3D 模型，都可以归入以下五大类别之一。选错类别，你将与数学推导死磕数周；选对类别，该领域过去十二年的进展脉络将清晰地印在你的脑海中。

**类型：** 学习
**语言：** Python
**前置知识：** 第二阶段（机器学习基础）、第三阶段（深度学习核心）、第七阶段 · 第14节（Transformers）
**时长：** 约 45 分钟

## 核心问题

生成模型只负责一项任务：给定从某个未知分布 `p_data(x)` 中采样的训练样本，输出看起来同样源自该分布的新样本。无论是人脸、句子、MIDI 文件还是蛋白质结构，只要稍加抽象，本质上都是同一个问题。

难点在于，`p_data` 存在于数百万维的空间中（一张 512x512 的 RGB 图像约有 78.6 万维），样本仅分布在该空间内一个极薄的流形（Manifold）上，而你手头可能只有约 1000 万个样本。试图暴力求解概率密度是毫无希望的。每一种生成模型都是一种折中方案，用一个相对可控的难题去替换另一个无解的难题。

过去十二年间，仅有五大模型家族经受住了时间的考验。了解每个家族所做的折中取舍，你就能明白为何它们在某些任务上表现优异，而在另一些任务上却会彻底失效。

## 核心概念

![生成模型的五大类别——按建模对象分类](../assets/taxonomy.svg)

**1. 显式密度，可处理（Explicit density, tractable）。** 将 `log p(x)` 表示为实际可计算的求和形式。自回归模型（Autoregressive models）（如 PixelCNN、WaveNet、GPT）将 `p(x)` 分解为 `p(x) = ∏ p(x_i | x_<i)`。归一化流（Normalizing flows）（如 RealNVP、Glow）通过对简单基分布进行可逆变换来构建 `p(x)`。优点：似然（Likelihood）精确，训练损失明确。缺点：自回归推理需顺序执行（长序列速度慢），流模型依赖可逆架构（架构设计受限）。

**2. 显式密度，近似计算（Explicit density, approximate）。** 从下方对 `log p(x)` 设定边界（即证据下界 ELBO），并优化该边界。变分自编码器（Variational Autoencoders, VAEs）（Kingma 2013）采用带有变分后验（Variational posterior）的编码器-解码器结构。扩散模型（Diffusion models）（DDPM, Ho 2020）通过训练去噪器（Denoiser）来隐式优化加权 ELBO。截至 2026 年，扩散模型已成为图像、视频和 3D 生成领域的主流骨干架构。

**3. 隐式密度（Implicit density）。** 完全跳过密度建模；直接学习一个用于生成样本的生成器 `G(z)` 和一个用于区分真实与伪造样本的判别器 `D(x)`。生成对抗网络（Generative Adversarial Networks, GANs）（Goodfellow 2014）即属此类。其推理速度快（仅需一次前向传播），但训练过程以极不稳定著称。即使在 2026 年，StyleGAN 1/2/3 在特定领域的照片级真实感生成（如人脸、卧室场景）方面仍保持最先进水平（State of the art）。

**4. 基于分数/连续时间（Score-based / continuous-time）。** 直接学习对数密度的梯度 `∇_x log p(x)`（即分数 Score）。Song & Ermon（2019）证明了分数匹配（Score matching）可将扩散过程推广至随机微分方程（Stochastic Differential Equation, SDE）。流匹配（Flow matching）（Lipman 2023）是 2024-2026 年的热门技术：支持无模拟训练、生成轨迹更直、采样速度比 DDPM 快 4-10 倍。Stable Diffusion 3、Flux 和 AudioCraft 2 均采用了流匹配技术。

**5. 基于离散词元的自回归（Token-based autoregressive over discrete codes）。** 使用 VQ-VAE 或残差量化器（Residual quantizer）将高维数据压缩为较短的离散词元（Token）序列，随后使用 Transformer 对该词元序列进行建模。Parti、MuseNet、AudioLM、VALL-E 以及 Sora 的图像块分词器（Patch tokenizer）均采用此方法。这本质上是第 1 类方法结合一个学习得到的分词器（Tokenizer）。

## 简要历史

| 年份 | 模型 | 重要意义 |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 首个具备可用训练损失（training loss）的深度生成模型（deep generative model）。 |
| 2014 | GAN (Goodfellow) | 采用隐式密度（implicit density），无需计算似然（likelihood）——生成样本清晰度惊人。 |
| 2015 | DRAW, PixelCNN | 实现序列图像生成。 |
| 2017 | Glow, RealNVP | 引入可逆流（invertible flows）；在深层网络中实现精确似然计算。 |
| 2017 | Progressive GAN | 首次生成百万像素级人脸。 |
| 2019 | StyleGAN / StyleGAN2 | 在该特定领域，照片级真实感人脸生成至今仍难以被超越。 |
| 2020 | DDPM (Ho) | 扩散模型（diffusion model）走向实用化。 |
| 2021 | CLIP, DALL-E 1, VQGAN | 文生图（text-to-image）技术走向主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | 潜在扩散（latent diffusion）+ 文本条件控制（text conditioning）= 成为通用技术。 |
| 2022 | ControlNet, LoRA | 实现对预训练扩散模型的精细控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | 模型规模扩大 + 更优的训练动态（training dynamics）。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 视频扩散；流匹配（flow matching）技术胜出。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 达到生产级视频生成标准。 |
| 2026 | Consistency + Rectified Flow | 基于扩散骨干网络（diffusion backbones）实现单步采样（one-step sampling）。 |

## 五问快速评估法

当一篇新的生成模型论文发布时，在阅读方法部分之前，请先回答以下五个问题。

1. **建模对象是什么？** 像素、潜在变量（latents）、离散词元（discrete tokens）、3D 高斯（3D Gaussians）、网格（meshes）还是波形（waveforms）？
2. **密度是显式还是隐式？** 作者是否明确写出了 `log p(x)`？
3. **采样方式：单步（one-shot）还是迭代（iterative）？** 迭代意味着推理速度较慢；单步通常意味着采用了对抗训练（adversarial）或蒸馏（distilled）技术。
4. **条件控制（conditioning）：无条件、类别、文本、图像还是姿态？** 这决定了损失函数和架构的支撑设计。
5. **评估指标：FID、CLIP 分数、IS、人类偏好还是任务准确率？** 每种指标都有其已知的失效模式（参见第 14 课）。

在本阶段的每一课中，你都需要重新回答这五个问题。到课程结束时，它们将成为你的本能反应。

## 动手实践

本课的代码是一个轻量级可视化示例：使用三种简易方法（核密度估计（kernel density）、离散直方图（discrete histogram）以及基于最近邻样本的“类 GAN（GAN-ish）”生成器）从样本中拟合一维高斯混合模型（mixture-of-Gaussians）。这样，你就能在一个单屏即可完整显示的简单问题上，直观地看到显式密度与隐式密度的区别。

运行 `code/main.py`。它会从一个双峰高斯混合分布中抽取 2000 个样本，然后打印：

explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)

注意：前两种方法允许你询问“该点出现的概率有多大？”而第三种方法则无法做到。这就是*显式与隐式*的区别，它将在未来的每一课中都至关重要。

## 实际应用

2026年，针对何种任务应选择何种模型家族？

| 任务 | 最佳模型家族 | 原因 |
|------|-------------|-----|
| 照片级真实人脸，窄领域 | StyleGAN 2/3 | 依然是清晰度最高、推理速度最快的。 |
| 通用文生图 (Text-to-Image) | 潜在扩散模型 (Latent Diffusion) + 流匹配 (Flow Matching) | SD3, Flux.1, DALL-E 3。 |
| 快速文生图 | 整流流 (Rectified Flow) + 蒸馏 (Distillation) | SDXL-Turbo, SD3-Turbo, LCM。 |
| 文生视频 (Text-to-Video) | 扩散 Transformer (DiT) + 流匹配 | Sora, Veo 2, Kling。 |
| 语音与音乐生成 | 基于 Token 的自回归模型 (Token-based AR)（如 AudioLM, VALL-E, MusicGen）或流匹配模型（如 AudioCraft 2） | 离散 Token 的扩展成本较低。 |
| 3D 场景 | 高斯泼溅 (Gaussian Splatting) 拟合 + 扩散先验 (Diffusion Prior) | 3D-GS 用于重建，扩散模型用于新视角合成。 |
| 密度估计（无需采样） | 流模型 (Normalizing Flows) | 唯一能提供精确 `log p(x)` 的模型家族。 |
| 仿真 / 物理模拟 | 流匹配，得分随机微分方程 (Score SDE) | 路径呈直线，向量场平滑。 |

## 交付上线

保存为 `outputs/skill-model-chooser.md`。

该技能模块接收任务描述，并输出：(1) 应选用的模型家族；(2) 三个开源方案与三个托管方案的排名列表；(3) 需警惕的潜在失败模式；(4) 算力/时间预算。

## 练习

1. **简单。** 针对以下五款产品，分别指出其模型家族与骨干网络 (Backbone)：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。依据应来自公开的技术报告。
2. **中等。** 你明天要读的论文声称其采样速度比扩散模型快 100 倍。请写下三个问题，以验证该加速效果在条件生成 (Conditioning) 和高分辨率下是否依然成立。
3. **困难。** 选取一个你关注的领域（例如蛋白质结构、CAD、分子、轨迹）。针对该领域当前的 SOTA (State-of-the-Art) 模型回答“五问快速评估”，并勾勒出一个更优模型应做出哪些改进。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 生成模型 (Generative Model) | “它能生成新东西” | 学习 `p_data(x)` 的采样器，可选择性地暴露 `log p(x)`。 |
| 显式密度 (Explicit Density) | “你可以计算它的概率” | 模型提供闭式或可计算的 `log p(x)`。 |
| 隐式密度 (Implicit Density) | “GAN 风格” | 仅提供采样器——无法评估给定点的 `p(x)`。 |
| ELBO (证据下界) | “证据下界” | `log p(x)` 的一个可计算下界；VAE 和扩散模型均对其进行优化。 |
| 得分函数 (Score) | “对数密度的梯度” | `∇_x log p(x)`；扩散模型和 SDE 模型学习的就是该向量场。 |
| 流形假设 (Manifold Hypothesis) | “数据存在于某个曲面上” | 高维数据集中在低维流形上；这也是降维方法有效的原因。 |
| 自回归 (Autoregressive) | “预测下一个片段” | 将联合分布分解为条件分布的乘积。 |
| 潜在表示 (Latent) | “压缩编码” | 低维表示，解码器可据此重建输入。 |

## 生产环境备注：五大模型家族，五种推理形态

每个模型族都对应着不同的推理服务器成本曲线（inference-server cost curve）。工业界推理文献（production-inference literature）通常将大语言模型（LLM）推理分解为预填充（prefill）与解码（decode）两个阶段；此处的分解方式同样适用：

- **自回归（Autoregressive，分组 1 和 5）。** 顺序解码（sequential decode）主导了延迟；键值缓存（KV-cache）、连续批处理（continuous batching）和投机解码（speculative decoding）等技术均可直接应用。
- **变分自编码器（VAE）/ 扩散模型（diffusion）/ 流匹配（flow-matching，分组 2 和 4）。** 不存在大语言模型意义上的解码过程。成本 = `num_steps × step_cost`，其中 `step_cost` 是在完整潜在空间分辨率（latent resolution）下执行一次 Transformer 或 U-Net 的前向传播（forward）。工程调优的关键参数包括步数（通过 DDIM / DPM-Solver / 模型蒸馏（distillation）调整）、批大小（batch size）以及计算精度（precision，如 bf16 / fp8 / int4）。
- **生成对抗网络（GAN，分组 3）。** 仅需一次前向传播（forward pass）。无需时间步调度（schedule），也无需 KV-cache。首字延迟（TTFT）约等于总延迟。这正是 StyleGAN 在垂直领域用户体验（narrow-domain UX）上依然保持优势的原因。

当你在论文摘要中看到“比扩散模型更快”的表述时，应将其理解为“更少的步数 × 相同的单步成本”或“相同的步数 × 更低的单步成本”。其余说法多为营销话术。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 的开山之作。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 的奠基论文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM 的原始论文。
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — 将扩散过程建模为随机微分方程（SDE）。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 流匹配（flow matching）的核心论文。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3 的技术基础。
# ControlNet、LoRA 与条件控制（Conditioning）

> 仅靠文本是一种笨拙的控制信号。ControlNet 允许你克隆一个预训练的扩散模型（Diffusion Model），并通过深度图（Depth Map）、姿态骨架（Pose Skeleton）、草图（Scribble）或边缘图像（Edge Image）来引导它。LoRA 让你只需训练 1000 万参数就能微调一个 20 亿参数的模型。两者结合，将 Stable Diffusion 从一个玩具变成了 2026 年每家机构都在部署的图像生成流水线（Image Pipeline）。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 8 阶段 · 07（潜在扩散模型（Latent Diffusion）），第 10 阶段（从零构建大语言模型（LLMs from Scratch）—— 用于 LoRA 基础）
**Time:** 约 75 分钟

## 核心问题

像“一个穿红裙的女人在繁忙的街道上遛狗”这样的提示词（Prompt），无法告诉模型狗具体在*哪里*、女人处于*什么姿态*，或者街道的*透视角度*是怎样的。文本只能确定生成图像所需信息的约 10%。其余部分属于视觉信息，无法用语言高效描述。

为每种控制信号（姿态、深度、Canny 边缘检测、语义分割）从头训练一个新的条件模型（Conditional Model）成本极高，几乎不可行。你希望保持 26 亿参数的 SDXL 主干网络（Backbone）冻结，附加一个读取条件信息的小型旁路网络（Side-network），让它去微调主干网络的中间特征。这就是 ControlNet。

你还希望在不重新训练完整模型的情况下，教会模型新概念（如你的脸、你的产品、你的风格）。你希望参数更新量（Delta）缩小 100 倍。这就是 LoRA —— 低秩自适应（Low-Rank Adaptation），一种插入现有注意力权重（Attention Weights）中的低秩适配器（Low-Rank Adapters）。

ControlNet + LoRA + 文本 = 2026 年从业者的标准工具包。大多数生产级图像生成流水线会在 SDXL / SD3 / Flux 基础模型之上，叠加 2-5 个 LoRA、1-3 个 ControlNet 以及一个 IP-Adapter。

## 核心概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

以预训练的 Stable Diffusion (SD) 为基础。*克隆* U-Net 的编码器部分。冻结原始权重。训练克隆模块以接收额外的条件输入 (conditioning input)（如边缘、深度、姿态）。通过*零卷积* (zero-convolution) 跳跃连接 (skip connections) 将克隆模块接回原始解码器部分（使用初始化为零的 1×1 卷积——初始阶段相当于无操作，随后学习增量）。

SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))

零卷积初始化意味着 ControlNet 初始表现为恒等映射 (identity)——即使在训练前也不会产生干扰。使用 100 万组（提示词、条件、图像）三元组，配合标准扩散损失 (diffusion loss) 进行训练。

每种模态的 ControlNet 均以小型旁路模型的形式发布（SDXL 约 360M，SD 1.5 约 70M）。你可以在推理阶段将它们组合使用：

features += weight_a * control_a(depth) + weight_b * control_b(pose)

### LoRA (Hu et al., 2021)

对于模型中的任意线性层 (linear layer) `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量 (low-rank delta)：

W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}

其中 `r << d`。注意力机制通常采用 4-16 的秩，深度微调则常用 64-128。新增参数量为 `2 · d · r`，而非 `d²`。以 SDXL 注意力层为例，当 `d=640`、`r=16` 时：每个适配器仅需 2 万参数，而非 41 万，参数量缩减 20 倍。就整个模型而言：LoRA 通常仅占 20-200MB，而基础模型约为 5GB。

在推理阶段，你可以对 LoRA 进行缩放：`W' = W + α · B @ A`。`α` 通常取值为 0.5-1.5。多个 LoRA 可以以相加方式叠加（需要注意的是，它们之间会产生非线性交互）。

### IP-Adapter (Ye et al., 2023)

一种轻量级适配器，可接受*图像*作为条件输入（与文本并行）。它利用 CLIP 图像编码器生成图像 token，并将其与文本 token 一同注入交叉注意力 (cross-attention) 机制中。每个基础模型仅需约 20MB。无需训练 LoRA 即可实现“生成具有该参考图风格的图像”。

## 可组合性矩阵

| 工具 | 控制内容 | 体积 | 适用场景 |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局与构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化定制、风格迁移 |
| IP-Adapter | 参考图像的风格或主体 | 20MB | 文本难以描述的外观 |
| Textual Inversion | 将单一概念表示为新 token | 10KB | 传统方法，基本已被 LoRA 取代 |
| DreamBooth | 针对特定主体的全量微调 | 2-5GB | 强身份特征、高算力需求 |
| T2I-Adapter | ControlNet 的轻量替代方案 | 70MB | 边缘设备、推理资源受限 |

ControlNet ≈ 空间控制。LoRA ≈ 语义控制。建议结合使用。

## 动手构建

`code/main.py` 在一维空间上模拟了这两种机制：

1. **低秩自适应（LoRA）**。一个预训练的线性层 `W`。将其冻结。训练一个低秩矩阵 `B @ A`，使得 `W + BA` 能够匹配目标线性层。证明 `r = 1` 足以完美学习秩为 1 的修正。

2. **轻量级控制网络（ControlNet-lite）**。一个“冻结的基础”预测器与一个读取额外信号的“辅助网络”。辅助网络的输出由一个初始化为零的可学习标量进行门控控制（这是我们实现的零卷积（zero-conv）版本）。开始训练并观察该门控值逐渐增大的过程。

### 步骤 1：LoRA 数学原理

def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))

### 步骤 2：零初始化辅助网络

side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated

在第 0 步时，输出与基础模型完全一致。训练初期会缓慢更新 `gate` 值——从而避免灾难性漂移（catastrophic drift）。

## 常见陷阱

- **过度放大 LoRA 权重**。将 `α = 2` 或 `α = 3` 是一种常见的“增强效果”的取巧做法，但这会导致输出过度风格化或出现破损。建议保持 `α ≤ 1.5`。
- **ControlNet 权重冲突**。同时以 1.0 的权重使用姿态控制网络（Pose ControlNet）和深度控制网络（Depth ControlNet）通常会导致效果过冲。将权重总和控制在 ≈ 1.0 是一个安全的默认设置。
- **在错误的基础模型上使用 LoRA**。由于注意力机制的维度不匹配，为 SDXL 训练的 LoRA 在 SD 1.5 上会静默失效（no-op）。在 Diffusers 0.30+ 版本中将会发出警告。
- **文本反转（Textual Inversion）漂移**。在一个检查点（checkpoint）上训练的词元（token）在另一个检查点上使用时会出现严重漂移。相比之下，LoRA 具有更好的可移植性。
- **LoRA 权重合并与存储**。你可以将 LoRA 权重烘焙（bake）到基础模型权重中以加快推理速度（无需运行时额外计算），但这会失去在运行时动态调整 `α` 的能力。建议同时保留合并前与合并后的两个版本。

## 使用指南

| 目标 | 2026 工作流（pipeline） |
|------|---------------|
| 复现某品牌的艺术风格 | 使用约 30 张精选图像、秩（rank）为 32 训练的 LoRA |
| 将我的脸放入生成图像中 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 指定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本提示 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示词 | IP-Adapter + 文本提示 |
| 精确布局控制 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 图像修复（Inpainting）（第 09 课） |
| 快速单步风格化 | 基于 SDXL-Turbo 的 LCM-LoRA |

## 交付

保存 `outputs/skill-sd-toolkit-composer.md`。该技能（Skill）接收一个任务（输入资产包括：提示词、可选的参考图像、可选的姿态、可选的深度图、可选的草图），并输出工具栈、权重配置以及可复现的随机种子（seed）协议。

## 练习

1. **简单。** 在 `code/main.py` 中，将 LoRA（低秩自适应）的秩（rank）`r` 从 1 调整至 4。当秩为多少时，LoRA 能够精确匹配秩为 2 的目标增量（delta）？
2. **中等。** 针对两个目标变换（transforms）分别训练两个独立的 LoRA。将它们同时加载并展示其叠加交互效应。在何种情况下该交互会打破线性关系？
3. **困难。** 使用 diffusers（扩散模型库）堆叠以下组件：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。在调整堆叠权重时，测量 FID（Fréchet Inception Distance）与提示词遵循度（prompt adherence）之间的权衡关系。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| ControlNet（控制网络） | “空间控制” | 克隆的编码器（encoder）+ 零卷积（zero-conv）跳跃连接；读取条件图像（conditioning image）。 |
| 零卷积（Zero convolution） | “初始为恒等映射” | 初始化为零的 1×1 卷积；ControlNet 初始状态下相当于无操作（no-op）。 |
| LoRA（低秩自适应） | “低秩适配器” | `W + B @ A`，`r << d`；参数量比全量微调（full fine-tune）少 100 倍。 |
| 秩 r（rank r） | “调节旋钮” | LoRA 的压缩维度；通常设为 4-16，重度个性化定制（personalization）需 64 以上。 |
| α | “LoRA 强度” | 运行时对 LoRA 增量（delta）的缩放系数。 |
| IP-Adapter | “参考图像” | 通过 CLIP 图像令牌（CLIP-image tokens）实现的小型图像条件适配器。 |
| DreamBooth | “全量主体微调” | 使用约 30 张主体图像对完整模型进行训练。 |
| Textual Inversion（文本反转） | “新词元” | 仅学习新的词嵌入（word embedding）；属于旧版技术，现大多已被替代。 |

## 生产环境注意事项：LoRA 热切换、ControlNet 通道与多租户服务

真实的文生图（text-to-image）SaaS 服务会在同一个基础检查点（checkpoint）上提供数百个 LoRA 和十几个 ControlNet。其服务部署问题与大语言模型（LLM）的多租户场景非常相似（生产环境文献通常在持续批处理（continuous batching）和 LoRAX / S-LoRA 的框架下讨论 LLM 案例）：

- **热切换 LoRA，而非合并。** 将 `W' = W + α·B·A` 合并到基础模型中可使单步推理速度提升约 3-5%，但会固定 `α` 和基础模型。应将 LoRA 作为秩为 r 的增量常驻显存（VRAM）中；diffusers 提供了 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 接口用于按请求激活。切换开销仅为 `2 · d · r · num_layers` 个权重——规模在 MB 级别，耗时亚秒级。
- **将 ControlNet 作为第二条注意力通道。** 克隆的编码器与基础模型并行运行。两个权重均为 1.0 的 ControlNet 意味着每步需要额外进行两次前向传播（forward passes），而非一次合并传播。批量大小（batch-size）的余量会呈二次方下降。每个激活的 ControlNet 需预留约 1.5 倍的单步计算开销。
- **LoRA 同样支持量化。** 如果你对基础模型进行了量化（参见第 07 课：8GB 显存运行 Flux），LoRA 增量也能干净地量化至 8-bit 或 4-bit。采用 QLoRA 风格的加载方式，允许你在 4-bit 的 Flux 基础模型上堆叠 5-10 个 LoRA 而不会导致内存溢出。

Flux 专属说明：Niels 的 Flux-on-8GB 笔记本将基础模型量化至 4-bit；在该量化基础模型上堆叠风格 LoRA（`pipe.load_lora_weights("user/style-lora")`），并指定 `weight_name="pytorch_lora_weights.safetensors"` 依然有效。这正是 2026 年大多数 SaaS 服务商采用的标准方案。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). 为文本到图像扩散模型 (Text-to-Image Diffusion Models) 添加条件控制](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu 等 (2021). LoRA：大语言模型 (Large Language Models) 的低秩自适应 (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) — LoRA（最初面向大语言模型；已移植至扩散模型）。
- [Ye 等 (2023). IP-Adapter：兼容文本的图像提示适配器](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou 等 (2023). T2I-Adapter：学习适配器以挖掘更强的可控能力](https://arxiv.org/abs/2302.08453) — ControlNet 的轻量级替代方案。
- [Ruiz 等 (2023). DreamBooth：面向主体驱动生成的文本到图像扩散模型微调](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter 文档](https://huggingface.co/docs/diffusers/training/controlnet) — 参考流水线 (Reference Pipelines)。
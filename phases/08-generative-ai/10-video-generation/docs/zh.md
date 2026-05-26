# 视频生成

> 图像是二维张量（tensor），视频则是三维张量。其底层理论相同，但计算开销（compute）高出 10 到 100 倍。OpenAI 的 Sora（2024 年 2 月）证明了其可行性。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 已能够以 1080p 分辨率从文本生成生产级视频——而开源权重技术栈（open-weights stack，如 CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）则落后约 12 个月。

**Type:** 构建实践
**Languages:** Python
**Prerequisites:** 第 8 阶段 · 07（潜在扩散模型 (Latent Diffusion)）、第 7 阶段 · 09（视觉 Transformer (ViT)）、第 8 阶段 · 06（去噪扩散概率模型 (DDPM)）
**Time:** 约 45 分钟

## 核心挑战

一段 24fps 的 10 秒 1080p 视频包含 240 帧，每帧分辨率为 1920×1080×3 像素。这意味着每个视频片段约有 1.5 GB 的原始数据。在像素空间进行扩散（pixel-space diffusion）是不可行的。你需要：

1. **时空压缩（Spatiotemporal compression）。** 使用变分自编码器（VAE）对完整视频而非单帧进行编码，将其转换为时空图块（spatial-temporal patches）序列。
2. **时间连贯性（Temporal coherence）。** 视频在数秒内的各帧需要保持内容、光照和物体身份的一致性。网络必须能够对运动进行建模。
3. **算力预算（Compute budget）。** 在相同模型规模下，视频训练的成本是图像训练的 10 到 100 倍。
4. **条件控制（Conditioning）。** 支持文本、图像（首帧）、音频或其他视频作为条件。大多数生产级模型同时支持这四种输入。

解决这一问题的架构是应用于时空图块的**扩散 Transformer（Diffusion Transformer, DiT）**，并在海量（提示词、描述文本、视频）数据集上进行训练。其扩散损失（diffusion loss）与第 06 课相同。

## 核心概念

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### 图块划分（Patchify）

使用三维变分自编码器（3D VAE）对视频进行编码（即学习到的时空压缩）。潜在表示（latent）的形状为 `[T_latent, H_latent, W_latent, C_latent]`。将其划分为大小为 `[t_p, h_p, w_p]` 的图块。对于 Sora 风格的模型，`t_p = 1`（逐帧划分）或 `t_p = 2`（每两帧划分）。一段 10 秒的 1080p 视频将被压缩为约 20,000 到 100,000 个图块。

### 时空 DiT（Spatiotemporal DiT）

Transformer 处理展平后的图块序列。每个图块都包含一个三维位置编码（3D positional embedding，时间 + y 轴 + x 轴）。注意力机制（Attention）通常采用分解设计：

- **空间注意力（Spatial attention）：** 作用于单帧内的各个图块。
- **时间注意力（Temporal attention）：** 作用于相同空间位置的不同帧之间。
- **完整 3D 注意力（Full 3D attention）：** 计算开销高出 16 到 100 倍，仅用于低分辨率场景或学术研究。

### 文本条件控制（Text conditioning）

通过交叉注意力（Cross-attention）与大型文本编码器结合（Sora 使用 T5-XXL，CogVideoX-5B 同样使用 T5-XXL）。长提示词至关重要——Sora 的训练集包含由 GPT 生成的密集重描述文本，平均每个视频片段约 200 个词元（tokens）。

### 训练（Training）

在时空潜在表示上应用标准扩散损失（ε 预测或 v 预测）。数据：网络视频 + 约 1 亿条精选片段 + 合成文本描述。算力：即便是小型研究实验也需要 10,000+ GPU 小时；Sora 级别的训练则超过 100,000 GPU 小时。

## 2026 年生产级应用格局

| 模型 | 发布日期 | 最大时长 | 最大分辨率 | 是否开源权重？ | 亮点 |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | 否 | 首个在大规模上展现世界模拟器（World Simulator）特性的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | 否 | 生产版 Sora，推理速度提升 5 倍 |
| Veo 2 (Google) | 2024-12 | 8s | 4K | 否 | 2025 年画质与物理模拟最佳 |
| Veo 3 | 2025 Q3 | 15s | 4K | 否 | 原生音频支持与更强的运镜控制 |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | 否 | 2025 年第一季度人体动作生成最佳 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 否 | 内置专业级视频编辑工具 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 否 | 角色一致性最强 |
| CogVideoX (THUDM) | 2024 | 10s | 720p | 是 (2B, 5B) | 首个开源的 5B 参数量视频模型 |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | 是 (13B) | 2024 年底开源领域当前最优（State of the Art, SOTA） |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | 是 (10B) | 许可证限制最宽松 |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | 是 | 2025 年中表现最强的开源模型 |

开源权重（Open Weights）模型的性能差距正在以快于图像生成领域的速度缩小：截至 2026 年中，HunyuanVideo 与 WAN 2.2 的低秩自适应（Low-Rank Adaptation, LoRA）已支撑起大多数开源工作流。

## 构建实现

`code/main.py` 模拟了时空扩散 Transformer（Spatiotemporal Diffusion Transformer, DiT）的核心思想：将一段小型合成视频进行分块处理（Patchify），为每个块添加位置编码（Position Embedding），并利用基于块的 Transformer 风格注意力机制（Transformer-style Attention）对整个序列进行去噪（Denoise）。全程不依赖 NumPy，仅使用纯 Python。我们证明了，即使在一维（1-D）场景下，只要相邻帧的块共享同一个去噪器（Denoiser）与位置编码，模型也能涌现出时间连贯性（Temporal Coherence）。

### 步骤 1：对合成的一维“视频”进行分块处理

def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]

### 步骤 2：为每一帧添加位置编码

def pos_embed(t, dim):
    return sinusoidal(t, dim)

### 步骤 3：去噪器感知完整序列

与独立对每一帧进行去噪不同，我们的微型网络会将所有帧的数值及其位置编码进行拼接，并联合预测所有帧的噪声。

### 步骤 4：时间连贯性测试

训练完成后，采样生成一段视频。测量帧间差值（Frame-to-frame Delta）。如果模型已学习到时间结构，该差值将小于独立采样每一帧时的结果。

## 常见陷阱

- **逐帧独立采样 = 画面闪烁。** 如果对每一帧单独运行图像扩散模型（Image Diffusion），输出会出现闪烁，因为每帧的噪声是相互独立的。视频扩散模型（Video Diffusion）通过注意力机制（Attention）或共享噪声将各帧耦合，从而解决该问题。
- **朴素 3D 注意力 = 内存溢出（OOM）。** 对 10 秒 1080p 潜变量（Latent）进行完整的 3D 注意力计算需要数千亿次运算。需将其分解为空间（Spatial）与时间（Temporal）注意力。
- **数据描述（Captioning）质量比数据规模更重要。** Sora 相较于先前工作的主要升级在于使用了约 10 倍更详细的文本描述进行训练（由 GPT-4 重新标注视频片段）。OpenAI 的技术报告对此有明确说明。
- **首帧条件控制（First-frame Conditioning）。** 大多数生产级模型也支持将一张图像作为首帧输入。这属于“图生视频（Image-to-Video）”模式；模型训练已包含该变体。
- **物理规律漂移（Physics Drift）。** 长视频片段（>10秒）会累积细微的不一致性。采用滑动窗口生成（Sliding-window Generation）结合关键帧锚定（Keyframe Anchoring）可有效缓解。

## Use It

| 应用场景 | 2026 年推荐方案 |
|----------|-----------|
| 最高质量的文生视频（Text-to-Video），云端托管 | Veo 3 或 Sora |
| 摄像机控制的电影级画面 | 搭配运动笔刷（Motion Brushes）的 Runway Gen-3 |
| 跨片段角色一致性 | Pika 2.0 或 Kling 2.1 |
| 开源权重，快速微调（Fine-tune） | WAN 2.2 + LoRA |
| 图生视频（Image-to-Video） | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音画同步口型生成 | Veo 3（原生支持音频）或专用口型同步模型 |
| 视频编辑 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（静态帧） |

在画质相当的前提下，2024 年至 2026 年间，每秒视频的生成成本已降至原来的 1/20。

## Ship It

保存 `outputs/skill-video-brief.md`。Skill 工具接收视频需求简报（包含时长、宽高比、风格、运镜方案、主体一致性、音频等），并输出：模型选择与托管方案、提示词框架（Prompt Scaffolding，含运镜术语、主体描述、运动描述符）、随机种子（Seed）与可复现性协议，以及逐帧质量检查（QA）清单。

## Exercises

1. **简单。** 在 `code/main.py` 中，对比以下两种方式的帧间差值（Delta）：(a) 逐帧独立采样，(b) 联合序列采样。报告差值的均值与方差。
2. **中等。** 添加首帧条件控制：将第 0 帧固定为给定值，并对剩余帧进行采样。测量该固定值在序列中的传播情况。
3. **困难。** 使用 HuggingFace `diffusers` 库在本地 GPU 上运行 CogVideoX-2B。对一段 6 秒 720p 视频进行 20 步推理（Inference）并计时。对时空注意力（Spatiotemporal Attention）进行性能剖析（Profiling）以定位瓶颈。

## Key Terms

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 视频变分自编码器 (Video VAE) | “3D VAE” | 将 `(T, H, W, C)` 压缩为时空潜在表示 (spatiotemporal latent) 的编码器。 |
| 块 (Patches) | “Token” | 潜在表示中固定大小的 3D 块；作为扩散 Transformer (DiT) 的输入。 |
| 分解注意力 (Factorized Attention) | “空间 + 时间” | 先在空间维度计算注意力，再在时间维度计算；跳过完整的 3D 注意力计算。 |
| 图生视频 (Image-to-Video, I2V) | “让这张照片动起来” | 模型接收图像与文本，生成以该图像为起始帧的视频。 |
| 关键帧条件控制 (Keyframe Conditioning) | “锚定帧” | 固定特定帧以控制视频的整体轨迹。 |
| 运动笔刷 (Motion Brush) | “方向提示” | 一种 UI 输入方式，允许用户在图像上绘制运动向量。 |
| 重标注 (Re-captioning) | “密集描述” | 使用大语言模型 (LLM) 为训练视频片段重新生成详细的提示词标签。 |
| 闪烁 (Flicker) | “时间伪影” | 帧间不一致现象；可通过耦合去噪 (coupled denoising) 修复。 |

## 生产环境提示：视频潜在表示 (video latents) 是内存带宽问题

一段 10 秒、1080p 分辨率、24 fps 的视频包含 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 的原始像素数据。经过 4 倍视频 VAE 压缩（空间 2 倍 × 时间 2 倍）后，每次请求的潜在表示约为 100 MB。在批次大小 (batch size) 为 1 的情况下，将其输入时空 DiT 进行 30 步推理，每一步需要在高带宽内存 (HBM) 中传输约 3 GB 数据——此时瓶颈在于内存带宽，而非浮点运算次数 (FLOPs)。

三个生产环境调优参数，均直接源自生产推理文献的推理章节：

- **跨 DiT 的张量并行 (Tensor Parallelism, TP)。** 文生视频模型的参数量通常 ≥100 亿。在 4 张 H100 上配置 TP=4 是标准做法；对于 405B 级别模型，则采用流水线并行 (Pipeline Parallelism, PP)=2 × TP=2。在达到 all-reduce 通信墙 (all-reduce wall) 之前，单步延迟大致随 TP 数量呈线性下降。
- **帧批处理 = 连续批处理 (Continuous Batching)。** 在生成阶段，视频在概念上是由注意力机制关联的帧批次。连续批处理（在途调度，in-flight scheduling）同样适用：如果模型架构支持滑动窗口生成 (sliding-window generation)，可以在返回第 `t-1` 帧的同时，开始渲染第 `t+1` 帧。
- **片段级预填充缓存 (Clip-level Prefill Cache)。** 对于图生视频任务，首帧条件控制类似于大语言模型 (LLM) 的提示词预填充 (prompt prefill)：只需计算一次，即可在时间解码器的多次传递中复用。这实际上就是视频领域的 KV 缓存 (KV-cache)。

## 延伸阅读

- [Brooks 等人（2024）。视频生成模型（Video Generation Models）作为世界模拟器](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告。
- [Yang 等人（2024）。CogVideoX：基于专家 Transformer 的文本到视频扩散模型（Text-to-Video Diffusion Models）](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong 等人（2024）。HunyuanVideo：大型视频生成模型的系统化框架](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo（2024）。Mochi-1 技术报告](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [阿里巴巴（2025）。WAN 2.2](https://wanvideo.io/) — 2025 年中开源的 SOTA（State of the Art，最先进）模型。
- [Ho、Salimans、Gritsenko 等人（2022）。视频扩散模型（Video Diffusion Models）](https://arxiv.org/abs/2204.03458) — 视频扩散领域的奠基性论文。
- [Blattmann 等人（2023）。对齐潜变量（Video LDM）](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的前身。
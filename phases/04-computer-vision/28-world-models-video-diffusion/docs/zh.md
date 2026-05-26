# 世界模型（World Models）与视频扩散（Video Diffusion）

> 能够预测场景接下来数秒画面的视频模型，本质上就是一个世界模拟器（World Simulator）。若以动作作为条件来引导该预测过程，你便得到了一个经过学习训练的游戏引擎。

**Type:** 学习 + 构建
**Languages:** Python
**Prerequisites:** 第4阶段第10课（扩散模型 Diffusion）、第4阶段第12课（视频理解 Video Understanding）、第4阶段第23课（DiT + 整流流模型 Rectified Flow）
**Time:** 约75分钟

## 学习目标

- 阐明纯视频生成模型（如 Sora 2）与动作条件化（action-conditioned）世界模型（如 Genie 3、DreamerV3）之间的差异
- 解析视频 DiT（Diffusion Transformer）架构：时空图块（spatio-temporal patches）、三维位置编码（3D position encoding）以及跨时间、高度、宽度（T, H, W）标记（tokens）的联合注意力机制（joint attention）
- 梳理世界模型接入机器人系统的链路：视觉语言模型（VLM）规划 → 视频模型模拟 → 逆动力学（inverse dynamics）输出动作
- 针对具体应用场景（创意视频、交互式模拟、自动驾驶数据合成），在 Sora 2、Genie 3、Runway GWM-1 Worlds、Wan-Video 与 HunyuanVideo 之间进行技术选型

## 问题背景

视频生成与世界建模在 2026 年走向融合。从某种意义上说，能够生成连贯一分钟视频的模型，已经在某种程度上掌握了物理世界的运行规律：物体恒存性（object permanence）、重力、因果关系与视觉风格。若将动作指令（如向左行走、开门）作为条件输入该预测过程，视频模型便会转化为一个可学习的模拟器，进而替代传统游戏引擎、驾驶模拟器或机器人仿真环境。

其落地价值已十分具体。Genie 3 能够从单张图像生成可交互游玩的环境；Runway GWM-1 Worlds 可合成无限的可探索场景；Sora 2 能够生成具备同步音频与物理建模的分钟级视频；NVIDIA Cosmos-Drive、Wayve Gaia-2 以及 Tesla DrivingWorld 则为自动驾驶训练生成逼真的驾驶视频。世界模型范式正悄然接管机器人领域的“仿真到现实”（sim-to-real）迁移流程。

本课是第4阶段的“宏观架构”导论。它将图像生成、视频理解与智能体推理（agentic reasoning）融会贯通，勾勒出当前前沿研究正集中演进的核心架构范式。

## 核心概念

### 世界建模（World-Modelling）的三大类别

flowchart LR
    subgraph GEN["Pure video generation"]
        G1["Text / image prompt"] --> G2["Video DiT"] --> G3["Video frames"]
    end
    subgraph ACTION["Action-conditioned world model"]
        A1["Past frames + action"] --> A2["Latent-action video DiT"] --> A3["Next frames"]
        A3 --> A1
    end
    subgraph RL["World models for RL (DreamerV3)"]
        R1["State + action"] --> R2["Latent transition model"] --> R3["Next latent + reward"]
        R3 --> R1
    end

    style GEN fill:#dbeafe,stroke:#2563eb
    style ACTION fill:#fef3c7,stroke:#d97706
    style RL fill:#dcfce7,stroke:#16a34a

- **Sora 2** 属于纯提示词条件（Prompt-Conditioned）视频生成模型。它没有动作接口，无法在生成推演（Rollout）过程中进行“引导”或干预。
- **Genie 3**、**GWM-1 Worlds** 和 **Mirage / Magica** 属于动作条件（Action-Conditioned）世界模型。它们从观测视频中推断潜在动作（Latent Action），并以此作为条件来预测未来帧。具备交互性——当你按下按键或移动视角时，场景会做出相应响应。
- **DreamerV3** 及经典的强化学习（Reinforcement Learning, RL）世界模型家族在潜在空间（Latent Space）中进行预测，并带有显式的动作条件，通过奖励信号（Reward Signal）进行训练。其视觉表现较弱，但在样本高效型强化学习（Sample-Efficient RL）中更具实用价值。

### 视频 DiT（Diffusion Transformer）架构

Video latent:          (C, T, H, W)
Patchify (spatial):    grid of P_h x P_w patches per frame
Patchify (temporal):   group P_t frames into a temporal patch
Resulting tokens:      (T / P_t) * (H / P_h) * (W / P_w) tokens

位置编码（Positional Encoding）采用三维形式：为每个 `(t, h, w)` 坐标分配旋转位置编码（Rotary Embedding）或可学习嵌入（Learned Embedding）。注意力机制（Attention）可分为以下几种：

- **全连接注意力（Full Joint）** — 所有 Token 相互关注。计算复杂度为 O(N^2)（N 为 Token 数量），对于长视频而言计算成本过高。
- **分离式注意力（Divided）** — 交替进行时间注意力（同一空间位置跨时间：`(H*W) * T^2`）和空间注意力（同一时间步跨空间：`T * (H*W)^2`）。TimeSformer 和大多数视频 DiT 均采用此设计。
- **窗口注意力（Window）** — 在 `(t, h, w)` 维度上划分局部窗口。Video Swin 采用此方案。

截至 2026 年，所有视频扩散模型（Video Diffusion Model）均采用上述三种模式之一，并结合自适应层归一化条件（AdaLN Conditioning，见第 23 课）与整流流（Rectified Flow）技术。

### 动作条件机制：潜在动作模型（Latent Action Models）

Genie 通过判别式预测连续两帧之间的动作，为每一帧学习一个**潜在动作（Latent Action）**。随后，模型的解码器（Decoder）以推断出的潜在动作作为条件，而非依赖明确的键盘按键。在推理阶段，用户可以指定一个潜在动作（或从新的先验分布中采样），模型便会生成与该动作一致的下一帧。

Sora 则完全跳过了动作接口。其解码器直接根据过去的时空 Token 预测未来的时空 Token。提示词仅用于初始化生成起点，在生成过程中无法进行任何干预或引导。

### 物理合理性（Physical Plausibility）

Sora 2 在 2026 年的发布中明确将**物理合理性**作为核心卖点：包括重量感、平衡性、物体恒存性（Object Permanence）以及因果关系。团队通过人工评分进行量化评估；相较于 Sora 1，该模型在物体掉落、角色碰撞以及故意失败场景（如跳跃失误）的表现上均有显著提升。

物理合理性不足仍是当前模型最主要的失效模式。2024 至 2025 年间生成的吃意大利面或从玻璃杯喝水的视频，暴露出模型缺乏对物体的持续表征能力。2026 年的模型（如 Sora 2、Runway Gen-5、HunyuanVideo）虽已大幅缓解此类问题，但尚未彻底根除。

### 自动驾驶世界模型

自动驾驶世界模型能够根据轨迹、边界框（Bounding Box）或导航地图生成逼真的道路场景。典型应用包括：

- **Cosmos-Drive-Dreams**（NVIDIA）— 生成数分钟的驾驶视频，用于强化学习训练。
- **Gaia-2**（Wayve）— 基于轨迹条件的场景合成，用于策略评估。
- **DrivingWorld**（Tesla）— 模拟多变的天气、时段及交通状况。
- **Vista**（ByteDance）— 具备响应能力的驾驶场景合成。

这些模型替代了针对长尾场景（Corner Case）的高昂真实数据采集工作——例如夜间行人乱穿马路、结冰路口、特殊车型等，否则这些场景通常需要数百万英里的实际路测才能覆盖。

### 机器人技术栈：VLM + 视频模型 + 逆动力学（Inverse Dynamics）

新兴的三组件机器人控制闭环如下：

1. **视觉语言模型（Vision-Language Model, VLM）** 解析任务目标（如“拿起红色杯子”），并规划高层动作序列。
2. **视频生成模型** 模拟执行每个动作的视觉结果——预测未来 N 帧的观测画面。
3. **逆动力学模型** 提取能够产生上述观测结果的具体电机控制指令。

该架构取代了传统的奖励塑形（Reward Shaping）与高样本消耗的强化学习。世界模型负责“想象”推演，逆动力学模型则完成执行层面的闭环控制。Genie Envisioner 是这一架构的具体实现之一，目前众多研究团队正逐渐向该范式靠拢。

### 评估指标

- **视觉质量** — FVD（Fréchet Video Distance，弗雷歇视频距离）、用户调研。
- **提示词对齐度** — 逐帧 CLIPScore、视觉问答（Visual Question Answering, VQA）式评估。
- **物理合理性** — 基于基准测试套件的人工评分（如 Sora 2 内部基准、VBench）。
- **可控性**（针对交互式世界模型）— 动作与观测的一致性；系统是否支持回退至先前状态？

### 2026 年模型生态概览

| Model | Use | Parameters | Output | License |
|-------|-----|------------|--------|---------|
| Sora 2 | 文本/音频转视频 | — | 1分钟 1080p + 音频 | 仅限 API |
| Runway Gen-5 | 文本/图像转视频 | — | 10秒片段 | API |
| Runway GWM-1 Worlds | 交互式世界 | — | 无限 3D 推演 | API |
| Genie 3 | 基于图像的交互式世界 | 11B+ | 可交互帧 | 研究预览版 |
| Wan-Video 2.1 | 开源文本转视频 | 14B | 高质量片段 | 非商业用途 |
| HunyuanVideo | 开源文本转视频 | 13B | 10秒片段 | 宽松许可 |
| Cosmos / Cosmos-Drive | 自动驾驶仿真 | 7-14B | 驾驶场景 | NVIDIA 开源 |
| Magica / Mirage 2 | AI 原生游戏引擎 | — | 可修改世界 | 商业产品 |

## 构建

### 步骤 1：视频的 3D 分块（3D Patchify）

import torch
import torch.nn as nn


class VideoPatch3D(nn.Module):
    def __init__(self, in_channels=4, dim=64, patch_t=2, patch_h=2, patch_w=2):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):
        # x: (N, C, T, H, W)
        x = self.proj(x)
        n, c, t, h, w = x.shape
        tokens = x.reshape(n, c, t * h * w).transpose(1, 2)
        return tokens, (t, h, w)

步长（stride）等于卷积核（kernel）大小的 3D 卷积（3D Convolution）充当了时空分块器（spatio-temporal patchifier）。它将输入划分为 `(T/2, H/2, W/2)` 网格的词元（tokens）。

### 步骤 2：3D 旋转位置编码（3D Rotary Position Encoding）

旋转位置嵌入（Rotary Position Embeddings, RoPE）分别沿 `t`、`h`、`w` 轴独立应用：

def rope_3d(tokens, t_dim, h_dim, w_dim, grid):
    """
    tokens: (N, T*H*W, D)
    grid: (T, H, W) sizes
    t_dim + h_dim + w_dim == D
    """
    T, H, W = grid
    n, seq, d = tokens.shape
    if t_dim + h_dim + w_dim != d:
        raise ValueError(f"t_dim+h_dim+w_dim ({t_dim}+{h_dim}+{w_dim}) must equal D={d}")
    assert seq == T * H * W
    t_idx = torch.arange(T, device=tokens.device).repeat_interleave(H * W)
    h_idx = torch.arange(H, device=tokens.device).repeat_interleave(W).repeat(T)
    w_idx = torch.arange(W, device=tokens.device).repeat(T * H)
    # Simplified: just scale channels by frequencies. Real RoPE rotates pairs.
    freqs_t = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(t_dim // 2, device=tokens.device) / (t_dim // 2))
    freqs_h = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(h_dim // 2, device=tokens.device) / (h_dim // 2))
    freqs_w = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(w_dim // 2, device=tokens.device) / (w_dim // 2))
    emb_t = torch.cat([torch.sin(t_idx[:, None] * freqs_t), torch.cos(t_idx[:, None] * freqs_t)], dim=-1)
    emb_h = torch.cat([torch.sin(h_idx[:, None] * freqs_h), torch.cos(h_idx[:, None] * freqs_h)], dim=-1)
    emb_w = torch.cat([torch.sin(w_idx[:, None] * freqs_w), torch.cos(w_idx[:, None] * freqs_w)], dim=-1)
    return tokens + torch.cat([emb_t, emb_h, emb_w], dim=-1)

此处为简化的加法形式。实际的 RoPE 会在不同频率下对成对的通道进行旋转操作，但两者所携带的位置信息是相同的。

### 步骤 3：分离注意力模块（Divided Attention Block）

class DividedAttentionBlock(nn.Module):
    def __init__(self, dim=64, heads=2):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, grid):
        T, H, W = grid
        n, seq, d = x.shape
        # time attention: same (h, w), across t
        xt = x.view(n, T, H * W, d).permute(0, 2, 1, 3).reshape(n * H * W, T, d)
        a, _ = self.time_attn(self.ln1(xt), self.ln1(xt), self.ln1(xt), need_weights=False)
        xt = (xt + a).reshape(n, H * W, T, d).permute(0, 2, 1, 3).reshape(n, seq, d)
        # space attention: same t, across (h, w)
        xs = xt.view(n, T, H * W, d).reshape(n * T, H * W, d)
        a, _ = self.space_attn(self.ln2(xs), self.ln2(xs), self.ln2(xs), need_weights=False)
        xs = (xs + a).reshape(n, T, H * W, d).reshape(n, seq, d)
        xs = xs + self.mlp(self.ln3(xs))
        return xs

时间注意力（time attention）在时间维度上对每个空间位置进行关注；空间注意力（space attention）在每一帧内对不同空间位置进行关注。这种设计将计算复杂度从单次 `O((THW)^2)` 降低为两次 `O(T^2 + (HW)^2)` 操作。这是 TimeSformer 以及所有现代视频 DiT（Diffusion Transformer）的核心机制。

### 步骤 4：构建微型视频 DiT

class TinyVideoDiT(nn.Module):
    def __init__(self, in_channels=4, dim=64, depth=2, heads=2):
        super().__init__()
        self.patch = VideoPatch3D(in_channels=in_channels, dim=dim, patch_t=2, patch_h=2, patch_w=2)
        self.blocks = nn.ModuleList([DividedAttentionBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, in_channels * 2 * 2 * 2)

    def forward(self, x):
        tokens, grid = self.patch(x)
        for blk in self.blocks:
            tokens = blk(tokens, grid)
        return self.out(tokens), grid

这并非一个可直接运行的视频生成器，而是一个结构演示，旨在确保各个组件的张量形状（tensor shapes）能够正确匹配。

### 步骤 5：检查张量形状

vid = torch.randn(1, 4, 8, 16, 16)  # (N, C, T, H, W)
model = TinyVideoDiT()
out, grid = model(vid)
print(f"input  {tuple(vid.shape)}")
print(f"tokens grid {grid}")
print(f"output {tuple(out.shape)}")

分块操作后，预期输出为 `grid = (4, 8, 8)` 和 `out = (1, 256, 32)`；随后投影头（head）会将结果映射为每个词元对应的时空分块，以便后续通过逆分块操作（un-patchify）还原为视频。

## 实际应用

2026 年生产环境接入方案：

- **Sora 2 API**（OpenAI）—— 文本生成视频（Text-to-Video），支持音画同步。采用高级定价策略。
- **Runway Gen-5 / GWM-1**（Runway）—— 图像生成视频（Image-to-Video），支持交互式世界。
- **Wan-Video 2.1 / HunyuanVideo** —— 开源模型，支持本地部署（Self-host）。
- **Cosmos / Cosmos-Drive**（NVIDIA）—— 驾驶仿真模型，开放权重（Open Weights）。
- **Genie 3** —— 研究预览版，需申请访问权限。

构建交互式世界模型（World Model）演示：建议以 Wan-Video 为基础保障生成质量，并叠加潜在动作适配器（Latent-Action Adapter）以实现交互性。针对自动驾驶仿真：Cosmos-Drive 是 2026 年的开源参考基准。

在机器人领域，实际落地的技术栈如下：

1. 语言目标 -> 视觉语言模型（Vision-Language Model, VLM）(Qwen3-VL) -> 高层规划。
2. 规划 -> 潜在动作视频模型（Latent-Action Video Model） -> 想象推演（Imagined Rollout）。
3. 推演结果 -> 逆动力学模型（Inverse Dynamics Model） -> 底层动作。
4. 执行动作 -> 观测结果反馈至步骤 1。

## 交付物

本章节将生成以下文件：

- `outputs/prompt-video-model-picker.md` —— 根据任务需求、许可证类型和延迟要求，在 Sora 2 / Runway / Wan / HunyuanVideo / Cosmos 之间进行模型选择。
- `outputs/skill-physical-plausibility-checks.md` —— 定义自动化检查规则（如物体恒存性、重力、连续性）的技能文件，用于在视频交付前对生成内容进行校验。

## 练习

1. **（简单）** 计算一段 5 秒、360p 分辨率视频在 `patch-t=2`、`patch-h=8`、`patch-w=8` 参数下的 Token 数量。分析在此规模下注意力机制（Attention）的内存占用情况。
2. **（中等）** 将上述的分离注意力模块（Divided Attention Block）替换为完整的联合注意力模块（Full Joint Attention Block），并测量其张量形状与参数量。解释为何分离注意力机制在实际视频模型中是必需的。
3. **（困难）** 构建一个最小化的潜在动作视频模型：使用包含 `(frame_t, action_t, frame_{t+1})` 三元组的数据集（任意简单的 2D 游戏均可），训练一个以动作嵌入（Action Embeddings）为条件的微型视频扩散 Transformer（Video DiT），并验证不同动作能够生成不同的下一帧。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 世界模型（World Model） | “学习到的模拟器” | 给定状态和动作，能够预测未来观测结果的模型 |
| 视频 DiT（Video DiT） | “时空 Transformer” | 采用三维分块（3D Patchification）和分离注意力机制的扩散 Transformer |
| 潜在动作（Latent Action） | “推断出的控制信号” | 从帧对推断出的离散或连续动作潜在变量；用于条件化下一帧的生成 |
| 分离注意力（Divided Attention） | “先时间后空间” | 每个模块包含两次注意力操作——先沿时间维度，再沿空间维度——以控制 O(N^2) 的计算复杂度 |
| 物体恒存性（Object Permanence） | “物体保持真实存在” | 视频模型必须学习的场景属性；在食物、玻璃器皿等物体上常出现经典失效模式 |
| FVD | “Fréchet 视频距离” | 视频领域的 FID 等效指标；主要的视觉质量评估指标 |
| 逆动力学模型（Inverse Dynamics Model） | “从观测到动作” | 给定（当前状态，下一状态），输出连接两者的动作；用于闭环机器人控制 |
| Cosmos-Drive | “NVIDIA 驾驶仿真器” | 开放权重的自动驾驶世界模型，用于强化学习（RL）与评估 |

## 延伸阅读

- [Sora 技术报告 (OpenAI)](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Genie：生成式交互环境 (Bruce et al., 2024)](https://arxiv.org/abs/2402.15391) — 潜在动作世界模型 (latent action world models)
- [TimeSformer (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) — 视频 Transformer 的分割注意力机制 (divided attention)
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) — 面向强化学习 (Reinforcement Learning, RL) 的世界模型 (world models)
- [Cosmos-Drive-Dreams (NVIDIA, 2025)](https://research.nvidia.com/labs/toronto-ai/cosmos-drive-dreams/) — 驾驶世界模型 (driving world model)
- [2026 年十大视频生成模型 (DataCamp)](https://www.datacamp.com/blog/top-video-generation-models)
- [从视频生成到世界模型 — 综述资源库 (survey repo)](https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model/)
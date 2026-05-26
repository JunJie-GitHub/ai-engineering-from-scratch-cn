# 任意分辨率视觉：分块打包（patch-n'-pack）与 NaFlex

> 真实图像并非 224x224 的正方形。收据的宽高比约为 9:16，图表为 16:9，医学扫描图可能高达 4096x4096，而手机截图则是 9:19.5。2024 年之前的视觉语言模型（Vision-Language Model, VLM）的通用解法是将所有图像强制缩放为固定正方形，但这会破坏支撑光学字符识别（Optical Character Recognition, OCR）、文档理解与高分辨率场景解析的关键信号。NaViT（Google, 2023）证明了可以通过块对角掩码（block-diagonal masking）将不同分辨率的图像块（patch）打包至同一个 Transformer 批次（batch）中。Qwen2-VL 的 M-RoPE（2024）则彻底移除了绝对位置编码表。LLaVA-NeXT 的 AnyRes 方案将高分辨率图像划分为基础图与子图进行平铺。SigLIP 2 的 NaFlex 变体（2025）现已成为开源 VLM 的默认视觉编码器，旨在让单一模型检查点（checkpoint）兼容任意宽高比。本课程将端到端实现分块打包（patch-n'-pack）流程。

**类型：** 构建实践
**编程语言：** Python（标准库，图像块打包器 + 块对角掩码）
**前置知识：** 第 12 阶段 · 01（ViT 图像块），第 12 阶段 · 05（LLaVA）
**预计耗时：** 约 120 分钟

## 学习目标

- 将一批不同分辨率图像的图像块（patch）打包为单一序列，并构建块对角注意力掩码（block-diagonal attention mask）。
- 针对特定任务，在 AnyRes 平铺（LLaVA-NeXT）、NaFlex（SigLIP 2）与 M-RoPE（Qwen2-VL）之间做出选择。
- 在不进行图像缩放的前提下，计算 OCR、图表与摄影任务的 Token 预算。
- 指出固定正方形缩放的三种失效模式：文本挤压变形、内容裁剪丢失、填充（padding）区域浪费 Token。

## 问题背景

Transformer 架构期望输入为序列数据。一个批次（batch）由长度相同的序列堆叠而成。如果你的图像都是 224x224，每次都会固定生成 196 个图像块 Token，无需填充，任务完成。在 224 分辨率下训练，在 224 分辨率下推理，从此无需再考虑分辨率问题。

但现实世界并不配合。文档通常是纵向的（8.5x11 英寸，宽高比约 2:3）。图表截图是横向的（16:9）。收据细长（1:3）。医学影像通常以 2048x2048 或更高分辨率交付。移动设备截图则为 1170x2532（宽高比约 0.46:1）。

2024 年之前的三种主流方案及其缺陷：

1. 缩放至固定正方形（224x224 或 336x336）。强制压缩会导致文本和人脸变形。降采样会破坏图表标签与 OCR 内容。这是 LLaVA-1.5 之前的标准做法。
2. 裁剪至固定宽高比。这会丢弃图像的大部分区域，且选择裁剪位置本身就是一个独立的视觉难题。
3. 按最长边进行填充（padding）。虽能避免形变，但处理纵向图像时会有超过 50% 的 Token 浪费在填充区域上。且注意力机制的计算成本与序列长度呈平方关系，这些填充 Token 会带来巨大的计算开销。

2024-2025 年的解决方案：让 Transformer 直接处理图像原始分辨率下的图像块，并研究如何将异构批次高效打包为单一序列，从而避免计算资源的浪费。

## 核心概念

### NaViT 与 patch-n'-pack（分块打包）

NaViT（Dehghani 等人，2023）这篇论文证明了该方法在大规模场景下的有效性。其核心机制如下：

1. 针对批次中的每张图像，按选定的补丁尺寸（例如 14）计算其原生补丁网格 (native patch grid)。
2. 将每张图像的补丁展平为各自的可变长度序列 (variable-length sequence)。
3. 将所有图像的补丁拼接成批次中的一个长序列。
4. 构建块对角注意力掩码 (block-diagonal attention mask)，确保图像 A 的补丁仅关注图像 A 内部。
5. 携带每个补丁的位置信息（二维旋转位置编码 2D RoPE 或分数位置嵌入 fractional position embeddings）。

一个包含三张图像（尺寸分别为 336x336/576 个 token、224x224/256 个 token、448x336/768 个 token）的批次，将被合并为一个包含 1600 个 token 的序列，并配备 1600x1600 的块对角掩码。无需填充 (padding)，无计算浪费。Transformer 能够处理任意宽高比的图像。

NaViT 还在训练中引入了分数补丁丢弃 (fractional patch dropping) 策略——在批次中随机丢弃 50% 的补丁——这既能起到正则化作用，又能加速训练。SigLIP 2 继承了这一设计。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是一种更务实的替代方案。给定一张高分辨率图像和一个固定编码器（如 336 分辨率的 CLIP 或 SigLIP），对图像进行分块 (tiling) 处理：

1. 从预定义的网格布局集合（如 1x1、1x2、2x1、1x3、3x1、2x2 等）中，选择最匹配图像宽高比的布局。
2. 将完整图像按网格切分；每个图块 (tile) 变为 336x336 的裁剪区域。
3. 同时生成缩略图：将整张图像缩放至 336x336，作为全局上下文 token (global-context token)。
4. 通过冻结的 336 分辨率编码器对每个图块进行编码。拼接图块 token 与缩略图 token。

对于 672x672 的图像采用 2x2 网格加缩略图：4 * 576 + 576 = 2880 个视觉 token。计算成本较高但效果显著——大语言模型 (LLM) 既能看到局部细节，又能掌握全局上下文。

当你的编码器处于冻结状态且仅支持单一分辨率时，AnyRes 是首选方案。但它会导致大图像的 token 数量激增（例如 1344x1344 的图像采用 4x4 网格会产生 9216 + 576 ≈ 9800 个 token，几乎占满 8k 上下文长度的 LLM）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置编码 (Multimodal Rotary Position Embedding)。与 NaViT 的分数位置或 AnyRes 的“图块+缩略图”不同，每个补丁携带一个三维位置信息（时间、高度、宽度）。查询/键 (query/key) 的旋转操作能够处理任意的高度 (H)、宽度 (W) 和时间长度。

M-RoPE 原生支持动态分辨率，且无需重新训练。在推理阶段，输入任意 HxW 尺寸的图像，补丁嵌入器 (patch embedder) 会生成 H/14 x W/14 个 token，每个 token 分配其 `(t=0, r=row, c=col)` 位置，RoPE 使用正确的频率旋转注意力机制，即可完成。Qwen2.5-VL 和 Qwen3-VL 延续了这一设计。InternVL3 的 V2PE 也是相同思路，但针对每种模态采用可变编码。

与 AnyRes 不同，M-RoPE 在原生分辨率下的 token 复杂度为 O(H x W / P^2)——没有图块带来的乘法级开销 (multiplicative tile overhead)。与 NaViT 不同，它在前向传播时仍期望每次处理单张图像。跨分辨率的批次处理仍需在顶层叠加 patch-n'-pack 技术。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 模型检查点 (checkpoint) 的原生灵活模式 (native-flex mode)。单个模型在推理时可支持多种序列长度（256、729、1024 个 token）。其内部在训练时采用 NaViT 风格的 patch-n'-pack，并为每个补丁分配绝对分数位置。核心卖点：仅需一个检查点，即可在推理时根据任务需求灵活选择 token 预算 (token budget)。

对于语义任务（分类、检索），使用 256 个 token；对于 OCR 或图表理解，使用 1024 个 token。全程无需重新训练。

### 打包掩码

块对角掩码是大多数实现容易出错的地方。对于长度为 `N_total` 的打包序列，涵盖图像 `i=0..B-1` 且各自长度为 `n_i`，形状为 `(N_total, N_total)` 的掩码 `M` 在两个索引均落在同一图像块内时为 1，否则为 0。你可以通过累积长度列表来构建它：

offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]

在 PyTorch 中，使用 `torch.block_diag` 或显式的 gather 操作只需一行代码即可实现。FlashAttention 的可变长度路径 (`cu_seqlens`) 完全跳过了掩码计算，直接利用累积长度张量在序列内部进行注意力计算——在典型批次中，其速度比密集掩码快约 10 倍。

### Token 预算

根据任务选择策略：

- OCR / 文档处理：1024-4096 个 token。可使用 SigLIP 2 NaFlex（1024 token 模式），或 AnyRes 3x3 网格加缩略图。
- 图表与 UI 界面：在 384-448 原生分辨率下使用 729-1024 个 token。Qwen2.5-VL 支持带最大像素上限 (max-pixels cap) 的动态分辨率。
- 自然照片：256-576 个 token 即可。下游 LLM 获取的信息已足够。将 token 预算分配给内容密度高的区域。
- 视频：空间池化 (spatial pooling) 后每帧 64-128 个 token，帧率 2-8 FPS。第 12.17 课将详细讲解此部分。

2026 年的生产环境准则：为每个任务设定最大像素上限，按原生宽高比编码至该上限，打包批次，并跳过填充操作。Qwen2.5-VL 提供了 `min_pixels` 和 `max_pixels` 参数，正是为了精确调节这一旋钮。

## 实际使用

`code/main.py` 实现了针对具有整数像素坐标的异构批次（heterogeneous batch）图像的补丁打包（patch-n'-pack）方法。它：

- 接收一个包含 (H, W) 图像尺寸的列表。
- 计算在补丁尺寸（patch size）为 14 时，每张图像的补丁序列长度。
- 将它们打包成一个总长度为 `sum(n_i)` 的序列。
- 构建块对角注意力掩码（block-diagonal attention mask）（为清晰起见使用稠密形式）。
- 对比打包成本与方形缩放（square-resize）及 AnyRes 分块（AnyRes tiling）的成本。
- 打印混合批次（收据、图表、截图、照片）的 Token 预算（token budget）表。

运行它。输出的数据正是 2026 年所有开源视觉语言模型（Vision-Language Model, VLM）均采用补丁打包策略的原因。

## 交付部署

本教程将生成 `outputs/skill-resolution-budget-planner.md`。给定混合长宽比的工作负载（OCR、图表、照片、视频帧）和总 Token 预算，它会选择合适的策略（NaFlex、AnyRes、M-RoPE 或固定方形），并输出每个请求的配置。在为产品规划 VLM 规模时使用此技能——它能避免导致延迟预算崩溃的隐性 10 倍 Token 膨胀。

## 练习题

1. 一张收据尺寸为 600x1500（长宽比 1:2.5）。在补丁尺寸为 14 时，原始分辨率下需要多少 Token？方形缩放至 336 后需要多少 Token？在实际应用中，哪种方式损失的 OCR 准确率更高？

2. 为包含四张长度分别为 256、576、729、1024 的图像批次构建块对角注意力掩码。验证注意力矩阵是否为 2585x2585，且恰好包含 `256^2 + 576^2 + 729^2 + 1024^2` 个非零元素。

3. 对于一张 1792x896 的图像（补丁尺寸为 14），对比以下方案：(a) 方形缩放至 336 后进行编码，(b) AnyRes 2x1 分块 + 缩略图，(c) 原始分辨率下使用 M-RoPE。哪种方案使用的 Token 最少？哪种方案保留的细节最多？

4. 实现分数补丁丢弃（fractional patch dropping）：给定一个已打包的序列，均匀随机丢弃 50% 的 Token，并相应更新块对角注意力掩码。测量掩码稀疏度（sparsity）的变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）的第 3.2 节。用两句话说明 `min_pixels` 和 `max_pixels` 控制的内容，以及为何这两个边界都至关重要。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Patch-n'-pack | “NaViT 风格的打包” | 将来自不同图像的变长图像块（patch）序列拼接至同一个批次（batch）维度中 |
| Block-diagonal mask | “打包掩码” | 注意力掩码（attention mask），用于限制每张图像的图像块仅关注自身，而不关注打包序列中的相邻图像块 |
| AnyRes | “LLaVA-NeXT 分块” | 将高分辨率图像划分为固定大小的网格图块（tile），并附加一个全局缩略图；使用固定编码器对每个图块进行编码 |
| NaFlex | “SigLIP 2 原生弹性” | 单个 SigLIP 2 模型检查点（checkpoint），在推理时无需重新训练即可支持 256/729/1024 个词元（token）的预算 |
| M-RoPE | “多模态 RoPE” | 三维旋转位置编码（rotary position encoding，涵盖时间、行、列），无需位置表即可处理任意高度（H）、宽度（W）和时间（T）维度 |
| cu_seqlens | “FlashAttention 打包” | 累积长度张量（cumulative-length tensor），FlashAttention 的变长路径（varlen path）使用它来替代密集的分块对角掩码 |
| min_pixels / max_pixels | “分辨率边界” | Qwen2.5-VL 中针对每次请求的调节参数，用于限制极小或极大输入时的词元数量上限 |
| Visual token budget | “每张图像的词元数量” | 每张图像生成的图像块词元的大致数量；决定了大语言模型（LLM）的提示词预算和注意力计算成本 |

## 延伸阅读

- [Dehghani 等人 — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang 等人 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon 等人 — 构建视觉语言模型时哪些因素至关重要？(Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen 团队 — Qwen2.5-VL 技术报告 (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
# LLaVA-OneVision：单图像、多图像、视频统一模型

> 在 LLaVA-OneVision（Li 等人，2024 年 8 月）问世之前，开放视觉语言模型（open-VLM）领域存在各自独立的发展路线：LLaVA-1.5 专注于单图像，Mantis 和 VILA 等模型处理多图像，Video-LLaVA 和 Video-LLaMA 等模型则针对视频。这些模型各自在特定基准测试中表现优异，却在其他场景下表现不佳。LLaVA-OneVision 提出，通过单一的训练课程（curriculum）即可让一个模型在所有三种场景中占据主导地位，并且其涌现式任务迁移效应（emergent task-transfer effects）（将单图像技能迁移至视频，将多图像推理能力迁移至单图像）能够超越多个专用模型（specialists）的总和。该方案看似简单，实则精妙：设定一个跨场景保持恒定的视觉 token 预算（visual-token budget），并配合明确的训练课程，从单图像逐步过渡到 OneVision（多图像），再到视频。本节将深入解析该预算机制、训练课程设计以及涌现出的模型行为。

**类型：** 构建
**编程语言：** Python（标准库，token 预算求解器 + 课程规划器）
**前置要求：** 第 12 阶段 · 05（LLaVA），第 12 阶段 · 06（任意分辨率）
**预计耗时：** 约 180 分钟

## 学习目标

- 设计一种视觉 token 预算（visual-token budget），使其在单图像、多图像和视频输入中保持恒定。
- 规划训练课程（training curriculum），实现从单图像到视频的技能迁移，同时避免灾难性遗忘（catastrophic forgetting）。
- 解释在训练课程设计合理的情况下，为何单一模型能在相同参数量下超越专用模型（specialists）。
- 列举 LLaVA-OneVision 报告中提到的三种涌现能力（emergent capabilities）：多摄像头推理（multi-camera reasoning）、标记集合提示（set-of-mark prompting）以及 iPhone 截图智能体（iPhone-screenshot agent）。

## 核心问题

单图像、多图像和视频对模型的压力点各不相同。

单图像任务需要高分辨率 token（AnyRes，约 2880 个视觉 token）以捕捉 OCR 文本和精细细节。单样本预算：1 张图像，2880 个 token。

多图像任务需要多张中等分辨率的图像（每张约 576 个 token），以便跨图像推理能够适应上下文窗口。单样本预算：4-8 张图像，每张 576 个 token，总计 2300-4600 个 token。

视频任务需要大量低分辨率帧（池化后每帧约 196 个 token）以捕捉时间动态特征。单样本预算：8-32 帧，每帧 196 个 token，总计 1600-6200 个 token。

如果分别训练独立模型，你只需为每种场景选定一种预算。但如果要训练单一统一模型，则必须让预算在不同场景间合理扩展，同时避免超出上下文窗口限制。

在 OneVision 出现之前，业界的默认做法是“专攻单一场景，忽略其他场景”。Video-LLaVA 通过增加额外的训练阶段，将视频能力“嫁接”到图像模型上；LLaVA-NeXT 则通过分块（tiling）技术引入了多图像支持。但没有任何一种方案能干净利落地同时兼顾这三种场景。

## 核心概念

### OneVision 的 Token 预算（Token Budget）
LLaVA-OneVision 为每个样本设定了统一的视觉 Token 预算（Visual-Token Budget），约为 3000-4000 个 Token，并根据不同场景进行差异化分配：

- 单张图像：采用 AnyRes-9（3x3 图块 + 缩略图），每个图块分辨率为 384，包含 729 个图像块（Patch）。采用激进的 2x2 双线性池化（Bilinear Pooling）→ 每个图块降至 182 个。总计：9 * 182 + 182 = 1820 个 Token。或者采用 AnyRes-4，每个图块 729 个 Token = 2916 + 729。
- 多张图像：每张图像采用中等分辨率（384，不切分图块），无池化处理，每张 729 个 Token。预算支持 6 张图像 → 4374 个 Token。
- 视频：32 帧，分辨率 384，采用激进的 3x3 双线性池化 → 每帧 81 个 Token。总计：32 * 81 = 2592 个 Token。

该分配策略使总 Token 数量大致保持恒定。大语言模型（LLM）永远不会遇到超出其上下文（Context）限制的批次（Batch）。尽管编码器（Encoder）在不同场景下会生成不同的几何结构，但 LLM 消耗的 Token 预算始终保持一致。

### 三阶段课程学习（Curriculum Learning）
LLaVA-OneVision 的训练分为三个阶段：

1. 单图像监督微调（Single-Image SFT，阶段 SI）。所有数据均为“单张图像+文本”格式。使用高分辨率 AnyRes 输入进行训练。此阶段旨在培养模型的感知能力、光学字符识别（OCR）能力以及细粒度理解能力。数据采用 LLaVA-NeXT 数据集加上 OneVision 专属的单图像数据。
2. OneVision 监督微调（OneVision SFT，阶段 OV）。混合单图像、多图像与视频（均匀采样帧）数据。在统一的 Token 预算下进行训练。此阶段旨在让模型学会处理异构的批次形状（Batch Shapes）。不重置权重——直接从阶段 SI 继续训练。
3. 任务迁移（Task Transfer，阶段 TT）。继续使用目标任务混合数据进行训练，通常根据具体产品需求，增加多图像或视频数据的比重。此为面向部署的可选微调阶段。

关键点：课程学习的顺序至关重要。即使使用相同的数据，先训练视频或多图像，其单图像性能也会劣于先训练单图像的方案。论文中对此进行了明确的消融实验（Ablation Study）验证。

### 课程学习为何有效
单图像训练构建了感知基础。图像块 Token 携带细粒度的视觉特征，LLM 学习将其与文本进行融合。多图像和视频引入了结构性挑战（例如区分哪张图像对应哪个视角、事件发生的先后顺序），若缺乏强大的感知基础，这些挑战将难以学习。

如果从一开始就将所有场景混合训练，模型会在感知上欠拟合（Underfit）（每个批次中单图像数据有限），而在结构上过拟合（Overfit）（多图像/视频数据过多）。最终结果是：模型虽然能遵循跨图像推理模式，但视觉理解能力流于表面。

课程学习的顺序安排使模型在阶段 SI 获得强大的感知能力，随后在阶段 OV 习得组合/时序推理能力，且两者均不会丢失。

### 涌现的跨场景能力
LLaVA-OneVision 论文报告了三种涌现能力（Emergent Capabilities）：

1. 多摄像头推理。模型分别使用多图像和视频数据进行训练；在推理阶段，被要求对多摄像头驾驶场景进行推理。尽管训练时从未见过完全相同的格式，模型仍能正确整合不同视角的信息。
2. 标记集合提示（Set-of-Mark Prompting）。用户使用带编号的标记对图像中的物体进行标注；模型需推理“标记 3 相对于标记 7 在做什么”。模型既未针对标记也未针对标注进行过专门训练；该能力源于空间定位（Spatial Grounding）与多图像参考能力的结合。
3. iPhone 截图智能体（Agent）。用户提供 iPhone 屏幕截图并要求规划下一步点击操作。模型使用 UI 截图、用户工作流视频以及多图像“操作前/后”对比对进行训练。该能力成功泛化至智能体应用场景。

这些并非专门训练的任务，而是从课程学习的组合结构中自然涌现出来的。

### 视觉 Token 池化
Token 预算的限制要求进行池化处理。OneVision 在二维图像块网格（2D Patch Grid）上使用双线性插值（Bilinear Interpolation）：24x24 = 576 个图像块可缩减为 12x12 = 144 个（2 倍因子）或 8x8 = 64 个（3 倍因子）。池化操作在图像块网格空间而非 Token 空间中进行，以保留局部性（Locality）。

每个场景的池化因子选择本身就是一个超参数（Hyperparameter）。池化程度越低 = Token 越多 = 表征（Representation）越丰富。池化程度越高 = Token 越少 = 能容纳的帧/图像数量越多。

### LLaVA-OneVision-1.5
2025 年的后续版本（LLaVA-OneVision-1.5，arXiv 2509.23661）在训练数据、模型权重和代码方面实现了“完全开源”。该版本在部分基准测试中缩小了与闭源模型的差距，并普及了该训练方案。其采用相同的课程学习策略，增加了数据量，并使用了更优的基础 LLM。模型架构未作更改。

### 与 Qwen2.5-VL 的对比
Qwen2.5-VL（Lesson 12.09）做出了不同的技术选择。它采用 M-RoPE 和动态帧率（Dynamic FPS），而非固定池化。其 Token 预算随输入动态变化——1 分钟视频消耗的 Token 多于 5 秒视频。而 LLaVA-OneVision 固定预算，通过调整池化比例来适应输入。两种方案均有效，本质上是在配置灵活性与可预测性之间进行权衡。

## 使用方法

`code/main.py` 是一个面向 OneVision 风格视觉语言模型（Vision-Language Model, VLM）的课程学习（Curriculum Learning）与预算规划器。在给定单样本 Token 预算（Token Budget）和目标场景混合比例（例如 40% 单图像、30% 多图像、30% 视频）的情况下，它会：

- 为每个场景分配分辨率、池化因子（Pooling Factor）和帧数。
- 检查每个场景是否都在共享预算范围内。
- 报告预期的 Token 数量、大语言模型（Large Language Model, LLM）浮点运算次数（FLOPs），以及哪些场景存在 Token 分配不足（Under-tokenized）的情况。
- 打印分阶段的训练计划。

可使用它来规划 OneVision 的微调（Fine-tuning），或用于合理性校验 VLM 部署时的单次请求成本。

## 交付与发布

本课时将生成 `outputs/skill-onevision-budget-planner.md`。在给定目标任务分布和单样本预算的情况下，它会输出 AnyRes 因子、逐帧池化（Per-frame Pooling）参数、视频帧数以及课程学习阶段权重。每当您训练或微调统一场景（Unified-scenario）VLM 时，均可使用此工具。

## 练习

1. 您的产品支持 80% 单图像、10% 多图像（2-4 张）和 10% 视频（8-16 帧）。请设计 Token 预算。您会将因减少重度多图像处理而节省下来的额外预算分配到哪里？
2. 阅读 LLaVA-OneVision 论文第 4.3 节（涌现能力）。提出一种课程学习可能解锁但论文未报告的第四种涌现技能（Emergent Skill）。
3. 交换课程学习顺序——先训练多图像，再训练单图像，最后训练视频。预测哪些基准测试（Benchmark）的性能会下降，并说明原因。
4. 论文报告的视频基准测试仅使用每样本 8 帧进行训练。这能否泛化（Generalize）到推理阶段的 30 秒视频？首先失效的会是 Token 预算还是时间推理（Temporal Reasoning）能力？
5. 将 24x24 的图像块（Patch）通过双线性池化（Bilinear Pooling）缩减至 12x12，相当于每个维度减少 4 倍。请使用 Python 标准库实现该池化操作，并验证每个 2x2 块的平均值是否与双线性输出一致。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| OneVision 场景（OneVision Scenario） | “单图、多图或视频” | 统一 VLM 处理的三种输入形态之一；各场景间预算保持恒定 |
| Token 预算（Token Budget） | “每个样本包含多少 Token” | 大语言模型在每次训练/推理样本中看到的视觉 Token 总数，通常为 3000-4000 |
| 课程学习（Curriculum） | “训练顺序” | 为促成能力涌现与迁移而设定的阶段顺序（单图像 → 多图像 → 视频） |
| 双线性池化（Bilinear Pooling） | “Token 压缩” | 对二维图像块网格应用双线性插值，在保留局部特征的同时减少 Token 数量 |
| 涌现技能（Emergent Skill） | “未专门训练却依然有效” | 由于课程组合而在推理阶段显现的能力，无需匹配的训练数据即可生效 |
| AnyRes-k | “k 瓦片配置” | k 个固定分辨率的子瓦片（Sub-tiles）加上一个缩略图，通常 k ∈ {4, 9} |
| 任务迁移（Task Transfer） | “跨场景泛化” | 通过共享骨干网络（Backbone），将在单图像上学到的技能应用于视频（反之亦然） |

## 扩展阅读

- [Li 等 — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5：完全开放框架 (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin 等 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin 等 — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang 等 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
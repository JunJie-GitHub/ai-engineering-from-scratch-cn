# 视频-语言模型 (Video-Language Models)：时序词元 (Temporal Tokens) 与定位 (Grounding)

> 视频并非照片的简单堆叠。一段 5 秒的片段包含因果顺序、动作动词和事件时序，这些是图像模型无法表征的。Video-LLaMA（Zhang 等人，2023 年 6 月）发布了首个具备视听定位 (audio-visual grounding) 能力的开源视频大语言模型 (video-LLM)。VideoChat 和 Video-LLaVA 沿袭并扩展了这一范式。到 2025 年，Qwen2.5-VL 的 TMRoPE 缩小了与前沿闭源模型之间的差距。各系统对时序词元 (temporal tokens) 的处理方式各不相同——Video-LLaMA 采用按片段 (per clip) 的 Q-former，Video-LLaVA 采用按帧 (per frame) 的拼接池化 (concat-pool)，而 Qwen2.5-VL 则采用按词元 (per token) 的 TMRoPE。本课程将剖析这些设计模式，构建均匀采样与动态采样的帧采样器 (frame sampler)，并在时序定位 (temporal grounding) 任务上进行评估。

**类型：** 构建 (Build)
**语言：** Python（标准库，帧采样器 + 时序定位评估器）
**前置要求：** 第 12 阶段 · 08 (LLaVA-OneVision)
**时长：** 约 180 分钟

## 学习目标

- 解释为何时序位置编码 (temporal positional encoding) 会独立于视觉编码器 (vision encoder) 影响视频视觉语言模型 (video VLM) 的性能。
- 对比均匀采样、动态帧率 (dynamic-FPS) 采样与事件驱动 (event-driven) 帧采样在每秒词元数 (tokens-per-second) 与定位准确率 (grounding accuracy) 之间的权衡。
- 描述按片段使用 Q-former（Q-former-per-clip，Video-LLaMA）、按帧池化（pooled-per-frame，Video-LLaVA）与按词元使用 M-RoPE（M-RoPE-per-token，Qwen2.5-VL）的设计差异。
- 列举四大视频基准测试：VideoMME、TempCompass、EgoSchema 和 Video-MMMU。

## 问题背景

一段 1 分钟、30 FPS 的视频包含 1800 帧。若每帧生成 196 个视觉词元 (visual tokens)（基于 224 分辨率的 ViT-B），总计将达 35.2 万个词元——这超出了 2024 年时期任何大语言模型 (LLM) 的上下文窗口 (context) 限制。

目前存在三种缩减策略 (reduction strategies)：

1. 帧下采样 (Subsample frames)（根据内容降至 1-8 FPS）。
2. 对每帧的图像块词元 (patch tokens) 进行激进池化（采用 3x3 或 4x4 双线性池化 (bilinear pool)）。
3. 通过 Q-former 进行压缩（输入 16 帧片段，输出 64 个词元）。

每种策略的权衡 (trade-off) 各不相同。下采样会损失时序细节，池化会损失空间细节，而 Q-former 虽会轻微损失两者，但能大幅节省词元数量。

时序位置编码 (temporal position encoding) 是另一个关键维度：模型如何知道第 5 帧在第 6 帧之前？可选方案包括简单的一维时序旋转位置编码 (1D temporal RoPE，Video-LLaMA)、可学习时序嵌入 (learned temporal embeddings，Video-LLaVA)，以及完整的三维 TMRoPE（Qwen2.5-VL，full 3D）。

## 核心概念

### Video-LLaMA：按片段划分的 Q-former + 音频分支

Video-LLaMA（2023）是首个开源的视频大语言模型（Video-LLM）。其架构如下：

- 以 2 FPS（帧率）采样，每段包含 16 帧（即 8 秒时长）。
- 逐帧的视觉变换器（ViT）特征 -> 视频 Q-former（对所有 16 帧执行交叉注意力（cross-attention）） -> 32 个可学习查询向量（learned queries） -> 大语言模型（LLM）。
- 并行音频分支：音频波形 -> ImageBind 音频编码器 -> 音频 Q-former -> 32 个查询向量 -> LLM。

优势：视听联合推理（audio-visual joint reasoning）。劣势：片段长度固定，无法进行任意时间点的定位（time grounding）。

### VideoChat 与 Video-LLaVA

VideoChat 延续了 Video-LLaMA 的思路，但去除了音频模块并进行了简化。Video-LLaVA（Lin 等人，2023）在图像和视频帧上共同训练单一视觉编码器（“先对齐后投影”），从而获得统一的表征。两者均采用“冻结的 CLIP 编码器 + 多层感知机（MLP） + 大语言模型（LLM）”的架构。

两者均无法处理长视频，都是基于 8-16 帧的系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE（时序-模态旋转位置嵌入，Temporal-Modality Rotary Position Embedding）。每个图像块令牌（patch token）携带一个 `(t, h, w)` 位置坐标，其中 `t` 为实际时间戳（而非帧索引）。

与简单时序嵌入（temporal embedding）的关键区别在于：

- 绝对时间而非索引。模型感知的是“在 4.2 秒处”，而非“在第 15 帧处”。
- 按令牌旋转而非按片段旋转。每个视觉令牌根据其时间戳独立进行旋转。
- 兼容动态帧率（dynamic FPS）。若某处采样率为 2 FPS，另一处为 4 FPS，TMRoPE 能原生处理这种非均匀间隔。

TMRoPE 使得模型能够回答“猫在第几秒跳跃？”这类查询。模型可直接输出“在 4.2 秒处”。而 Video-LLaMA 只能回答“在片段早期”。

### 帧采样策略

均匀采样（Uniform）：在视频时长内均匀抽取 N 帧。实现简单，但会丢失运动峰值信息。

动态帧率（Dynamic FPS）：根据运动强度自适应采样。通过光流法（optical flow）或帧差分法选取高运动片段进行密集采样。Qwen2.5-VL 即采用此策略进行训练。

事件驱动（Event-driven）：运行轻量级检测器，在动作发生区域增加采样密度。VideoAgent 采用了该方法。

关键帧 + 上下文（Keyframe + context）：在镜头边界处采样，并附加少量相邻帧。适用于影视类内容。

### 逐帧池化

在 1 FPS 且每帧 576 个令牌的设定下，一段 5 分钟的视频将产生 172,800 个令牌。虽然 Qwen2.5-VL-72B 的 128k 上下文窗口能够容纳，但计算成本极高。

采用 3x3 双线性池化（bilinear pooling）可将每帧降至 64 个令牌 -> 5 分钟视频共 19,200 个令牌。这是大多数任务的最佳平衡点。

对于对空间细节要求较低的智能体工作流（agent workflows），可采用更激进的池化策略（6x6 -> 每帧 16 个令牌）。

### 四大视频基准测试

- VideoMME：综合性视频理解，涵盖短、中、长视频。
- TempCompass：细粒度时序推理（temporal reasoning），侧重“之前”/“之后”类问题。
- EgoSchema：长程第一人称视角视频。
- Video-MMMU：多模态多学科视频问答。

完整的视频视觉语言模型（video-VLM）评估需覆盖全部四项。它们侧重不同维度：TempCompass 专注于事件排序，EgoSchema 考验 3 分钟以上的长程推理能力，VideoMME 则覆盖不同时长跨度。

### 定位输出格式

时序定位（temporal grounding）的输出格式包括：

- 自由文本：“猫大约在第 4 秒跳跃。”易于解析但精度较低。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 针对此格式进行了训练。
- 基于令牌（Token-based）：在回答中穿插特殊的 `<time>4.1</time>` 令牌。这是 Qwen2.5-VL 的内部格式。

基于令牌的格式在下游应用中最精确。Qwen2.5-VL 的 JSON 输出格式则可直接解析。

### 2026 年最佳实践

针对 2026 年的视频视觉语言模型（video VLMs）：

- 编码器：采用 M-RoPE 或 TMRoPE 的 SigLIP 2（如 Qwen2.5-VL）。
- 帧采样：动态帧率（根据运动强度在 1-4 FPS 间调整），并设置最大帧数上限。
- 逐帧池化：3x3 双线性池化。
- 输出：包含时间与事件字段的结构化 JSON。
- 基准测试：通用场景使用 VideoMME + TempCompass；长程场景使用 EgoSchema。

## 上手使用

`code/main.py` 包含：

- 均匀（Uniform）与动态帧率（Dynamic FPS）帧采样器。
- 一个示例级时间定位（Temporal Grounding）评估器：给定时间 T 的“真实标签（Ground Truth）”事件与模型输出，在容差（Tolerance）范围内计算准确率得分。
- 针对 Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）与 Qwen2.5-VL（动态帧率 + TMRoPE）的横向对比。

## 产出物

本课时将生成 `outputs/skill-video-vlm-frame-planner.md`。针对具体的视频任务（如监控、动作识别、时间定位、视频摘要），它会为你选定帧采样器、池化因子（Pooling Factor）、输出格式以及预期的准确率等级。

## 练习题

1. 针对一段 3 分钟的烹饪演示视频，在均匀采样与动态帧率（Dynamic FPS）采样之间做出选择。请结合 Token 数量说明理由。

2. TMRoPE 具体引入了哪些机制，是简单的时间嵌入表（Temporal Embedding Table）无法实现的？

3. 编写一个用于时间定位（Temporal Grounding）的 JSON Schema，使视觉语言模型（VLM）能够学习并输出该格式。需包含异常场景（Error Cases）的定义。

4. 阅读 Video-LLaVA 论文第 3 节关于“投影前对齐（Alignment Before Projection）”的论述。为何该方法优于分别训练独立的图像与视频编码器？

5. 参考 VideoMME 排行榜，截至 2026 年，顶尖开源模型与顶尖闭源（Proprietary）模型之间的性能差距有多大？其中有多少差距可归因于时间编码（Temporal Encoding）与基础大语言模型（Base LLM）参数规模的差异？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 时间定位（Temporal Grounding） | “带时间戳的答案” | VLM 输出事件发生的具体时间戳范围 |
| TMRoPE | “时间多模态旋转位置编码” | 结合绝对时间戳的 3D 旋转位置编码，由 Qwen2.5-VL 采用 |
| 动态帧率（Dynamic FPS） | “运动感知采样” | 在运动剧烈的片段采样更多帧，在静态片段采样较少帧 |
| 帧池化（Frame Pooling） | “逐帧空间压缩” | 在输入 LLM 前，通过双线性插值减少每帧的图像块（Patch）数量 |
| 视频 Q-former（Video Q-former） | “视频片段压缩器” | 通过交叉注意力（Cross-Attention）瓶颈层，将 N 帧映射为 K 个可学习查询向量 |
| VideoMME | “视频评测基准” | 涵盖短/中/长视频的综合评测基准，包含 2500+ 样本 |

## 延伸阅读

- [Zhang 等人 — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li 等人 — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin 等人 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen 团队 — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin 等人 — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
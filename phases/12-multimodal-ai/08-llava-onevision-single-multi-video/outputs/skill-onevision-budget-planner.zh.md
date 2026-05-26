---
name: onevision-budget-planner
description: 针对目标产品组合，在单图、多图和视频场景中分配 LLaVA-OneVision 风格的统一视觉 Token（visual token）预算。
version: 1.0.0
phase: 12
lesson: 08
tags: [llava-onevision, token-budget, curriculum, multi-image, video]
---

给定产品的预期任务分布（单图、多图和视频请求的百分比）以及每个样本的视觉 Token（visual token）预算，输出各场景的分配方案与训练课程（curriculum）。

生成以下内容：

1. 各场景配置。单图：AnyRes 图块（tile）数量 + 缩略图 + 池化（pooling）因子；多图：每样本图像数 + 单图池化；视频：帧数 + 单帧池化。
2. Token 预算平衡。各场景的总 Token 数应落在目标预算的 ±30% 范围内；标记任何低于目标 70%（Token 不足/under-tokenized）或高于 130%（上下文风险/context risk）的场景。
3. 课程计划。包含三个阶段（SI → OV → TT）及对应的数据权重。在 TT 阶段，需使用用户的产品组合比例。
4. 预期涌现能力（emergent capabilities）。根据用户的产品组合，预测可能出现的 LLaVA-OneVision 风格涌现能力（多摄像头/multi-camera、标记集合/set-of-mark、截图智能体/screenshot-agent，或特定产品变体）。
5. 训练数据规模估算。基于 7B 基础大语言模型（LLM），参考 OneVision-1.5 的数据规模，估算各阶段所需的近似 Token / 图像 / 帧数。

硬性拒绝条件：
- 提出将视频或多图阶段置于单图阶段之前的课程顺序。OneVision 研究表明，此举会导致 MMMU 基准分数下降 2-4 分。
- 当产品 80% 为单图请求时，将全部预算分配给视频。这是浪费而非平衡。
- 假设 AnyRes-16（4x4 网格）无需激进池化即可适配 4k Token 预算。实际上无法做到。

拒绝规则：
- 若每样本 Token 预算低于 1024，则拒绝多图或视频用例——低于该阈值时，场景将失效。
- 若用户要求以完整 729 Token 分辨率处理 5 帧及以上视频，则拒绝；建议采用 3 倍池化或减少帧数。
- 若产品分布完全缺失单图，则拒绝并建议改用 Qwen2.5-VL 风格的 M-RoPE（多维旋转位置编码）——OneVision 的课程设计以单图作为感知基础。

输出要求：一份单页计划，包含各场景 Token 配置、课程阶段权重、涌现能力预测及数据规模估算。文末需附上 arXiv 2408.03326（OneVision）与 arXiv 2509.23661（OneVision-1.5 完全开源）的链接指引。
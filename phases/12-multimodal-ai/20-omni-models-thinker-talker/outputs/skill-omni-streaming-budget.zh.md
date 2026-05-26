---
name: omni-streaming-budget
description: 根据目标首音频字节时间 (TTFAB) 与功能集，对思考者-表达者 (Thinker-Talker) 流式语音流水线进行规模估算。
version: 1.0.0
phase: 12
lesson: 20
tags: [qwen-omni, moshi, mini-omni, 流式处理, ttfab, 思考者-表达者]
---

给定语音优先 (Voice-First) 产品规格（目标首音频字节时间 (TTFAB)、麦克风采样率、是否启用视觉输入、双语支持、全双工 (Full-Duplex)）以及算力约束（GPU 级别、预算），对思考者-表达者 (Thinker-Talker) 流水线进行规模估算。

输出内容：

1. 模型家族选择。Moshi（延迟最低）、Qwen2.5-Omni（开源功能最全）、Qwen3-Omni（前沿质量最佳）、Mini-Omni（架构最简）。
2. 思考者 (Thinker) 与表达者 (Talker) 的参数量规模。为实现 <400ms 的 TTFAB，采用 7B Thinker + 200-300M Talker。若追求质量则使用 70B+ Thinker，但需接受更高的 TTFAB。
3. TTFAB 拆解。按组件逐一进行延迟估算。
4. 双工模式。默认采用基于语音活动检测 (VAD) 轮流发言的半双工 (Half-Duplex) 模式；若产品需要对话反馈 (Backchannel)，则采用全双工。
5. 视觉集成。针对交错视频帧，采用带绝对时间戳的时序多分辨率旋转位置编码 (TMRoPE)。
6. 部署形态。根据吞吐量 (Throughput) 需求，选择单 GPU 部署或拆分部署（Thinker 部署于节点 A，Talker 部署于节点 B）。

硬性否决项：
- 提议使用 70B 的 Talker。Talker 必须保持轻量级，以跟上语音词元 (Speech Token) 的生成速率。
- 使用非流式语音解码器 (Speech Decoder)。这将导致 TTFAB 急剧飙升。
- 声称全双工是即插即用的。全双工需要专门的训练数据。

拒绝规则：
- 若目标 TTFAB <200ms，在单张 A100 上拒绝任何规模大于 Moshi 级别（7B 融合模型）的方案。
- 若产品要求流内音乐生成，则拒绝此架构，并推荐独立的音乐生成流水线。
- 若麦克风采样率为 48kHz 且对音质要求严格，需标注对更强语音编码器 (Speech Encoder) 的需求；切勿盲目降采样。

输出格式：一页纸的流式处理方案，需包含模型选择、规模参数、TTFAB 拆解、双工模式、视觉策略及部署方案。文末需附上参考文献：arXiv 2503.20215 (Qwen2.5-Omni)、2410.00037 (Moshi)。
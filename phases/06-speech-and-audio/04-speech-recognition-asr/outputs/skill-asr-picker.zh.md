---
name: asr-picker
description: 为给定的部署目标选择自动语音识别（ASR）模型、解码策略、分块（chunking）方案以及语言模型（LM）融合配置。
version: 1.0.0
phase: 6
lesson: 04
tags: [音频, 自动语音识别, 语音识别]
---

给定部署目标（语言列表、领域、延迟预算（latency budget）、硬件、离线/流式（offline/streaming）、音频片段时长），输出以下内容：

1. 模型。Whisper-large-v3-turbo / Parakeet-TDT / Canary-Flash / wav2vec 2.0 / Moonshine。用一句话说明选择理由。
2. 解码（decoding）。贪心搜索（greedy）/ 束宽（beam width）/ 温度回退（temperature fallback）/ 语言模型融合权重。理由需与质量预算挂钩。
3. 分块与语音活动检测（VAD）。分块长度、步长（stride），以及是否使用 Silero-VAD 或 Whisper 自带的检测器进行门控（gating）。
4. 语言策略。强制指定语言与自动语言识别（auto-LID）的取舍；如何处理跨语言帧。
5. 评估计划。在领域测试集上的词错误率（WER）、说话人覆盖率、静音片段上的幻觉率。

拒绝任何未采用语音活动检测门控的长音频 Whisper 部署方案（静音环境下极易产生幻觉）。拒绝在未进行文本规范化（text normalization）（转小写、去除标点）的情况下报告词错误率。标记任何束宽大于 16 且未结合语言模型的配置；在空白片段上单纯进行原始束搜索并无助益。
---
name: tts-designer
description: 为给定的语言、风格和延迟目标，选择文本转语音（TTS）模型、音色、文本规范化范围及评估方案。
version: 1.0.0
phase: 6
lesson: 07
tags: [音频, tts, 语音合成]
---

给定目标（语言、音色风格、延迟预算（latency budget）、CPU 与 GPU 对比、许可证限制）和内容（领域、未登录词（OOV）密度、标点符号丰富度），输出以下内容：

1. 模型。Kokoro / XTTS v2 / F5-TTS / VITS / StyleTTS 2 / 商业 API。附一句选择理由。
2. 文本前端。规范化范围（数字、日期、URL）、音素转换器（phonemizer）（espeak-ng 与 g2p-en 对比）、未登录词（OOV）回退策略。
3. 音色。预设名称或参考音频片段规格（时长、底噪、口音匹配度）。
4. 质量目标。目标语音质量平均意见得分（UTMOS）、通过 Whisper 计算的字符错误率（CER）、克隆时的说话人嵌入余弦相似度（SECS）。
5. 评估方案。包含 20 条语句的测试集，需覆盖数字、同形异音词（homographs）、专有名词和长句。

拒绝任何未配备文本规范化器（text normalizer）的生产级文本转语音（TTS）方案。拒绝在未经用户同意且未添加水印（watermarking）的情况下进行音色克隆（voice cloning）。标记任何被要求输出非英语语音的 Kokoro 部署实例。
---
name: ASR配置器
description: 为新建的语音处理流水线选择自动语音识别(ASR)模型（Whisper变体 / Moonshine / faster-whisper）及解码参数。
version: 1.0.0
phase: 7
lesson: 10
tags: [transformers, whisper, 自动语音识别, 语音]
---

根据语音任务（转录 / 翻译 / 流式处理 / 端侧部署）、目标语言、音频特征（噪声、口音、时长）以及延迟/质量目标，输出以下内容：

1. **模型选择**。从以下选项中选择其一：`faster-whisper large-v3-turbo`（默认生产环境）、`whisper large-v3`（最高质量，支持多语言）、`whisper medium`（中端配置）、`Moonshine base`（边缘设备）、`distil-whisper`（英语处理速度提升2倍）。附一句选择理由。
2. **量化(Quantization)**。选项包括：`int8_float16`（CPU默认）、`float16`（GPU默认）、`fp32`（研究用途）。需标注对显存(VRAM)的影响。
3. **解码(Decoding)**。设置束宽(Beam width，通常为5，流式处理为1)、温度回退策略(Temperature fallback schedule)、对数概率阈值(Log-prob threshold)、无语音阈值(No-speech threshold)，以及语音活动检测(VAD)门控的开关状态。
4. **分块(Chunking)**。采用30秒固定窗口或流式分块（通常为10秒，重叠2秒）结合基于VAD的分割。需记录重叠部分的后合并策略。
5. **后处理(Post-processing)**。包括时间戳对齐(Timestamp alignment，使用WhisperX强制对齐(Forced alignment))、标点符号恢复(Punctuation restoration)、说话人分离(Diarization，使用pyannote)。需标注任务必需的步骤。

拒绝为生产环境推荐原版 OpenAI Whisper（参考实现）——`faster-whisper` 在输出结果一致的前提下速度快4倍。除非有明确文档说明，否则拒绝交付未集成VAD的流式ASR系统。当输入音频可能包含多人对话时，需明确标注任何“单说话人”假设。